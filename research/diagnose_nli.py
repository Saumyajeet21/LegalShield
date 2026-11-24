"""Audit frozen NLI ordering, token coverage and retrieval on development data."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .datasets import LABELS
from .evidence_retrieval import Retriever, coverage_report, original_select
from .nli import INDEX, metrics


def read_partition(path, split):
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["partition"] == split:
                yield row


def diagnose(path, model_path, device):
    validation = list(read_partition(path, "validation"))
    training = list(read_partition(path, "train"))
    report = coverage_report(training, validation, Retriever(training))
    rng = random.Random(13)
    chosen = [row for label in LABELS for row in rng.sample(
        [item for item in validation if item["label"] == label], 25)]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True).to(device).eval()
    native = {label.lower(): index for index, label in model.config.id2label.items()}
    order = [native[key] for key in ("entailment", "contradiction", "neutral")]
    report.update(native_labels=model.config.id2label, canonical_logit_order=order,
                  frozen_validation_sample=len(chosen), frozen_sample_design="25 per class, seed 13")
    parts, overflow, sequences = 0, 0, 0
    variants = {"correct_order_mean_logits": [], "correct_order_top1": [], "reversed_order_mean_logits": []}
    y = []
    with torch.inference_mode():
        for row in chosen:
            windows = [part["text"] for part in original_select(row)]
            h = [row["claim"]] * len(windows)
            y.append(INDEX[row["label"]])
            for reversed_order in (False, True):
                a, b = (h, windows) if reversed_order else (windows, h)
                batch = tokenizer(a, b, max_length=512, truncation="only_second" if reversed_order else "only_first",
                                  stride=24, return_overflowing_tokens=True, padding=True, return_tensors="pt")
                origin = batch.pop("overflow_to_sample_mapping")
                if not reversed_order:
                    parts += len(windows)
                    overflow += int((origin.bincount() > 1).sum())
                    sequences += len(origin)
                    expected = tokenizer(h[0], add_special_tokens=False)["input_ids"]
                    for idx in range(len(origin)):
                        kept = [int(token) for token, kind in zip(batch.input_ids[idx], batch.sequence_ids(idx), strict=True) if kind == 1]
                        if kept != expected:
                            raise ValueError("The complete hypothesis was not preserved as sequence 2.")
                logits = model(**batch.to(device)).logits[:, order].float().cpu()
                per_window = torch.stack([logits[origin == i].mean(0) for i in range(len(windows))])
                if reversed_order:
                    variants["reversed_order_mean_logits"].append(per_window.mean(0).softmax(0).numpy())
                else:
                    variants["correct_order_mean_logits"].append(per_window.mean(0).softmax(0).numpy())
                    variants["correct_order_top1"].append(per_window[0].softmax(0).numpy())
    report["tokenization"] = {"lexical_windows": parts, "windows_with_tokenizer_overflow": overflow,
                             "encoded_pairs": sequences, "hypothesis_tokens_verified": True}
    report["frozen_validation"] = {key: metrics(y, np.array(p)) for key, p in variants.items()}
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    result = diagnose(args.data, args.model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("original_lexical_top4", "learned_top5", "tokenization")}))
    print({key: (item["accuracy"], item["macro"]["f1"]) for key, item in result["frozen_validation"].items()})
