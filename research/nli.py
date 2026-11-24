"""Train and evaluate document-level ContractNLI classifiers."""
import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from .datasets import LABELS, digest, read_jsonl, tokens, windows

INDEX = dict(zip(LABELS, range(len(LABELS)), strict=True))


@lru_cache(maxsize=768)
def lexical_windows(text):
    return [(part["text"], set(tokens(part["text"]))) for part in windows(text)]


def contexts(row, track, limit):
    if track == "oracle" and row["evidence"]:
        evidence = "\n".join(span["text"] for span in row["evidence"])
        return [part["text"] for part in windows(evidence)]
    parts = lexical_windows(row["text"])
    claim = set(tokens(row["claim"]))
    return [part[0] for part in sorted(parts, key=lambda part: -len(claim & part[1]))[:limit]] or [""]


def metrics(y, probabilities):
    p = np.asarray(probabilities)
    pred = p.argmax(axis=1)
    labels = list(range(len(LABELS)))
    matrix = confusion_matrix(y, pred, labels=labels)
    report = {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
              "mcc": float(matthews_corrcoef(y, pred)), "confusion_matrix": matrix.tolist(), "count": len(y)}
    for average in ("macro", "micro", "weighted", None):
        precision, recall, f1, support = precision_recall_fscore_support(
            y, pred, labels=labels, average=average, zero_division=0)
        if average is not None:
            report[average] = dict(precision=float(precision), recall=float(recall), f1=float(f1))
            continue
        report["classes"] = {}
        for i, label in enumerate(LABELS):
            actual = np.asarray(y) == i
            positives, negatives = int(actual.sum()), int((~actual).sum())
            fp = int(matrix[:, i].sum() - matrix[i, i])
            pr = precision_recall_curve(actual, p[:, i]) if positives and negatives else None
            report["classes"][label] = {
                "support": int(support[i]), "precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "specificity": (negatives - fp) / negatives if negatives else None,
                "roc_auc": float(roc_auc_score(actual, p[:, i])) if positives and negatives else None,
                "pr_auc": float(auc(pr[1], pr[0])) if pr else None,
                "average_precision": float(average_precision_score(actual, p[:, i])) if positives and negatives else None,
            }
    return report


class BiLSTM(nn.Module):
    def __init__(self, vocabulary, embedding=128, hidden=128, dropout=.3):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, embedding, padding_idx=0)
        self.encoder = nn.LSTM(embedding, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hidden * 8, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 3))

    def encode(self, sequences):
        padded = pad_sequence(sequences, batch_first=True).to(self.embedding.weight.device)
        packed = pack_padded_sequence(self.embedding(padded), [len(s) for s in sequences],
                                      batch_first=True, enforce_sorted=False)
        _, (state, _) = self.encoder(packed)
        return torch.cat((state[0], state[1]), dim=-1)

    def forward(self, evidence, claims):
        a, b = self.encode(evidence), self.encode(claims)
        return self.head(torch.cat((a, b, (a-b).abs(), a*b), dim=-1))


def prepare(rows, track, count):
    return [(contexts(row, track, count), row["claim"], INDEX[row["label"]], row["sample_id"]) for row in rows]


def classify_batches(model, samples, collate, device, batch_size, optimizer=None,
                     accumulation=1, scheduler=None, scaler=None, amp_dtype=None, latency=None):
    training = optimizer is not None
    model.train(training)
    predictions, gold, logits_record = [], [], []
    total_loss = 0.0
    starts = list(range(0, len(samples), batch_size))
    if training:
        optimizer.zero_grad()
    for i, start in enumerate(starts):
        batch = samples[start:start + batch_size]
        premises, claims, sizes, labels, _ = [], [], [], [], []
        for spans, claim, label, _ in batch:
            premises.extend(spans)
            claims.extend([claim] * len(spans))
            sizes.append(len(spans))
            labels.append(label)
        y = torch.tensor(labels, device=device)
        if latency is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_start = time.perf_counter()
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_dtype is not None, dtype=amp_dtype):
                raw = collate(model, premises, claims)
                logits = torch.stack([chunk.float().mean(dim=0) for chunk in raw.split(sizes)])
                loss = nn.functional.cross_entropy(logits, y)
            if latency is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latency.append((len(batch), time.perf_counter() - batch_start))
            batch_loss = float(loss.detach())
            if not math.isfinite(batch_loss):
                raise FloatingPointError("Non-finite NLI loss. Check the checkpoint and try FP32 training.")
            if training:
                group_size = min(accumulation, len(starts) - (i // accumulation) * accumulation)
                scaler.scale(loss / group_size).backward()
                if (i + 1) % accumulation == 0 or i + 1 == len(starts):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
        total_loss += batch_loss * len(batch)
        predictions.extend(logits.detach().softmax(-1).cpu().tolist())
        logits_record.extend(logits.detach().cpu().tolist())
        gold.extend(labels)
    return total_loss / len(samples), predictions, gold, logits_record


def save_checkpoint(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    resume_path = args.output / "last.pt"
    if args.resume and (args.model not in ("bilstm", "deberta") or not resume_path.is_file()):
        raise ValueError("--resume requires last.pt from this training run.")
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new output directory for every run or use --resume.")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.dataset)
    subsets = {key: prepare([row for row in rows if row["partition"] == key], args.track, args.windows)
               for key in ("train", "validation", "calibration", "test")}
    if args.smoke:
        for key in subsets:
            subsets[key] = [row for k in range(3) for row in [x for x in subsets[key] if x[2] == k][:3]]
    train, valid, test = (subsets[key] for key in ("train", "validation", "test"))
    if not train or not valid or not test:
        raise ValueError("Supply nonempty train, validation and test partitions.")
    if len({row[2] for row in train}) != len(LABELS):
        raise ValueError("Training requires support for all three labels.")
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.perf_counter()
    histories, inference_times = [], []
    fit_seconds = None
    extra_predictions = {}
    metadata = {"model": args.model, "seed": args.seed, "evidence_track": args.track,
                "dataset_sha256": digest(args.dataset), "labels": list(LABELS), "smoke": args.smoke,
                "device": str(device), "windows": args.windows, "source_model": args.source_model,
                "source_revision": args.revision, "batch_size": args.batch_size, "epochs": args.epochs,
                "accumulation": args.accumulation, "token_limit": args.max_length,
                "precision": args.precision, "gradient_checkpointing": args.gradient_checkpointing,
                "patience": args.patience, "split_counts": {key: len(value) for key, value in subsets.items()},
                "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    if args.model == "tfidf":
        import joblib
        features = ColumnTransformer([
            ("evidence", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), 0),
            ("claim", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), 1),
        ])
        model = make_pipeline(features, LogisticRegression(max_iter=1000, random_state=args.seed))
        def design(samples):
            return np.asarray([["\n".join(spans), claim] for spans, claim, _, _ in samples], dtype=object)
        fit_start = time.perf_counter()
        model.fit(design(train), [item[2] for item in train])
        fit_seconds = time.perf_counter() - fit_start
        for key, sample in subsets.items():
            if key in ("validation", "calibration", "test") and sample:
                probabilities = []
                for offset in range(0, len(sample), args.batch_size):
                    batch = sample[offset:offset + args.batch_size]
                    timer = time.perf_counter()
                    probabilities.extend(model.predict_proba(design(batch)).tolist())
                    if key == "test":
                        inference_times.append((len(batch), time.perf_counter() - timer))
                extra_predictions[key] = (probabilities, [], [item[2] for item in sample])
        joblib.dump(model, args.output / "tfidf.joblib")
    else:
        tokenizer = None
        if args.model == "bilstm":
            vocabulary = Counter(word for spans, claim, _, _ in train for text in [*spans, claim] for word in tokens(text))
            vocab = {word: i for i, (word, _) in enumerate(vocabulary.most_common(30000), start=2)}
            model = BiLSTM(len(vocab) + 2)
            @lru_cache(maxsize=36000)
            def encode(text):
                ids = [vocab.get(word, 1) for word in tokens(text)[:args.max_length]]
                return torch.tensor(ids or [1])

            def collate(model, premises, claims):
                return model([encode(text) for text in premises], [encode(text) for text in claims])
            (args.output / "vocabulary.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            checkpoint = args.source_model or ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
                                                if args.model == "general" else "microsoft/deberta-v3-base")
            metadata["source_model"] = checkpoint
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=args.revision, trust_remote_code=False)
            overrides = {} if args.model == "general" else {"num_labels": 3, "id2label": dict(enumerate(LABELS)), "label2id": INDEX}
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint, revision=args.revision, trust_remote_code=False,
                **({"use_safetensors": True} if args.model == "general" else {"dtype": torch.float32}), **overrides)
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            order = [0, 1, 2]
            if args.model == "general":
                lookup = {value.lower(): key for key, value in model.config.id2label.items()}
                order = [lookup["entailment"], lookup["contradiction"], lookup["neutral"]]
            def collate(model, premises, claims):
                batch = tokenizer(premises, claims, truncation="only_first", padding=True,
                                  max_length=args.max_length, return_overflowing_tokens=True,
                                  stride=min(24, args.max_length // 8), return_tensors="pt")
                origin = batch.pop("overflow_to_sample_mapping").to(device)
                logits = model(**batch.to(device)).logits[:, order]
                return torch.stack([logits[origin == i].mean(dim=0) for i in range(len(premises))])
        model.to(device)
        precision = args.precision
        if args.mixed_precision and precision == "fp32" and device.type == "cuda":
            precision = "auto"
        if precision != "fp32" and device.type != "cuda":
            raise ValueError("Choose fp32 on CPU.")
        if precision == "auto":
            precision = "bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16"
        if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("The selected CUDA device does not support BF16.")
        amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        metadata["precision"] = precision
        if args.model != "general":
            lr = args.learning_rate if args.learning_rate is not None else .001 if args.model == "bilstm" else 2e-5
            metadata["learning_rate"] = lr
            optimizer = (torch.optim.Adam(model.parameters(), lr=lr) if args.model == "bilstm"
                         else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01))
            scheduler = None
            if args.model == "deberta":
                from transformers import get_linear_schedule_with_warmup
                steps = args.epochs * math.ceil(math.ceil(len(train) / args.batch_size) / args.accumulation)
                scheduler = get_linear_schedule_with_warmup(optimizer, int(.1 * steps), steps)
            scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
            best, stalled, first_epoch = -1., 0, 1
            fit_seconds = 0.
            if args.resume:
                state = torch.load(resume_path, map_location="cpu", weights_only=True)
                if state["configuration"] != metadata or not (args.output / "best.pt").is_file():
                    raise ValueError("Resume configuration does not match the checkpoint.")
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                if scheduler is not None:
                    scheduler.load_state_dict(state["scheduler"])
                scaler.load_state_dict(state["scaler"])
                histories, best, stalled = state["history"], state["best"], state["stalled"]
                fit_seconds, first_epoch = state["fit_seconds"], state["epoch"] + 1
                torch.set_rng_state(state["cpu_rng"])
                if device.type == "cuda":
                    torch.cuda.set_rng_state(state["gpu_rng"], device)
                del state
            for epoch in range(first_epoch, args.epochs + 1):
                if stalled >= args.patience:
                    break
                epoch_start = time.perf_counter()
                train_order = train.copy()
                random.Random(args.seed + epoch).shuffle(train_order)
                tl, tp, ty, _ = classify_batches(model, train_order, collate, device, args.batch_size, optimizer,
                                                 args.accumulation, scheduler, scaler, amp_dtype)
                vl, vp, vy, _ = classify_batches(model, valid, collate, device, args.batch_size, amp_dtype=amp_dtype)
                validation = metrics(vy, vp)
                key = validation["macro"]["f1"]
                histories.append({"epoch": epoch, "train_loss": tl, "validation_loss": vl,
                                  "train_accuracy": metrics(ty, tp)["accuracy"], "validation_accuracy": validation["accuracy"],
                                  "validation_macro_f1": key, "learning_rate": optimizer.param_groups[0]["lr"]})
                fit_seconds += time.perf_counter() - epoch_start
                if key > best:
                    best, stalled = key, 0
                    save_checkpoint(model.state_dict(), args.output / "best.pt")
                else:
                    stalled += 1
                print(json.dumps(histories[-1]), flush=True)
                save_checkpoint({
                    "configuration": metadata, "epoch": epoch, "history": histories, "best": best,
                    "stalled": stalled, "fit_seconds": fit_seconds, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
                    "scaler": scaler.state_dict(), "cpu_rng": torch.get_rng_state(),
                    "gpu_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                }, resume_path)
            checkpoint_path = args.output / "best.pt"
            metadata["checkpoint_sha256"] = digest(checkpoint_path)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            if tokenizer is not None:
                export = args.output / "model"
                model.save_pretrained(export, safe_serialization=True)
                tokenizer.save_pretrained(export)
                metadata["model_weights_sha256"] = digest(export / "model.safetensors")
        for key, sample in subsets.items():
            if key in ("validation", "calibration", "test") and sample:
                _, probabilities, y, logits = classify_batches(
                    model, sample, collate, device, args.batch_size, amp_dtype=amp_dtype,
                    latency=inference_times if key == "test" else None)
                extra_predictions[key] = (probabilities, logits, y)
    for key, (probabilities, logits, y) in extra_predictions.items():
        with (args.output / (f"predictions-{key}.csv")).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "gold", "prediction", *LABELS, "logits"])
            for i, (sample, p) in enumerate(zip(subsets[key], probabilities, strict=True)):
                writer.writerow([sample[3], LABELS[y[i]], LABELS[int(np.argmax(p))], *p,
                                 json.dumps(logits[i]) if logits else ""])
    probabilities, _, y = extra_predictions["test"]
    metadata["elapsed_seconds"] = time.perf_counter() - start_time
    metadata["fit_seconds"] = fit_seconds
    metadata["mean_test_ms_per_example"] = 1000 * sum(seconds for _, seconds in inference_times) / len(test)
    metadata["test_batch_latency_p95_ms"] = float(np.percentile([seconds for _, seconds in inference_times], 95) * 1000)
    for filename, data in (("run.json", metadata), ("metrics.json", metrics(y, probabilities)), ("history.json", histories)):
        (args.output / filename).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("tfidf", "bilstm", "general", "deberta"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--track", choices=("retrieved", "oracle"), default="retrieved")
    parser.add_argument("--source-model")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16", "auto"), default="fp32")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.batch_size = args.batch_size if args.batch_size is not None else 32 if args.model == "bilstm" else 4
    args.epochs = args.epochs if args.epochs is not None else 20 if args.model == "bilstm" else 4
    if min(args.batch_size, args.epochs, args.patience, args.windows, args.max_length, args.accumulation) < 1:
        parser.error("Training limits must be positive.")
    run(args)


if __name__ == "__main__":
    main()
