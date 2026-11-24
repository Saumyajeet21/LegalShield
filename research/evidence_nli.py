"""Train evidence-weighted DeBERTa on ContractNLI document bags."""
import argparse
import csv
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from .datasets import LABELS, digest
from .fast_evidence import admit_context, preencode
from .diagnose_nli import read_partition
from .evidence_retrieval import Retriever, covers
from .nli import INDEX, metrics, save_checkpoint

PROTOCOL = "document-evidence-pooling-v1"
BASE_CHECKPOINT = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


@dataclass
class Bag:
    id: str
    hypothesis: str
    windows: list
    label: int
    targets: list


def bags(rows, retriever, count, supervise=False):
    result = []
    for row in rows:
        items = retriever.select(row["text"], row["claim"], row["hypothesis_id"], count)
        targets = [-1] * len(items)
        if supervise and row["evidence"]:
            for index, item in enumerate(items):
                if any(covers(item, span) for span in row["evidence"]):
                    targets[index] = 1
                elif not any(item["start"] < span["end"] and span["start"] < item["end"]
                             for span in row["evidence"]):
                    targets[index] = 0
        result.append(Bag(row["sample_id"], row["claim"], [item["text"] for item in items],
                          INDEX[row["label"]], targets))
    return result


def load_model(source, revision, gradient_checkpointing):
    backbone = AutoModelForSequenceClassification.from_pretrained(
        source, revision=revision, dtype=torch.float32, trust_remote_code=False, use_safetensors=True)
    labels = {name.lower(): int(index) for index, name in backbone.config.id2label.items()}
    order = [labels[name] for name in ("entailment", "contradiction", "neutral")]
    if backbone.classifier.out_features != 3 or sorted(order) != [0, 1, 2]:
        raise ValueError("Expected a three-label NLI checkpoint with semantic labels.")
    with torch.no_grad():
        backbone.classifier.weight.copy_(backbone.classifier.weight[order].clone())
        backbone.classifier.bias.copy_(backbone.classifier.bias[order].clone())
    backbone.config.id2label = dict(enumerate(LABELS))
    backbone.config.label2id = INDEX
    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return backbone


class EvidenceNLI(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.relevance = nn.Linear(backbone.classifier.in_features, 1)

    def forward(self, encoded):
        hidden = self.backbone.deberta(**encoded).last_hidden_state
        pooled = self.backbone.pooler(hidden)
        return self.backbone.classifier(self.backbone.dropout(pooled)), self.relevance(pooled).squeeze(-1)


def make_batcher(tokenizer, limit, device, cache=None):
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer required for overlapping premise overflow.")

    @lru_cache(maxsize=48000)
    def encode(premise, hypothesis):
        batch = tokenizer(premise, hypothesis, truncation="only_first", max_length=limit,
                          stride=64, return_overflowing_tokens=True)
        return tuple(torch.tensor(ids) for ids in batch.input_ids)

    def collate(batch):
        ids, overflow_sizes, bag_sizes, targets = [], [], [], []
        for bag in batch:
            bag_sizes.append(len(bag.windows))
            targets.extend(bag.targets)
            for window in bag.windows:
                key = (window, bag.hypothesis)
                derived = cache[key] if cache is not None else encode(*key)
                ids.extend(derived)
                overflow_sizes.append(len(derived))
        padded = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        lengths = torch.tensor([len(sequence) for sequence in ids])
        mask = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
        encoded = {"input_ids": padded.to(device), "attention_mask": mask.to(device)}
        return encoded, overflow_sizes, bag_sizes, torch.tensor(targets, device=device, dtype=torch.float32)

    return collate


def epoch(model, rows, batcher, device, batch_size, dtype, class_weights, *, optimiser=None,
          scheduler=None, scaler=None, accumulation=1, cutoff=None, benchmark=False, max_epochs=1):
    training = optimiser is not None
    model.train(training)
    probabilities, gold, total_loss = [], [], 0.0
    step_count = math.ceil(len(rows) / batch_size)
    skipped_examples = 0
    group_examples = 0
    if training:
        optimiser.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for step, start in enumerate(range(0, len(rows), batch_size)):
        if cutoff is not None and step % accumulation == 0 and time.monotonic() >= cutoff:
            break
        batch = rows[start:start + batch_size]
        labels = torch.tensor([sample.label for sample in batch], device=device)
        encoded, overflow_sizes, bag_sizes, targets = batcher(batch)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                scores, evidence = model(encoded)
            scores = torch.stack([value.mean(0) for value in scores.float().softmax(-1).split(overflow_sizes)])
            relevance = torch.stack([value.mean() for value in evidence.float().split(overflow_sizes)])
            weights = .05 + relevance.sigmoid()
            p = torch.stack([w.sum(0) / z.sum() for w, z in zip(
                (scores * weights[:, None]).split(bag_sizes), weights.split(bag_sizes), strict=True)])
            loss = nn.functional.nll_loss(p.clamp_min(1e-9).log(), labels, weight=class_weights)
            if training:
                active = targets >= 0
                relevance_loss = nn.functional.binary_cross_entropy_with_logits(
                    relevance, targets.clamp_min(0), reduction="none")
                loss = loss + .35 * (relevance_loss * active).sum() / active.sum().clamp_min(1)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError("Non-finite NLI loss. Verify the precision and checkpoint.")
            if training:
                group_examples += len(batch)
                group = min(accumulation, step_count - (step // accumulation) * accumulation)
                scaler.scale(loss / group).backward()
                if (step + 1) % accumulation == 0 or step + 1 == step_count:
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    old_scale = scaler.get_scale()
                    scaler.step(optimiser)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()
                    else:
                        skipped_examples += group_examples
                    group_examples = 0
                    optimiser.zero_grad(set_to_none=True)
        probabilities.extend(p.detach().cpu().tolist())
        gold.extend(labels.cpu().tolist())
        total_loss += value * len(batch)
        if benchmark and step + 1 == min(step_count, max(8, accumulation * 2)):
            seconds = time.perf_counter() - started
            estimate = seconds * step_count / (step + 1)
            remaining = max(0, cutoff - time.monotonic()) if cutoff is not None else None
            print(json.dumps({"benchmark_microsteps": step + 1, "seconds": seconds,
                              "estimated_full_epoch_minutes": round(estimate / 60, 1),
                              "estimated_max_training_minutes": round(estimate * max_epochs / 60, 1),
                              "estimated_session_minutes_with_30_min_overhead": round(estimate * max_epochs / 60 + 30, 1),
                              "full_epoch_likely": remaining is None or estimate - seconds <= remaining}), flush=True)
    elapsed = time.perf_counter() - started
    return (total_loss / len(gold) if gold else None), np.array(probabilities), gold, elapsed, skipped_examples


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save_predictions(path, rows, probabilities):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["sample_id", "gold", "prediction", *LABELS, "logits"])
        for row, p in zip(rows, probabilities, strict=True):
            writer.writerow([row.id, LABELS[row.label], LABELS[int(np.argmax(p))], *p,
                             json.dumps(np.log(np.maximum(p, 1e-9)).tolist())])


def setup(args):
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    dtype = None
    if device.type == "cuda":
        dtype = torch.bfloat16 if not args.fast and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    torch.manual_seed(13)
    tokenizer = AutoTokenizer.from_pretrained(args.source, revision=args.revision, trust_remote_code=False)
    model = EvidenceNLI(load_model(args.source, args.revision, args.gradient_checkpointing)).to(device)
    configuration = {
        "model": "deberta-legal", "source_model": args.source, "source_revision": args.revision,
        "dataset_sha256": digest(args.data), "evidence_track": "supervised retrieved",
        "inference_protocol": PROTOCOL, "labels": list(LABELS), "seed": 13, "smoke": args.smoke,
        "token_limit": args.max_length, "windows": args.windows, "batch_size": args.batch_size,
        "accumulation": args.accumulation, "epochs": args.epochs, "patience": args.patience,
        "learning_rate": args.lr, "class_weight_power": args.weight_power, "gradient_checkpointing": args.gradient_checkpointing,
        "device": device.type, "precision": str(dtype), "torch": str(torch.__version__),
    }
    if args.fast:
        configuration['fast'] = True
    return device, dtype, tokenizer, model, configuration, device_name


def export_model(output, model, tokenizer, retriever):
    exported = output / "model"
    model.backbone.save_pretrained(exported, safe_serialization=True)
    tokenizer.save_pretrained(exported)
    save_file({key: value.detach().cpu().contiguous() for key, value in model.relevance.state_dict().items()},
              exported / "evidence_head.safetensors")
    save_json(output / "retriever.json", retriever.artifact())
    return exported


def try_batch(model, rows, batcher, device, dtype, size):
    # Leave space for FP32 AdamW moments and the allocator at real steps.
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(device)
    baseline = torch.cuda.memory_allocated(device)
    reserve = 2 * sum(p.numel() * p.element_size() for p in model.parameters()) + 768 * 2**20
    torch.cuda.reset_peak_memory_stats(device)
    fits = False
    try:
        encoded, *_ = batcher(rows[:size])
        with torch.autocast(device_type="cuda", dtype=dtype):
            scores, relevance = model(encoded)
        (scores.float().square().mean() + relevance.float().square().mean()).backward()
        torch.cuda.synchronize(device)
        fits = torch.cuda.max_memory_allocated(device) - baseline + reserve < free
    except torch.OutOfMemoryError:
        pass
    model.zero_grad(set_to_none=True)
    return fits


def choose_fast_batch(model, rows, batcher, device, dtype, cache):
    heavy = sorted(rows, key=lambda bag: sum(ids.numel() for text in bag.windows
                   for ids in cache[(text, bag.hypothesis)]), reverse=True)[:8]
    for checkpoint in (False, True):
        if checkpoint:
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for size in (8, 4, 2, 1):
            if size <= len(heavy) and try_batch(model, heavy, batcher, device, dtype, size):
                torch.cuda.empty_cache()
                return size, checkpoint
    raise RuntimeError("A one-document FP16 batch did not fit with optimizer headroom.")


def apply_profile(args, profile):
    for key in ("batch_size", "accumulation", "windows", "max_length", "gradient_checkpointing"):
        setattr(args, key, profile[key])


def train(args):
    output = args.output
    profile_path = output / "fast-profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if args.fast and profile_path.is_file() else None
    cutoff = None
    if args.fast:
        if profile:
            apply_profile(args, profile)
        limit = min(time.time() + args.budget_seconds, args.stop_at or float("inf"))
        if profile:
            limit = min(limit, profile["ends_at"])
        cutoff = time.monotonic() + max(0, limit - time.time())
    device, dtype, tokenizer, model, config, hardware = setup(args)
    output.mkdir(parents=True, exist_ok=True)
    last = output / "last.pt"
    if args.resume != last.is_file():
        raise ValueError("Use --resume only with an existing last.pt in the same run directory.")
    cache_path = output / "token-cache.pt"
    reuse_cache = args.fast and profile is not None and cache_path.is_file()
    retriever = None
    if not reuse_cache or args.smoke:
        training = list(read_partition(args.data, "train"))
        validation = list(read_partition(args.data, "validation"))
        retriever = Retriever(training)
    if args.fast and profile is None:
        coverage = admit_context(tokenizer, retriever, validation)
        args.windows = coverage["selected"]["windows"]
        args.max_length = coverage["selected"]["max_length"]
        save_json(output / "fast-coverage.json", coverage)
        print(json.dumps({"fast_context": coverage["selected"]}), flush=True)
    if not reuse_cache:
        if args.smoke:
            training, validation = (
                [next(row for row in rows if row["label"] == label) for label in LABELS] for rows in (training, validation))
        training = bags(training, retriever, args.windows, supervise=True)
        validation = bags(validation, retriever, args.windows)
    cache = None
    if args.fast:
        identity = [config["dataset_sha256"], args.source, args.revision, args.windows, args.max_length, args.smoke]
        if cache_path.is_file():
            packed = torch.load(cache_path, map_location="cpu", weights_only=True)
            if packed["identity"] != identity:
                raise ValueError("The fast token cache does not match this dataset or context.")
            training = [Bag(**row) for row in packed["training"]]
            validation = [Bag(**row) for row in packed["validation"]]
            cache = packed["tokens"]
            del packed
        else:
            cache = preencode(tokenizer, training + validation, args.max_length)
            save_checkpoint({"identity": identity, "tokens": cache, "training": [vars(row) for row in training],
                             "validation": [vars(row) for row in validation]}, cache_path)
    weights = torch.bincount(torch.tensor([sample.label for sample in training]), minlength=3).float()
    weights = weights.pow(-args.weight_power)
    weights = (weights / weights.mean()).to(device)
    batcher = make_batcher(tokenizer, args.max_length, device, cache)
    if args.fast and profile is None:
        args.batch_size, args.gradient_checkpointing = choose_fast_batch(model, training, batcher, device, dtype, cache)
        args.accumulation = max(1, 16 // args.batch_size)
        profile = {"batch_size": args.batch_size, "accumulation": args.accumulation, "windows": args.windows,
                   "max_length": args.max_length, "gradient_checkpointing": args.gradient_checkpointing,
                   "ends_at": limit}
        save_json(profile_path, profile)
        print(json.dumps({"fast_gpu_profile": profile, "training_examples": len(training)}), flush=True)
    if args.fast:
        config.update(batch_size=args.batch_size, accumulation=args.accumulation, windows=args.windows,
                      token_limit=args.max_length, gradient_checkpointing=args.gradient_checkpointing)
        torch.manual_seed(13)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01,
                                  **({"fused": True} if args.fast else {}))
    steps = math.ceil(len(training) / (args.batch_size * args.accumulation)) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimiser, int(.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    start, score, stale, history, fit = 1, -1., 0, [], 0.
    best_path = None
    if args.resume:
        state = torch.load(last, weights_only=True, map_location="cpu")
        if state["config"] != config:
            raise ValueError("Resume requires the same dataset, checkpoint and training configuration.")
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, score, stale, history, fit = state["epoch"]+1, state["best"], state["stale"], state["history"], state["fit"]
        best_path = state["best_path"]
        torch.set_rng_state(state["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["gpu_rng"], device)
        del state
    reserve = 0
    if args.fast and (not history or start <= args.epochs):
        probe = validation[:min(len(validation), 16)]
        _, _, _, seconds, _ = epoch(model, probe, batcher, device, args.batch_size, dtype, None)
        reserve = 1.6 * seconds * len(validation) / len(probe) + (180 if "/content/drive/" in str(output) else 90)
        if args.smoke:
            reserve = 0
        print(json.dumps({"validation_and_checkpoint_reserve_seconds": round(reserve),
                          "remaining_training_phase_seconds": round(max(0, cutoff - time.monotonic()))}), flush=True)
    for step in range(start, args.epochs + 1):
        if stale >= args.patience:
            break
        shuffled = random.Random(13 + step).sample(training, len(training))
        loss, pred, gold, seconds, skipped = epoch(model, shuffled, batcher, device, args.batch_size, dtype, weights,
            optimiser=optimiser, scheduler=scheduler, scaler=scaler, accumulation=args.accumulation,
            cutoff=cutoff - reserve if cutoff is not None else None,
            benchmark=args.fast and not history, max_epochs=args.epochs)
        if not gold:
            break
        validation_loss, p, y, val_seconds, _ = epoch(model, validation, batcher, device, args.batch_size, dtype, None)
        fit += seconds + val_seconds
        measured = metrics(y, p)
        current = measured["macro"]["f1"]
        row = {"epoch": step, "train_loss": loss, "train_accuracy": float((pred.argmax(1) == gold).mean()),
               "validation_loss": validation_loss, "validation_accuracy": measured["accuracy"],
               "validation_macro_f1": current, "learning_rate": optimiser.param_groups[0]["lr"], "fit_seconds": fit,
               "training_examples_processed": len(gold), "eligible_training_examples": len(training),
               "fp16_skipped_examples": skipped, "partial_epoch": len(gold) < len(training)}
        history.append(row)
        print(json.dumps(row), flush=True)
        previous_best = best_path
        if current > score:
            score, stale, best_path = current, 0, f"best-epoch-{step}.pt"
            save_checkpoint({"model": model.state_dict(), "config": config, "metrics": measured, "epoch": step},
                            output / best_path)
        else:
            stale += 1
        save_checkpoint({"config": config, "model": model.state_dict(), "optimiser": optimiser.state_dict(),
                         "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                         "rng": torch.get_rng_state(), "gpu_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                         "epoch": step, "history": history, "best": score, "best_path": best_path,
                         "stale": stale, "fit": fit}, last)
        if previous_best and previous_best != best_path:
            (output / previous_best).unlink(missing_ok=True)
        save_json(output / "history.json", history)
        if row["partial_epoch"]:
            break
    if not best_path:
        raise RuntimeError("Time budget expired before a train step could be validated and saved.")
    winner = torch.load(output / best_path, weights_only=True, map_location="cpu")
    save_json(output / "history.json", history)
    if args.smoke:
        model.load_state_dict(winner["model"])
        export_model(output, model, tokenizer, retriever)
    for entry in history:
        entry.setdefault("training_examples_processed", len(training))  # Legacy full runs completed each saved epoch.
        entry.setdefault("partial_epoch", False)
    through_winner = history[:winner["epoch"]]
    training_coverage = {
        "eligible_pairs": len(training),
        "largest_epoch_pairs_processed": max(row["training_examples_processed"] for row in history),
        "largest_epoch_pairs_processed_through_winner": max(row["training_examples_processed"] for row in through_winner),
        "total_pairs_processed": sum(row["training_examples_processed"] for row in history),
        "fp16_skipped_examples": (sum(row["fp16_skipped_examples"] for row in history)
                                  if all("fp16_skipped_examples" in row for row in history) else None),
        "completed_epochs": sum(not row["partial_epoch"] for row in history),
        "partial_epoch": history[-1]["partial_epoch"],
    }
    print(json.dumps({"training_coverage": training_coverage}), flush=True)
    save_json(output / "completed.json", {"config": config, "checkpoint": best_path, "gpu": hardware,
              "best_epoch": winner["epoch"], "validation": winner["metrics"], "fit_seconds": fit,
              "training_coverage": training_coverage})


def finalize(args):
    if args.smoke:
        raise ValueError("Validation-only smoke trials never evaluate official test.")
    trial = args.trial
    outputs = json.loads((trial / "completed.json").read_text(encoding="utf-8"))
    if args.fast:
        apply_profile(args, json.loads((trial / "fast-profile.json").read_text(encoding="utf-8")))
    device, dtype, tokenizer, model, config, hardware = setup(args)
    if config != outputs["config"]:
        raise ValueError("Final evaluation requires the frozen winning configuration.")
    checkpoint = torch.load(trial / outputs["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint["config"] != config:
        raise ValueError("Winner checkpoint configuration mismatch.")
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    retriever = Retriever(list(read_partition(args.data, "train")))
    batcher = make_batcher(tokenizer, args.max_length, device)
    output = args.output
    if (output / "metrics.json").exists():
        raise ValueError("Final test results already exist for this output.")
    output.mkdir(parents=True, exist_ok=True)
    held_out = {}
    for name in ("calibration", "test"):
        examples = list(read_partition(args.data, name))
        if name == "test" and (len(examples) != 2091 or len({row["sample_id"] for row in examples}) != 2091):
            raise ValueError("Final evaluation requires all 2,091 official test IDs.")
        selected = bags(examples, retriever, args.windows)
        if args.fast:
            batcher = make_batcher(tokenizer, args.max_length, device, preencode(tokenizer, selected, args.max_length))
        loss, p, y, seconds, _ = epoch(model, selected, batcher, device, args.batch_size, dtype, None)
        save_predictions(output / f"predictions-{name}.csv", selected, p)
        held_out[name] = {**metrics(y, p), "nll": loss, "inference_seconds": seconds}
    exported = export_model(output, model, tokenizer, retriever)
    shutil.copyfile(trial / "history.json", output / "history.json")
    save_json(output / "run.json", {**config, "gpu": hardware, "fit_seconds": outputs["fit_seconds"],
              "best_epoch": outputs["best_epoch"], "validation": outputs["validation"],
              "training_coverage": outputs.get("training_coverage"),
              "checkpoint_sha256": digest(trial / outputs["checkpoint"]),
              "model_weights_sha256": digest(exported / "model.safetensors"),
              "evidence_head_sha256": digest(exported / "evidence_head.safetensors"),
              "retriever_sha256": digest(output / "retriever.json"),
              "mean_test_ms_per_example": held_out["test"]["inference_seconds"] * 1000 / 2091})
    save_json(output / "metrics.json", held_out["test"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "finalize"))
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trial", type=Path)
    parser.add_argument("--source", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    parser.add_argument("--revision", default="6f5cf0a2b59cabb106aca4c287eed12e357e90eb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-power", type=float, choices=(0., .5), default=.5)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--budget-seconds", type=int, default=5400)
    parser.add_argument("--stop-at", type=float, help="Absolute UTC epoch-seconds deadline")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.accumulation, args.windows) < 1 or args.max_length < 128:
        parser.error("Invalid training or context settings.")
    if args.fast and (not 0 < args.budget_seconds <= 5400 or args.epochs > 3 or
                      not args.device.startswith("cuda") or args.gradient_checkpointing or
                      (args.source != BASE_CHECKPOINT and not args.smoke)):
        parser.error("FAST requires CUDA, base NLI, a budget up to 5400 seconds, at most three epochs and automatic checkpointing.")
    if args.stage == "finalize" and args.trial is None:
        parser.error("Specify the frozen winning --trial directory.")
    {"train": train, "finalize": finalize}[args.stage](args)
