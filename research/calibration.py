"""Fit temperature and decision thresholds on grouped calibration predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score

from .datasets import LABELS, digest


def quality(p, y, bins=15):
    confidence = p.max(axis=1)
    prediction = p.argmax(axis=1)
    correct = prediction == y
    edges = np.linspace(0, 1, bins + 1)
    reliability = []
    for i in range(bins):
        mask = (confidence >= edges[i]) & (confidence < edges[i + 1] if i + 1 < bins else confidence <= 1)
        if mask.any():
            reliability.append({"count": int(mask.sum()), "accuracy": float(correct[mask].mean()),
                                "confidence": float(confidence[mask].mean())})
    order = np.argsort(-confidence, kind="stable")
    counts = np.arange(1, len(y) + 1)
    risks = np.cumsum(~correct[order]) / counts
    return {
        "brier": float(np.mean(np.sum((p - np.eye(len(LABELS))[y]) ** 2, axis=1))),
        "nll": float(-np.log(p[np.arange(len(y)), y].clip(min=1e-15)).mean()),
        "ece": float(sum(item["count"] * abs(item["accuracy"] - item["confidence"]) for item in reliability) / len(y)),
        "reliability": reliability,
        "risk_coverage": [{"risk": float(risk), "coverage": float(count / len(y))}
                          for count, risk in zip(counts, risks, strict=True)],
        "aurc": float(risks.mean()),
        "incorrect_auroc": float(roc_auc_score(~correct, 1-confidence)) if 0 < correct.sum() < len(y) else None,
    }


def calibrate(logits, y):
    if len(y) < 3 or set(y) != set(range(len(LABELS))):
        raise ValueError("Calibration requires observations for every class.")
    def objective(log_temperature):
        z = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y]))
    result = minimize_scalar(objective, bounds=(-5, 5), method="bounded")
    if not result.success:
        raise ValueError("Temperature optimization failed.")
    return float(np.exp(result.x))


def threshold_for_precision(p, labels, index, target, minimum_support=5):
    scores = p[:, index]
    for cutoff in sorted(set(scores[scores >= .5])):
        chosen = scores >= cutoff
        if int(chosen.sum()) >= minimum_support and (labels[chosen] == index).mean() >= target:
            return float(min(cutoff, 1. - 1e-9))
    return None


def load_predictions(path):
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records or any(not row["logits"] for row in records):
        raise ValueError("A nonempty CSV of neural logits is required.")
    ids = {row["sample_id"] for row in records}
    if len(ids) != len(records):
        raise ValueError("Duplicate sample IDs.")
    y = np.asarray([LABELS.index(row["gold"]) for row in records])
    z = np.asarray([json.loads(row["logits"]) for row in records])
    if z.shape != (len(y), len(LABELS)) or not np.isfinite(z).all():
        raise ValueError("Invalid logits.")
    return ids, y, z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--target-precision", type=float, default=.9)
    args = parser.parse_args()
    if not 0 < args.target_precision <= 1:
        parser.error("Target precision must be between 0 and 1.")
    ids, y, z = load_predictions(args.calibration)
    temperature = calibrate(z, y)
    p = softmax(z / temperature, axis=1)
    entailment = threshold_for_precision(p, y, 0, args.target_precision)
    contradiction = threshold_for_precision(p, y, 1, args.target_precision)
    run_path = args.calibration.parent / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    weights = run.get("model_weights_sha256")
    data = {
        "model_weights_sha256": weights, "dataset_sha256": run.get("dataset_sha256"),
        "inference_protocol": run.get("inference_protocol", "pair"),
        "max_length": run.get("token_limit", 512),
        "temperature": temperature, "target_precision": args.target_precision,
        "entailment_threshold": entailment, "contradiction_threshold": contradiction,
        "deployable": (bool(weights) and run.get("smoke") is False and entailment is not None
                       and contradiction is not None and run.get("inference_protocol", "pair") == "pair"),
        "calibration_sha256": digest(args.calibration),
        "calibration": {"before": quality(softmax(z, axis=1), y), "after": quality(p, y)},
    }
    if args.test:
        test_ids, test_y, test_z = load_predictions(args.test)
        if ids & test_ids:
            raise ValueError("Calibration and test share sample IDs.")
        data["test_sha256"] = digest(args.test)
        data["test"] = {"before": quality(softmax(test_z, axis=1), test_y),
                        "after": quality(softmax(test_z / temperature, axis=1), test_y)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, allow_nan=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
