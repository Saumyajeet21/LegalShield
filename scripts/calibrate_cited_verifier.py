"""Calibrate the deployed cited-passage protocol on independently reviewed cases."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from legalshield.verification import CITED_PROTOCOL, DocumentVerifier
from research.calibration import calibrate, quality, threshold_for_precision
from research.datasets import LABELS, digest, read_jsonl


def predict(verifier, rows):
    if (not rows or any(row.get("human_review_status") != "approved" for row in rows)
        or len({row["sample_id"] for row in rows}) != len(rows)
        or min(sum(row["label"] == label for row in rows) for label in LABELS) < 10):
        raise ValueError("Use independently reviewed, unique cases with at least ten examples per label.")
    probs = []
    for row in rows:
        p = verifier.classify_passages(row["passages"], row["claim"])
        if p is None:
            raise ValueError(f"Cited set exceeds the serving protocol: {row['sample_id']}")
        probs.append([p[label] for label in LABELS])
    return np.log(np.maximum(probs, 1e-9)), np.array([LABELS.index(row["label"]) for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("calibration_cases", type=Path)
    parser.add_argument("validation_cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    left, right = read_jsonl(args.calibration_cases), read_jsonl(args.validation_cases)
    if ({row["family_id"] for row in left} & {row["family_id"] for row in right}
        or {row["sample_id"] for row in left} & {row["sample_id"] for row in right}):
        raise ValueError("Calibration and staging validation must use disjoint document families and case IDs.")
    verifier = DocumentVerifier.load_research(args.model_dir, args.model_dir.parent / "calibration.json")
    verifier.temperature = 1.0
    logits, gold = predict(verifier, left)
    vz, vy = predict(verifier, right)
    temperature = calibrate(logits, gold)
    train_p, val_p = softmax(logits / temperature, axis=1), softmax(vz / temperature, axis=1)
    thresholds = [threshold_for_precision(train_p, gold, i, .9, minimum_support=10) for i in (0, 1)]
    masks = [val_p[:, i] >= threshold if threshold is not None else np.zeros(len(vy), dtype=bool)
             for i, threshold in enumerate(thresholds)]
    # The deployed policy checks contradiction before entailment.
    masks[0] &= ~masks[1]
    outcomes = {}
    for i, mask in enumerate(masks):
        outcomes[LABELS[i]] = {"selected": int(mask.sum()),
                              "precision": float((vy[mask] == i).mean()) if mask.any() else None}
    data = {
        "inference_protocol": CITED_PROTOCOL, "source_calibration": "reviewed_cited_evidence",
        "dataset_sha256": verifier.dataset_sha256,
        "model_weights_sha256": digest(args.model_dir / "model.safetensors"),
        "evidence_head_sha256": digest(args.model_dir / "evidence_head.safetensors"),
        "calibration_sha256": digest(args.calibration_cases), "validation_sha256": digest(args.validation_cases),
        "max_length": verifier.max_length, "temperature": temperature,
        "entailment_threshold": thresholds[0], "contradiction_threshold": thresholds[1],
        "validation_decisions": outcomes, "validation": quality(val_p, vy),
        "deployable": all(v["selected"] >= 10 and v["precision"] >= .9 for v in outcomes.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    print(json.dumps({"deployable": data["deployable"], "validation_decisions": outcomes}))


if __name__ == "__main__":
    main()
