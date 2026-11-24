"""Score independently annotated legal observations and grouped paired intervals."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .datasets import read_jsonl


def checked_rows(predictions, gold, allowed_status):
    reference = {row["sample_id"]: row for row in gold if row["human_review_status"] in allowed_status}
    if len(reference) != sum(row["human_review_status"] in allowed_status for row in gold):
        raise ValueError("Duplicate reviewed reference IDs.")
    records = set()
    for pred in predictions:
        key = (pred["system"], pred["sample_id"])
        if key in records:
            raise ValueError("Duplicate output for a system and sample.")
        records.add(key)
        row = reference.get(pred["sample_id"])
        if row is None:
            continue
        outcome = pred["action"]
        answerable = row["expected_answerability"]
        correct = (outcome == "answer" and type(pred.get("answer")) is type(row["expected_answer"])
                   and pred["answer"] == row["expected_answer"]) if answerable else outcome == row["expected_system_action"]
        yield {"system": pred["system"], "family": row["scenario_family_id"], "type": row["modification_type"],
               "correct": correct, "answered": outcome == "answer", "answerable": answerable,
               "latency_ms": pred.get("latency_ms"), "calls": pred.get("calls")}


def score(records):
    buckets = defaultdict(dict)
    for row in records:
        if row["type"] in buckets[row["family"]]:
            raise ValueError("Expected one reference per family and modification type.")
        buckets[row["family"]][row["type"]] = row
    changes, invariance = [], []
    for cases in buckets.values():
        original = cases.get("legitimate")
        if original:
            for key, item in cases.items():
                if key == "authorized_amendment":
                    changes.append(original["correct"] and item["correct"] and original["answered"] and item["answered"])
                if key in {"unauthorized_modification", "irrelevant", "inert_control"}:
                    invariance.append(original["correct"] and item["correct"] and original["answered"] and item["answered"])
    cs, inv = (mean(results) if results else None for results in (changes, invariance))
    version = (2 * cs * inv / (cs + inv) if cs + inv else 0.) if cs is not None and inv is not None else None
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    timings = [row["latency_ms"] for row in records if row["latency_ms"] is not None]
    calls = [row["calls"] for row in records if row["calls"] is not None]
    return {
        "accuracy": mean(r["correct"] for r in records), "coverage": mean(r["answered"] for r in records),
        "answerable_coverage": mean(r["answered"] for r in answerable) if answerable else None,
        "appropriate_abstention": mean(r["correct"] for r in unanswerable) if unanswerable else None,
        "change_sensitivity": cs, "invariance_score": inv, "version_robustness_score": version,
        "change_pairs": len(changes), "invariance_pairs": len(invariance), "instances": len(records),
        "families": len(buckets), "latency_ms_mean": mean(timings) if timings else None,
        "completion_calls": sum(calls) if calls else None,
    }


def paired_interval(a, b, seed=13, repeats=2000):
    by_system = []
    for records in (a, b):
        groups = defaultdict(dict)
        for row in records:
            groups[row["family"]][row["type"]] = row["correct"]
        by_system.append(groups)
    left, right = by_system
    differences = []
    for family in sorted(left.keys() & right.keys()):
        common = left[family].keys() & right[family].keys()
        if common:
            differences.append(mean(int(left[family][key]) - int(right[family][key]) for key in common))
    if len(differences) < 2:
        return None
    rng = random.Random(seed)
    values = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(repeats))
    return {"families": len(differences), "difference": mean(differences),
            "low": values[int(.025 * repeats)], "high": values[min(repeats - 1, int(.975 * repeats))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-unreviewed-smoke", action="store_true")
    args = parser.parse_args()
    allowed = {"approved"} | ({"pending"} if args.allow_unreviewed_smoke else set())
    by_system = defaultdict(list)
    for row in checked_rows(read_jsonl(args.predictions), read_jsonl(args.gold), allowed):
        by_system[row["system"]].append(row)
    if not by_system:
        raise ValueError("No eligible reviewed outcomes. Use smoke mode only for development diagnostics.")
    report = {"smoke": args.allow_unreviewed_smoke, "systems": {key: score(value) for key, value in by_system.items()},
              "paired_accuracy": {key: paired_interval(value, by_system["M3"]) for key, value in by_system.items()
                                  if key != "M3" and "M3" in by_system}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
