"""Aggregate test-set results from a LegalShield NDJSON experiment export."""
import json
import sys
from pathlib import Path
from statistics import mean


def rate(values):
    known = [value for value in values if value is not None]
    return mean(known) if known else None


def summarize(runs):
    development = {run["sha256"] for run in runs if run["split"] == "development"}
    tested = [run for run in runs if run["split"] == "test"]
    if development & {run["sha256"] for run in tested}:
        raise ValueError("An identical document is present in both the development and test splits.")
    detectors = {}
    for name in ("rule_based", "semantic", "laven"):
        records = [run["metrics"][name] for run in tested]
        tp, fp, fn = (sum(record[key] for record in records) for key in ("tp_characters", "fp_characters", "fn_characters"))
        detectors[name] = {
            "character_precision": tp / (tp + fp) if tp + fp else None,
            "character_recall": tp / (tp + fn) if tp + fn else None,
            "character_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
            "mean_exact_span_f1": rate(row["exact_span_f1"] for row in records),
            "mean_clause_false_positive_rate": rate(row["false_positive_clause_rate"] for row in records),
        }
    exposed = [run for run in tested if run["injection_retrieved"] and run["marker_retrieved"]]
    return {"test_runs": len(tested), "paired_runs_with_marker_retrieved": len(exposed), "detectors": detectors,
        "raw_marker_success_rate": rate(run["raw_marker_observed"] for run in exposed),
        "protected_marker_success_rate": rate(run["protected_marker_observed"] for run in exposed),
        "reference_phrase_match_rate": rate(run["answer_phrase_match"] for run in tested),
        "mean_citation_supported_claim_fraction": rate(run["citation_supported_claim_fraction"] for run in tested),
        "mean_legitimate_character_retention": rate(run["legitimate_character_retention"] for run in tested),
        "mean_latency_ms": rate(run["latency_ms"] for run in tested)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/summarize_runs.py export.ndjson")
    try:
        data = [json.loads(line)["report"] for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
        print(json.dumps(summarize(data), indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error))
