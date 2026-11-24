"""Build the Research Laboratory's aggregate JSON from measured artifacts."""
import argparse
import csv
import json
from pathlib import Path

from .datasets import digest


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build(manifests, runs, comparisons, compare_systems=False):
    figures, sources = {}, []

    def push(id_, label, value):
        if value is not None:
            figures.setdefault(id_, []).append({"label": label, "value": value})

    for path in manifests:
        data = load(path)
        name = data["dataset"]
        if name == "legalbench":
            name = next(iter(data["inputs"]), path.stem).removesuffix(".tsv")
        sources.append(digest(path)[:12])
        push(1, name, data["records"])
        push(5, "Fictional" if name == "laven" else "Public", data["records"])
        for key, count in data["labels"].items():
            push(2, f"{name} · {key}", count)
        for key, count in data["partitions"].items():
            push(3, f"{name} · {key}", count)
    from sklearn.metrics import precision_recall_curve, roc_curve

    comparable, eligible = None, None
    for path in runs:
        metadata = load(path / "run.json")
        if metadata.get("smoke"):
            continue
        identity = (metadata["dataset_sha256"], metadata["evidence_track"])
        if compare_systems:
            identity = identity[:1]
        if comparable and identity != comparable:
            raise ValueError("Classification plots require the same dataset and evidence track.")
        comparable = identity
        sources.append(digest(path / "run.json")[:12])
        metrics = load(path / "metrics.json")
        history = load(path / "history.json")
        name = metadata["model"]
        if compare_systems:
            name += " · " + metadata["evidence_track"]
        for epoch in history:
            offset = 6 if name == "bilstm" else 8
            for split in ("train", "validation"):
                for metric, index in (("accuracy", offset), ("loss", offset + 1)):
                    push(index, f"{name} · {split} · ep {epoch['epoch']}", epoch[f"{split}_{metric}"])
            push(10, f"{name} · ep {epoch['epoch']}", epoch["learning_rate"])
            push(11, f"{name} · ep {epoch['epoch']}", epoch["validation_macro_f1"])
        if metadata.get("fit_seconds") is not None:
            push(12, name, metadata["fit_seconds"])
        push(13, name, metrics["accuracy"])
        for metric, value in metrics["macro"].items():
            push(14, f"{name} · {metric}", value)
        for label, values in metrics["classes"].items():
            push(15, f"{name} · {label}", values["f1"])
        for i, row in enumerate(metrics["confusion_matrix"]):
            for j, count in enumerate(row):
                push(16, f"{name} · {metadata['labels'][i]} → {metadata['labels'][j]}", count)
        with (path / "predictions-test.csv").open(encoding="utf-8", newline="") as source:
            predictions = list(csv.DictReader(source))
        ids = {row["sample_id"]: row["gold"] for row in predictions}
        if not ids or len(predictions) != len(ids) or (eligible is not None and ids != eligible):
            raise ValueError("Classification plots require matching held-out IDs and gold labels.")
        eligible = ids
        for label in metadata["labels"]:
            truth = [row["gold"] == label for row in predictions]
            if len(set(truth)) < 2:
                continue
            probability = [float(row[label]) for row in predictions]
            fpr, tpr, _ = roc_curve(truth, probability)
            precision, recall, _ = precision_recall_curve(truth, probability)
            for id_, xs, ys in ((17, fpr, tpr), (18, recall, precision)):
                count = 16
                indexes = sorted({int((len(xs) - 1) * i / (count - 1)) for i in range(count)})
                figures.setdefault(id_, []).extend({"label": name + " · " + label,
                    "x": float(xs[i]), "value": float(ys[i])} for i in indexes)
        calibration_path = path / "calibration.json"
        if calibration_path.is_file():
            calibration = load(calibration_path)
            if calibration.get("test_sha256") != digest(path / "predictions-test.csv"):
                raise ValueError("Calibration report test predictions differ from the run.")
            result = calibration["test"]["after"]
            push(38, name, result["incorrect_auroc"])
            for id_, key, x, y in (
                (36, "reliability", "confidence", "accuracy"),
                (37, "risk_coverage", "coverage", "risk"),
            ):
                trace = result[key]
                indexes = sorted({int((len(trace) - 1) * i / 15) for i in range(16)})
                figures.setdefault(id_, []).extend({"label": name, "x": trace[i][x], "value": trace[i][y]}
                                                  for i in indexes if trace)
    if comparisons:
        report = load(comparisons)
        if not report.get("smoke") and not report.get("partial"):
            sources.append(digest(comparisons)[:12])
            for name, value in report["systems"].items():
                push(19, name, value["accuracy"])
                push(31, name + " · CS", value["change_sensitivity"])
                push(31, name + " · IS", value["invariance_score"])
                push(32, name, value["version_robustness_score"])
                push(41 if name.startswith("M") else 42, name, value["accuracy"])
                push(44, name, value["latency_ms_mean"])
            for name, bounds in report["paired_accuracy"].items():
                if bounds:
                    for key in ("low", "difference", "high"):
                        push(43, name + " · " + key, bounds[key])
    return {"version": "Aggregate source hashes " + ", ".join(sources[:8]) if sources else "No eligible results",
            "figures": [{"id": key, "points": rows[:250]} for key, rows in sorted(figures.items())]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--comparisons", type=Path)
    parser.add_argument("--compare-systems", action="store_true")
    args = parser.parse_args()
    summary = build(args.manifest, args.run, args.comparisons, args.compare_systems)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
