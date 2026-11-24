"""Export real NLI, calibration and system figures without filling missing results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from .datasets import LABELS

PALETTE = ('#175575', '#ae572c', '#287861', '#5d527e')


def save(fig, directory, title):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(directory / f"{title}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def nli(run, out):
    if load(run / "run.json").get("smoke"):
        print("Rendering development smoke figures. Do not report as benchmark results.")
    history = load(run / "history.json")
    for metric in ("accuracy", "loss"):
        if history:
            fig, ax = plt.subplots()
            for split in ("train", "validation"):
                ax.plot([row["epoch"] for row in history], [row[f"{split}_{metric}"] for row in history],
                        marker="o", label=split.title())
            ax.set(xlabel="Epoch", ylabel=metric.title())
            ax.legend()
            save(fig, out, metric)
    report = load(run / "metrics.json")
    fig, ax = plt.subplots()
    matrix = ax.imshow(report["confusion_matrix"], cmap="Blues")
    ax.set_xticks(range(3), LABELS, rotation=15)
    ax.set_yticks(range(3), LABELS)
    ax.set(xlabel="Prediction", ylabel="Reference")
    fig.colorbar(matrix)
    save(fig, out, "confusion_matrix")
    with (run / "predictions-test.csv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    for title, curve in (("roc", roc_curve), ("precision_recall", precision_recall_curve)):
        fig, ax = plt.subplots()
        for label in LABELS:
            true = [row["gold"] == label for row in records]
            if len(set(true)) < 2:
                continue
            a, b, _ = curve(true, [float(row[label]) for row in records])
            ax.plot(a if title == "roc" else b, b if title == "roc" else a, label=label)
        if ax.lines:
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate" if title == "roc" else "Recall",
                   ylabel="True-positive rate" if title == "roc" else "Precision")
            ax.legend()
            save(fig, out, title)
        else:
            plt.close(fig)


def calibration(path, out):
    content = load(path)
    for split in ("calibration", "test"):
        if split not in content:
            continue
        for chart in ("reliability", "risk_coverage"):
            fig, ax = plt.subplots()
            for stage, report in content[split].items():
                if chart == "reliability":
                    x, y = "confidence", "accuracy"
                else:
                    x, y = "coverage", "risk"
                pairs = report[chart]
                ax.plot([item[x] for item in pairs], [item[y] for item in pairs], label=stage.title())
            if chart == "reliability":
                ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Reference")
            ax.set(xlabel=x.title(), ylabel=y.title(), xlim=(0, 1), ylim=(0, 1))
            ax.legend()
            save(fig, out, f"{split}_{chart}")


def comparisons(path, out):
    report = load(path)
    for metric in ("accuracy", "coverage", "change_sensitivity", "invariance_score",
                   "version_robustness_score", "latency_ms_mean", "completion_calls"):
        entries = [(key, value[metric]) for key, value in report["systems"].items() if value.get(metric) is not None]
        if not entries:
            continue
        fig, ax = plt.subplots()
        ax.bar(*zip(*entries, strict=True))
        ax.set(ylabel=metric.replace("_", " ").capitalize())
        save(fig, out, metric)


def dataset(path, out):
    details = load(path)["split_details"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = [0] * len(details)
    for label, color in zip(LABELS, PALETTE):
        heights = [split["labels"][label] for split in details.values()]
        ax.bar(list(details), heights, bottom=bottom, label=label, color=color)
        bottom = [a+b for a, b in zip(bottom, heights, strict=True)]
    ax.set(ylabel="Official contract–hypothesis pairs")
    ax.legend(frameon=False)
    save(fig, out, "contractnli_class_distribution")


def model_comparison(paths, out, compare_systems=False):
    observed = []
    samples = None
    identity = None
    for path in paths:
        run = load(path / "run.json")
        if run["smoke"]:
            raise ValueError("Smoke runs cannot be published as benchmark comparisons.")
        current = (run["dataset_sha256"], run["evidence_track"], run["windows"], run["token_limit"])
        with (path / "predictions-test.csv").open(encoding="utf-8", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        rows = {(row["sample_id"], row["gold"]) for row in predictions}
        if compare_systems:
            current = current[:1]
        if not rows or len({sample for sample, _ in rows}) != len(predictions) or (
            identity and (identity != current or rows != samples)
        ):
            raise ValueError("Dataset and held-out IDs/labels must match. Use --compare-systems for differing retrieval tracks.")
        identity, samples = current, rows
        observed.append((run, load(path / "metrics.json")))
    with (out / "model_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "evidence_track", "train_examples", "test_examples", "accuracy", "balanced_accuracy",
                         "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "micro_f1",
                         "mcc", "fit_seconds", "mean_test_ms_per_example", "device"])
        for run, metrics in observed:
            writer.writerow([run["model"], run["evidence_track"],
                             run.get("split_counts", {}).get("train", run.get("training_coverage", {}).get("eligible_pairs")),
                             metrics["count"],
                             metrics["accuracy"], metrics["balanced_accuracy"], metrics["macro"]["precision"],
                             metrics["macro"]["recall"], metrics["macro"]["f1"], metrics["weighted"]["f1"],
                             metrics["micro"]["f1"], metrics["mcc"], run["fit_seconds"],
                             run["mean_test_ms_per_example"], run["device"]])
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [run["model"] if not compare_systems else run["model"] + "\n" + run["evidence_track"]
             for run, _ in observed]
    for i, (name, key) in enumerate((("Accuracy", "accuracy"), ("Macro F1", "macro"), ("Weighted F1", "weighted"))):
        values = [m[key] if key == "accuracy" else m[key]["f1"] for _, m in observed]
        ax.bar([n + .24 * (i-1) for n in range(len(names))], values, width=.23, label=name, color=PALETTE[i])
    ax.set(xticks=range(len(names)), xticklabels=names, ylabel="Held-out score", ylim=(0, 1))
    ax.legend(frameon=False)
    save(fig, out, "model_comparison")
    for name, keys, caption in (
        ("macro_prf_comparison", ("precision", "recall", "f1"), "Macro-averaged score"),
        ("class_f1_comparison", LABELS, "Per-class F1"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, key in enumerate(keys):
            values = [m["macro"][key] if name == "macro_prf_comparison" else m["classes"][key]["f1"]
                      for _, m in observed]
            ax.bar([n + .24 * (i-1) for n in range(len(names))], values, width=.23,
                   label=key, color=PALETTE[i])
        ax.set(xticks=range(len(names)), xticklabels=names, ylabel=caption, ylim=(0, 1))
        ax.legend(frameon=False)
        save(fig, out, name)
    for name, key, caption in (
        ("training_time", "fit_seconds", "Fit and validation time (s)"),
        ("inference_time", "mean_test_ms_per_example", "Mean test time per example (ms)"),
    ):
        values = [((run["model"] + " (" + run["device"] + ")"), run[key])
                  for run, _ in observed if run[key] is not None]
        if values:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(*zip(*values, strict=True), color=PALETTE[0])
            ax.set(ylabel=caption)
            save(fig, out, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("nli", "calibration", "comparisons", "dataset", "model-comparison"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--compare-systems", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.kind == "model-comparison":
        model_comparison([args.input, *args.run], args.output, args.compare_systems)
    else:
        {"nli": nli, "calibration": calibration, "comparisons": comparisons,
         "dataset": dataset}[args.kind](args.input, args.output)


if __name__ == "__main__":
    main()
