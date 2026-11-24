"""Render presentation-sized PNG charts from the measured ContractNLI comparison."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

NAMES = {"tfidf": "TF-IDF", "bilstm": "BiLSTM", "general": "Frozen DeBERTa", "deberta-legal": "Fine-tuned legal DeBERTa"}
KEYS = (("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
        ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1"))


def render(source, output):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [NAMES[row["model"]] for row in rows]
    colors = ["#008477" if row["model"] == "deberta-legal" else "#617d9d" for row in rows]
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 13, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "semibold"}):
        figure, axes = plt.subplots(2, 2, figsize=(15, 7.8), layout="constrained")
        for axis, (key, title) in zip(axes.flat, KEYS, strict=True):
            values = [float(row[key]) for row in rows]
            plotted = axis.barh(names, values, color=colors, height=.58, zorder=3)
            axis.bar_label(plotted, labels=[f"{value:.2%}" for value in values], padding=6)
            axis.invert_yaxis()
            axis.set(xlim=(0, 1.05), xticks=[0, .25, .5, .75, 1], title=title)
            axis.xaxis.set_major_formatter(PercentFormatter(1))
            axis.grid(alpha=.2, axis="x", zorder=1)
        figure.suptitle(f"ContractNLI | {int(rows[0]['test_examples']):,} official test pairs", fontsize=20, weight="bold")
        figure.savefig(output / "dashboard_model_metrics.png", dpi=170)
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(12, 5.5), layout="constrained")
        for row, label, color in zip(rows, names, colors, strict=True):
            axis.scatter(float(row["mean_test_ms_per_example"]), float(row["macro_f1"]), s=130, color=color, label=label)
        axis.set(xscale="log", xlabel="Mean measured inference time per test pair (ms, log scale)",
                 ylabel="Macro F1", title="Measured inference latency and classification F1")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.22)
        axis.legend(loc="best", title="Measurements span CPU and GPU runs")
        figure.savefig(output / "dashboard_latency_f1.png", dpi=170)
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.source, args.output)
