"""Read a local index of published aggregate research measurements."""
import csv
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "contractnli"
MODELS = {"tfidf": ("TF-IDF", "tfidf"), "bilstm": ("BiLSTM", "bilstm"),
          "general": ("Frozen DeBERTa", "general"), "deberta-legal": ("Fine-tuned legal DeBERTa", "deberta-colab")}
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
CHARTS = {
    "comparison-accuracy": "system-comparison/model_comparison.png",
    "comparison-prf": "system-comparison/macro_prf_comparison.png",
    "comparison-overview": "system-comparison/dashboard_model_metrics.png",
    "comparison-tradeoff": "system-comparison/dashboard_latency_f1.png",
    "comparison-class-f1": "system-comparison/class_f1_comparison.png",
    "comparison-inference": "system-comparison/inference_time.png",
    "comparison-training": "system-comparison/training_time.png",
    "dataset-classes": "dataset-figures/contractnli_class_distribution.png",
    **{f"{model}-{name}": f"{folder}/figures/{name}.png" for model, (_, folder) in MODELS.items()
       for name in ("accuracy", "loss", "confusion_matrix", "roc", "precision_recall", "calibration_reliability",
                    "calibration_risk_coverage", "test_reliability", "test_risk_coverage")},
}


def model_comparison():
    path = ARTIFACTS / "system-comparison" / "model_comparison.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = []
        for row in csv.DictReader(source):
            key = row["model"]
            if key not in MODELS:
                continue
            scores = {metric: float(row[metric]) for metric in METRICS}
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
                raise ValueError("Invalid published comparison score.")
            rows.append({"id": key, "name": MODELS[key][0], "test_examples": int(row["test_examples"]), **scores})
        return rows


def chart_path(chart):
    path = ARTIFACTS / CHARTS[chart] if chart in CHARTS else None
    return path if path and path.is_file() else None


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=100)
    value: float = Field(allow_inf_nan=False)
    x: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class Figure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int = Field(ge=1, le=45)
    points: list[Point] = Field(min_length=1, max_length=250)


class ResearchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(min_length=1, max_length=150)
    figures: list[Figure] = Field(max_length=45)


def read_index(location):
    if not location:
        return None
    path = Path(location)
    try:
        with path.open("r", encoding="utf-8-sig") as source:
            data = source.read(512_001)
        if len(data) > 512_000:
            raise ValueError("Research aggregate index exceeds its size limit.")
        summary = ResearchSummary.model_validate_json(data)
        ids = [item.id for item in summary.figures]
        if len(set(ids)) != len(ids):
            raise ValueError("Repeated figure in research aggregate index.")
        return summary
    except FileNotFoundError:
        return None
    except (ValidationError, ValueError, OSError) as error:
        raise ValueError("Research aggregate index failed validation.") from error
