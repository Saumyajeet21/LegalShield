"""Verify the legal DeBERTa export and unpack only known data artifacts."""
import argparse
import csv
import json
import math
import stat
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

from .datasets import LABELS, digest
from .nli import metrics

PROTOCOL = "document-evidence-pooling-v1"
FILES = {
    "run.json", "metrics.json", "history.json", "calibration.json", "selection.json", "fast-profile.json",
    "fast-coverage.json", "dataset.manifest.json", "labels.json", "predictions-test.csv",
    "predictions-calibration.csv", "retriever.json", "model/config.json", "model/model.safetensors",
    "model/tokenizer.json", "model/tokenizer_config.json", "model/evidence_head.safetensors",
    "model/special_tokens_map.json", "model/spm.model", "model/added_tokens.json",
}
OPTIONAL = {"model/special_tokens_map.json", "model/spm.model", "model/added_tokens.json",
            "fast-profile.json", "fast-coverage.json"}


def verify(path, dataset):
    load = lambda name: json.loads((path / name).read_text(encoding="utf-8"))
    run, cal, reported = (load(name) for name in ("run.json", "calibration.json", "metrics.json"))
    hashes = {
        "model_weights_sha256": digest(path / "model/model.safetensors"),
        "evidence_head_sha256": digest(path / "model/evidence_head.safetensors"),
        "retriever_sha256": digest(path / "retriever.json"),
    }
    if any(run.get(key) != value for key, value in hashes.items()):
        raise ValueError("Model, evidence head or retriever digest mismatch.")
    data_hash = digest(dataset)
    if (run.get("smoke") is not False or run.get("inference_protocol") != PROTOCOL
        or cal.get("inference_protocol") != PROTOCOL or cal.get("deployable") is not False
        or run.get("dataset_sha256") != data_hash or cal.get("dataset_sha256") != data_hash
        or load("dataset.manifest.json")["output_sha256"] != data_hash
        or cal.get("model_weights_sha256") != hashes["model_weights_sha256"]
        or any(tuple(labels) != LABELS for labels in (run["labels"], load("labels.json")))):
        raise ValueError("Dataset, label map or research protocol mismatch.")
    selection = load("selection.json")
    winner = selection["trials"][selection["winner"]]
    if winner["validation"] != run["validation"] or winner["best_epoch"] != run["best_epoch"]:
        raise ValueError("Selected validation checkpoint mismatch.")

    expected = {"test": {}, "calibration": {}}
    with dataset.open(encoding="utf-8") as source:
        for item in map(json.loads, source):
            if item["partition"] in expected:
                expected[item["partition"]][item["sample_id"]] = item["label"]
    for split, count in (("test", 2091), ("calibration", 408)):
        file = path / f"predictions-{split}.csv"
        if cal.get(f"{split}_sha256") != digest(file):
            raise ValueError(f"{split} prediction hash mismatch.")
        with file.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != count or {row["sample_id"]: row["gold"] for row in rows} != expected[split]:
            raise ValueError(f"{split} predictions do not match every official ID and raw label.")
        p = np.array([[float(row[label]) for label in LABELS] for row in rows])
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
            raise ValueError(f"Invalid {split} probabilities.")
        if any(row["prediction"] != LABELS[i] for row, i in zip(rows, p.argmax(1), strict=True)):
            raise ValueError(f"{split} prediction label differs from maximum probability.")
        if split == "test":
            observed = metrics([LABELS.index(row["gold"]) for row in rows], p)
    if (reported["count"] != observed["count"] or reported["confusion_matrix"] != observed["confusion_matrix"]
        or not math.isclose(reported["accuracy"], observed["accuracy"])
        or not math.isclose(reported["macro"]["f1"], observed["macro"]["f1"])):
        raise ValueError("Reported metrics differ from recomputed test results.")
    return observed


def ingest(zip_path, output, dataset):
    if output.exists():
        raise FileExistsError(f"Choose a fresh import directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as stage, ZipFile(zip_path) as archive:
        extracted = Path(stage) / "payload"
        checked = set()
        total = 0
        for entry in archive.infolist():
            name = entry.filename
            member = PurePosixPath(name)
            figure = len(member.parts) == 2 and member.parts[0] == "figures" and member.suffix in {".png", ".svg", ".pdf"}
            if (name in checked or member.as_posix() != name or ".." in member.parts or member.is_absolute() or "\\" in name or ":" in name
                or entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16)
                or (name not in FILES and not figure)):
                raise ValueError(f"Unexpected archive member: {name}")
            total += entry.file_size
            if total > 1_200_000_000:
                raise ValueError("Archive exceeds the expected uncompressed size.")
            checked.add(name)
        if FILES - OPTIONAL - checked:
            raise ValueError("Model export is incomplete.")
        for entry in archive.infolist():
            archive.extract(entry, extracted)
        observed = verify(extracted, dataset)
        verification = {"archive_sha256": digest(zip_path), "test_metrics": observed, "stage_only": True}
        (extracted / "import-verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
        extracted.replace(output)
        return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    outcome = ingest(args.zip, args.output, args.dataset)
    print(json.dumps({"archive_sha256": outcome["archive_sha256"], "accuracy": outcome["test_metrics"]["accuracy"],
                      "macro": outcome["test_metrics"]["macro"], "count": outcome["test_metrics"]["count"]}))
