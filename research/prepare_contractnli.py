"""Prepare one cross-platform, leakage-audited ContractNLI comparison dataset."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from zipfile import Path as ArchivePath, ZipFile

from .datasets import LABELS, OFFICIAL_SPLITS, contractnli, digest, leakage

URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"


def prepare(archive, output, expected_digest=None):
    sha256 = digest(archive)
    if expected_digest and sha256 != expected_digest.lower():
        raise ValueError("ContractNLI archive checksum differs from the pinned release.")
    with ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        folders = {
            name.removesuffix("train.json")
            for name in names if name.split("/")[-1] == "train.json"
            and all(name.removesuffix("train.json") + file + ".json" in names for file in OFFICIAL_SPLITS)
        }
        if len(folders) != 1:
            raise ValueError("Expected one directory containing official train/dev/test JSON.")
        rows = list(contractnli(ArchivePath(bundle, at=folders.pop())))
    raw_docs = {split: len({r["document_id"] for r in rows if r["partition"] == split})
                for split in OFFICIAL_SPLITS.values()}
    if raw_docs != {"train": 423, "validation": 61, "test": 123} or len(rows) != 607 * 17:
        raise ValueError(f"Unexpected official release counts: {raw_docs}; {len(rows)} pairs")
    for row in rows:
        row["official_partition"] = row["partition"]

    # Preserve test and retain each document's original assignment in the
    # manifest. Exclude lower-priority documents connected to a held-out clone.
    priority = {"train": 0, "validation": 1, "test": 2}
    overlaps = leakage(rows)
    if any(item["kind"] != "text_overlap" for item in overlaps):
        raise ValueError("An official document family appears across splits.")
    excluded = {min(zip(item["partitions"], item["families"], strict=True),
                    key=lambda pair: priority[pair[0]])[1] for item in overlaps}
    eligible = [row for row in rows if row["family_id"] not in excluded]
    dev = sorted({row["family_id"] for row in eligible if row["partition"] == "validation"})
    random.Random(13).shuffle(dev)
    calibration = set(dev[:round(.4 * len(dev))])
    for row in eligible:
        if row["family_id"] in calibration:
            row["partition"] = "calibration"
    audit = leakage(eligible)
    if audit:
        raise ValueError("Eligible data contains overlap after grouping calibration.")
    splits = {}
    for split in ("train", "validation", "calibration", "test"):
        subset = [row for row in eligible if row["partition"] == split]
        count = Counter(row["label"] for row in subset)
        if set(count) != set(LABELS):
            raise ValueError(f"Missing label support in {split}: {count}")
        splits[split] = {
            "documents": len({row["document_id"] for row in subset}),
            "records": len(subset), "labels": dict(count),
            "evidence_spans": sum(len(row["evidence"]) for row in subset),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in eligible:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "dataset": "ContractNLI", "revision": "2021-10-05", "source": URL,
        "license": "CC-BY-4.0 (official release terms apply)", "archive_sha256": sha256,
        "output_sha256": digest(output), "official_documents": raw_docs, "official_records": len(rows),
        "records": len(eligible), "labels": dict(Counter(row["label"] for row in eligible)),
        "partitions": dict(Counter(row["partition"] for row in eligible)), "split_details": splits,
        "cross_split_audit_threshold": .90, "official_cross_split_links": overlaps,
        "excluded_families": sorted(excluded), "eligible_overlap_count": len(audit),
        "calibration_seed": 13, "calibration_fraction": .4,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    result = prepare(args.archive, args.output, args.sha256)
    print(json.dumps({"archive_sha256": result["archive_sha256"], "dataset_sha256": result["output_sha256"],
                      "split_details": result["split_details"], "excluded_families": result["excluded_families"]},
                     indent=2))
