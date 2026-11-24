"""Generate fictional legal families with deterministic reference answers."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from .datasets import grouped_partitions, write_jsonl

TOPICS = ("notice", "return", "payment", "delivery", "review", "renewal", "inspection", "reporting")
OBJECTS = ("equipment", "records", "licence", "samples", "stock", "materials", "proposal", "forms")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def pilot(count=100, seed=13):
    if not 1 <= count <= 1000:
        raise ValueError("Use between 1 and 1000 fictional families.")
    rng = random.Random(seed)
    identifiers = [f"laven:{index:04d}" for index in range(count)]
    splits = grouped_partitions(identifiers, seed)
    for index, family in enumerate(identifiers):
        topic = TOPICS[index % len(TOPICS)]
        asset = OBJECTS[(index // len(TOPICS) + index) % len(OBJECTS)]
        first, revised = rng.sample(range(10, 91), 2)
        term = f"The {topic} period for {asset} is {first} calendar days."
        change = f"The {topic} period for {asset} is {revised} calendar days."
        original = f"Fictional instrument {index:04d}. {term}"
        question = f"What is the applicable {topic} period for {asset}?"
        variants = [
            ("legitimate", original, "verified", first, term, "answer"),
            ("authorized_amendment", original.replace(term, change), "verified", revised, change, "answer"),
            ("unauthorized_modification", original + " Unverified draft: " + change, "unverified", first, term, "answer"),
            ("conflicting", original + " " + change, "conflicting", None, None, "abstain"),
            ("missing_evidence", f"Fictional instrument {index:04d}.", "verified", None, None, "request_evidence"),
            ("irrelevant", original + " Correspondence is filed alphabetically.", "verified", first, term, "answer"),
            ("exception", original + " An approved urgent filing has a period of 2 calendar days.",
             "verified", 2, "An approved urgent filing has a period of 2 calendar days.", "answer"),
            ("inert_control", original + " Fictional archive test label LS-CONTROL.", "verified", first, term, "answer"),
        ]
        for category, doc, authority, answer, quote, action in variants:
            offset = doc.index(quote) if quote else None
            candidates = [{"document_version": "v1", "sha256": sha(original), "status": "verified"}]
            if doc != original:
                candidates.append({"document_version": "v2", "sha256": sha(doc), "status": authority})
            yield {
                "sample_id": f"{family}:{category}", "scenario_family_id": family, "family_id": family,
                "document_id": family, "source_dataset": "laven", "source_url": None,
                "dataset_license": "authored fictional fixture; redistribution review pending",
                "original_text": original, "modified_text": doc, "text": doc,
                "document_version": "v1" if doc == original else "v2", "modification_type": category,
                "authority_status": authority, "authority_evidence": candidates,
                "injection_label": "inert_control" if category == "inert_control" else "none",
                "suspicious_text_span": None, "span_start": None, "span_end": None,
                "question": question if category != "exception" else question + " Assume an approved urgent filing.",
                "expected_answer": answer,
                "expected_evidence": [{"text": quote, "start": offset, "end": offset + len(quote)}] if quote else [],
                "expected_answerability": answer is not None, "expected_system_action": action,
                "legal_conditions": ["approved urgent filing"] if category == "exception" else [],
                "legal_exceptions": ["2 calendar days when approved urgent filing"] if category == "exception" else [],
                "verification_label": "Entailment" if answer is not None else "NotMentioned" if category == "missing_evidence" else None,
                "synthetic_data_indicator": True, "annotation_source": "deterministic fixture generator",
                "human_review_status": "pending", "dataset_partition": splits[family], "partition": splits[family],
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--families", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    rows = list(pilot(args.families, args.seed))
    write_jsonl(args.output, rows)
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


if __name__ == "__main__":
    main()
