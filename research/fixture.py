"""Create tiny deterministic data for checking the training pipeline only."""
import argparse
from pathlib import Path

from .datasets import LABELS, write_jsonl


def examples():
    for split, offset in (("train", 0), ("validation", 20), ("calibration", 40), ("test", 60)):
        for item in range(9):
            i = offset + item
            label = LABELS[item % 3]
            claim = f"Instrument {i} requires 30 calendar days of notice."
            text = (claim if label == "Entailment" else f"Instrument {i} requires 60 calendar days of notice."
                    if label == "Contradiction" else f"Instrument {i} references correspondence without a notice period.")
            yield {"sample_id": f"smoke-{i}", "document_id": str(i), "family_id": f"smoke-{i}",
                   "text": text, "claim": claim, "label": label, "partition": split,
                   "evidence": [{"text": text, "start": 0, "end": len(text)}] if label != "NotMentioned" else []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    write_jsonl(parser.parse_args().output, examples())
