"""Check a LAVEN JSON digest and exact quotations against extracted pages."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from legalshield.verification import Atom, Rule, seal, solve


def replay(record, pages, original_paths=None):
    content = dict(record)
    provided = content.pop("record_sha256", None)
    if provided != seal(content)["record_sha256"]:
        raise ValueError("Record checksum mismatch.")
    checked = 0
    checked_originals = set()
    for attempt in record["attempts"]:
        for claim in attempt.get("claims", []):
            for cite in claim["citations"]:
                page = pages[f"{cite['document_id']}:{cite['page']}"]
                if page[cite["start_offset"]:cite["end_offset"]] != cite["quote"]:
                    raise ValueError("Citation does not match the extracted page.")
                document_id = str(cite["document_id"])
                if original_paths is not None and document_id not in checked_originals:
                    with Path(original_paths[document_id]).open("rb") as original:
                        if hashlib.file_digest(original, "sha256").hexdigest() != cite["sha256"]:
                            raise ValueError("Original bytes do not match the recorded document digest.")
                    checked_originals.add(document_id)
                checked += 1
            if claim.get("symbolic") not in (None, "disabled"):
                rules = [Rule.model_validate(rule) for rule in claim["rules"]]
                result = solve(rules, Atom.model_validate(claim["conclusion"]) if claim["conclusion"] else None)
                if result != claim["symbolic"]:
                    raise ValueError("Solver replay differs from recorded result.")
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("pages", type=Path, help='JSON map keyed "document-id:page" containing exact extracted text')
    parser.add_argument("--originals", type=Path, help="JSON mapping document IDs to original file paths")
    args = parser.parse_args()
    count = replay(json.loads(args.record.read_text(encoding="utf-8-sig")),
                   json.loads(args.pages.read_text(encoding="utf-8-sig")),
                   json.loads(args.originals.read_text(encoding="utf-8-sig")) if args.originals else None)
    print(json.dumps({"record_and_spans_checked": True, "citations_checked": count}))
