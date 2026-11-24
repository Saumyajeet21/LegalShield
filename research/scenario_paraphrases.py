"""Request limited Groq Qwen question paraphrases for human review."""
import argparse
import asyncio
import os
import sys
from pathlib import Path

from .datasets import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from legalshield.groq import Groq


async def generate(rows, output):
    os.chdir(ROOT / "backend")
    groq = Groq()
    proposed = []
    try:
        for row in rows:
            raw = await groq.json(
                'Return a JSON object with one field "question". Reword this fictional contract '
                'research question in plain English. Preserve all names, dates, scope and conditions. '
                'Do not add an expected answer, an example answer, or new facts.',
                {"question": row["question"]}, max_tokens=160, temperature=.3)
            question = raw.get("question")
            if set(raw) != {"question"} or not isinstance(question, str) or not 5 <= len(question) <= 300:
                continue
            if "\n" in question or question.casefold().strip() == row["question"].casefold().strip():
                continue
            new_row = {**row, "sample_id": row["sample_id"] + ":qwen-paraphrase", "question": question.strip(),
                       "annotation_source": "inherited provisional reference; requires semantic review",
                       "human_review_status": "pending", "generation_model": groq.model,
                       "paraphrase_of": row["sample_id"]}
            proposed.append(new_row)
    finally:
        await groq.client.aclose()
        write_jsonl(output, proposed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 20:
        parser.error("Limit must be between 1 and 20.")
    rows = [row for row in read_jsonl(args.pilot) if row["source_dataset"] == "laven"
            and row["synthetic_data_indicator"] is True and row["partition"] == "train"
            and row["modification_type"] == "legitimate"][:args.limit]
    print({"candidate_families": len(rows), "logical_completions_at_most": len(rows)})
    if args.execute and rows:
        if args.output.exists():
            parser.error("Choose a new output path.")
        asyncio.run(generate(rows, args.output.resolve()))


if __name__ == "__main__":
    main()
