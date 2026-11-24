"""Run Groq Qwen on opted-in, reviewed fictional contract questions."""
import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path

from .datasets import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from legalshield import laven, verification
from legalshield.groq import Groq

SYSTEMS = ("M1", "M2", "M3", "M4", "M5", "M6", "M7",
           "without_authority", "without_neural", "without_symbolic", "without_uncertainty", "without_regeneration")
SYSTEM = """Return JSON with action answer, abstain, or request_evidence, and integer answer_days
or null. For research questions grounded in quoted evidence, use only supplied excerpts.
If evidence is missing, ask for it. If the evidence is conflicting, abstain."""
DAY = re.compile(r"\b(\d+)\s+(?:calendar\s+)?days?\b", re.I)


class BudgetUsed(RuntimeError):
    pass


class Counted:
    def __init__(self, groq, maximum):
        self.groq, self.maximum, self.calls = groq, maximum, 0

    @property
    def model(self):
        return self.groq.model

    async def json(self, *args, **kwargs):
        if self.calls >= self.maximum:
            raise BudgetUsed("The logical completion budget was reached.")
        self.calls += 1
        return await self.groq.json(*args, **kwargs)


def fixture(row, authority):
    text = row["modified_text"]
    if authority and row["authority_status"] == "unverified":
        text = row["original_text"]
    # The fixture uses fictional strings and recorded authority metadata. It
    # does not substitute adjudicator reference answers for system output.
    return [{"document_id": row["document_id"], "name": "Fictional contract", "page": 1,
             "clause": None, "revision": 1, "parts": [{"text": text, "start": 0, "end": len(text)}]}]


def parse_claims(claims, empty_action="abstain"):
    values = {int(match) for claim in claims for match in DAY.findall(claim["text"])}
    if len(values) == 1:
        return "answer", values.pop()
    return empty_action, None


async def execute(groq, row, system, verifier):
    question = row["question"]
    if system in ("M1", "M2"):
        data = {"question": question, "closed_book": system == "M1"}
        if system == "M2":
            data["evidence"] = [row["modified_text"]]
        raw = await groq.json(SYSTEM + " In closed-book mode answer only from your knowledge.", data, max_tokens=200)
        result = raw.get("answer_days")
        if raw.get("action") == "answer" and type(result) is int:
            return "answer", result
        if raw.get("action") == "request_evidence":
            return "request_evidence", None
        return "abstain", None
    evidence = fixture(row, authority=system in ("M7", "without_neural", "without_symbolic",
                                                "without_uncertainty", "without_regeneration"))
    if system in ("M3", "without_neural"):
        if system == "M3":
            pages = [laven.Page(1, evidence[0]["parts"][0]["text"])]
            _, _, findings = await laven.analyze_page(pages[0], groq)
            evidence[0]["parts"], _ = laven.retain(pages[0], 0, len(pages[0].text), findings)
        return parse_claims(await laven.grounded(groq, question, evidence, []))
    versions = {row["document_id"]: {"authority_status": row["authority_status"],
                                     "version_label": row["document_version"]}}
    claims, record = await verification.verified_answer(
        groq, verifier, question, evidence, [], versions,
        max_retries=int(system in ("M7", "without_authority", "without_symbolic", "without_uncertainty")),
        sample_count=3 if system in ("M6", "M7", "without_regeneration", "without_symbolic", "without_authority") else 1,
        symbolic=system not in ("M4", "M6", "without_symbolic"))
    return parse_claims(claims, "request_evidence" if record["status"] == "request_evidence" else "abstain")


async def run(args, rows):
    verifier = None
    if any(system not in ("M1", "M2", "M3", "without_neural") for system in args.systems):
        if not args.model_path or not args.calibration:
            raise ValueError("The selected systems require a trained verifier and calibration.")
        verifier = verification.Verifier.load(args.model_path, args.calibration)
    os.chdir(ROOT / "backend")
    client = Groq()
    groq = Counted(client, args.max_calls)
    outputs = []
    try:
        for row in rows:
            for system in args.systems:
                before, start = groq.calls, time.perf_counter()
                try:
                    with client.track_usage() as usage:
                        action, answer = await execute(groq, row, system, verifier)
                except BudgetUsed:
                    print("Stopped at the completion budget. The saved run is partial.")
                    return outputs
                outputs.append({"system": system, "sample_id": row["sample_id"], "action": action,
                                "answer": answer, "latency_ms": (time.perf_counter()-start)*1000,
                                "calls": groq.calls-before, "generation_model": groq.model, "groq_usage": usage})
        return outputs
    finally:
        await client.client.aclose()
        write_jsonl(args.output, outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=["M1", "M2", "M3"])
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or len(set(args.systems)) != len(args.systems):
        parser.error("Use a positive limit and distinct systems.")
    rows = [r for r in read_jsonl(args.data) if r["partition"] == args.partition
            and r["human_review_status"] == "approved" and r["synthetic_data_indicator"] is True][:args.limit]
    print({"reviewed_instances": len(rows), "systems": args.systems,
           "illustrative_logical_calls": len(rows) * len(args.systems) * 6,
           "execute": args.execute, "transport_attempts_per_call_at_most": 3})
    if not args.execute:
        return
    if not rows or not args.max_calls or args.max_calls < 1:
        parser.error("Execution requires reviewed fictional cases and a positive explicit completion budget.")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("Choose a new output file for a new run.")
    for item in ("model_path", "calibration"):
        if getattr(args, item):
            setattr(args, item, getattr(args, item).resolve())
    asyncio.run(run(args, rows))


if __name__ == "__main__":
    main()
