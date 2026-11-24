"""Owner-scoped version resolution and verification record export."""
from datetime import date
from html import escape
from typing import Literal
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator


class VersionRegistration(BaseModel):
    family_id: UUID
    version_label: str = Field(min_length=1, max_length=80)
    authority_status: Literal["verified", "unverified", "revoked"]
    source_reference: str = Field(min_length=1, max_length=500)
    effective_from: date
    effective_to: date | None = None
    precedence: int = Field(ge=0, le=10000)

    @model_validator(mode="after")
    def dates(self):
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("Effective interval must be nonempty.")
        return self


def select_versions(selected, rows, as_of):
    """Full-document versions only. A tie at the governing priority is unresolved."""
    registered = {row["document_id"]: row for row in rows}
    chosen = {item for item in selected if item not in registered}
    families = {registered[item]["family_id"] for item in selected if item in registered}
    issues = []
    for family in sorted(families):
        eligible = [row for row in rows if row["family_id"] == family
                    and row["authority_status"] == "verified"
                    and row["effective_from"] <= as_of
                    and (row["effective_to"] is None or as_of < row["effective_to"])]
        if not eligible:
            issues.append("No reviewed version applies to a selected document family.")
            continue
        priority = max(row["precedence"] for row in eligible)
        governing = [row for row in eligible if row["precedence"] == priority]
        if len(governing) != 1:
            issues.append("A selected document family has multiple applicable versions at the same precedence.")
            continue
        chosen.add(governing[0]["document_id"])
    return sorted(chosen), issues


async def applicable_versions(store, selected, as_of):
    ids = [str(item) for item in selected]
    matches = await store.read("legal_document_versions", document_id="in.(" + ",".join(ids) + ")", limit=len(ids))
    if not matches:
        return ids, {}, []
    families = sorted({row["family_id"] for row in matches})
    rows = await store.read("legal_document_versions", family_id="in.(" + ",".join(families) + ")", limit=501)
    if len(rows) > 500:
        raise HTTPException(409, "The selected version history exceeds the verification budget.")
    chosen, issues = select_versions(ids, rows, as_of)
    return chosen, {row["document_id"]: row for row in rows if row["document_id"] in chosen}, issues


def report_pdf(record):
    import pymupdf
    from io import BytesIO

    def paragraph(text):
        return f"<p>{escape(str(text))}</p>"

    body = ["<h1>LAVEN verification record</h1>", paragraph(f"Question {record['question_id']}"),
            paragraph(record["question"]), paragraph(f"Outcome {record['status']} · as of {record['as_of']}"),
            paragraph(f"Record SHA-256 {record['record_sha256']}")]
    for attempt in record["attempts"]:
        body.append(f"<h2>Verification attempt {int(attempt['attempt'])}</h2>")
        for claim in attempt.get("claims", []):
            body.extend([paragraph(claim["text"]), paragraph(f"Verification status {claim['status']}")])
            for source in claim["citations"]:
                body.extend([paragraph(f"{source['name']} · page {source['page']} · "
                                       f"characters {source['start_offset']}–{source['end_offset']}"),
                             paragraph(source["quote"])])
    output = BytesIO()
    writer = pymupdf.DocumentWriter(output)
    story = pymupdf.Story("".join(body))
    story.write(writer, lambda *_: (pymupdf.Rect(0, 0, 595, 842), pymupdf.Rect(44, 44, 551, 798), None))
    writer.close()
    return output.getvalue()



def summarize_version_changes(before_pages, after_pages, before_findings, after_findings):
    from collections import Counter

    from .documents import chunk_ranges

    def passages(pages):
        return Counter(" ".join(page["text"][start:end].split())
                       for page in pages for start, end, _ in chunk_ranges(page["text"]))

    def signatures(items):
        return {(item["category"], " ".join((item.get("quote") or "").split()))
                for item in items if item.get("quote")}

    before, after = passages(before_pages), passages(after_pages)
    introduced = signatures(after_findings) - signatures(before_findings)
    localized = [{key: item[key] for key in ("id", "document_id", "page", "start_offset", "end_offset", "category")}
                 for item in after_findings if (item["category"], " ".join((item.get("quote") or "").split()))
                 in introduced]
    return {"added_passages": sum((after - before).values()),
            "removed_passages": sum((before - after).values()), "newly_observed_findings": localized[:100]}
