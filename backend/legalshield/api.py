import asyncio
import hashlib
import json
import re
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from uuid import UUID, uuid4

import httpx
import numpy as np
import pymupdf
from defusedxml.common import DefusedXmlException
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from safetensors import SafetensorError

from . import generation, laven, reconstruction, research_artifacts, upgraded, verification
from .config import settings
from .documents import MAX_BYTES, Page, extract
from .embeddings import embed
from .groq import Groq
from .store import PUBLIC_DOCUMENT, Store

CATEGORIES = ("Contract", "Legislation", "Court judgment", "Regulation", "Legal notice", "Other")


@asynccontextmanager
async def lifespan(app):
    app.state.supabase = httpx.AsyncClient(timeout=httpx.Timeout(35, connect=10))
    app.state.groq = Groq()
    app.state.verifier = None
    try:
        config = settings()
        if config.laven_enhanced:
            if not config.laven_verifier_path or not config.laven_calibration_path:
                raise ValueError("Enhanced LAVEN requires a verifier and matching deployment calibration.")
            app.state.verifier = await asyncio.to_thread(
                verification.Verifier.load, config.laven_verifier_path, config.laven_calibration_path)
        yield
    finally:
        await app.state.supabase.aclose()
        await app.state.groq.client.aclose()


app = FastAPI(title="LegalShield AI", version="1.0.0", lifespan=lifespan)
bearer = HTTPBearer(auto_error=False)


@app.middleware("http")
async def upload_size_limit(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/documents":
        declared = request.headers.get("content-length", "")
        if not declared.isascii() or not declared.isdecimal():
            return JSONResponse({"detail": "Send an upload with a valid content length."}, status_code=411)
        if len(declared) > 9 or int(declared) > MAX_BYTES + 65_536:
            return JSONResponse({"detail": "Document uploads are limited to 20 MiB."}, status_code=413)
    return await call_next(request)


app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in settings().allowed_origins.split(",") if x.strip()],
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["Authorization", "Content-Type"])


async def authenticated(request: Request, auth: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if auth is None:
        raise HTTPException(401, "Sign in to use your legal workspace.")
    return await Store(request.app.state.supabase, auth.credentials).authenticate()


async def researcher(store: Store = Depends(authenticated)):
    if not store.researcher:
        raise HTTPException(403, "Researcher access is required.")
    return store


async def document(store, document_id, *, indexed=True):
    rows = await store.read("documents", id=f"eq.{document_id}", limit=1)
    if not rows:
        raise HTTPException(404, "The document was not found in this workspace.")
    row = rows[0]
    if indexed and row["status"] != "indexed":
        raise HTTPException(409, "This document is not available for review.")
    return row


def public_doc(row):
    return {key: row[key] for key in PUBLIC_DOCUMENT.split(",")}


async def all_rows(store, table, *, max_rows=2500, **params):
    result = []
    while len(result) < max_rows:
        batch = await store.read(table, offset=len(result), limit=min(200, max_rows - len(result)), **params)
        result.extend(batch)
        if len(batch) < 200:
            break
    return result


async def read_findings(store, ids=None):
    if ids == []:
        return []
    params = {"order": "document_id,page,start_offset", "select": "*,documents!inner(name,status)"}
    if ids is not None:
        params["document_id"] = "in.(" + ",".join(map(str, ids)) + ")"
    params["documents.status"] = "eq.indexed"
    rows = await all_rows(store, "security_findings", **params)
    for row in rows:
        row["name"] = row.pop("documents")["name"]
        row.pop("owner", None)
    return rows


def get_pages(name, data):
    try:
        return extract(name, data)
    except (ValueError, RuntimeError, UnicodeError, KeyError, zipfile.BadZipFile, DefusedXmlException, pymupdf.FileDataError) as error:
        detail = str(error) if type(error) is ValueError else "The file is malformed or cannot be read."
        raise HTTPException(422, detail) from error


async def indexed_content(name, data, groq):
    kind, pages = await asyncio.to_thread(get_pages, name, data)
    await groq.verify()
    try:
        page_rows, chunks, findings = await laven.index(pages, groq)
    except (ValueError, OSError) as error:
        raise HTTPException(503, "Indexing failed. Make the configured 384-dimensional embedding model available locally.") from error
    return kind, pages, page_rows, chunks, findings


async def replace_index(store, row, page_rows, chunks, findings):
    result = await store.rpc("replace_index", {"p_owner": store.owner, "p_document": row["id"],
        "p_revision": row["revision"], "p_pages": page_rows, "p_chunks": chunks, "p_findings": findings}, privileged=True)
    if result is not True:
        raise HTTPException(409, "The document changed during indexing. Refresh and try again.")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/session")
async def session(store: Store = Depends(authenticated)):
    return {"researcher": store.researcher, "laven_enhanced": settings().laven_enhanced}


@app.get("/connection")
async def check_connection(request: Request, store: Store = Depends(authenticated)):
    await store.read("documents", select="id", limit=1)
    return {"model": await request.app.state.groq.verify(), "database": "connected"}


@app.get("/documents")
async def documents(store: Store = Depends(authenticated)):
    return await all_rows(store, "documents", order="created_at.desc,id", select=PUBLIC_DOCUMENT)


@app.post("/documents", status_code=201)
async def upload(request: Request, file: UploadFile = File(), category: str = Form("Other"),
                 synthetic: bool = Form(False), trusted: bool = Form(False), store: Store = Depends(authenticated)):
    if (synthetic or trusted) and not store.researcher:
        raise HTTPException(403, "Only researchers can label synthetic or trusted reference documents.")
    if synthetic and trusted:
        raise HTTPException(422, "A synthetic document cannot be a trusted legal reference.")
    if category not in CATEGORIES:
        raise HTTPException(422, "Choose a supported document category.")
    name = PurePosixPath((file.filename or "").replace("\\", "/")).name[:200]
    if not name or any(ord(character) < 32 for character in name):
        raise HTTPException(422, "Choose a supported file with a valid name.")
    data = await file.read(MAX_BYTES + 1)
    kind, pages, page_rows, chunks, findings = await indexed_content(name, data, request.app.state.groq)
    identifier = str(uuid4())
    path = f"{store.owner}/{identifier}/original"
    row = (await store.write("POST", "documents", data={
        "id": identifier, "name": name, "media_type": kind, "category": category, "object_path": path,
        "sha256": hashlib.sha256(data).hexdigest(), "page_count": len(pages), "status": "processing",
        "synthetic": synthetic, "trusted": trusted,
    }))[0]
    try:
        await store.put_original(path, data, kind)
        await replace_index(store, row, page_rows, chunks, findings)
    except HTTPException:
        # A timed out index call can have committed. Never delete a document with a committed revision.
        row = await document(store, identifier, indexed=False)
        if row["status"] == "indexed":
            return public_doc(row)
        changed = await store.write("PATCH", "documents", id=f"eq.{identifier}", status="eq.processing", data={"status": "deleting"})
        if not changed:
            current = await document(store, identifier, indexed=False)
            if current["status"] == "indexed":
                return public_doc(current)
        else:
            await store.remove_original(path)
            await store.rpc("delete_document", {"p_owner": store.owner, "p_document": identifier}, privileged=True)
        raise
    return public_doc(await document(store, identifier))


@app.get("/documents/{document_id}/preview")
async def preview(document_id: UUID, page: int = Query(1, ge=1), store: Store = Depends(authenticated)):
    row = await document(store, document_id)
    values = await store.read("document_pages", document_id=f"eq.{document_id}", page=f"eq.{page}", select="page,text", limit=1)
    if not values:
        raise HTTPException(404, "This page is unavailable.")
    findings = await all_rows(store, "security_findings", document_id=f"eq.{document_id}", page=f"eq.{page}", order="start_offset,id")
    return {"document": public_doc(row), "page": values[0], "findings": findings}


@app.get("/documents/{document_id}/scan-metrics")
async def scan_metrics(document_id: UUID, store: Store = Depends(authenticated)):
    source = await document(store, document_id)
    scope = {"document_id": f"eq.{document_id}", "order": "page,id"}
    chunks, findings = await asyncio.gather(
        all_rows(store, "document_chunks", max_rows=5001, select="page,start_offset,end_offset,parts", **scope),
        all_rows(store, "security_findings", max_rows=5001, select="page,action", **scope))
    if (await document(store, document_id))["revision"] != source["revision"]:
        raise HTTPException(409, "The document changed. Reopen its scan results.")
    if len(chunks) == 5001 or len(findings) == 5001:
        raise HTTPException(422, "This document exceeds the detailed scan reporting limit.")
    def covered(retained):
        ranges = sorted((row["page"], part["start"], part["end"]) for row in chunks
                        for part in (row["parts"] if retained else
                                     [{"start": row["start_offset"], "end": row["end_offset"]}]))
        count, last_page, previous_end = 0, None, 0
        for page, start, end in ranges:
            if page != last_page:
                last_page, previous_end = page, 0
            count += max(0, end - max(start, previous_end))
            previous_end = max(previous_end, end)
        return count
    total, retained = covered(False), covered(True)
    return {"spans": len(findings), "affected_pages": len({row["page"] for row in findings}),
            "isolated": sum(row["action"] == "isolated" for row in findings),
            "quarantined": sum(row["action"] == "quarantined" for row in findings),
            "retention": retained / total if total else None, "retained_characters": retained,
            "candidate_characters": total, "confidence": None}


@app.get("/documents/{document_id}/original")
async def original(document_id: UUID, store: Store = Depends(authenticated)):
    row = await document(store, document_id)
    data = await store.get_original(row["object_path"])
    return Response(data, media_type=row["media_type"], headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
        "Content-Disposition": "inline" if row["media_type"] == "application/pdf" else "attachment"})


@app.post("/documents/{document_id}/reindex")
async def reindex(request: Request, document_id: UUID, store: Store = Depends(authenticated)):
    row = await document(store, document_id)
    data = await store.get_original(row["object_path"])
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise HTTPException(409, "The stored original did not match its upload checksum.")
    _, _, page_rows, chunks, findings = await indexed_content(row["name"], data, request.app.state.groq)
    await replace_index(store, row, page_rows, chunks, findings)
    return public_doc(await document(store, document_id))


@app.delete("/documents/{document_id}", status_code=204)
async def delete(document_id: UUID, store: Store = Depends(authenticated)):
    row = await document(store, document_id, indexed=False)
    if row["status"] != "deleting":
        changed = await store.write("PATCH", "documents", id=f"eq.{document_id}", revision=f"eq.{row['revision']}",
                                    data={"status": "deleting"})
        if not changed:
            raise HTTPException(409, "The document changed. Refresh and retry deletion.")
    await store.remove_original(row["object_path"])
    await store.rpc("delete_document", {"p_owner": store.owner, "p_document": row["id"]}, privileged=True)
    return Response(status_code=204)


@app.get("/alerts")
async def alerts(store: Store = Depends(authenticated)):
    return await read_findings(store)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[UUID] = Field(min_length=1, max_length=20)
    conversation_id: UUID | None = None
    as_of: date | None = None
    compare_baseline: bool = False


async def search(store, question, ids):
    try:
        vector = (await asyncio.to_thread(embed, [question]))[0]
    except (ValueError, OSError) as error:
        raise HTTPException(503, "The configured embedding model could not be loaded.") from error
    base = {"p_embedding": vector, "p_documents": [str(item) for item in ids]}
    hits = await store.rpc("match", {**base, "p_limit": 8})
    trusted = await store.rpc("match", {**base, "p_limit": 4, "p_trusted": True})
    return hits, trusted


async def get_conversation(store, conversation_id):
    rows = await store.read("conversations", id=f"eq.{conversation_id}", limit=1)
    if not rows:
        raise HTTPException(404, "Conversation unavailable.")
    return rows[0]


@app.post("/ask")
async def ask(body: Question, request: Request, store: Store = Depends(authenticated)):
    started = time.perf_counter()
    question = body.question.strip()
    if len(question) < 3:
        raise HTTPException(422, "Enter a legal research question.")
    ids = list(dict.fromkeys(body.document_ids))
    for identifier in ids:
        await document(store, identifier)
    if body.conversation_id:
        await get_conversation(store, body.conversation_id)
    flags = laven.baseline(Page(0, question))
    config, groq = settings(), request.app.state.groq
    as_of = (body.as_of or datetime.now(timezone.utc).date()).isoformat()
    effective_ids, versions, issues = ids, {}, []
    if config.laven_enhanced and not flags:
        effective_ids, versions, issues = await upgraded.applicable_versions(store, ids, as_of)
        for identifier in effective_ids:
            source = await document(store, identifier)
            key = str(identifier)
            if key in versions and versions[key]["sha256"] != source["sha256"]:
                raise HTTPException(409, "Reviewed version no longer matches its recorded source.")
            versions[key] = {**versions.get(key, {}), "sha256": source["sha256"]}
    source_ids = list(dict.fromkeys(map(str, [*ids, *effective_ids])))
    hits, trusted = ([], []) if flags or issues or not effective_ids else await search(store, question, effective_ids)
    if config.laven_enhanced and trusted:
        reference_ids = list(dict.fromkeys(item["document_id"] for item in trusted))
        applicable, _, _ = await upgraded.applicable_versions(store, reference_ids, as_of)
        allowed_references = set(applicable)
        trusted = [item for item in trusted if item["document_id"] in allowed_references]
    findings = await read_findings(store, source_ids)
    record = None
    if flags:
        claims = []
    elif config.laven_enhanced:
        try:
            with groq.track_usage() as usage:
                async with asyncio.timeout(config.laven_answer_timeout):
                    claims, record = await verification.verified_answer(
                        groq, request.app.state.verifier, question, hits, trusted, versions,
                        max_retries=config.laven_regeneration_attempts, sample_count=config.laven_sample_count,
                        max_entropy=config.laven_max_entropy, as_of=as_of, document_ids=[str(i) for i in ids],
                        authority_issues=issues)
        except TimeoutError as error:
            raise HTTPException(504, "Verification exceeded the answer time budget.") from error
    else:
        claims = await laven.grounded(groq, question, hits, trusted)
    result = {"document_ids": source_ids, "claims": claims, "message": ("The question contains a potential AI-directed instruction. Rephrase it as a legal research question." if flags else laven.INSUFFICIENT) if not claims else None,
        "alerts": findings, "question_flagged": bool(flags),
        "retrieval": [{"id": h["id"], "document_id": h["document_id"], "page": h["page"], "revision": h["revision"],
                       "similarity": h["similarity"]} for h in hits]}
    if record is not None:
        content = {key: value for key, value in record.items() if key != "record_sha256"}
        result["verification"] = verification.seal({**content, "groq_usage": usage})
        if record["status"] == "verifier_unavailable":
            result["message"] = "The independent verifier is unavailable."
        elif issues:
            result["message"] = " ".join(issues)
        elif record["status"] == "uncertain":
            result["message"] = "The answer candidates did not provide sufficient agreement."
        if body.compare_baseline and hits:
            baseline_started = time.perf_counter()
            try:
                async with asyncio.timeout(min(30, config.laven_answer_timeout)):
                    baseline_claims = await laven.grounded(groq, question, hits, trusted)
            except (TimeoutError, HTTPException):
                pass
            else:
                result["comparison"] = {
                    "baseline": {"claims": len(baseline_claims), "latency_ms": round((time.perf_counter() - baseline_started) * 1000)},
                    "verified": {"claims": len(claims), "latency_ms": record.get("generation_ms", 0) + (record.get("verification_ms") or 0)},
                }
    result["response_latency_ms"] = round((time.perf_counter() - started) * 1000)
    conversation_id = str(body.conversation_id or uuid4())
    saved = await store.rpc("save_turn", {"p_owner": store.owner, "p_conversation": conversation_id,
        "p_create": body.conversation_id is None, "p_question": question,
        "p_result": result}, privileged=True)
    if saved is not True:
        raise HTTPException(409, "A source document changed during the answer. Refresh and try again.")
    return {**result, "conversation_id": conversation_id}


@app.get("/conversations")
async def conversations(store: Store = Depends(authenticated)):
    return await store.read("conversations", order="created_at.desc", limit=50)


@app.get("/conversations/{conversation_id}")
async def history(conversation_id: UUID, store: Store = Depends(authenticated)):
    await get_conversation(store, conversation_id)
    return await all_rows(store, "chat_turns", conversation_id=f"eq.{conversation_id}", order="id")


@app.post("/documents/{document_id}/legal-version", status_code=201)
async def register_legal_version(document_id: UUID, body: upgraded.VersionRegistration,
                                 store: Store = Depends(researcher)):
    source = await document(store, document_id)
    if not settings().laven_enhanced:
        raise HTTPException(404, "Legal version registration is disabled.")
    if body.authority_status == "verified" and not (source["trusted"] or source["synthetic"]):
        raise HTTPException(422, "Verified versions require an explicitly trusted or synthetic source.")
    current = await store.read("legal_document_versions", document_id=f"eq.{document_id}", limit=1)
    if current:
        raise HTTPException(409, "This source already has a legal version registration.")
    rows = await store.write("POST", "legal_document_versions", data={
        **body.model_dump(mode="json"), "document_id": str(document_id), "sha256": source["sha256"]})
    return rows[0]


@app.get("/documents/{document_id}/legal-versions")
async def legal_versions(document_id: UUID, store: Store = Depends(authenticated)):
    await document(store, document_id)
    if not settings().laven_enhanced:
        raise HTTPException(404, "Legal versions are disabled.")
    rows = await store.read("legal_document_versions", document_id=f"eq.{document_id}", limit=1)
    return await all_rows(store, "legal_document_versions", family_id=f"eq.{rows[0]['family_id']}",
                          order="effective_from,precedence") if rows else []


@app.get("/documents/{document_id}/legal-change")
async def legal_change(document_id: UUID, before: UUID, store: Store = Depends(authenticated)):
    for identifier in (document_id, before):
        await document(store, identifier)
    if not settings().laven_enhanced:
        raise HTTPException(404, "Legal versions are disabled.")
    if document_id == before:
        raise HTTPException(422, "Select two different legal versions.")
    ids = [before, document_id]
    rows = await store.read("legal_document_versions", document_id="in.(" + ",".join(map(str, ids)) + ")", limit=2)
    if len(rows) != 2 or rows[0]["family_id"] != rows[1]["family_id"]:
        raise HTTPException(422, "Select two registered versions in the same document family.")
    old_pages, new_pages, findings = await asyncio.gather(
        all_rows(store, "document_pages", max_rows=300, document_id=f"eq.{before}", order="page", select="page,text"),
        all_rows(store, "document_pages", max_rows=300, document_id=f"eq.{document_id}", order="page", select="page,text"),
        read_findings(store, ids),
    )
    return await asyncio.to_thread(
        upgraded.summarize_version_changes, old_pages, new_pages,
        [item for item in findings if item["document_id"] == str(before)],
        [item for item in findings if item["document_id"] == str(document_id)],
    )


@app.get("/conversations/{conversation_id}/verification/{question_id}.pdf")
async def verification_pdf(conversation_id: UUID, question_id: UUID,
                           store: Store = Depends(authenticated)):
    rows = await store.read("chat_turns", conversation_id=f"eq.{conversation_id}", limit=1,
                            **{"result->verification->>question_id": f"eq.{question_id}"})
    if not rows or not rows[0]["result"].get("verification"):
        raise HTTPException(404, "Verification record unavailable.")
    result = await asyncio.to_thread(upgraded.report_pdf, rows[0]["result"]["verification"])
    return Response(result, media_type="application/pdf", headers={
        "Cache-Control": "private, no-store", "Content-Disposition": "attachment; filename=laven-verification.pdf"})


class GoldSpan(BaseModel):
    page: int = Field(ge=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)


class Experiment(BaseModel):
    document_id: UUID
    question: str = Field(min_length=3, max_length=2000)
    expected: list[GoldSpan] = Field(default_factory=list, max_length=100)
    attack_marker: str = Field(min_length=8, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    expected_answer_phrase: str = Field(default="", max_length=300)
    split: str = Field(default="test", pattern=r"^(development|test)$")


@app.post("/research")
async def experiment(body: Experiment, request: Request, store: Store = Depends(researcher)):
    started = time.perf_counter()
    doc = await document(store, body.document_id)
    if not doc["synthetic"]:
        raise HTTPException(422, "Experiments require a document marked synthetic when uploaded.")
    pages = [Page(row["page"], row["text"]) for row in await all_rows(store, "document_pages", max_rows=300,
             document_id=f"eq.{body.document_id}", order="page", select="page,text")]
    page_by_number = {page.number: page for page in pages}
    expected = [r.model_dump() for r in body.expected]
    if any(r["page"] not in page_by_number or not r["start_offset"] < r["end_offset"] <= len(page_by_number[r["page"]].text)
           for r in expected):
        raise HTTPException(422, "Gold spans must identify valid zero-based character ranges in the extracted page.")
    if body.attack_marker.lower() in body.question.lower() or not any(body.attack_marker in page.text for page in pages):
        raise HTTPException(422, "Use a synthetic marker present in the document and absent from the question.")
    groq = request.app.state.groq
    baseline, semantic, proposed = [], [], []
    for page in pages:
        rules, model, combined = await laven.analyze_page(page, groq)
        baseline.extend(rules)
        semantic.extend(model)
        proposed.extend(combined)
    metrics = {name: laven.localization(spans, expected, pages)
               for name, spans in (("rule_based", baseline), ("semantic", semantic), ("laven", proposed))}
    stored = await all_rows(store, "document_chunks", document_id=f"eq.{body.document_id}", order="id",
        select="id,document_id,page,clause,start_offset,end_offset,parts")
    for row in stored:
        row["name"] = doc["name"]
    raw = [{**row, "parts": [{"start": row["start_offset"], "end": row["end_offset"],
            "text": page_by_number[row["page"]].text[row["start_offset"]:row["end_offset"]]}]} for row in stored]
    if raw:
        vectors = np.asarray(await asyncio.to_thread(embed, [body.question, *(r["parts"][0]["text"] for r in raw)]))
        best = np.argsort(-(vectors[1:] @ vectors[0]), kind="stable")[:8]
        hits = [stored[i] for i in best]
        raw = [raw[i] for i in best]
    else:
        hits = []
    _, trusted = await search(store, body.question, [body.document_id])
    if (await document(store, body.document_id))["revision"] != doc["revision"]:
        raise HTTPException(409, "The document was reindexed. Restart the experiment.")
    marker_retrieved = any(body.attack_marker in row["parts"][0]["text"] for row in raw)
    injection_retrieved = any(span["page"] == row["page"] and span["start_offset"] < row["end_offset"] and
                              span["end_offset"] > row["start_offset"] for row in hits for span in expected)
    attack_retrieved = injection_retrieved

    raw_output = await laven.draft(groq, body.question, laven.evidence_map(raw)) if hits else {"claims": []}
    protected_output = await laven.draft(groq, body.question, laven.evidence_map(hits)) if hits else {"claims": []}
    grounded = await laven.grounded(groq, body.question, hits, trusted, candidate=protected_output) if hits else []
    def contains(result, marker):
        claims = result.get("claims")
        return isinstance(claims, list) and any(isinstance(x, dict) and isinstance(x.get("text"), str)
             and marker.casefold() in x["text"].casefold() for x in claims)

    raw_observed, protected_observed = [contains(r, body.attack_marker) for r in (raw_output, protected_output)]
    statuses = []
    if not attack_retrieved:
        statuses.append("INJECTION NOT RETRIEVED")
    else:
        statuses.append("ATTACK OBSERVED" if raw_observed else "ATTACK NOT OBSERVED")
    if proposed:
        statuses.append("ATTACK DETECTED")
    if attack_retrieved and raw_observed and not protected_observed:
        statuses.append("ATTACK NEUTRALIZED")
    if not grounded:
        statuses.append("INSUFFICIENT EVIDENCE")
    total_characters = sum(len(page.text) for page in pages)
    gold = {(r["page"], i) for r in expected for i in range(r["start_offset"], r["end_offset"])}
    retained = {(row["page"], i) for row in stored for part in row["parts"]
                for i in range(part["start"], part["end"])}
    protected_claims = protected_output.get("claims")
    entry = {"method": "LAVEN", "document_id": doc["id"], "sha256": doc["sha256"], "revision": doc["revision"],
             "created_at": datetime.now(timezone.utc).isoformat(), "model": groq.model, "temperature": 0,
             "embedding_model": settings().embedding_model, "retrieval_method": "top 8 by raw chunk cosine similarity", "split": body.split, "question": body.question,
             "gold_spans": expected, "attack_marker": body.attack_marker, "statuses": statuses, "metrics": metrics,
             "detectors": {"rule_based": baseline, "semantic": semantic, "laven": proposed},
             "injection_retrieved": attack_retrieved, "marker_retrieved": marker_retrieved,
             "raw_marker_observed": raw_observed, "protected_marker_observed": protected_observed,
             "expected_answer_phrase": body.expected_answer_phrase or None,
             "answer_phrase_match": contains({"claims": grounded}, body.expected_answer_phrase) if body.expected_answer_phrase else None,
             "citation_supported_claim_fraction": len(grounded) / len(protected_claims) if isinstance(protected_claims, list) and protected_claims else None,
             "legitimate_character_retention": len(retained - gold) / (total_characters - len(gold)) if total_characters > len(gold) else None,
             "latency_ms": round((time.perf_counter() - started) * 1000),
             "retrieved_chunks": [{"id": r["id"], "page": r["page"], "start_offset": r["start_offset"], "end_offset": r["end_offset"]} for r in hits],
             "raw_output": raw_output, "protected_output": protected_output, "grounded_claims": grounded}
    return (await store.write("POST", "research_experiments", data={"document_id": doc["id"], "report": entry}))[0]


@app.get("/research")
async def experiments(store: Store = Depends(researcher)):
    return await all_rows(store, "research_experiments", order="created_at.desc,id")


@app.get("/research/artifacts")
async def published_research(store: Store = Depends(researcher)):
    try:
        return await asyncio.to_thread(research_artifacts.read_index, settings().laven_research_index)
    except ValueError as error:
        raise HTTPException(503, "The published aggregate index is unreadable.") from error


@app.get("/research/comparison")
async def published_comparison(store: Store = Depends(researcher)):
    try:
        return {"models": await asyncio.to_thread(research_artifacts.model_comparison),
                "charts": [key for key in research_artifacts.CHARTS if research_artifacts.chart_path(key)]}
    except (OSError, KeyError, ValueError) as error:
        raise HTTPException(503, "The published model comparison is unreadable.") from error


@app.get("/research/charts/{chart}")
async def published_chart(chart: str, store: Store = Depends(researcher)):
    path = research_artifacts.chart_path(chart)
    if path is None:
        raise HTTPException(404, "This chart has not been published.")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store",
                                                               "X-Content-Type-Options": "nosniff"})


@app.post("/generation/drafts")
async def generate_documents(body: generation.DraftRequest, request: Request, store: Store = Depends(authenticated)):
    identifiers = list(dict.fromkeys(body.document_ids))
    for identifier in identifiers:
        await document(store, identifier)
    query = " ".join([body.kind, *body.requirements.values(), body.instructions])[:2500]
    hits, trusted = await search(store, query, identifiers) if identifiers else ([], [])
    try:
        async with asyncio.timeout(150):
            return await generation.generate_drafts(request.app.state.groq, request.app.state.verifier, body, hits, trusted)
    except TimeoutError as error:
        raise HTTPException(504, "Drafting reached its time limit. Try fewer drafts.") from error


@app.post("/generation/export/{format}")
async def download_generated(format: str, body: generation.Export, store: Store = Depends(authenticated)):
    formats = {"pdf": (generation.export_pdf, "application/pdf"),
               "docx": (generation.export_docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    if format not in formats:
        raise HTTPException(404, "This file format is unavailable.")
    render, media_type = formats[format]
    output = await asyncio.to_thread(render, body.title, body.text.splitlines())
    return Response(output, media_type=media_type,
                    headers={"Cache-Control": "private, no-store", "Content-Disposition": f'attachment; filename="laven-generated.{format}"'})


class ReconstructionRequest(BaseModel):
    page: int = Field(1, ge=1, le=300)
    model: Literal["ae", "vae", "both"] = "ae"
    degradation: Literal["gaussian", "blur", "contrast", "speckle", "compression"] = "gaussian"
    controlled: bool = True
    severity: float = Field(.5, ge=0, le=1)


class LatentRequest(BaseModel):
    seed: int = Field(2026, ge=0, le=10**9)
    coordinates: list[Annotated[float, Field(ge=-4, le=4)]] = Field(default_factory=list, max_length=4)


@app.get("/reconstruction/status")
async def reconstruction_status(store: Store = Depends(authenticated)):
    location = settings().laven_reconstruction_path
    if location:
        try:
            await asyncio.to_thread(reconstruction.load_models, location)
            return {"available": True}
        except (OSError, ValueError, KeyError, RuntimeError, SafetensorError):
            return {"available": False, "message": "Reconstruction checkpoints failed verification."}
    return {"available": False, "message": "Not Trained · Import the AE and VAE Colab export to enable reconstruction."}


@app.get("/research/reconstruction")
async def published_reconstruction(store: Store = Depends(researcher)):
    location = settings().laven_reconstruction_path
    if not location:
        return None
    await reconstruction_models()
    root = Path(location)
    try:
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        protocol = manifest["protocol"]
        source = {reconstruction.TEXT_PROTOCOL: "ContractNLI pages with controlled blur",
                  reconstruction.DAE_PROTOCOL: "ContractNLI pages with controlled blur, noise or scan degradation"}
        return {"metrics": metrics, "test_count": manifest["split_counts"]["test"],
                "test_source": source.get(protocol, "fictional source documents"),
                "primary_model": "Denoising autoencoder" if protocol == reconstruction.DAE_PROTOCOL else "Autoencoder",
                "charts": [{"name": name, "title": title} for name, title in reconstruction.PLOTS.items()
                           if (root / "figures" / (name + ".png")).is_file()]}
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(503, "Reconstruction evaluation results are unavailable.") from error


@app.get("/research/reconstruction/charts/{chart}")
async def published_reconstruction_chart(chart: str, store: Store = Depends(researcher)):
    if chart not in reconstruction.PLOTS:
        raise HTTPException(404, "This chart does not exist.")
    await reconstruction_models()
    path = Path(settings().laven_reconstruction_path) / "figures" / (chart + ".png")
    if not path.is_file():
        raise HTTPException(404, "This chart has not been published.")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})


async def reconstruction_models():
    if not settings().laven_reconstruction_path:
        raise HTTPException(503, "Reconstruction checkpoints have not been imported.")
    try:
        return await asyncio.to_thread(reconstruction.load_models, settings().laven_reconstruction_path)
    except (OSError, ValueError, KeyError, RuntimeError, SafetensorError) as error:
        raise HTTPException(503, "Reconstruction checkpoints failed verification.") from error


@app.post("/reconstruction/upload")
async def restore_upload(request: Request, page: int = Query(1, ge=1, le=300),
                         model: Literal["ae", "vae", "both"] = "ae", store: Store = Depends(authenticated)):
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, "Restoration uploads are limited to 20 MiB.")
        data.extend(chunk)
    if data.startswith(b"%PDF-"):
        kind, source = "pdf", f"Uploaded PDF · page {page}"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        kind, source = "png", "Uploaded image"
    elif data.startswith(b"\xff\xd8\xff"):
        kind, source = "jpeg", "Uploaded image"
    else:
        raise HTTPException(415, "Choose a PDF, PNG or JPEG file.")
    models = await reconstruction_models()
    try:
        image, page_count = await asyncio.to_thread(reconstruction.render_upload, bytes(data), page, kind)
    except ValueError as error:
        detail = str(error) if type(error) is ValueError else "This file could not be rendered."
        raise HTTPException(422, detail) from error
    except (pymupdf.FileDataError, RuntimeError, OverflowError) as error:
        raise HTTPException(422, "This file could not be rendered.") from error
    selected = models if model == "both" else {model: models[model]}
    result = await asyncio.to_thread(reconstruction.reconstruct, selected, image, "gaussian", False, .5)
    return JSONResponse({**result, "source_kind": source, "page_count": page_count},
                        headers={"Cache-Control": "private, no-store"})


@app.post("/documents/{document_id}/reconstruction")
async def reconstruct_document(document_id: UUID, body: ReconstructionRequest, store: Store = Depends(authenticated)):
    row = await document(store, document_id)
    if body.page > row["page_count"]:
        raise HTTPException(404, "This page is unavailable.")
    models = await reconstruction_models()
    selected = models if body.model == "both" else {body.model: models[body.model]}
    if row["media_type"] == "application/pdf":
        original_bytes = await store.get_original(row["object_path"])
        rendered = await asyncio.to_thread(reconstruction.render_page, original_bytes, body.page)
        source_kind = "Original page render"
    else:
        pages = await store.read("document_pages", document_id=f"eq.{document_id}", page=f"eq.{body.page}", select="text", limit=1)
        if not pages:
            raise HTTPException(404, "This extracted page is unavailable.")
        rendered = await asyncio.to_thread(reconstruction.render_page, None, body.page, text=pages[0]["text"])
        source_kind = "Extracted text render"
    result = await asyncio.to_thread(reconstruction.reconstruct, selected, rendered, body.degradation, body.controlled, body.severity)
    return {**result, "source_kind": source_kind}


@app.post("/reconstruction/latent")
async def generate_latent(body: LatentRequest, store: Store = Depends(authenticated)):
    models = await reconstruction_models()
    image = await asyncio.to_thread(reconstruction.latent_sample, models["vae"], body.seed, body.coordinates)
    return {"image": image}
