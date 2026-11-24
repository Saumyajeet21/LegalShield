import asyncio
import hashlib
import re
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import PurePosixPath
from uuid import UUID, uuid4

import httpx
import numpy as np
import pymupdf
from defusedxml.common import DefusedXmlException
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from . import laven
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
    yield
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
    return {"researcher": store.researcher}


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
    question = body.question.strip()
    if len(question) < 3:
        raise HTTPException(422, "Enter a legal research question.")
    ids = list(dict.fromkeys(body.document_ids))
    for identifier in ids:
        await document(store, identifier)
    if body.conversation_id:
        await get_conversation(store, body.conversation_id)
    flags = laven.baseline(Page(0, question))
    hits, trusted = ([], []) if flags else await search(store, question, ids)
    findings = await read_findings(store, ids)
    claims = [] if flags else await laven.grounded(request.app.state.groq, question, hits, trusted)
    result = {"document_ids": [str(item) for item in ids], "claims": claims, "message": ("The question contains a potential AI-directed instruction. Rephrase it as a legal research question." if flags else laven.INSUFFICIENT) if not claims else None,
        "alerts": findings, "question_flagged": bool(flags),
        "retrieval": [{"id": h["id"], "document_id": h["document_id"], "page": h["page"], "revision": h["revision"],
                       "similarity": h["similarity"]} for h in hits]}
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
