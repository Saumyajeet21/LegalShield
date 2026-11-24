from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4
from zipfile import ZipFile
from io import BytesIO

import pymupdf
import torch
from fastapi.testclient import TestClient

from legalshield import api, reconstruction


def test_generation_and_published_analytics(monkeypatch):
    owner_doc, other_doc = str(uuid4()), str(uuid4())
    quote = "The supplier must give 30 days of written notice."

    class GroqFixture:
        model = "qwen-test"

        @contextmanager
        def track_usage(self):
            yield {"usage_responses": 0, "completion_attempts": 0, "prompt_tokens": 0, "completion_tokens": 0}

        async def json(self, instruction, data, **_):
            return {"title": "Notice", "clauses": [{"title": "Purpose", "text": quote}]}

    class StoreFixture:
        researcher = True

        async def read(self, table, **params):
            if table == "documents" and params["id"] == "eq." + owner_doc:
                return [{"id": owner_doc, "status": "indexed", "revision": 1}]
            if table == "document_chunks":
                return [{"page": 1, "start_offset": 0, "end_offset": 30, "parts": [{"start": 0, "end": 20}]},
                        {"page": 1, "start_offset": 15, "end_offset": 35, "parts": [{"start": 25, "end": 35}]}]
            if table == "security_findings":
                return [{"page": 1, "action": "isolated"}]
            return []

    async def authenticate():
        return StoreFixture()

    async def search(*_):
        return [{"document_id": owner_doc, "name": "source.txt", "page": 1, "clause": None,
                 "parts": [{"start": 0, "end": len(quote), "text": quote}]}], []

    monkeypatch.setitem(api.app.dependency_overrides, api.authenticated, authenticate)
    monkeypatch.setattr(api.app, "state", SimpleNamespace(groq=GroqFixture(), verifier=None))
    monkeypatch.setattr(api, "search", search)
    client = TestClient(api.app)

    generated = client.post("/generation/drafts", json={"kind": "notice", "requirements": {"Notice": "30 days"},
                                                        "document_ids": [owner_doc]})
    assert generated.status_code == 200
    assert generated.json()["drafts"][0]["metrics"]["requirements"][0]["matched"]
    assert generated.json()["drafts"][0]["metrics"]["tokens"] is None
    assert client.post("/generation/drafts", json={"kind": "notice", "requirements": {}, "document_ids": [other_doc]}).status_code == 404

    export = {"title": "Notice", "text": quote}
    pdf = client.post("/generation/export/pdf", json=export)
    with pymupdf.open(stream=pdf.content, filetype="pdf") as doc:
        assert "30 days" in doc[0].get_text()
    docx = client.post("/generation/export/docx", json=export)
    with ZipFile(BytesIO(docx.content)) as doc:
        assert b"30 days" in doc.read("word/document.xml")

    assert client.post("/simulation/scenarios", json={}).status_code == 404

    metrics = client.get("/documents/" + owner_doc + "/scan-metrics").json()
    assert metrics["spans"] == 1 and metrics["retention"] == 30 / 35 and metrics["confidence"] is None
    assert client.get("/documents/" + other_doc + "/scan-metrics").status_code == 404
    assert client.get("/research/charts/not-a-published-chart").status_code == 404


def test_image_reconstruction_inference_without_reference():
    image = torch.full((98, 105), .6)
    models = {"ae": reconstruction.EncoderDecoder().eval(), "vae": reconstruction.EncoderDecoder(True).eval()}
    result = reconstruction.reconstruct(models, image, "gaussian", False, .5)
    assert result["input_metrics"] is None and len(result["results"]) == 2
    assert all(output["image"].startswith("iVBOR") and output["metrics"] is None for output in result["results"])
    assert reconstruction.latent_sample(models["vae"], 42, [0, 1]).startswith("iVBOR")


def test_patch_alignment_and_text_reconstruction_skip():
    class Identity(torch.nn.Module):
        def forward(self, image):
            return image, None

    # Odd dimensions exercise padding and overlaps along both edges.
    page = torch.linspace(0, 1, 117 * 141).reshape(117, 141)
    torch.testing.assert_close(reconstruction.infer(Identity(), page), page)
    blank = torch.ones_like(page)
    for variational in (False, True):
        model = reconstruction.TextEncoderDecoder(variational).eval()
        with torch.no_grad():
            model.repair[-1].bias.fill_(-2)
        torch.testing.assert_close(reconstruction.infer(model, blank), blank)


def test_restore_uploaded_image_and_scanned_pdf(monkeypatch):
    image = torch.full((96, 100), .6)
    pixmap = reconstruction.image_pixmap(image)
    png = pixmap.tobytes("png")
    with pymupdf.open() as document:
        page = document.new_page(width=100, height=96)
        page.insert_image(page.rect, stream=png)
        assert not page.get_text()
        scan = document.tobytes()
    client = TestClient(api.app)
    endpoint = "/reconstruction/upload"
    assert client.post(endpoint, content=png).status_code == 401

    async def authenticated():
        return object()

    async def models():
        return {"ae": reconstruction.EncoderDecoder().eval(), "vae": reconstruction.EncoderDecoder(True).eval()}

    monkeypatch.setitem(api.app.dependency_overrides, api.authenticated, authenticated)
    monkeypatch.setattr(api, "reconstruction_models", models)
    for data in (png, pixmap.tobytes("jpeg"), scan):
        response = client.post(endpoint + "?model=both", content=data)
        assert response.status_code == 200
        result = response.json()
        assert result["page_count"] == 1 and result["original"].startswith("iVBOR")
        assert result["input"] is None and result["input_metrics"] is None
        assert [item["model"] for item in result["results"]] == ["ae", "vae"]
        assert all(item["image"].startswith("iVBOR") and item["metrics"] is None for item in result["results"])
    assert client.post(endpoint + "?page=2", content=scan).status_code == 422
    assert client.post(endpoint, content=b"wrong format").status_code == 415
    monkeypatch.setattr(api, "MAX_BYTES", 4)
    assert client.post(endpoint, content=png).status_code == 413
