import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from legalshield import api, verification
from legalshield.config import Settings


def test_real_model_through_ask_and_pdf(monkeypatch):
    location = os.environ.get("LAVEN_TEST_COLAB_EXPORT")
    if not location:
        pytest.skip("Point LAVEN_TEST_COLAB_EXPORT at the verified local Colab export.")
    root = Path(location)
    with pytest.raises(ValueError, match="deployable calibration"):
        verification.Verifier.load(root / "model", root / "calibration.json")
    verifier = verification.DocumentVerifier.load_research(root / "model", root / "calibration.json")
    document_id = str(uuid4())
    source = "The supplier must send written notice 30 calendar days before termination."
    passages = [source, "Payments are due upon receipt."]
    for claim, label in (
        (source, "Entailment"),
        ("The supplier may terminate without written notice.", "Contradiction"),
        ("The agreement grants a perpetual software license.", "NotMentioned"),
    ):
        scores = verifier.classify_passages(passages, claim)
        assert max(scores, key=scores.get) == label
    database = {}

    class Store:
        owner = str(uuid4())
        researcher = False

        async def read(self, table, **filters):
            if table == "documents" and filters["id"] == f"eq.{document_id}":
                return [{"id": document_id, "status": "indexed", "sha256": "a" * 64}]
            if table == "chat_turns" and filters["conversation_id"] == f"eq.{database.get('conversation')}":
                return [{"result": database["result"]}]
            return []

        async def rpc(self, name, args, **opts):
            assert name == "save_turn" and args["p_owner"] == self.owner and opts["privileged"]
            database.update(conversation=args["p_conversation"], result=args["p_result"])
            return True

    class Draft:
        model = "synthetic-qwen"

        @contextmanager
        def track_usage(self):
            yield {}

        async def json(self, instruction, data, **kwargs):
            assert data["evidence"][0]["id"] == "S1.1"
            return {"claims": [{
                "text": source, "semantic_key": "written notice 30 calendar days",
                "citations": [{"id": "S1.1", "quote": source}],
            }]}

    async def authenticate():
        return Store()

    async def search(store, question, ids):
        assert ids == [document_id]
        return [{"id": 1, "document_id": document_id, "name": "Fixture", "page": 1,
                 "clause": "Notice", "revision": 1, "similarity": .9,
                 "parts": [{"text": source, "start": 0, "end": len(source)}]}], []

    config = Settings(laven_enhanced=True)
    monkeypatch.setattr(api, "settings", lambda: config)
    monkeypatch.setattr(api, "search", search)
    monkeypatch.setattr(api.app.state, "groq", Draft(), raising=False)
    monkeypatch.setattr(api.app.state, "verifier", verifier, raising=False)
    monkeypatch.setitem(api.app.dependency_overrides, api.authenticated, authenticate)
    client = TestClient(api.app)
    response = client.post("/ask", json={"document_ids": [document_id], "question": "How much notice is required?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verification"]["status"] == "supported"
    assert body["verification"]["verifier_sha256"] == verifier.artifact
    assert body["claims"][0]["citations"][0]["quote"] == source
    record = body["verification"]
    uri = f"/conversations/{body['conversation_id']}/verification/{record['question_id']}.pdf"
    assert client.get(uri).content.startswith(b"%PDF-")
    assert client.get("/research/artifacts").status_code == 403
