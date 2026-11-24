from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from legalshield import api, verification


def test_upgrade_route_visibility_and_owner_scope(monkeypatch):
    document_id, conversation_id, question_id = (str(uuid4()) for _ in range(3))
    access = []

    class OwnerStore:
        owner = str(uuid4())
        researcher = False

        async def read(self, table, **params):
            access.append((table, params))
            if table == "documents":
                return [{"id": document_id, "status": "indexed"}] if params["id"] == f"eq.{document_id}" else []
            if table == "legal_document_versions":
                return []
            if table == "chat_turns" and params["conversation_id"] == f"eq.{conversation_id}":
                value = verification.seal({"question_id": question_id, "question": "What does the term say?",
                                           "status": "supported", "as_of": None, "attempts": []})
                return [{"result": {"verification": value}}] if params["result->verification->>question_id"] == f"eq.{question_id}" else []
            return []

    settings = SimpleNamespace(laven_enhanced=True)
    monkeypatch.setattr(api, "settings", lambda: settings)

    async def authenticate():
        return OwnerStore()

    endpoint = "/conversations/" + conversation_id + "/verification/" + question_id + ".pdf"
    api.app.dependency_overrides[api.authenticated] = authenticate
    try:
        client = TestClient(api.app)
        assert client.get("/research/artifacts").status_code == 403
        assert client.get("/documents/" + document_id + "/legal-versions").json() == []
        assert client.get("/documents/" + str(uuid4()) + "/legal-versions").status_code == 404
        assert client.get(endpoint).content.startswith(b"%PDF-")
        assert client.get(endpoint.replace(question_id, str(uuid4()))).status_code == 404
    finally:
        api.app.dependency_overrides.pop(api.authenticated)
    assert any(table == "chat_turns" for table, _ in access)
