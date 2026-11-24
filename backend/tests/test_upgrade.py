from types import SimpleNamespace

import pymupdf
import pytest

from legalshield import upgraded, verification


def test_exact_raw_offsets():
    text = "Header    before.\nIssue:\t\t30 calendar days applies."
    phrase = "Issue: 30 calendar days applies."
    start, end = verification.locate(text, phrase)
    assert text[start:end] == "Issue:\t\t30 calendar days applies."
    assert verification.locate(phrase + " " + phrase, phrase) is None


def test_authority_in_force_and_conflicts():
    old = {"document_id": "old", "family_id": "f", "authority_status": "verified",
           "effective_from": "2024-01-01", "effective_to": None, "precedence": 0}
    draft = {**old, "document_id": "draft", "authority_status": "unverified", "precedence": 20}
    change = {**old, "document_id": "change", "effective_from": "2025-01-01", "precedence": 2}
    assert upgraded.select_versions(["draft"], [old, draft, change], "2024-10-01")[0] == ["old"]
    assert upgraded.select_versions(["old"], [old, draft, change], "2025-01-01")[0] == ["change"]
    chosen, issues = upgraded.select_versions(["old"], [old, change, {**change, "document_id": "tie"}], "2026-01-01")
    assert not chosen and issues


def test_solver_under_conditions():
    pytest.importorskip("z3")
    fact = verification.Atom(key="approved", op="eq", value=True)
    target = verification.Atom(key="days", op="eq", value=30)
    rule = verification.Rule(when=[fact], unless=[], then=target, citations=["S1.1"])
    premise = verification.Rule(when=[], unless=[], then=fact, citations=["S1.1"])
    assert verification.solve([rule], target) == "insufficient_evidence"
    assert verification.solve([rule, premise], target) == "supported"
    contrary = verification.Rule(when=[], unless=[], then=verification.Atom(key="days", op="eq", value=60), citations=["S1.1"])
    assert verification.solve([rule, premise, contrary], target) == "inconsistent_premises"
    assert verification.solve([contrary], target) == "contradicted"


def test_bounded_regeneration_and_record():
    import asyncio

    class FakeGroq:
        model = "test-qwen"
        calls = 0

        async def json(self, instruction, evidence, **options):
            self.calls += 1
            return {"claims": [{"text": "The period is 30 days.", "semantic_key": "period 30 days",
                                "citations": [{"id": "S1.1", "quote": "The period is 30 days." if self.calls == 2
                                               else "An invalid source quote."}]}]}

    class FakeVerifier:
        artifact = "fixture"
        entailment_threshold = .8
        contradiction_threshold = .8

        def classify(self, *_):
            return dict(Entailment=.98, Contradiction=.01, NotMentioned=.01)

    rows = [{"document_id": "document", "name": "Fictional term", "page": 1, "clause": None,
             "parts": [{"start": 0, "end": 22, "text": "The period is 30 days."}]}]
    groq = FakeGroq()
    claims, record = asyncio.run(verification.verified_answer(
        groq, FakeVerifier(), "What is the period?", rows, [], {}, max_retries=1))
    assert groq.calls == 2 and claims
    assert record["attempts"][0]["claims"][0]["status"] == "invalid_citation"
    assert record["attempts"][1]["claims"][0]["status"] == "supported"
    assert record["status"] == "supported"
    with pymupdf.open(stream=upgraded.report_pdf(record), filetype="pdf") as rendered:
        assert "The period is 30 days" in rendered[0].get_text()


def test_verifier_missing_abstains_without_a_model_call():
    import asyncio

    rows = [{"document_id": "document", "parts": [{"text": "Some fictional text."}]}]
    claims, record = asyncio.run(verification.verified_answer(
        SimpleNamespace(model="qwen"), None, "Question?", rows, [], {}))
    assert claims == []
    assert record["status"] == "verifier_unavailable"


def test_semantic_entropy():
    assert verification.empirical_entropy([]) is None
    assert verification.empirical_entropy(["a", "b"]) == 1
    assert verification.empirical_entropy(["a", "a"]) == 0



def test_version_changes_preserve_judgment_about_missing_history():
    old = [{"text": "Original term provides 30 calendar days."}]
    new = [{"text": "Revised term provides 60 calendar days."}]
    result = upgraded.summarize_version_changes(old, new, [], [])
    assert result == {"added_passages": 1, "removed_passages": 1, "newly_observed_findings": []}
