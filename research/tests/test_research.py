import numpy as np

from research import calibration, datasets, experiments, pilot


def test_grouped_pilot():
    records = list(pilot.pilot())
    assert len(records) == 800
    assert {r["human_review_status"] for r in records} == {"pending"}
    grouped = {}
    for row in records:
        grouped.setdefault(row["family_id"], set()).add(row["partition"])
        for span in row["expected_evidence"]:
            assert row["text"][span["start"]:span["end"]] == span["text"]
    assert len(grouped) == 100
    assert all(len(splits) == 1 for splits in grouped.values())


def test_leakage_exact_and_group():
    rows = [{"sample_id": str(i), "family_id": "family", "partition": split, "text": "A contract term."}
            for i, split in enumerate(["train", "test"])]
    assert {item["kind"] for item in datasets.leakage(rows)} == {"family", "text_overlap"}
    source = "  Unicode: café; sections 1 and 2."
    for span in datasets.windows(source, size=3, overlap=1):
        assert source[span["start"]:span["end"]] == span["text"]


def test_version_robustness_needs_correct_original():
    def row(category, correct):
        return {"family": "one", "type": category, "correct": correct, "answered": True,
                "answerable": True, "calls": None, "latency_ms": None}
    scenarios = [row("legitimate", False), row("authorized_amendment", True), row("irrelevant", True)]
    assert experiments.score(scenarios)["version_robustness_score"] == 0
    scenarios[0]["correct"] = True
    assert experiments.score(scenarios)["version_robustness_score"] == 1


def test_calibration():
    p = np.array([[.9,.05,.05], [.05,.9,.05], [.05,.05,.9]])
    y = np.arange(3)
    assert calibration.calibrate(np.log(p), y) > 0
    report = calibration.quality(p, y)
    assert report["aurc"] == 0
    assert report["incorrect_auroc"] is None
