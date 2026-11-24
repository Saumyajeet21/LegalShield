"""ContractNLI evidence retrieval and validation-only coverage diagnostics."""
from collections import Counter
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .datasets import tokens, windows


@lru_cache(maxsize=768)
def candidates(text):
    return tuple(windows(text, size=160, overlap=80)) or ({"start": 0, "end": 0, "text": ""},)


def covers(window, span):
    return window["start"] <= span["start"] and window["end"] >= span["end"]


class Retriever:
    def __init__(self, training):
        if any(row["partition"] != "train" for row in training):
            raise ValueError("Fit the evidence retriever on training contracts only.")
        documents = {row["document_id"]: row["text"] for row in training}
        chunks = {key: candidates(text) for key, text in documents.items()}
        offsets = {}
        corpus = []
        for key, items in chunks.items():
            offsets[key] = len(corpus)
            corpus.extend(item["text"] for item in items)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        matrix = self.vectorizer.fit_transform(corpus)
        self.ranking = {}
        by_hypothesis = {}
        for row in training:
            if row["evidence"]:
                by_hypothesis.setdefault(row["hypothesis_id"], []).append(row)
        for hypothesis, rows in by_hypothesis.items():
            indexes, labels = [], []
            for row in rows:
                for i, item in enumerate(chunks[row["document_id"]]):
                    indexes.append(offsets[row["document_id"]] + i)
                    labels.append(int(any(covers(item, span) for span in row["evidence"])))
            if len(set(labels)) == 2:
                ranker = LogisticRegression(class_weight="balanced", max_iter=160, random_state=13)
                self.ranking[hypothesis] = ranker.fit(matrix[indexes], labels)

    @lru_cache(maxsize=768)
    def features(self, text):
        return self.vectorizer.transform([part["text"] for part in candidates(text)])

    def artifact(self):
        return {"schema_version": 1, "window_tokens": 160, "overlap_tokens": 80, "lexical_weight": .35,
                "ngram_range": [1, 2], "sublinear_tf": True, "lowercase": True, "norm": "l2",
                "vocabulary": self.vectorizer.vocabulary_, "idf": self.vectorizer.idf_.tolist(),
                "rankers": {key: {"coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}
                            for key, model in self.ranking.items()}}

    def select(self, text, hypothesis, hypothesis_id, count):
        parts = candidates(text)
        features = self.features(text)
        query = self.vectorizer.transform([hypothesis])
        lexical = (features @ query.T).toarray().ravel()
        ranker = self.ranking.get(hypothesis_id)
        scores = lexical if ranker is None else ranker.decision_function(features) + .35 * lexical
        return [parts[i] for i in np.argsort(-scores, kind="stable")[:count]]


def original_select(row):
    claim = set(tokens(row["claim"]))
    return sorted(windows(row["text"]), key=lambda window: -len(claim & set(tokens(window["text"]))))[:4]


def recall(rows, select):
    count, hit, spans, recovered, complete = 0, 0, 0, 0, 0
    for row in rows:
        if not row["evidence"]:
            continue
        count += 1
        chosen = select(row)
        checks = [any(covers(part, span) for part in chosen) for span in row["evidence"]]
        spans += len(checks)
        recovered += sum(checks)
        hit += any(checks)
        complete += all(checks)
    return {"annotated_pairs": count, "any_evidence_recall": hit/count, "all_evidence_recall": complete/count,
            "span_recall": recovered/spans}


def coverage_report(training, validation, retriever):
    if any(row["partition"] != "validation" for row in validation):
        raise ValueError("Evidence diagnostics require validation only.")
    return {
        "train_labels": dict(Counter(row["label"] for row in training)),
        "validation_labels": dict(Counter(row["label"] for row in validation)),
        "original_lexical_top4": recall(validation, original_select),
        "learned_top4": recall(validation, lambda r: retriever.select(r["text"], r["claim"], r["hypothesis_id"], 4)),
        "learned_top5": recall(validation, lambda r: retriever.select(r["text"], r["claim"], r["hypothesis_id"], 5)),
    }
