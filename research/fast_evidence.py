"""Validation-only context admission and batched evidence tokenization."""
from collections import defaultdict
from functools import lru_cache

import torch

from .evidence_retrieval import covers, recall


def admit_context(tokenizer, retriever, validation):
    """Admit only configurations within the coverage tolerances of top-5/512."""
    examples = [row for row in validation if row["evidence"]]
    ranked = {row["sample_id"]: retriever.select(
        row["text"], row["claim"], row["hypothesis_id"], 5) for row in examples}
    trials = []
    for limit in (256, 384, 512):
        encoded = {}
        for start in range(0, len(examples), 64):
            rows = examples[start:start + 64]
            ordered = [(row, part) for row in rows for part in ranked[row["sample_id"]]]
            batch = tokenizer([part["text"] for _, part in ordered],
                              [row["claim"] for row, _ in ordered],
                              max_length=limit, truncation="only_first", stride=64,
                              return_overflowing_tokens=True, return_offsets_mapping=True)
            spans = defaultdict(list)
            for index, original in enumerate(batch["overflow_to_sample_mapping"]):
                row, part = ordered[original]
                offsets = [(a, b) for (a, b), seq in zip(batch["offset_mapping"][index],
                           batch.sequence_ids(index), strict=True) if seq == 0 and a < b]
                if offsets:
                    a, b = part["start"], part["start"]
                    spans[original].append(({"start": a + offsets[0][0], "end": b + offsets[-1][1]},
                                             len(batch["input_ids"][index])))
            for i, (row, _) in enumerate(ordered):
                encoded.setdefault(row["sample_id"], []).append(spans[i])
        for count in (3, 4, 5):
            recovered = recall(examples, lambda r: [span for pieces in encoded[r["sample_id"]][:count]
                                                    for span, _ in pieces])
            tokens = sum(length for parts in encoded.values() for pieces in parts[:count] for _, length in pieces)
            trials.append({"windows": count, "max_length": limit, "token_proxy": tokens, **recovered})
    baseline = next(row for row in trials if row["windows"] == 5 and row["max_length"] == 512)
    accepted = [row for row in trials if
                row["any_evidence_recall"] >= baseline["any_evidence_recall"] - .03 and
                row["all_evidence_recall"] >= baseline["all_evidence_recall"] - .05 and
                row["span_recall"] >= baseline["span_recall"] - .05]
    return {"selected": min(accepted, key=lambda row: (row["token_proxy"], row["max_length"])),
            "reference": baseline, "candidates": trials}


def preencode(tokenizer, rows, limit):
    """Encode unique premise/hypothesis pairs in batches, without fixed padding."""
    keys = list(dict.fromkeys((text, row.hypothesis) for row in rows for text in row.windows))
    result = defaultdict(list)
    for start in range(0, len(keys), 256):
        batch_keys = keys[start:start + 256]
        batch = tokenizer([key[0] for key in batch_keys], [key[1] for key in batch_keys],
                          truncation="only_first", max_length=limit, stride=64,
                          return_overflowing_tokens=True)
        for index, ids in zip(batch["overflow_to_sample_mapping"], batch["input_ids"], strict=True):
            result[batch_keys[index]].append(torch.tensor(ids))
    return dict(result)
