"""Download pinned archives and preserve official document identity and offsets."""
import argparse
import hashlib
import json
import math
import random
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

LABELS = ("Entailment", "Contradiction", "NotMentioned")
OFFICIAL_SPLITS = {"train": "train", "dev": "validation", "test": "test"}
PINNED_SOURCES = {
    "contractnli": ("https://stanfordnlp.github.io/contract-nli/", "CC-BY-4.0"),
    "cuad": ("https://github.com/The-Atticus-Project/cuad", "CC-BY-4.0"),
    "legalbench": ("https://github.com/HazyResearch/legalbench", "task-specific"),
    "legalbench-rag": ("https://github.com/zeroentropy-ai/legalbenchrag", "source-specific"),
    "bipia": ("https://github.com/microsoft/BIPIA", "task-specific"),
    "laven": (None, "review before redistribution"),
}
TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def acquire(url, destination, sha256):
    if urlsplit(url).scheme != "https" or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Use an HTTPS release URL and a pinned SHA-256.")
    if destination.is_file():
        if digest(destination) != sha256:
            raise ValueError("Existing archive checksum differs.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        candidate = Path(tmp) / "download"
        with urlopen(url, timeout=45) as response, candidate.open("wb") as output:
            if urlsplit(response.geturl()).scheme != "https":
                raise ValueError("Redirect must use HTTPS.")
            shutil.copyfileobj(response, output)
        if digest(candidate) != sha256:
            raise ValueError("Downloaded archive checksum differs.")
        candidate.replace(destination)


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokens(text):
    return TOKEN.findall(normalized(text))


def windows(text, size=230, overlap=55):
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size.")
    offsets = list(TOKEN.finditer(text))
    result = []
    for start in range(0, len(offsets), size - overlap):
        last = min(start + size, len(offsets))
        a, b = offsets[start].start(), offsets[last - 1].end()
        result.append({"start": a, "end": b, "text": text[a:b]})
        if last == len(offsets):
            break
    return result


def validate_span(text, span):
    if len(span) != 2 or any(type(x) is not int for x in span):
        raise ValueError("Invalid span coordinates.")
    start, end = span
    if not 0 <= start < end <= len(text):
        raise ValueError(f"Span out of bounds: {span}")
    return {"start": start, "end": end, "text": text[start:end]}


def contractnli(directory):
    official = {}
    for filename, split in OFFICIAL_SPLITS.items():
        with (directory / f"{filename}.json").open(encoding="utf-8-sig") as source:
            data = json.load(source)
        for doc in data["documents"]:
            doc_id, text = str(doc["id"]), doc["text"]
            if doc_id in official:
                raise ValueError(f"Repeated official document ID: {doc_id}")
            official[doc_id] = split
            annotations = doc["annotation_sets"]
            if len(annotations) != 1:
                raise ValueError(f"Expected one adjudicated set in {doc_id}.")
            clauses = [validate_span(text, span) for span in doc["spans"]]
            for key, item in annotations[0]["annotations"].items():
                label = item["choice"]
                evidence = [clauses[index] for index in item["spans"]]
                if label not in LABELS or (label == "NotMentioned") != (not evidence):
                    raise ValueError(f"Inconsistent label and evidence in {doc_id}/{key}.")
                yield {
                    "sample_id": f"contractnli:{doc_id}:{key}", "document_id": doc_id,
                    "family_id": f"contractnli:{doc_id}", "partition": split,
                    "hypothesis_id": key, "claim": data["labels"][key]["hypothesis"],
                    "text": text, "label": label, "evidence": evidence,
                }


def cuad(path, partition):
    with path.open(encoding="utf-8-sig") as source:
        data = json.load(source)
    for doc in data["data"]:
        title = doc["title"]
        for para in doc["paragraphs"]:
            text = para["context"]
            for qa in para["qas"]:
                gold = []
                for answer in qa["answers"]:
                    start = answer["answer_start"]
                    span = validate_span(text, [start, start + len(answer["text"])])
                    if span["text"] != answer["text"]:
                        raise ValueError(f"CUAD source span differs for {qa['id']}")
                    gold.append(span)
                yield {
                    "sample_id": qa["id"], "document_id": title, "family_id": f"cuad:{title}",
                    "partition": partition, "text": text, "question": qa["question"], "evidence": gold,
                }


def legalbench(path, task, partition):
    import csv
    with path.open(encoding="utf-8-sig", newline="") as source:
        for index, row in enumerate(csv.DictReader(source, delimiter="\t")):
            if not row.get("answer"):
                raise ValueError(f"Missing LegalBench answer in row {index}")
            yield {"sample_id": f"{task}:{partition}:{index}", "partition": partition,
                   "task": task, "source_row": row, "independent_test": False}


def retrieval(path, corpus):
    with path.open(encoding="utf-8-sig") as source:
        data = json.load(source)
    tests = data["tests"] if isinstance(data, dict) else data
    root = corpus.resolve()
    documents = {}
    for index, item in enumerate(tests):
        gold = []
        for item_span in item["snippets"]:
            key = item_span["file_path"]
            ref = (root / key).resolve()
            if not ref.is_relative_to(root):
                raise ValueError("Corpus reference escapes the corpus directory.")
            if key not in documents:
                with ref.open(encoding="utf-8", newline="") as source:
                    documents[key] = source.read()
            span = validate_span(documents[key], item_span["span"])
            if item_span.get("answer") is not None and item_span["answer"] != span["text"]:
                raise ValueError(f"Retrieved gold offset mismatch in {key}")
            gold.append({"document_id": key, **span})
        yield {"sample_id": f"{path.stem}:{index}", "question": item["query"],
               "evidence": gold, "independent_test": False}


def clean_external(path, task):
    """Import only an independently reviewed clean derivative."""
    for row in read_jsonl(path):
        fields = {"id", "context", "question", "answer", "reviewed_clean", "license"}
        if set(row) != fields or row["reviewed_clean"] is not True:
            raise ValueError("Supply a reviewed clean subset with per-row provenance.")
        if any(not isinstance(row[field], str) or not row[field].strip() for field in fields - {"reviewed_clean"}):
            raise ValueError("Clean derivative fields must be nonempty strings.")
        yield {"sample_id": f"bipia:{task}:{row['id']}", "task": task,
               "license": row["license"], "text": row["context"],
               "question": row["question"], "answer": row["answer"]}


def grouped_partitions(groups, seed=13):
    ordered = sorted(set(groups))
    random.Random(seed).shuffle(ordered)
    raw = [len(ordered) * ratio for ratio in (.7, .1, .1, .1)]
    sizes = [math.floor(count) for count in raw]
    for index in sorted(range(4), key=lambda i: raw[i] - sizes[i], reverse=True)[:len(ordered) - sum(sizes)]:
        sizes[index] += 1
    partition = {}
    cursor = 0
    for name, size in zip(("train", "validation", "calibration", "test"), sizes, strict=True):
        partition.update(dict.fromkeys(ordered[cursor:cursor + size], name))
        cursor += size
    return partition


def leakage(records, near_threshold=.9):
    """Audit exact and near links without rewriting official splits."""
    families = {}
    texts = {}
    seen_ids = set()
    for row in records:
        sample, family, split = (row[key] for key in ("sample_id", "family_id", "partition"))
        if sample in seen_ids:
            raise ValueError(f"Repeated sample: {sample}")
        seen_ids.add(sample)
        families.setdefault(family, set()).add(split)
        given = row.get("text")
        if isinstance(given, str) and given.strip():
            value = normalized(given)
            texts[(family, split, hashlib.sha256(value.encode()).hexdigest())] = value
    issues = [{"kind": "family", "families": [family], "partitions": sorted(splits)}
              for family, splits in families.items() if len(splits) > 1]
    grams = {}
    for key, value in texts.items():
        words = value.split()
        grams[key] = set(zip(words, words[1:], words[2:], words[3:], words[4:]))
    keys = list(texts)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            if left[1] == right[1]:
                continue
            if texts[left] == texts[right]:
                score = 1.0
            elif grams[left] and grams[right] and min(len(grams[left]), len(grams[right])) >= near_threshold * max(len(grams[left]), len(grams[right])):
                score = len(grams[left] & grams[right]) / len(grams[left] | grams[right])
            else:
                continue
            if score >= near_threshold:
                issues.append({"kind": "text_overlap", "families": [left[0], right[0]],
                               "partitions": [left[1], right[1]], "similarity": score})
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=(*PINNED_SOURCES, "download", "audit"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--partition")
    parser.add_argument("--task")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--url")
    parser.add_argument("--sha256")
    parser.add_argument("--calibration-fraction", type=float, default=0.0)
    args = parser.parse_args()
    if args.dataset == "download":
        if not args.url or not args.sha256:
            parser.error("download requires --url and --sha256")
        acquire(args.url, args.output, args.sha256)
        return
    if args.dataset == "audit":
        args.output.write_text(json.dumps(leakage(read_jsonl(args.input)), indent=2), encoding="utf-8")
        return
    if not args.revision:
        parser.error("Record the release or source commit with --revision.")
    if args.dataset in {"cuad", "legalbench"} and not args.partition:
        parser.error("Supply the official --partition.")
    if args.dataset in {"legalbench", "bipia"} and not args.task:
        parser.error("Supply --task.")
    if args.dataset == "legalbench-rag" and args.corpus is None:
        parser.error("Supply --corpus.")
    loaders = {
        "contractnli": lambda: contractnli(args.input),
        "cuad": lambda: cuad(args.input, args.partition),
        "legalbench": lambda: legalbench(args.input, args.task, args.partition),
        "legalbench-rag": lambda: retrieval(args.input, args.corpus),
        "bipia": lambda: clean_external(args.input, args.task),
        "laven": lambda: read_jsonl(args.input),
    }
    rows = list(loaders[args.dataset]())
    if args.calibration_fraction:
        if args.dataset != "contractnli" or not 0 < args.calibration_fraction < 1:
            parser.error("--calibration-fraction requires ContractNLI and a value between 0 and 1.")
        groups = sorted({row["family_id"] for row in rows if row["partition"] == "validation"})
        random.Random(13).shuffle(groups)
        held = set(groups[:round(len(groups) * args.calibration_fraction)])
        if not held or len(held) == len(groups):
            parser.error("The grouped calibration allocation leaves an empty partition.")
        for row in rows:
            if row["family_id"] in held:
                row["partition"] = "calibration"
    write_jsonl(args.output, rows)
    paths = [args.input / f"{name}.json" for name in OFFICIAL_SPLITS] if args.dataset == "contractnli" else [args.input]
    source, license_name = PINNED_SOURCES[args.dataset]
    manifest = {"dataset": args.dataset, "revision": args.revision, "source": source, "license": license_name,
                "inputs": {p.name: digest(p) for p in paths}, "output_sha256": digest(args.output),
                "records": len(rows), "documents": len({r["document_id"] for r in rows if "document_id" in r}),
                "partitions": Counter(r.get("partition", "unspecified") for r in rows),
                "labels": Counter(r["label"] for r in rows if "label" in r)}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
