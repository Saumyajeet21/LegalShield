"""Verify and import a local LAVEN reconstruction export without general ZIP extraction."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from backend.legalshield.reconstruction import PLOTS, load_models
from .reconstruction import aggregate, dump

REQUIRED = ("manifest.json", "history.json", "dataset.json", "test-predictions.json",
            "metrics.json", "ae.safetensors", "vae.safetensors")
OPTIONAL = ("protocol-comparison.json", "protocol-predictions.json", *(f"figures/{chart}.png" for chart in PLOTS))


def import_export(archive, destination):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError("Choose a new reconstruction directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as package, TemporaryDirectory(prefix=".laven-import-", dir=destination.parent) as temporary:
        members = package.namelist()
        expected = set(REQUIRED) | set(OPTIONAL)
        if (len(set(members)) != len(members) or not set(REQUIRED) <= set(members)
                or set(members) - expected or sum(info.file_size for info in package.infolist()) > 100_000_000):
            raise ValueError("The reconstruction export has missing, excessive or unexpected entries.")
        stage = (Path(temporary) / "verified").resolve()
        stage.mkdir()
        for name in (*REQUIRED, *OPTIONAL):
            if name not in members:
                continue
            # Paths originate from constants above. Check containment as a second guard.
            path = (stage / name).resolve()
            if not path.is_relative_to(stage):
                raise ValueError("Unexpected reconstruction output path.")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(package.read(name))
        load_models(str(stage))
        load_models.cache_clear()
        manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
        dataset = json.loads((stage / "dataset.json").read_text(encoding="utf-8"))
        records = json.loads((stage / "test-predictions.json").read_text(encoding="utf-8"))
        groups = {name: set(dataset["source_doc_splits"][name]) for name in ("train", "validation", "test")}
        if (sum(map(len, groups.values())) != len(set.union(*groups.values()))
                or {name: len(group) for name, group in groups.items()} != manifest["split_counts"]):
            raise ValueError("The reconstruction split manifest failed validation.")
        pairs = {(row["source_id"], row["model"]) for row in records}
        if len(records) != len(pairs) or pairs != {(source, model) for source in groups["test"] for model in ("input", "ae", "vae")}:
            raise ValueError("The held-out reconstruction evaluations are incomplete.")
        dump(stage / "metrics.json", aggregate(records))
        stage.rename(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print("Verified reconstruction models:", import_export(args.archive, args.destination))
