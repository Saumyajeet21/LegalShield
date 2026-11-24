"""Train text-preserving reconstructions on real ContractNLI pages with controlled blur."""
import argparse
import json
import random
import shutil
import subprocess
import time
import unicodedata
from pathlib import Path
from urllib.parse import unquote
from zipfile import ZIP_DEFLATED, ZipFile

import pymupdf
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from backend.legalshield import reconstruction as r
from .reconstruction import aggregate, dump, plot

SEED = 2026
BETA = .0001
COUNTS = {"train": 30, "validation": 8, "test": 12}


def select_pages(archive, excluded_train, counts=COUNTS, excluded_test=()):
    members = {unquote(name.split("/")[-1]): name for name in archive.namelist() if name.startswith("contract-nli/raw/")}
    groups = {}
    for split, size in counts.items():
        official = "dev" if split == "validation" else split
        candidates = json.loads(archive.read(f"contract-nli/{official}.json"))["documents"]
        candidates = sorted(candidates, key=lambda row: row["file_name"])
        random.Random(SEED).shuffle(candidates)
        selected = []
        for row in candidates:
            name = row["file_name"]
            if split == "test" and name in excluded_test:
                continue
            member = members.get(name, members.get(unquote(name)))
            if not member or row["document_type"] != "search-pdf" or (split == "train" and row["id"] in excluded_train):
                continue
            with pymupdf.open(stream=archive.read(member), filetype="pdf") as document:
                for page in list(document)[:3]:
                    text = " ".join(page.get_text(sort=True).split())
                    words = page.get_text("words")
                    if 65 <= len(words) <= 700 and len(text) > 300:
                        image = r.render_grayscale(page, r.TEXT_RENDER_SIDE)
                        if min(image.shape) >= r.SIZE:
                            selected.append((name, page.number + 1, image, text))
                            break
            if len(selected) == size:
                break
        if len(selected) != size:
            raise ValueError(f"Only {len(selected)} eligible pages for {split}.")
        groups[split] = selected
    ids = [item[0] for rows in groups.values() for item in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("A source document occurs in more than one split.")
    return groups


def blur(image, level, style):
    sigma = 0.65 + 1.4 * level
    x = torch.arange(-5, 6, dtype=image.dtype)
    kernel = (-(x / sigma).square() / 2).exp()
    kernel /= kernel.sum()
    value = image[None, None]
    value = F.conv2d(F.pad(value, (5, 5, 0, 0), mode="replicate"), kernel[None, None, None, :])
    if style == "defocus":
        value = F.conv2d(F.pad(value, (0, 0, 5, 5), mode="replicate"), kernel[None, None, :, None])
    return value[0, 0]


def corrupt(clean, level, style, seed):
    source = blur(clean, level, "horizontal" if style == "motion" else "defocus")
    if style == "noise":
        return r.degrade(source, "gaussian", seed, .12 + .20 * level)
    if style == "scan":
        faded = r.degrade(source, "contrast", seed, .15 + .15 * level)
        return r.degrade(faded, "compression", seed, .36 + .30 * level)
    return source


DEGRADATIONS = ("defocus", "motion", "noise", "scan")


def patches(pages, count, mixed=False):
    rng = random.Random(SEED)
    examples, targets = [], []
    for page_index, (_, _, clean, _) in enumerate(pages):
        alternatives = ([corrupt(clean, level, style, SEED + page_index) for level in (.08, .38, .72)
                         for style in DEGRADATIONS] if mixed else
                        [blur(clean, level, style) for level in (.15, .42, .72) for style in ("defocus", "horizontal")])
        height, width = clean.shape
        for index in range(count):
            for _ in range(25):
                top, left = rng.randrange(height - r.SIZE + 1), rng.randrange(width - r.SIZE + 1)
                patch = clean[top:top + r.SIZE, left:left + r.SIZE]
                if patch.lt(.65).float().mean() > .014:
                    break
            noisy = alternatives[index % len(alternatives)]
            examples.append(noisy[top:top + r.SIZE, left:left + r.SIZE])
            targets.append(patch)
    return TensorDataset(torch.stack(examples)[:, None], torch.stack(targets)[:, None])


def loss_parts(prediction, target, kl, generated):
    prediction = prediction.float()
    weights = 1 + 4 * (1 - target)
    mse = F.mse_loss(prediction, target)
    pixel = ((prediction - target).abs() * weights).mean()
    dx, dy = prediction.diff(dim=3), prediction.diff(dim=2)
    edge = F.l1_loss(dx, target.diff(dim=3)) + F.l1_loss(dy, target.diff(dim=2))
    kld = kl.float() if kl is not None else mse.new_zeros(())
    aux = F.mse_loss(generated.float(), target) if generated is not None else mse.new_zeros(())
    return 2 * mse + pixel + .5 * edge + BETA * kld + .02 * aux, mse, kld


def train(kind, prepared, device, out, epochs, dae=False):
    torch.manual_seed(SEED)
    model = (r.DocumentDAE() if dae else r.TextEncoderDecoder(kind == "vae")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.0003)
    training = DataLoader(prepared["train"], batch_size=32, shuffle=True)
    validation = DataLoader(prepared["validation"], batch_size=32)
    checkpoint = out / (kind + ".safetensors")
    use_fp16 = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    best, history = float("inf"), []
    for epoch in range(1, epochs + 1):
        began = time.perf_counter()
        row = {"epoch": epoch}
        for mode, loader in (("train", training), ("validation", validation)):
            model.train(mode == "train")
            running = [0., 0., 0.]
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(model.training), torch.autocast(device.type, torch.float16, enabled=use_fp16):
                    if dae:
                        output, kl = model(inputs)
                        generated = None
                    else:
                        output, kl, generated = model.training_outputs(inputs, auxiliary=model.variational)
                    values = loss_parts(output, targets, kl, generated)
                    if dae:
                        with torch.autocast(device.type, enabled=False):
                            similarity = r.structural_similarity(output.float(), targets)
                        values = (values[0] + .35 * (1 - similarity), values[1], values[2])
                if model.training:
                    scaler.scale(values[0]).backward()
                    scaler.step(optimizer)
                    scaler.update()
                for index, value in enumerate(values):
                    running[index] += value.item() * len(inputs)
            row[mode] = dict(zip(("loss", "reconstruction", "kl"), (value / len(loader.dataset) for value in running), strict=True))
        scheduler.step()
        row["seconds"] = time.perf_counter() - began
        history.append(row)
        validation_mse = row["validation"]["reconstruction"]
        print(f"{kind} epoch {epoch}/{epochs}, validation MSE {validation_mse:.6f} ({row['seconds']:.1f}s)", flush=True)
        if validation_mse < best:
            best = validation_mse
            save_file({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint)
    model = model.cpu().eval()
    model.load_state_dict(load_file(checkpoint))
    return model, history


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def edit_distance(observed, reference):
    """Exact bit-parallel Levenshtein distance for characters or words."""
    if not reference:
        return len(observed)
    masks = {}
    for index, token in enumerate(reference):
        masks[token] = masks.get(token, 0) | (1 << index)
    positive, negative, score = -1, 0, len(reference)
    last = 1 << (score - 1)
    for token in observed:
        matched = masks.get(token, 0)
        vertical = matched | negative
        horizontal = (((matched & positive) + positive) ^ positive) | matched
        add = negative | ~(horizontal | positive)
        subtract = positive & horizontal
        score += bool(add & last) - bool(subtract & last)
        add, subtract = (add << 1) | 1, subtract << 1
        positive, negative = subtract | ~(vertical | add), add & vertical
    return score


def ocr_batch(folder, out):
    script = Path(__file__).with_name("windows_ocr.ps1").resolve()
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(script),
                    "-Folder", str(folder), "-Output", str(out)], check=True, timeout=180)
    return json.loads(out.read_text(encoding="utf-8-sig"))


def test_models(pages, models, legacy, out, mixed=False):
    records, previews, audit = [], [], []
    ocr_dir = out / "ocr-inputs"
    ocr_dir.mkdir(exist_ok=True)
    for index, (source, page, clean, text) in enumerate(pages):
        severity = (.26, .50, .70)[(index // len(DEGRADATIONS) if mixed else index) % 3]
        style = DEGRADATIONS[index % len(DEGRADATIONS)] if mixed else "defocus"
        degraded = corrupt(clean, severity, style, SEED + index) if mixed else blur(clean, severity, style)
        variants = {"clean": clean, "input": degraded}
        timings = {}
        for name, model in {**models, **{"legacy_" + name: model for name, model in legacy.items()}}.items():
            start = time.perf_counter()
            variants[name] = r.infer(model, degraded)
            timings[name] = (time.perf_counter() - start) * 1000
        for name, image in variants.items():
            key = f"{index:02}-{name}"
            (ocr_dir / (key + ".png")).write_bytes(r.image_pixmap(image).tobytes("png"))
            scores = r.quality(image, clean)
            if name in timings:
                scores["inference_ms"] = timings[name]
            if name in models:
                scores["parameters"] = sum(value.numel() for value in models[name].parameters())
            scores["foreground_fraction"] = image.lt(.85).float().mean().item()
            measured = {"source_id": source, "page": page, "model": name, "strength": severity, "degradation": style, **scores}
            audit.append((key, text, measured))
            if name in ("input", "ae", "vae"):
                records.append(measured)
        if index < 3:
            previews.append((f"controlled {style} ({severity})", {label: variants[name] for label, name in
                             (("Original", "clean"), ("Degraded", "input"), ("DAE" if mixed else "AE", "ae"), ("VAE", "vae"))}))
        print(f"Held-out page {index + 1}/{len(pages)} rendered", flush=True)
    ocr = ocr_batch(ocr_dir, out / "ocr-transcripts.json")
    for key, text, row in audit:
        expected, recognized = normalized(text), normalized(ocr[key])
        row["cer"] = edit_distance(recognized, expected) / len(expected)
        row["wer"] = edit_distance(recognized.split(), expected.split()) / len(expected.split())
    dump(out / "protocol-comparison.json", aggregate([row for _, _, row in audit]))
    dump(out / "protocol-predictions.json", [row for _, _, row in audit])
    return records, previews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--dae", action="store_true", help="Train a mixed-degradation DAE and retain the existing VAE.")
    parser.add_argument("--report-only", action="store_true", help="Rebuild figures and ZIP from the completed measured run.")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)
    eligibility = json.loads((args.archive.parent / "eligible.manifest.json").read_text(encoding="utf-8-sig"))
    excluded_train = {int(family.rsplit(":", 1)[1]) for family in eligibility["excluded_families"]}
    counts = {"train": 54, "validation": 12, "test": 18} if args.dae else COUNTS
    excluded_test = (json.loads((args.legacy / "dataset.json").read_text())["source_doc_splits"]["test"] if args.dae else ())
    with ZipFile(args.archive) as archive:
        pages = select_pages(archive, excluded_train, counts, excluded_test)
    dataset = {"source": "ContractNLI official source PDFs, one native-text page per document",
               "seed": SEED, "target": "clean native PDF raster",
               "input": "matched raster with defocus, motion blur, noise or scan compression" if args.dae else "matched raster with controlled blur",
               "source_doc_splits": {key: [row[0] for row in values] for key, values in pages.items()}}
    dump(out / "dataset.json", dataset)
    prepared = {"validation": patches(pages["validation"], 36, args.dae)}
    models = {}
    if args.report_only:
        histories = json.loads((out / "history.json").read_text())
        if len(histories["ae"]) != args.epochs or (not args.dae and len(histories["vae"]) != args.epochs):
            raise ValueError("The report must use the same epoch count as training.")
        records = json.loads((out / "test-predictions.json").read_text())
        if {row["source_id"] for row in records} != set(dataset["source_doc_splits"]["test"]):
            raise ValueError("The measured report does not match the selected test documents.")
        for kind in ("ae", "vae"):
            model = (r.DocumentDAE() if args.dae and kind == "ae" else r.TextEncoderDecoder(kind == "vae")).eval()
            model.load_state_dict(load_file(out / (kind + ".safetensors")))
            models[kind] = model
        previews = [(f"held-out page {i + 1}", {label: r.tensor_image(pymupdf.Pixmap(
            str(out / "ocr-inputs" / f"{i:02}-{name}.png")))
            for label, name in (("Original", "clean"), ("Degraded", "input"), ("AE", "ae"), ("VAE", "vae"))})
            for i in range(3)]
    else:
        prepared["train"] = patches(pages["train"], 72 if args.dae else 48, args.dae)
        histories = {}
        legacy = r.load_models(str(args.legacy.resolve())).copy()
        models["ae"], histories["ae"] = train("ae", prepared, device, out, args.epochs, args.dae)
        if args.dae:
            models["vae"] = legacy.pop("vae")
            shutil.copyfile(args.legacy / "vae.safetensors", out / "vae.safetensors")
            histories["vae"] = json.loads((args.legacy / "history.json").read_text())["vae"]
        else:
            models["vae"], histories["vae"] = train("vae", prepared, device, out, args.epochs)
        dump(out / "history.json", histories)
        records, previews = test_models(pages["test"], models, legacy, out, args.dae)
    val_noisy, val_target = prepared["validation"].tensors
    baseline_mse = F.mse_loss(val_noisy, val_target).item()
    validation = {kind: min(row["validation"]["reconstruction"] for row in history)
                  for kind, history in histories.items() if not args.dae or kind == "ae"}
    validation_passed = all(value < baseline_mse for value in validation.values())
    print(f"Validation baseline MSE {baseline_mse:.6f}, checkpoints {validation}", flush=True)
    metrics = aggregate(records)
    dump(out / "test-predictions.json", records)
    dump(out / "metrics.json", metrics)
    plot(out, histories, metrics, records, previews, models["vae"])
    manifest = {"protocol": r.DAE_PROTOCOL if args.dae else r.TEXT_PROTOCOL, "patch_size": r.SIZE, "latent_size": r.LATENT,
                "split_counts": counts, "device": str(device), "epochs": args.epochs,
                "validation_baseline_mse": baseline_mse, "validation_model_mse": validation,
                "ocr": "Windows Media OCR en-US; compared with exported PDF text",
                "deployable": validation_passed and args.epochs >= 2,
                "sha256": {filename: r.file_hash(out / filename) for filename in ("ae.safetensors", "vae.safetensors")}}
    dump(out / "manifest.json", manifest)
    with ZipFile(out / "laven-text-restoration.zip", "w", ZIP_DEFLATED) as export:
        for path in [*(out / name for name in ("manifest.json", "metrics.json", "history.json", "dataset.json",
                                               "test-predictions.json", "protocol-comparison.json",
                                               "protocol-predictions.json", *manifest["sha256"])),
                     *sorted((out / "figures").glob("*.png"))]:
            export.write(path, path.relative_to(out))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
