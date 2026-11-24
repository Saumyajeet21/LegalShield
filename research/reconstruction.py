"""Prepare fictional legal layout patches, train AE/VAE, and evaluate held-out corruptions."""
import argparse
import json
import random
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pymupdf
import torch
from safetensors.torch import save_file
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from backend.legalshield.reconstruction import (LATENT, PROTOCOL, SIZE, EncoderDecoder, degrade, file_hash,
                                                image_pixmap, quality, tensor_image)

KINDS = ("gaussian", "blur", "contrast", "speckle", "compression")
PRESETS = {"smoke": (10, 5, 5, 1), "fast": (480, 120, 120, 8), "full": (3000, 500, 500, 20)}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def page_for(source, seed):
    rng = random.Random(seed + 100_003 * source)
    words = ("Supplier", "Customer", "Licensor", "Reviewer")
    title = rng.choice(("AGREEMENT", "NOTICE", "ADDENDUM", "SCHEDULE"))
    actor = rng.choice(words)
    lines = [title, f"Ref: {source:05d}", "", f"{actor} must give", f"{rng.randint(5, 90)} days of notice.",
             "", f"Fee: USD {rng.randint(5, 920)}", "Internal template"]
    font = rng.choice(("helv", "tiro", "cour"))
    x, y = rng.randint(4, 10), rng.randint(8, 17)
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=SIZE, height=SIZE)
        for index, line in enumerate(lines):
            if line:
                page.insert_text((x, y + 8 * index), line, fontname=font,
                                 fontsize=7 if index < 2 else 6, color=(0.06, 0.06, 0.06))
        image = tensor_image(page.get_pixmap(colorspace=pymupdf.csGRAY, alpha=False))
    return image, "\n".join(line for line in lines if line)


def make_split(ids, seed):
    inputs, originals, references = [], [], []
    for index, source in enumerate(tqdm(ids, desc="Rendering fictional source documents")):
        clean, text = page_for(source, seed)
        kind, severity = KINDS[index % len(KINDS)], (.25, .5, .75)[(index // len(KINDS)) % 3]
        inputs.append(degrade(clean, kind, 100_003 * seed + source, severity))
        originals.append(clean)
        references.append({"source_id": source, "degradation": kind, "strength": severity, "text": text})
    return TensorDataset(torch.stack(inputs)[:, None], torch.stack(originals)[:, None]), references


def run_epoch(model, loader, device, beta, epoch, epochs, optimizer, scaler):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    with torch.set_grad_enabled(training):
        progress = tqdm(loader, desc=f"{'Train' if training else 'Validate'} {epoch}/{epochs}", leave=False)
        for degraded, clean in progress:
            degraded, clean = degraded.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                predicted, kl = model(degraded)
                reconstruction = F.mse_loss(predicted.float(), clean)
                kl = kl.float() if kl is not None else clean.new_zeros(())
                loss = reconstruction + beta * kl
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            totals += np.array([loss.item(), reconstruction.item(), kl.item()]) * len(clean)
            progress.set_postfix(loss=f"{loss.item():.4f}")
    return dict(zip(("loss", "reconstruction", "kl"), (totals / len(loader.dataset)).tolist(), strict=True))


def train(kind, data, args, epochs, device, manifest):
    torch.manual_seed(args.seed + (kind == "vae"))
    model = EncoderDecoder(kind == "vae").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.fp32)
    signature = {"dataset": manifest, "beta": args.beta, "batch_size": args.batch_size, "fp16": scaler.is_enabled()}
    checkpoint = args.out / (kind + "-resume.pt")
    weights = args.out / (kind + ".safetensors")
    best, history = float("inf"), []
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Resume requires the same split, precision, beta and batch size.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best = state["history"], state["best"]
        if not weights.is_file():
            raise ValueError("The best weights accompanying the resume checkpoint are missing.")
    validation = DataLoader(data["validation"][0], batch_size=args.batch_size, pin_memory=device.type == "cuda")
    for epoch in range(len(history) + 1, epochs + 1):
        generator = torch.Generator().manual_seed(args.seed + 7919 * epoch)
        torch.manual_seed(args.seed + 1231 * epoch)
        loader = DataLoader(data["train"][0], batch_size=args.batch_size, shuffle=True, generator=generator,
                            pin_memory=device.type == "cuda")
        began = time.perf_counter()
        train_metrics = run_epoch(model, loader, device, args.beta, epoch, epochs, optimizer, scaler)
        valid_metrics = run_epoch(model, validation, device, args.beta, epoch, epochs, None, scaler)
        history.append({"epoch": epoch, "train": train_metrics, "validation": valid_metrics,
                        "seconds": time.perf_counter() - began})
        remaining = sum(row["seconds"] for row in history) / len(history) * (epochs - epoch)
        print(f"{kind.upper()} epoch {epoch}/{epochs} | validation {valid_metrics['loss']:.5f} | ETA {remaining / 60:.1f} min", flush=True)
        if valid_metrics["loss"] < best:
            best = valid_metrics["loss"]
            temporary = weights.with_suffix(".tmp")
            save_file({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
            temporary.replace(weights)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "best": best, "history": history}, temporary)
        temporary.replace(checkpoint)
    from safetensors.torch import load_file
    model.load_state_dict(load_file(weights))
    return model.eval(), history


def edit_distance(actual, expected):
    row = list(range(len(expected) + 1))
    for index, item in enumerate(actual):
        new = [index + 1]
        for position, target in enumerate(expected):
            new.append(min(row[position] + (item != target), row[position + 1] + 1, new[-1] + 1))
        row = new
    return row[-1]


def ocr_scores(image, text, enabled):
    if not enabled:
        return {}
    output = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "6"],
                            input=image_pixmap(image).tobytes("png"), capture_output=True, timeout=15, check=True)
    target, actual = " ".join(text.casefold().split()), " ".join(output.stdout.decode().casefold().split())
    return {"cer": edit_distance(actual, target) / len(target),
            "wer": edit_distance(actual.split(), target.split()) / len(target.split())}


@torch.inference_mode()
def evaluate(models, test, device, beta, ocr):
    dataset, references = test
    records, previews = [], []
    for (variant, clean), reference in tqdm(zip(dataset, references, strict=True), total=len(dataset), desc="Held-out test"):
        degraded, original = variant[0], clean[0]
        preview = {"Original": original, "Degraded": degraded}
        baseline = {**quality(degraded, original), **ocr_scores(degraded, reference["text"], ocr)}
        records.append({**reference, "model": "input", **baseline})
        for name, model in models.items():
            batch = variant[None].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            output, kl = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            milliseconds = (time.perf_counter() - started) * 1000
            image = output[0, 0].cpu()
            scores = {**quality(image, original), **ocr_scores(image, reference["text"], ocr)}
            kl_value = float(kl.item()) if kl is not None else None
            records.append({**reference, "model": name, **scores, "inference_ms": milliseconds,
                            "parameters": sum(weight.numel() for weight in model.parameters()), "kl": kl_value,
                            "loss": scores["mse"] + beta * (kl_value or 0)})
            preview[name.upper()] = image
        if len(previews) < 5:
            previews.append((reference["degradation"], preview))
    return records, previews


def aggregate(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["model"]].append(row)
    keys = ("mse", "mae", "psnr", "ssim", "cer", "wer", "inference_ms", "kl", "loss", "parameters")
    return {name: {key: float(np.mean(values)) if (values := [row[key] for row in group if row.get(key) is not None]) else None
                   for key in keys} for name, group in groups.items()}


def plot(out, histories, scores, records, previews, vae):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(figures / (name + ".png"), dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    for model, history in histories.items():
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        for split in ("train", "validation"):
            ax.plot([row["epoch"] for row in history], [row[split]["loss"] for row in history], marker="o", label=split.title())
        ax.set(title=model.upper() + " training and validation", xlabel="Epoch", ylabel="Training objective")
        ax.legend()
        save(fig, model + "-learning")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, key, title in zip(axes, ("reconstruction", "kl"), ("VAE reconstruction loss", "VAE KL divergence"), strict=True):
        for split in ("train", "validation"):
            ax.plot([row["epoch"] for row in histories["vae"]], [row[split][key] for row in histories["vae"]],
                    marker="o", label=split.title())
        ax.set(title=title, xlabel="Epoch"); ax.legend()
    save(fig, "vae-objective")
    names = {"input": "Degraded input", "ae": "Autoencoder", "vae": "VAE"}
    for key in ("mse", "mae", "psnr", "ssim", "cer", "wer", "inference_ms"):
        measured = [(name, row[key]) for name, row in scores.items() if row[key] is not None]
        if measured:
            fig, ax = plt.subplots(figsize=(10, 3), layout="constrained")
            ax.barh([names[name] for name, _ in measured], [value for _, value in measured], color="#167b70")
            ax.bar_label(ax.containers[0], fmt="%.4g", padding=4)
            ax.margins(x=.17); ax.set(title="Held-out mean " + key.upper(), xlabel=key.upper())
            save(fig, "test-" + key)
    fig, axes = plt.subplots(len(previews), 4, figsize=(10, 2.3 * len(previews)), squeeze=False, layout="constrained")
    for axs, (degradation, images) in zip(axes, previews, strict=True):
        for ax, (name, image) in zip(axs, images.items(), strict=True):
            ax.imshow(image.numpy(), cmap="gray", vmin=0, vmax=1); ax.set_title(name); ax.axis("off")
        axs[1].set_title("Degraded · " + degradation)
    save(fig, "test-reconstructions")
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    for name in names:
        levels = sorted({row["strength"] for row in records if row["model"] == name})
        ax.plot(levels, [np.mean([row["mse"] for row in records if row["model"] == name and row["strength"] == level])
                         for level in levels], marker="o", label=names[name])
    ax.set(title="Reconstruction quality by degradation severity", xlabel="Degradation strength", ylabel="Mean squared error")
    ax.legend(); save(fig, "test-strength")
    with torch.inference_mode():
        generator = torch.Generator().manual_seed(2026)
        latent = torch.randn(8, LATENT, generator=generator).to(next(vae.parameters()).device)
        outputs = vae.decode_latent(latent).cpu()
        layout_patches = []
        for _, preview in previews:
            page = preview["Original"]
            candidates = page.unfold(0, SIZE, 48).unfold(1, SIZE, 48).reshape(-1, SIZE, SIZE)
            layout_patches.append(candidates[candidates.var(dim=(1, 2)).argmax()])
        test_inputs = torch.stack(layout_patches)[:, None].to(latent.device)
        embeddings = vae.mean(vae.encoder(test_inputs).flatten(1)).cpu().numpy()
    fig, axes = plt.subplots(2, 4, figsize=(10, 5), layout="constrained")
    for index, ax in enumerate(axes.flat):
        ax.imshow(outputs[index, 0].numpy(), cmap="gray", vmin=0, vmax=1); ax.axis("off")
    fig.suptitle("Synthetic VAE layout samples"); save(fig, "vae-samples")
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c="#167b70")
    for index, (x, y) in enumerate(embeddings[:, :2]):
        ax.annotate(previews[index][0], (x, y))
    ax.set(title="Encoded held-out layouts", xlabel="Latent coordinate 1", ylabel="Latent coordinate 2")
    save(fig, "vae-latent")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=PRESETS, default="fast")
    parser.add_argument("--out", type=Path, required=True)
    for option in ("train", "validation", "test", "epochs"):
        parser.add_argument("--" + option, type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beta", type=float, default=.005)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--ocr", action="store_true", help="Evaluate OCR via an installed Tesseract binary.")
    args = parser.parse_args()
    counts = dict(zip(("train", "validation", "test", "epochs"), PRESETS[args.mode], strict=True))
    counts.update({key: getattr(args, key) for key in counts if getattr(args, key) is not None})
    epochs = counts.pop("epochs")
    if min(*counts.values(), epochs, args.batch_size) < 1 or not 0 <= args.beta <= 1:
        parser.error("Use positive sizes, epochs, batch size and beta between zero and one.")
    if args.ocr and not shutil.which("tesseract"):
        parser.error("--ocr requires Tesseract on PATH.")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}", flush=True)
    ids = list(range(sum(counts.values())))
    random.Random(args.seed).shuffle(ids)
    splits, start = {}, 0
    for split, count in counts.items():
        splits[split], start = ids[start:start + count], start + count
    dataset = {"source": "Locally generated fictional legal layout patches", "license": "Project-generated, no third-party contract text",
               "seed": args.seed, "size": SIZE, "degradations": list(KINDS), "strengths": [.25, .5, .75],
               "source_doc_splits": splits}
    dump(args.out / "dataset.json", dataset)
    data = {name: make_split(source_ids, args.seed) for name, source_ids in splits.items()}
    models, histories = {}, {}
    for kind in ("ae", "vae"):
        models[kind], histories[kind] = train(kind, data, args, epochs, device, dataset)
    records, previews = evaluate(models, data["test"], device, args.beta, args.ocr)
    metrics = aggregate(records)
    dump(args.out / "metrics.json", metrics)
    dump(args.out / "test-predictions.json", records)
    dump(args.out / "history.json", histories)
    plot(args.out, histories, metrics, records, previews, models["vae"])
    manifest = {"protocol": PROTOCOL, "mode": args.mode, "patch_size": SIZE, "latent_size": LATENT,
                "device": str(device), "beta": args.beta, "epochs": epochs, "split_counts": counts,
                "deployable": args.mode != "smoke" and epochs >= 2 and counts["train"] >= 100 and counts["test"] >= 30,
                "sha256": {name + ".safetensors": file_hash(args.out / (name + ".safetensors")) for name in models}}
    dump(args.out / "manifest.json", manifest)
    export = args.out / "laven-reconstruction.zip"
    public = [*manifest["sha256"], "metrics.json", "history.json", "manifest.json", "dataset.json", "test-predictions.json"]
    with ZipFile(export, "w", compression=ZIP_DEFLATED) as package:
        for path in [*(args.out / name for name in public), *sorted((args.out / "figures").glob("*.png"))]:
            package.write(path, path.relative_to(args.out))
    print(export, json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
