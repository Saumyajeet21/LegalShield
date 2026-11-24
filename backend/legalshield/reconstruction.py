"""Image-only denoising. Original evidence is never modified or reindexed."""
import base64
import hashlib
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pymupdf
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

SIZE = 96
LATENT = 32
PROTOCOL = "laven-reconstruction-v1"
TEXT_PROTOCOL = "laven-text-restoration-v2"
DAE_PROTOCOL = "laven-text-restoration-v3"
TEXT_RENDER_SIDE = 1700
PLOTS = {
    "ae-learning": "Autoencoder learning curve", "vae-learning": "Variational autoencoder learning curve",
    "vae-objective": "VAE reconstruction and KL loss", "test-reconstructions": "Held-out reconstruction gallery",
    "test-strength": "Quality by degradation strength", "vae-latent": "Held-out latent coordinates",
    "vae-samples": "Generated layout samples", "test-mse": "Held-out MSE", "test-mae": "Held-out MAE",
    "test-psnr": "Held-out PSNR", "test-ssim": "Held-out SSIM", "test-cer": "Held-out CER",
    "test-wer": "Held-out WER", "test-inference_ms": "Held-out inference latency",
}


class EncoderDecoder(nn.Module):
    def __init__(self, variational=False):
        super().__init__()
        self.variational = variational
        self.encoder = nn.Sequential(nn.Conv2d(1, 12, 4, 2, 1), nn.SiLU(),
                                     nn.Conv2d(12, 24, 4, 2, 1), nn.SiLU(),
                                     nn.Conv2d(24, 32, 4, 2, 1), nn.SiLU())
        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 24, 4, 2, 1), nn.SiLU(),
                                     nn.ConvTranspose2d(24, 12, 4, 2, 1), nn.SiLU(),
                                     nn.ConvTranspose2d(12, 1, 4, 2, 1), nn.Sigmoid())
        if variational:
            self.mean = nn.Linear(32 * 12 * 12, LATENT)
            self.log_variance = nn.Linear(32 * 12 * 12, LATENT)
            self.from_latent = nn.Linear(LATENT, 32 * 12 * 12)

    def decode_latent(self, latent):
        return self.decoder(self.from_latent(latent).reshape(-1, 32, 12, 12))

    def forward(self, image):
        features = self.encoder(image)
        if not self.variational:
            return self.decoder(features), None
        flat = features.flatten(1)
        mean, log_variance = self.mean(flat), self.log_variance(flat).clamp(-12, 12)
        latent = mean + torch.randn_like(mean) * torch.exp(.5 * log_variance) if self.training else mean
        kl = -.5 * (1 + log_variance - mean.square() - log_variance.exp()).mean()
        return self.decode_latent(latent), kl


class TextEncoderDecoder(EncoderDecoder):
    """Restore residual detail while retaining the supplied glyph locations."""

    def __init__(self, variational=False):
        super().__init__(variational)
        self.detail = nn.Sequential(nn.Conv2d(1, 12, 5, padding=2), nn.SiLU(),
                                    nn.Conv2d(12, 12, 3, padding=1), nn.SiLU())
        self.context = nn.Conv2d(32, 8, 1)
        self.repair = nn.Sequential(nn.Conv2d(20, 12, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(12, 1, 3, padding=1))
        nn.init.zeros_(self.repair[-1].weight)
        nn.init.zeros_(self.repair[-1].bias)

    def training_outputs(self, image, auxiliary=False):
        features = self.encoder(image)
        kl, generated = None, None
        if self.variational:
            mean = self.mean(features.flatten(1))
            log_var = self.log_variance(features.flatten(1)).clamp(-12, 12)
            latent = mean + torch.randn_like(mean) * torch.exp(.5 * log_var) if self.training else mean
            kl = -.5 * (1 + log_var - mean.square() - log_var.exp()).mean()
            features = self.from_latent(latent).reshape(-1, 32, 12, 12)
            if auxiliary:
                generated = self.decoder(features)
        context = F.interpolate(self.context(features), size=image.shape[-2:], mode="bilinear", align_corners=False)
        correction = .42 * self.repair(torch.cat((self.detail(image), context), dim=1)).tanh()
        ink = F.max_pool2d(1 - image, kernel_size=5, stride=1, padding=2)
        return (image + correction * (ink * 5).clamp(0, 1)).clamp(0, 1), kl, generated

    def forward(self, image):
        restored, kl, _ = self.training_outputs(image)
        return restored, kl


class DocumentDAE(nn.Module):
    """Multi-scale denoiser with spatial skips and a bounded input residual."""

    def __init__(self):
        super().__init__()
        self.full = nn.Sequential(nn.Conv2d(1, 16, 5, padding=2), nn.SiLU(), nn.Conv2d(16, 16, 3, padding=1), nn.SiLU())
        self.half_encoder = nn.Sequential(nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.SiLU(),
                                         nn.Conv2d(24, 24, 3, padding=1), nn.SiLU())
        self.quarter = nn.Sequential(nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.SiLU(),
                                    nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.half_decoder = nn.Sequential(nn.Conv2d(56, 24, 3, padding=1), nn.SiLU())
        self.full_decoder = nn.Sequential(nn.Conv2d(40, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 1, 3, padding=1))
        nn.init.zeros_(self.full_decoder[-1].weight)
        nn.init.zeros_(self.full_decoder[-1].bias)

    def forward(self, image):
        full = self.full(image)
        half = self.half_encoder(full)
        low = F.interpolate(self.quarter(half), size=half.shape[-2:], mode="bilinear", align_corners=False)
        recovered = self.half_decoder(torch.cat((low, half), dim=1))
        recovered = F.interpolate(recovered, size=full.shape[-2:], mode="bilinear", align_corners=False)
        correction = .48 * self.full_decoder(torch.cat((recovered, full), dim=1)).tanh()
        ink = F.max_pool2d(1 - image, 5, stride=1, padding=2)
        return (image + correction * (ink * 5).clamp(0, 1)).clamp(0, 1), None


def tensor_image(pixmap):
    return torch.from_numpy(np.frombuffer(pixmap.samples, dtype=np.uint8).copy().reshape(pixmap.height, pixmap.width)).float() / 255


def image_pixmap(image):
    pixels = image.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).contiguous()
    return pymupdf.Pixmap(pymupdf.csGRAY, pixels.shape[-1], pixels.shape[-2], pixels.numpy().tobytes(), False)


def render_page(original, page_number, *, text=None, max_side=TEXT_RENDER_SIDE):
    if text is not None:
        with pymupdf.open() as doc:
            page = doc.new_page(width=420, height=594)
            from html import escape
            page.insert_htmlbox(pymupdf.Rect(25, 25, 395, 570), "<pre>" + escape(text[:5000]) + "</pre>",
                                css="pre { white-space: pre-wrap; font-size: 10pt; }")
            return tensor_image(page.get_pixmap(colorspace=pymupdf.csGRAY, alpha=False))
    with pymupdf.open(stream=original, filetype="pdf") as doc:
        if not 1 <= page_number <= len(doc):
            raise ValueError("Select an available page.")
        return render_grayscale(doc[page_number - 1], max_side)


def render_grayscale(page, max_side):
    width, height = page.rect.width, page.rect.height
    if any(not math.isfinite(side) or side <= 0 for side in (width, height)):
        raise ValueError("The page dimensions are invalid.")
    scale = min(2, max_side / max(width, height))
    return tensor_image(page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY, alpha=False))


def render_upload(data, page_number, kind):
    with pymupdf.open(stream=data, filetype=kind) as doc:
        if doc.needs_pass:
            raise ValueError("Upload an unlocked copy of the PDF.")
        if not 1 <= page_number <= len(doc):
            raise ValueError(f"Select a page from 1 to {len(doc)}." if len(doc) else "This file contains no pages.")
        page = doc[page_number - 1]
        images = page.get_image_info()
        if any(image["width"] * image["height"] > 25_000_000 for image in images):
            raise ValueError("Use a page with images of up to 25 megapixels each.")
        if kind != "pdf" and not images:
            raise ValueError("This image could not be rendered.")
        return render_grayscale(page, TEXT_RENDER_SIDE), len(doc)


def degrade(image, kind, seed, strength=.5):
    random = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(image.shape, generator=random)
    if kind == "gaussian":
        image = image + noise * (.04 + .27 * strength)
    elif kind == "speckle":
        image = image + image * noise * (.05 + .3 * strength)
    elif kind == "contrast":
        image = .5 + (image - .5) * (1 - .82 * strength)
    elif kind == "blur":
        radius = 1 + round(strength * 2)
        batch = F.pad(image[None, None], (radius,) * 4, mode="replicate")
        image = F.avg_pool2d(batch, 2 * radius + 1, stride=1)[0, 0]
    elif kind == "compression":
        jpeg = image_pixmap(image).tobytes("jpeg", jpg_quality=max(5, round(90 - 80 * strength)))
        decoded = pymupdf.Pixmap(jpeg)
        image = tensor_image(decoded if decoded.n == 1 else pymupdf.Pixmap(pymupdf.csGRAY, decoded))
    return image.clamp(0, 1)


def structural_similarity(predicted, clean):
    pool = lambda value: F.avg_pool2d(value, 7, 1, 3, count_include_pad=False)
    first, second = pool(predicted), pool(clean)
    return ((2 * first * second + .0001) * (2 * (pool(predicted * clean) - first * second) + .0009) /
            ((first.square() + second.square() + .0001) *
             ((pool(predicted.square()) - first.square()).clamp_min(0) +
              (pool(clean.square()) - second.square()).clamp_min(0) + .0009))).mean()


def quality(reconstruction, reference):
    predicted, clean = reconstruction[None, None], reference[None, None]
    error = predicted - clean
    mse, mae = error.square().mean().item(), error.abs().mean().item()
    return {"mse": mse, "mae": mae, "psnr": -10 * float(np.log10(mse)) if mse > 0 else None,
            "ssim": structural_similarity(predicted, clean).item(), "cer": None, "wer": None}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=2)
def load_models(location):
    root = Path(location)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    protocol = manifest.get("protocol")
    if (protocol not in (PROTOCOL, TEXT_PROTOCOL, DAE_PROTOCOL) or manifest.get("deployable") is not True
            or manifest.get("patch_size") != SIZE or manifest.get("latent_size") != LATENT):
        raise ValueError("The reconstruction export is incompatible or is a smoke run.")
    models = {}
    for kind in ("ae", "vae"):
        path = root / (kind + ".safetensors")
        if path.stat().st_size > 20_000_000 or file_hash(path) != manifest["sha256"][path.name]:
            raise ValueError("Reconstruction checkpoint verification failed.")
        if protocol == DAE_PROTOCOL and kind == "ae":
            model = DocumentDAE()
        else:
            factory = TextEncoderDecoder if protocol in (TEXT_PROTOCOL, DAE_PROTOCOL) else EncoderDecoder
            model = factory(kind == "vae")
        model.load_state_dict(load_file(path), strict=True)
        models[kind] = model.eval()
    return models


@torch.inference_mode()
def infer(model, image):
    height, width = image.shape
    stride = 64
    rows = max(1, (height - SIZE + stride - 1) // stride + 1)
    columns = max(1, (width - SIZE + stride - 1) // stride + 1)
    pad_h, pad_w = (rows - 1) * stride + SIZE, (columns - 1) * stride + SIZE
    padded = F.pad(image[None, None], (0, pad_w - width, 0, pad_h - height), mode="replicate")
    patches = F.unfold(padded, SIZE, stride=stride).transpose(1, 2).reshape(-1, 1, SIZE, SIZE)
    batch = torch.cat([model(part)[0] for part in patches.split(16)])
    flat = batch.flatten(1).T[None]
    weight = torch.hann_window(SIZE, periodic=False).clamp_min(.05)
    window = (weight[:, None] * weight[None, :]).reshape(1, -1, 1)
    reconstructed = F.fold(flat * window, (pad_h, pad_w), SIZE, stride=stride)
    overlap = F.fold(torch.ones_like(flat) * window, (pad_h, pad_w), SIZE, stride=stride)
    return (reconstructed / overlap)[0, 0, :height, :width]


def png(image):
    return base64.b64encode(image_pixmap(image).tobytes("png")).decode("ascii")


def reconstruct(models, image, mode, controlled, severity):
    source = degrade(image, mode, 2026, severity) if controlled else image
    output = []
    for key, model in models.items():
        started = time.perf_counter()
        result = infer(model, source)
        elapsed = round((time.perf_counter() - started) * 1000)
        output.append({"model": key, "image": png(result), "inference_ms": elapsed,
                       "parameters": sum(weight.numel() for weight in model.parameters()),
                       "metrics": quality(result, image) if controlled else None})
    return {"original": png(image), "input": png(source) if controlled else None,
            "input_metrics": quality(source, image) if controlled else None, "results": output, "controlled": controlled}


@torch.inference_mode()
def latent_sample(model, seed, coordinates):
    latent = torch.randn(1, LATENT, generator=torch.Generator().manual_seed(seed))
    if coordinates:
        latent[0, :len(coordinates)] = torch.tensor(coordinates, dtype=latent.dtype)
    return png(model.decode_latent(latent)[0, 0])
