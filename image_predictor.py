"""
image_predictor.py
==================
Inference wrapper for the AI-generated-image detector.

Two backends, picked at load time:
  - "clip"   (preferred) frozen CLIP ViT-L/14 + trained head, from
             train_clip_head.py. Analyses 224x224 crops at native resolution,
             so 512px+ uploads are examined pixel-for-pixel.
  - "resnet" (fallback) the 32x32 CIFAKE ResNet-18 from train_image_model.py.

Loaded lazily so app.py still starts (with the image tab disabled) when no
model has been trained yet.
"""

import io
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
CLIP_HEAD_PATH = os.path.join(ROOT, "ai_image_model", "clip_head.pt")
RESNET_PATH = os.path.join(ROOT, "ai_image_model", "image_model.pt")
RESNET_SIZE = 32

# Uploads are shrunk to this short side before cropping. The training images
# top out at 1024px; a 4000px phone photo cropped at native resolution would
# show the head sensor-level detail it never saw in training.
MAX_SHORT_SIDE = 1024

_state = {"backend": None, "model": None, "head": None, "mean": None, "std": None,
          "temperature": 1.0, "meta": None, "device": "cpu"}


def is_available():
    return os.path.isfile(CLIP_HEAD_PATH) or os.path.isfile(RESNET_PATH)


def load_model(device="cpu"):
    if _state["backend"] is not None:
        return
    if os.path.isfile(CLIP_HEAD_PATH):
        _load_clip(device)
    elif os.path.isfile(RESNET_PATH):
        _load_resnet(device)
    else:
        raise FileNotFoundError(
            "No image model found. Run: python train_clip_head.py "
            "(or python train_image_model.py for the 32x32 baseline)"
        )


def _load_clip(device):
    from clip_features import load_clip
    from train_clip_head import make_head

    ckpt = torch.load(CLIP_HEAD_PATH, map_location="cpu", weights_only=False)
    head = make_head(ckpt["head"])
    head.load_state_dict(ckpt["state_dict"])
    metrics = ckpt.get("metrics", {})
    _state.update({
        "backend": "clip",
        "model": load_clip(device),
        "head": head.to(device).eval(),
        "temperature": float(ckpt.get("temperature", 1.0)),
        "device": device,
        "meta": {"arch": f"CLIP ViT-L/14 + {ckpt['head']} head",
                 "test_accuracy": metrics.get("test_accuracy"),
                 "device": device},
    })


def _load_resnet(device):
    from image_model import ARCHITECTURES

    ckpt = torch.load(RESNET_PATH, map_location=device, weights_only=False)
    model = ARCHITECTURES[ckpt["arch"]]()
    model.load_state_dict(ckpt["state_dict"])
    _state.update({
        "backend": "resnet",
        "model": model.eval().to(device),
        "mean": ckpt["mean"].to(device),
        "std": ckpt["std"].to(device),
        "device": device,
        "meta": {"arch": ckpt["arch"],
                 "test_accuracy": ckpt.get("test_accuracy"),
                 "device": device},
    })


def metadata():
    return _state["meta"] or {}


def predict(image_bytes, device=None):
    """Return (label, css_class, confidence, fake_prob, real_prob, info)."""
    load_model(device or "cpu")
    image = Image.open(io.BytesIO(image_bytes))
    image.load()

    if _state["backend"] == "clip":
        fake_prob, real_prob, info = _predict_clip(image)
    else:
        fake_prob, real_prob, info = _predict_resnet(image)

    if real_prob > fake_prob:
        label, css = "REAL PHOTO", "result-real"
    else:
        label, css = "AI GENERATED", "result-fake"
    return label, css, max(real_prob, fake_prob), fake_prob, real_prob, info


def _predict_clip(image):
    from clip_features import CROP, crops_tensor, embed

    image = image.convert("RGB")
    w, h = image.size
    work = image
    if min(w, h) > MAX_SHORT_SIDE:
        scale = MAX_SHORT_SIDE / min(w, h)
        work = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

    # 3x3 grid when there is room for distinct crops, else corners + centre.
    grid = 3 if min(work.size) >= 2 * CROP else 2
    crops = crops_tensor(work, grid=grid)
    feats = embed(_state["model"], crops)
    with torch.no_grad():
        probs = F.softmax(_state["head"](feats), dim=-1).mean(0)
        # Same calibration as train_clip_head.py: temperature on log mean prob.
        probs = F.softmax(probs.clamp_min(1e-6).log() / _state["temperature"], dim=-1)

    info = {
        "original_size": f"{w}x{h}",
        "analysed_as": f"{len(crops)} crops of {CROP}x{CROP}"
                       + (f" from {work.size[0]}x{work.size[1]}" if work is not image
                          else " at full resolution"),
        "low_resolution": min(w, h) < CROP,
        "backend": "clip",
    }
    return float(probs[0]), float(probs[1]), info


def _predict_resnet(image):
    """Centre-crop to square, then resize to the 32x32 the model was trained on."""
    image = image.convert("RGB")
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    small = image.crop((left, top, left + side, top + side)).resize(
        (RESNET_SIZE, RESNET_SIZE), Image.BICUBIC)

    # np.array (not asarray) gives a writable copy, which torch.from_numpy wants.
    tensor = torch.from_numpy(np.array(small, dtype=np.uint8)).permute(2, 0, 1)
    tensor = tensor.float().unsqueeze(0).to(_state["device"]) / 255.0
    tensor = (tensor - _state["mean"]) / _state["std"]
    with torch.no_grad():
        probs = F.softmax(_state["model"](tensor), dim=1)[0]

    info = {
        "original_size": f"{w}x{h}",
        "analysed_as": f"shrunk to {RESNET_SIZE}x{RESNET_SIZE}",
        # Trained only on 32x32 CIFAKE images; large uploads are far outside that.
        "low_resolution": False,
        "out_of_distribution": min(w, h) > 128,
        "backend": "resnet",
    }
    return float(probs[0]), float(probs[1]), info
