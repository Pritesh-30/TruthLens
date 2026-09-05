"""
image_predictor.py
==================
Inference wrapper for the AI-generated-image detector.

Loaded lazily so app.py still starts (with the image tab disabled) when the
model has not been trained yet.
"""

import io
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from image_model import ARCHITECTURES

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "ai_image_model", "image_model.pt")
IMG_SIZE = 32

_state = {"model": None, "mean": None, "std": None, "meta": None}


def is_available():
    return os.path.isfile(MODEL_PATH)


def load_model(device="cpu"):
    if _state["model"] is not None:
        return _state["model"]
    if not is_available():
        raise FileNotFoundError(
            "ai_image_model/image_model.pt not found. Run: python train_image_model.py"
        )

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = ARCHITECTURES[ckpt["arch"]]()
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    _state.update({
        "model": model,
        "mean": ckpt["mean"].to(device),
        "std": ckpt["std"].to(device),
        "meta": {"arch": ckpt["arch"],
                 "test_accuracy": ckpt.get("test_accuracy"),
                 "device": device},
    })
    return model


def metadata():
    return _state["meta"] or {}


def preprocess(image):
    """Centre-crop to square, then resize to the 32x32 the model was trained on.

    Cropping before resizing avoids squashing the aspect ratio of a photo, which
    would introduce distortions the model never saw in training.
    """
    image = image.convert("RGB")
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)

    # np.array (not asarray) gives a writable copy, which torch.from_numpy wants.
    tensor = torch.from_numpy(
        np.array(image, dtype=np.uint8)
    ).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor, (w, h)


def predict(image_bytes, device="cpu"):
    """Return (label, css_class, confidence, fake_prob, real_prob, info)."""
    model = load_model(device)
    image = Image.open(io.BytesIO(image_bytes))
    tensor, original_size = preprocess(image)

    tensor = ((tensor.to(device) - _state["mean"]) / _state["std"])
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]

    fake_prob = float(probs[0].item())
    real_prob = float(probs[1].item())

    if real_prob > fake_prob:
        label, css = "REAL PHOTO", "result-real"
    else:
        label, css = "AI GENERATED", "result-fake"

    info = {
        "original_size": f"{original_size[0]}x{original_size[1]}",
        "downscaled_to": f"{IMG_SIZE}x{IMG_SIZE}",
        # The model was trained only on 32x32 CIFAKE images. A large upload is
        # far outside that distribution, so flag it rather than quietly
        # reporting a confident number.
        "out_of_distribution": min(original_size) > 128,
    }
    return label, css, max(real_prob, fake_prob), fake_prob, real_prob, info
