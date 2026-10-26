"""
clip_features.py
================
Shared CLIP feature extraction for the AI-image detector, used by both
extract_clip_features.py (training) and image_predictor.py (serving), so the
two paths crop and normalise identically.

Approach (UnivFD - Ojha et al., CVPR 2023): a frozen CLIP ViT-L/14 turns an
image into a 768-d embedding and a small trained head classifies it. CLIP's
features generalise to generators the head never saw far better than a CNN
trained from scratch.

CLIP's native input is 224x224, but the images we care about are 512px and up.
Instead of shrinking the whole image to 224 (which erases the fine generation
traces) we take several 224x224 crops at native resolution and average the
per-crop predictions, so every crop is seen pixel-for-pixel.
"""

import numpy as np
import torch
from PIL import Image

CLIP_MODEL = "openai/clip-vit-large-patch14"
CROP = 224
EMBED_DIM = 768

# CLIP's own normalisation constants.
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def crop_positions(width, height, grid):
    """Top-left corners of CROP x CROP windows.

    grid=2 gives the four corners plus the centre (5 crops, used in training);
    grid=3 gives a 3x3 grid, which includes those 5 (9 crops, used for larger
    uploads). Windows overlap when the image is small - that is fine.
    """
    xs = np.linspace(0, width - CROP, grid).round().astype(int)
    ys = np.linspace(0, height - CROP, grid).round().astype(int)
    pos = [(int(x), int(y)) for y in ys for x in xs]
    centre = ((width - CROP) // 2, (height - CROP) // 2)
    if grid == 2:
        pos.append(centre)
    return pos


def crops_tensor(image, grid=2):
    """PIL image -> float tensor [K, 3, 224, 224], normalised for CLIP.

    Images smaller than 224 on either side are upscaled just enough to fit;
    callers should treat those results as low-confidence.
    """
    image = image.convert("RGB")
    w, h = image.size
    if min(w, h) < CROP:
        scale = CROP / min(w, h)
        image = image.resize((max(CROP, round(w * scale)), max(CROP, round(h * scale))),
                             Image.BICUBIC)
        w, h = image.size

    arr = np.asarray(image, dtype=np.uint8)
    crops = [arr[y:y + CROP, x:x + CROP] for x, y in crop_positions(w, h, grid)]
    t = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255.0
    return (t - MEAN) / STD


def load_clip(device="cuda"):
    """Frozen CLIP vision tower with its projection head (fp16 on GPU)."""
    from transformers import CLIPVisionModelWithProjection

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL, dtype=dtype)
    return model.to(device).eval()


@torch.no_grad()
def embed(model, crops):
    """[K, 3, 224, 224] -> L2-normalised [K, 768] float32 embeddings."""
    param = next(model.parameters())
    feats = model(pixel_values=crops.to(param.device, param.dtype)).image_embeds.float()
    return torch.nn.functional.normalize(feats, dim=-1)
