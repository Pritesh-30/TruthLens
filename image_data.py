"""
image_data.py
=============
Loads the CIFAKE image corpus into memory.

The dataset is 120,000 32x32 RGB JPEGs, which is only ~368 MB as raw uint8 -
small enough to hold entirely in RAM (and in GPU memory). Decoding it once into
a .npy cache means training does no disk I/O at all, which removes the usual
data-loading bottleneck for tiny images.

Layout expected (as produced by the CIFAKE download):
    archive/train/REAL, archive/train/FAKE
    archive/test/REAL,  archive/test/FAKE

Labels follow the same convention as the news model: 0 = FAKE, 1 = REAL.
"""

import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(ROOT, "archive")
CACHE = os.path.join(ROOT, "image_cache")
IMG_SIZE = 32
CLASSES = {"FAKE": 0, "REAL": 1}


def list_images(directory):
    """JPEGs only - WSL leaves a ':Zone.Identifier' stream file beside each."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and "Zone.Identifier" not in f
    )


def build_split(split):
    xs, ys = [], []
    for name, label in CLASSES.items():
        directory = os.path.join(ARCHIVE, split, name)
        files = list_images(directory)
        print(f"  {split}/{name}: {len(files)} images")
        for i, path in enumerate(files):
            with Image.open(path) as im:
                im = im.convert("RGB")
                if im.size != (IMG_SIZE, IMG_SIZE):
                    im = im.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
                xs.append(np.asarray(im, dtype=np.uint8))
            ys.append(label)
            if (i + 1) % 20000 == 0:
                print(f"    ...{i + 1}/{len(files)}")
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def load(split, rebuild=False):
    """Return (images uint8 NHWC, labels int64), decoding to cache on first use."""
    os.makedirs(CACHE, exist_ok=True)
    x_path = os.path.join(CACHE, f"{split}_x.npy")
    y_path = os.path.join(CACHE, f"{split}_y.npy")

    if not rebuild and os.path.isfile(x_path) and os.path.isfile(y_path):
        return np.load(x_path), np.load(y_path)

    print(f"Decoding {split} split (one-time)...")
    x, y = build_split(split)
    np.save(x_path, x)
    np.save(y_path, y)
    print(f"  cached {x.shape} -> {x_path} ({x.nbytes / 1e6:.0f} MB)")
    return x, y


if __name__ == "__main__":
    for split in ("train", "test"):
        x, y = load(split)
        print(f"{split}: {x.shape} labels {np.bincount(y)}")
