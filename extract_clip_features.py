"""
extract_clip_features.py
========================
Runs every Tiny-GenImage image (from prepare_genimage.py) through frozen CLIP
ViT-L/14 once and caches the per-crop embeddings, so the classifier head in
train_clip_head.py trains in seconds instead of re-running CLIP every epoch.

Each image yields 5 native-resolution 224x224 crops (four corners + centre).

Filtering:
  - BigGAN is dropped: its images are 128x128, too small for a 224 crop
    without upscaling, and it is an obsolete generator.
  - Images under 224px on the short side are dropped (about 5% of the real
    photos) for the same reason.

JPEG augmentation: half of the training images are re-compressed at a random
quality (65-95) before cropping, for both classes. This stops the head from
using compression differences as a shortcut. The test split is extracted
twice - clean, and re-compressed at quality 75 - to measure how well the
detector holds up on images that have been shared and re-saved.

Output: genimage/features/{train,test,test_jpeg75}.pt
"""

import glob
import io
import os
import random
import time

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from clip_features import CROP, EMBED_DIM, crops_tensor, embed, load_clip

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "genimage")
FEAT_DIR = os.path.join(DATA, "features")
SEED = 42
DROP_GENERATORS = {"BigGAN"}
N_CROPS = 5


def load_metadata():
    files = sorted(glob.glob(os.path.join(DATA, "meta", "*.csv")))
    if not files:
        raise FileNotFoundError("No metadata found - run: python prepare_genimage.py")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    short = df[["width", "height"]].min(axis=1)
    keep = ~df["generator"].isin(DROP_GENERATORS) & (short >= CROP)
    print(f"{len(df)} images, keeping {keep.sum()} "
          f"(dropped {(~keep).sum()}: BigGAN or under {CROP}px)")
    return df[keep].reset_index(drop=True)


def jpeg(image, quality):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


class CropDataset(Dataset):
    def __init__(self, frame, mode):
        self.paths = list(frame["path"])
        self.mode = mode  # "train" | "clean" | "jpeg75"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(os.path.join(DATA, self.paths[i])) as im:
            im = im.convert("RGB")
            if self.mode == "train":
                rng = random.Random(SEED * 1_000_003 + i)
                if rng.random() < 0.5:
                    im = jpeg(im, rng.randint(65, 95))
            elif self.mode == "jpeg75":
                im = jpeg(im, 75)
            return crops_tensor(im, grid=2)


def extract(model, frame, mode, name):
    out_path = os.path.join(FEAT_DIR, f"{name}.pt")
    if os.path.isfile(out_path):
        print(f"{name}: already extracted, skipping")
        return

    loader = DataLoader(CropDataset(frame, mode), batch_size=16, num_workers=6,
                        pin_memory=True, prefetch_factor=4)
    feats = torch.empty(len(frame), N_CROPS, EMBED_DIM, dtype=torch.float16)
    t0, done = time.time(), 0
    for crops in loader:
        b = crops.shape[0]
        f = embed(model, crops.view(-1, 3, CROP, CROP)).view(b, N_CROPS, EMBED_DIM)
        feats[done:done + b] = f.half().cpu()
        done += b
        if done % 1600 < b:
            rate = done / (time.time() - t0)
            print(f"  {name}: {done}/{len(frame)} images, {rate:.1f} img/s, "
                  f"ETA {(len(frame) - done) / rate / 60:.1f} min", flush=True)

    torch.save({
        "features": feats,
        "label": torch.tensor(frame["label"].values),
        "generator": list(frame["generator"]),
        "path": list(frame["path"]),
    }, out_path)
    print(f"{name}: {len(frame)} images in {(time.time() - t0) / 60:.1f} min -> {out_path}")


def main():
    os.makedirs(FEAT_DIR, exist_ok=True)
    df = load_metadata()
    for split, part in df.groupby("split"):
        print(f"  {split}: {len(part)} images | "
              + ", ".join(f"{g} {n}" for g, n in part["generator"].value_counts().items()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {device} CLIP...")
    model = load_clip(device)

    extract(model, df[df["split"] == "train"], "train", "train")
    extract(model, df[df["split"] == "test"], "clean", "test")
    extract(model, df[df["split"] == "test"], "jpeg75", "test_jpeg75")


if __name__ == "__main__":
    main()
