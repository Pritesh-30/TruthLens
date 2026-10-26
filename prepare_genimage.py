"""
prepare_genimage.py
===================
Downloads Tiny-GenImage (TheKernel01/Tiny-GenImage on Hugging Face) one parquet
shard at a time, writes each image to disk, then deletes the shard. Peak extra
disk use is one shard (~0.5 GB) rather than the full 8.4 GB download.

Images are kept at their native resolution - no resizing - because the
generation traces the detector looks for live in fine pixel detail.

Format handling (the main shortcut risk in GenImage): real images are ImageNet
JPEGs while generated ones are PNGs, so a model can learn "PNG = fake".
  - Real JPEGs are written byte-for-byte, so they are not compressed twice.
  - Generated PNGs are encoded once as JPEG at quality 95.
Both classes therefore end up as singly-compressed JPEGs. Training adds random
JPEG re-compression on top, which removes the remaining quality difference.

Output:
    genimage/images/<split>/<generator>/<shard>_<row>.jpg
    genimage/meta/<shard>.csv    one row per image, written when a shard is done
    genimage/done_shards.txt     shards already processed (the script resumes)

Labels follow the project convention: 0 = FAKE, 1 = REAL.
"""

import csv
import io
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

REPO = "TheKernel01/Tiny-GenImage"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "genimage")
RAW = os.path.join(ROOT, "genimage_raw")
DONE = os.path.join(OUT, "done_shards.txt")

# Stop before Windows itself runs short: the WSL disk grows into C:.
MIN_FREE_GB = 8
FREE_SPACE_PATH = "/mnt/c" if os.path.isdir("/mnt/c") else ROOT

# Dataset's own generator class ids.
GENERATORS = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney",
              "SD14", "SD15", "VQDM", "Wukong"]
FIELDS = ["path", "label", "generator", "split", "width", "height", "source_format"]


def free_gb():
    return shutil.disk_usage(FREE_SPACE_PATH).free / 1e9


def list_shards():
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet"))


def save_image(raw_bytes, out_path):
    """Write one image; return (width, height, source_format)."""
    with Image.open(io.BytesIO(raw_bytes)) as im:
        fmt = im.format
        width, height = im.size
        if fmt == "JPEG":
            with open(out_path, "wb") as fh:
                fh.write(raw_bytes)
        else:
            im.convert("RGB").save(out_path, "JPEG", quality=95)
    return width, height, fmt


def process_shard(shard, pool):
    split = "train" if "/train-" in shard else "test"
    stem = os.path.basename(shard).replace(".parquet", "")

    t0 = time.time()
    local = hf_hub_download(REPO, shard, repo_type="dataset", local_dir=RAW)
    size_mb = os.path.getsize(local) / 1e6
    took = time.time() - t0
    print(f"  downloaded {size_mb:.0f} MB in {took:.0f}s ({size_mb * 8 / max(took, 1e-6):.0f} Mbit/s)",
          flush=True)

    rows = []
    t1 = time.time()
    pf = pq.ParquetFile(local)
    row = 0
    for batch in pf.iter_batches(batch_size=64, columns=["image", "label", "generator"]):
        images = batch.column("image").to_pylist()
        gens = batch.column("generator").to_pylist()
        jobs = []
        for img, gen in zip(images, gens):
            gen_name = GENERATORS[gen]
            rel = os.path.join("images", split, gen_name, f"{stem}_{row:05d}.jpg")
            os.makedirs(os.path.join(OUT, os.path.dirname(rel)), exist_ok=True)
            jobs.append((rel, gen_name, pool.submit(save_image, img["bytes"],
                                                    os.path.join(OUT, rel))))
            row += 1
        for rel, gen_name, fut in jobs:
            try:
                w, h, fmt = fut.result()
            except Exception as exc:
                print(f"  skipped {rel}: {exc}")
                continue
            rows.append({
                "path": rel,
                # Dataset uses 1 = fake; the project uses 0 = FAKE, 1 = REAL.
                "label": 1 if gen_name == "Real" else 0,
                "generator": gen_name,
                "split": split,
                "width": w,
                "height": h,
                "source_format": fmt,
            })
        del images, gens

    os.makedirs(os.path.join(OUT, "meta"), exist_ok=True)
    with open(os.path.join(OUT, "meta", stem + ".csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    os.remove(local)
    print(f"  wrote {len(rows)} images in {time.time() - t1:.0f}s, deleted shard", flush=True)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.isfile(DONE):
        with open(DONE) as fh:
            done = {line.strip() for line in fh if line.strip()}

    shards = list_shards()
    todo = [s for s in shards if s not in done]
    print(f"{len(shards)} shards, {len(done)} already done, {len(todo)} to go. "
          f"Free space: {free_gb():.1f} GB", flush=True)

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, shard in enumerate(todo, 1):
            if free_gb() < MIN_FREE_GB:
                print(f"STOPPED: only {free_gb():.1f} GB free (minimum {MIN_FREE_GB} GB).")
                sys.exit(1)
            print(f"[{i}/{len(todo)}] {shard}", flush=True)
            process_shard(shard, pool)
            with open(DONE, "a") as fh:
                fh.write(shard + "\n")
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(todo) - i)
            print(f"  elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min, "
                  f"free {free_gb():.1f} GB", flush=True)

    shutil.rmtree(RAW, ignore_errors=True)
    print(f"DONE in {(time.time() - t_start) / 60:.1f} min. Free space: {free_gb():.1f} GB")


if __name__ == "__main__":
    main()
