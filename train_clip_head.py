"""
train_clip_head.py
==================
Trains the classifier head on top of cached CLIP features
(extract_clip_features.py). Every crop is a training sample; an image's
prediction is the mean of its crops' probabilities.

Two heads are compared on a validation slice of the training split - a linear
layer (UnivFD's choice) and a small MLP - and the better one is kept. The
test split is scored once at the end. A temperature is then fitted on the
validation slice so the reported confidence is calibrated, as with the news
model (see calibrate_model.py).

Output: ai_image_model/clip_head.pt and ai_image_model/clip_metrics.json
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from clip_features import CLIP_MODEL, CROP, EMBED_DIM

ROOT = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(ROOT, "genimage", "features")
OUT_DIR = os.path.join(ROOT, "ai_image_model")
SEED = 42



def load(name):
    d = torch.load(os.path.join(FEAT_DIR, f"{name}.pt"))
    return d["features"].float(), d["label"], np.array(d["generator"])


def image_logits(head, feats):
    """[N, K, 768] -> [N, 2] logits whose softmax is the mean crop probability."""
    with torch.no_grad():
        probs = F.softmax(head(feats), dim=-1).mean(1)
    return probs.clamp_min(1e-6).log()


def make_head(kind):
    if kind == "linear":
        return nn.Linear(EMBED_DIM, 2)
    return nn.Sequential(nn.Linear(EMBED_DIM, 256), nn.GELU(), nn.Dropout(0.3),
                         nn.Linear(256, 2))
    

def train_head(kind, x, y, device, epochs=40):
    torch.manual_seed(SEED)
    head = make_head(kind).to(device)
    # Flatten crops into samples; each keeps its image's label.
    xs = x.reshape(-1, EMBED_DIM).to(device)
    ys = y.repeat_interleave(x.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(xs), device=device)
        for i in range(0, len(xs), 512):
            idx = perm[i:i + 512]
            loss = F.cross_entropy(head(xs[idx]), ys[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return head.eval()


def accuracy(logits, y):
    return (logits.argmax(1).cpu() == y).float().mean().item()


def per_generator(logits, y, gens):
    pred = logits.argmax(1).cpu().numpy()
    return {g: float((pred[gens == g] == y.numpy()[gens == g]).mean())
            for g in sorted(set(gens))}


def fit_temperature(logits, y):
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    logits, y = logits.cpu(), y.cpu()

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y, gens = load("train")

    # Validation slice, stratified by generator so every source is represented.
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=0.1, random_state=SEED, stratify=gens)
    print(f"train {len(tr)} | val {len(va)} images")

    results = {}
    for kind in ("linear", "mlp"):
        head = train_head(kind, x[tr], y[tr], device)
        acc = accuracy(image_logits(head, x[va].to(device)), y[va])
        results[kind] = (acc, head)
        print(f"  {kind:<6} val accuracy {acc * 100:.2f}%")

    kind = max(results, key=lambda k: results[k][0])
    val_acc, head = results[kind]
    print(f"Keeping {kind} head")

    val_logits = image_logits(head, x[va].to(device))
    temperature = fit_temperature(val_logits, y[va])
    print(f"Temperature T = {temperature:.3f}")

    metrics = {"head": kind, "clip_model": CLIP_MODEL, "crop": CROP,
               "val_accuracy": val_acc, "temperature": temperature}
    print("\n=== Test (scored once) ===")
    for name in ("test", "test_jpeg75"):
        if not os.path.isfile(os.path.join(FEAT_DIR, f"{name}.pt")):
            continue
        tx, ty, tg = load(name)
        logits = image_logits(head, tx.to(device))
        acc = accuracy(logits, ty)
        gen_acc = per_generator(logits, ty, tg)
        metrics[f"{name}_accuracy"] = acc
        metrics[f"{name}_per_generator"] = gen_acc
        print(f"\n{name}: accuracy {acc * 100:.2f}%  ({len(ty)} images)")
        for g, a in gen_acc.items():
            print(f"   {g:<11} {a * 100:6.2f}%")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"head": kind, "state_dict": head.cpu().state_dict(),
                "temperature": temperature, "clip_model": CLIP_MODEL, "crop": CROP,
                "classes": {0: "FAKE", 1: "REAL"}, "metrics": metrics},
               os.path.join(OUT_DIR, "clip_head.pt"))
    with open(os.path.join(OUT_DIR, "clip_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nSaved {os.path.join(OUT_DIR, 'clip_head.pt')}")


if __name__ == "__main__":
    main()
