"""
train_image_model.py
====================
Trains the AI-generated-image detector on CIFAKE.

Everything lives on the GPU: the corpus is only ~368 MB as uint8, so both splits
are uploaded once and augmented with tensor ops per batch. No DataLoader, no
disk reads during training - for 32x32 images the usual Python-side pipeline
would be the bottleneck, not the GPU.

Model selection uses a validation slice carved out of the training set; the
20,000-image test split is scored exactly once, at the end, so the headline
number is not chosen by peeking at it.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import image_data
from image_model import ARCHITECTURES

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "ai_image_model")
SEED = 42


def augment(x, generator):
    """Random 32x32 crop from a reflect-padded 40x40, plus horizontal flip.

    Kept deliberately light: aggressive colour or blur augmentation would
    smear the high-frequency generation artifacts the model needs to see.
    """
    b, c, h, w = x.shape
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")

    i = torch.randint(0, 9, (b,), device=x.device, generator=generator)
    j = torch.randint(0, 9, (b,), device=x.device, generator=generator)
    rows = i[:, None] + torch.arange(h, device=x.device)
    cols = j[:, None] + torch.arange(w, device=x.device)

    out = padded.gather(2, rows.view(b, 1, h, 1).expand(b, c, h, padded.shape[3]))
    out = out.gather(3, cols.view(b, 1, 1, w).expand(b, c, h, w))

    flip = torch.rand(b, 1, 1, 1, device=x.device, generator=generator) < 0.5
    return torch.where(flip, out.flip(3), out)


@torch.no_grad()
def evaluate(model, x, y, mean, std, batch_size=512):
    model.eval()
    correct = 0
    preds_all = []
    for i in range(0, len(x), batch_size):
        xb = (x[i:i + batch_size].float() / 255.0 - mean) / std
        with torch.autocast("cuda", enabled=x.is_cuda):
            logits = model(xb)
        preds = logits.argmax(1)
        preds_all.append(preds)
        correct += (preds == y[i:i + batch_size]).sum().item()
    return correct / len(x), torch.cat(preds_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18", choices=list(ARCHITECTURES))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--benchmark", action="store_true",
                    help="Time a few steps, print a projected total, and exit.")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    train_x, train_y = image_data.load("train")
    test_x, test_y = image_data.load("test")

    # Validation slice held out of training, so the test split stays untouched
    # until the final score.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(train_x))
    n_val = int(0.05 * len(train_x))
    val_idx, tr_idx = order[:n_val], order[n_val:]

    def to_gpu(arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).to(device)

    xt = to_gpu(train_x[tr_idx])
    yt = torch.from_numpy(train_y[tr_idx]).to(device)
    xv = to_gpu(train_x[val_idx])
    yv = torch.from_numpy(train_y[val_idx]).to(device)
    xs = to_gpu(test_x)
    ys = torch.from_numpy(test_y).to(device)

    print(f"train {tuple(xt.shape)} | val {tuple(xv.shape)} | test {tuple(xs.shape)}")
    print(f"GPU memory held by data: {(xt.nbytes + xv.nbytes + xs.nbytes) / 1e6:.0f} MB")

    mean = (xt.float() / 255.0).mean((0, 2, 3)).view(1, 3, 1, 1)
    std = (xt.float() / 255.0).std((0, 2, 3)).view(1, 3, 1, 1)

    model = ARCHITECTURES[args.arch]().to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: {args.arch} ({n_params / 1e6:.2f}M params)")

    optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    steps_per_epoch = len(xt) // args.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=0.15,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    generator = torch.Generator(device=device).manual_seed(SEED)

    def run_steps(n_steps, perm, start):
        for k in range(n_steps):
            idx = perm[(start + k) * args.batch_size:(start + k + 1) * args.batch_size]
            xb = augment(xt[idx].float(), generator) / 255.0
            xb = ((xb - mean) / std).to(memory_format=torch.channels_last)
            with torch.autocast("cuda", enabled=(device == "cuda")):
                loss = criterion(model(xb), yt[idx])
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
        return float(loss.item())

    if args.benchmark:
        perm = torch.randperm(len(xt), device=device)
        run_steps(5, perm, 0)                      # warm-up / CUDA autotune
        torch.cuda.synchronize()
        t0 = time.time()
        run_steps(30, perm, 5)
        torch.cuda.synchronize()
        per_step = (time.time() - t0) / 30
        print(f"\n{per_step * 1000:.1f} ms/step  |  {steps_per_epoch} steps/epoch"
              f"  ->  {per_step * steps_per_epoch:.1f} s/epoch")
        print(f"Projected {args.epochs} epochs: "
              f"{per_step * steps_per_epoch * args.epochs / 60:.1f} minutes")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    best_val, best_state = 0.0, None
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(xt), device=device)
        loss = run_steps(steps_per_epoch, perm, 0)

        val_acc, _ = evaluate(model, xv, yv, mean, std)
        took = time.time() - t0
        flag = ""
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        print(f"epoch {epoch:>2}/{args.epochs}  loss {loss:.4f}  "
              f"val {val_acc * 100:.2f}%  {took:.1f}s{flag}")

    print(f"\nTraining took {(time.time() - t_start) / 60:.1f} minutes")

    model.load_state_dict(best_state)
    test_acc, preds = evaluate(model, xs, ys, mean, std)

    from sklearn.metrics import classification_report, confusion_matrix
    y_true = ys.cpu().numpy()
    y_pred = preds.cpu().numpy()
    print(f"\n=== Test set ({len(y_true)} images) - accuracy {test_acc * 100:.2f}% ===")
    print(classification_report(y_true, y_pred, target_names=["FAKE (AI)", "REAL"],
                                digits=4))
    print("confusion matrix [rows true, cols pred]:")
    print(confusion_matrix(y_true, y_pred))

    torch.save({
        "arch": args.arch,
        "state_dict": best_state,
        "mean": mean.cpu(),
        "std": std.cpu(),
        "val_accuracy": best_val,
        "test_accuracy": test_acc,
        "classes": {0: "FAKE", 1: "REAL"},
    }, os.path.join(OUT_DIR, "image_model.pt"))

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as fh:
        json.dump({"arch": args.arch, "params": n_params,
                   "val_accuracy": best_val, "test_accuracy": test_acc,
                   "epochs": args.epochs}, fh, indent=2)
    print(f"\nSaved model to {os.path.join(OUT_DIR, 'image_model.pt')}")


if __name__ == "__main__":
    main()
