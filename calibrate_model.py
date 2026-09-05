"""
calibrate_model.py
==================
Post-hoc temperature scaling (Guo et al., 2017, "On Calibration of Modern
Neural Networks").

A DistilBERT classifier trained to near-zero loss reports ~100% confidence on
almost everything, including the cases it gets wrong. That is miscalibration,
not accuracy: the softmax scores are not usable as probabilities.

Temperature scaling fits a single scalar T on held-out data and divides the
logits by it before the softmax. It cannot change any prediction (dividing all
logits by a positive constant preserves their order, so accuracy is identical)
- it only rescales how confident the reported probability is. T is fitted by
minimising negative log-likelihood, so the resulting numbers are the honest
ones, not chosen for appearance.

Writes the fitted value to calibration.json, which app.py loads at startup.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from news_cleaning import clean_text
from retrain_model import MAX_LEN, SEED, build_frame, make_short

MODEL_DIR = "fake_news_model_v2"


def expected_calibration_error(probs, correct, n_bins=10):
    """Mean gap between stated confidence and observed accuracy."""
    conf = probs.max(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return ece


def collect_logits(model, tokenizer, texts, device, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [clean_text(t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        padding=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            out.append(model(**enc).logits.float().cpu())
    return torch.cat(out)


def fit_temperature(logits, labels):
    """Minimise NLL over a single scalar T."""
    log_T = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_T], lr=0.1, max_iter=200)

    def closure():
        optimiser.zero_grad()
        loss = F.cross_entropy(logits / log_T.exp(), labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_T.exp().item())


def main():
    root = os.path.dirname(os.path.abspath(__file__)) or "."
    rng = np.random.default_rng(SEED)

    # Rebuild exactly the held-out slices retrain_model.py kept aside, using
    # the same seed. None of these rows were seen during training.
    fake = pd.read_csv(os.path.join(root, "Fake.csv")).fillna("")
    true = pd.read_csv(os.path.join(root, "True.csv")).fillna("")
    fake["label"] = 0
    true["label"] = 1
    isot = pd.concat([fake, true], ignore_index=True)[["title", "text", "label"]]
    _, isot_eval = train_test_split(isot, test_size=0.1, random_state=SEED,
                                    stratify=isot["label"])

    news = pd.read_csv(os.path.join(root, "news_dataset.csv")).fillna("")
    news["label"] = 1
    _, news_eval = train_test_split(news[["title", "text", "label"]],
                                    test_size=0.15, random_state=SEED)

    held_out = pd.concat([
        build_frame(isot_eval, rng, augment_short=False),
        pd.DataFrame({
            "content": [clean_text(make_short(t, b, rng))
                        for t, b in zip(isot_eval["title"], isot_eval["text"])],
            "label": isot_eval["label"].values,
        }),
        build_frame(news_eval, rng, augment_short=False),
    ], ignore_index=True)

    # Split the held-out data again: fit T on one half, report on the other,
    # so the reported calibration is measured on data T never saw.
    calib, test = train_test_split(held_out, test_size=0.5, random_state=SEED,
                                   stratify=held_out["label"])
    print(f"Calibration set: {len(calib)} articles | Test set: {len(test)} articles")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

    calib_logits = collect_logits(model, tokenizer, list(calib["content"]), device)
    test_logits = collect_logits(model, tokenizer, list(test["content"]), device)
    calib_labels = torch.tensor(calib["label"].values)
    test_labels = torch.tensor(test["label"].values)

    T = fit_temperature(calib_logits, calib_labels)
    print(f"\nFitted temperature T = {T:.3f}")

    correct = (test_logits.argmax(1) == test_labels).numpy()
    before = torch.softmax(test_logits, dim=1).numpy()
    after = torch.softmax(test_logits / T, dim=1).numpy()

    print(f"\nOn the held-out test half ({len(test)} articles):")
    print(f"  accuracy                  {correct.mean():.4f}  (unchanged - T cannot flip a prediction)")
    print(f"  mean confidence  before   {before.max(axis=1).mean():.4f}")
    print(f"  mean confidence  after    {after.max(axis=1).mean():.4f}")
    print(f"  calibration error before  {expected_calibration_error(before, correct):.4f}")
    print(f"  calibration error after   {expected_calibration_error(after, correct):.4f}")

    print(f"\n  confidence >99.5% before  {(before.max(axis=1) > 0.995).mean() * 100:.1f}% of articles")
    print(f"  confidence >99.5% after   {(after.max(axis=1) > 0.995).mean() * 100:.1f}% of articles")

    out = os.path.join(root, "calibration.json")
    with open(out, "w") as fh:
        json.dump({"temperature": T, "fitted_on": len(calib),
                   "test_accuracy": float(correct.mean())}, fh, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
