import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from news_cleaning import clean_text

# Base checkpoint used when no locally trained weights exist yet.
BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 128
SEED = 42


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def has_weights(path):
    """True if a directory holds actual model weights, not just a config."""
    return os.path.isdir(path) and any(
        os.path.isfile(os.path.join(path, f))
        for f in ("model.safetensors", "pytorch_model.bin")
    )


def map_label(val):
    if isinstance(val, str):
        val_lower = val.strip().lower()
        if val_lower in ["fake", "0", "false"]:
            return 0
        elif val_lower in ["real", "true", "1"]:
            return 1
        else:
            raise ValueError(f"Unknown label value: {val}")
    return int(val)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def make_short(title, body, rng):
    """A headline plus the first few words of the body.

    The app classifies RSS items that are 15-120 tokens long, but ISOT articles
    average ~400 words. Trained on full articles only, the model treats any
    short input as out-of-distribution and collapses to FAKE - including on
    genuine Reuters headlines. Generating short variants for BOTH classes puts
    that length range in-distribution without making length itself a class cue.
    """
    k = int(rng.choice([0, 8, 15, 25, 40]))
    words = str(body).split()[:k]
    return (str(title) + " " + " ".join(words)).strip()


def build_frame(df, rng, augment_short=True):
    """Cleaned full-length rows, plus a short variant of each when asked."""
    full = (df["title"].astype(str) + " " + df["text"].astype(str)).map(clean_text)
    rows = pd.DataFrame({"content": full, "label": df["label"].values})

    if augment_short:
        short = [
            clean_text(make_short(t, b, rng))
            for t, b in zip(df["title"].astype(str), df["text"].astype(str))
        ]
        rows = pd.concat(
            [rows, pd.DataFrame({"content": short, "label": df["label"].values})],
            ignore_index=True,
        )

    rows["content"] = rows["content"].str.strip()
    return rows[rows["content"].str.len() > 0].reset_index(drop=True)


def main():
    rng = np.random.default_rng(SEED)
    root = os.path.dirname(os.path.abspath(__file__)) or "."

    fake = pd.read_csv(os.path.join(root, "Fake.csv")).fillna("")
    true = pd.read_csv(os.path.join(root, "True.csv")).fillna("")
    fake["label"] = 0
    true["label"] = 1

    news_path = os.path.join(root, "news_dataset.csv")
    if os.path.isfile(news_path):
        news = pd.read_csv(news_path).fillna("")
        if "label" not in news.columns:
            if "target" in news.columns:
                news["label"] = news["target"]
            else:
                raise ValueError(
                    "news_dataset.csv must have a 'label' or 'target' column (0=fake, 1=true)"
                )
        news["label"] = news["label"].apply(map_label)
        for col in ("title", "text"):
            if col not in news.columns:
                news[col] = ""
    else:
        print("news_dataset.csv not found - run: python harvest_real_news.py")
        news = pd.DataFrame(columns=["title", "text", "label"])

    isot = pd.concat([fake, true], ignore_index=True)[["title", "text", "label"]]
    print(f"ISOT corpus:      {len(isot)} articles "
          f"({(isot.label == 0).sum()} fake, {(isot.label == 1).sum()} real)")
    print(f"Harvested corpus: {len(news)} articles (real, multi-outlet RSS)")

    # Hold out slices BEFORE augmentation so no eval article leaks into
    # training via one of its own short variants.
    isot_train, isot_eval = train_test_split(
        isot, test_size=0.1, random_state=SEED, stratify=isot["label"]
    )

    if len(news):
        news_train, news_eval = train_test_split(
            news[["title", "text", "label"]], test_size=0.15, random_state=SEED
        )
    else:
        news_train = news_eval = pd.DataFrame(columns=["title", "text", "label"])

    train_rows = pd.concat(
        [build_frame(isot_train, rng, augment_short=True),
         build_frame(news_train, rng, augment_short=True)],
        ignore_index=True,
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"\nTraining rows after augmentation: {len(train_rows)} "
          f"({(train_rows.label == 0).sum()} fake, {(train_rows.label == 1).sum()} real)")

    # Three separate eval views, because a single number hides the failure.
    eval_views = {
        "ISOT full-length": build_frame(isot_eval, rng, augment_short=False),
        "ISOT short-form": pd.DataFrame({
            "content": [clean_text(make_short(t, b, rng))
                        for t, b in zip(isot_eval["title"], isot_eval["text"])],
            "label": isot_eval["label"].values,
        }),
    }
    if len(news_eval):
        eval_views["RSS real news (held out)"] = build_frame(
            news_eval, rng, augment_short=False
        )

    # Default to a fresh base checkpoint. Continuing from local weights would
    # carry over shortcuts the previous run learned (Reuters style = real),
    # which is exactly what this training run exists to unlearn.
    model_dir = os.path.join(root, "fake_news_model")
    resume = "--continue" in sys.argv and has_weights(model_dir)
    source = model_dir if resume else BASE_MODEL
    print(f"Starting from: {source}" + ("" if resume else " (fresh; pass --continue to resume)"))

    tokenizer = DistilBertTokenizerFast.from_pretrained(source)
    model = DistilBertForSequenceClassification.from_pretrained(
        source,
        num_labels=2,
        id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1},
    )

    use_gpu = torch.cuda.is_available()
    print(f"CUDA available: {use_gpu}"
          + (f" ({torch.cuda.get_device_name(0)})" if use_gpu else ""))

    def encode(frame):
        enc = tokenizer(list(frame["content"]), truncation=True,
                        padding="max_length", max_length=MAX_LEN)
        return NewsDataset(enc, list(frame["label"]))

    train_dataset = encode(train_rows)
    eval_datasets = {name: encode(frame) for name, frame in eval_views.items()}
    combined_eval = encode(pd.concat(list(eval_views.values()), ignore_index=True))

    epochs = 2
    batch_size = 32
    total_steps = math.ceil(len(train_dataset) / batch_size) * epochs

    # transformers 5.x dropped warmup_ratio, so express warmup in steps.
    training_args = TrainingArguments(
        output_dir=os.path.join(root, "results"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        warmup_steps=int(0.06 * total_steps),
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=use_gpu,
        dataloader_num_workers=2,
        disable_tqdm=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=combined_eval,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n" + "=" * 62)
    for name, frame in eval_views.items():
        preds = np.argmax(trainer.predict(eval_datasets[name]).predictions, axis=-1)
        acc = accuracy_score(frame["label"], preds)
        print(f"\n--- {name}  ({len(frame)} articles) --- accuracy {acc:.4f}")
        labels_present = sorted(frame["label"].unique())
        print(classification_report(
            frame["label"], preds,
            labels=labels_present,
            target_names=[["FAKE", "REAL"][i] for i in labels_present],
            digits=4, zero_division=0,
        ))

    # app.py loads MODEL_DIR = "fake_news_model_v2"
    for out_dir in (os.path.join(root, "fake_news_model"),
                    os.path.join(root, "fake_news_model_v2")):
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"Saved DistilBERT model and tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
