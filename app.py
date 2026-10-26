print("[DEBUG] app.py started")
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
fake_data = pd.read_csv("Fake.csv")
print("[DEBUG] app.py started")

import json
import os
import re
import torch
from flask import Flask, render_template, request, jsonify
import feedparser
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

from news_cleaning import clean_text, strip_html
import image_predictor
print("[DEBUG] Flask import OK")
app = Flask(__name__)
# ...existing code...

# Add /fake route after app is defined
@app.route("/fake")
def fake():
    global fake_data
    if fake_data is None:
        fake_data = pd.read_csv("Fake.csv")
    row = fake_data.sample(1).iloc[0]
    fake_text = row["title"] + " " + row["text"]
    return render_template(
        "index.html",
        entered_text=fake_text,
        article_title=row["title"],
        article_summary=row["text"],
        article_date="",
        article_link="",
        result_text=None,
        result_class=None
    )


MODEL_DIR = "fake_news_model_v2"

RSS_SOURCES = [
    "https://feeds.feedburner.com/ndtvnews-top-stories",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://www.thehindu.com/news/national/feeder/default.rss"
]


# State for cycling fetches: two real, one fake
fetch_cycle = ["real", "real", "fake"]
fetch_cycle_index = 0
# Track shown real news to avoid repeats
shown_real_articles = set()
last_real_article = None


print("[DEBUG] Checking transformer model directory...")
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        "fake_news_model directory not found. Run: python retrain_model.py"
    )

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Temperature scaling (Guo et al., 2017), fitted by calibrate_model.py on
# held-out data. Dividing the logits by T rescales confidence to match observed
# accuracy; it preserves their order, so no prediction changes. T = 1.0 (a
# no-op) if the model has not been calibrated yet.
try:
    with open("calibration.json") as _fh:
        TEMPERATURE = float(json.load(_fh)["temperature"])
    print(f"Calibration loaded: T = {TEMPERATURE:.3f}")
except (OSError, ValueError, KeyError):
    TEMPERATURE = 1.0
    print("No calibration.json found - run: python calibrate_model.py")
print("Transformer model loaded successfully")

# The CLIP image detector runs ViT-L/14 over up to 9 crops per upload: well
# under a second on the GPU, several seconds on a laptop CPU.
IMAGE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_MODEL_READY = image_predictor.is_available()
if IMAGE_MODEL_READY:
    image_predictor.load_model(IMAGE_DEVICE)
    _meta = image_predictor.metadata()
    print(f"Image model loaded: {_meta.get('arch')} "
          f"(test accuracy {(_meta.get('test_accuracy') or 0) * 100:.2f}%)")
else:
    print("Image model not trained yet - run: python train_image_model.py")
print("Loaded model type:", type(model))


# ── feature-explanation helper ───────────────────────────────────
def get_top_features(text, n=5):
    # DistilBERT does not expose token-level linear coefficients.
    return []


def predict_news(text):
    try:
        cleaned = clean_text(text)

        if not cleaned:
            raise ValueError("No text provided")

        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits / TEMPERATURE, dim=1)[0]
        fake_prob = float(probs[0].item())
        real_prob = float(probs[1].item())

        if real_prob > fake_prob:
            label = "REAL"
            css = "result-real"
        else:
            label = "FAKE"
            css = "result-fake"

        confidence = max(real_prob, fake_prob)

        return label, css, confidence, fake_prob, real_prob

    except Exception as e:
        print("Ensemble error:", e)
        return "UNCERTAIN", "result-warning", 0.5, 0.5, 0.5

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({
            "prediction": "UNCERTAIN",
            "confidence": 0.0,
            "real_prob": 0.0,
            "fake_prob": 0.0,
            "error": "No text provided",
        }), 200

    try:
        label, _css, conf, p_fake, p_real = predict_news(text)
        return jsonify({
            "prediction": label,
            "confidence": float(conf),
            "real_prob": float(p_real),
            "fake_prob": float(p_fake),
        })
    except Exception as exc:
        print(f"Prediction route error: {exc}")
        return jsonify({
            "prediction": "UNCERTAIN",
            "confidence": 0.5,
            "real_prob": 0.5,
            "fake_prob": 0.5,
            "error": "Prediction failed",
        }), 200


def fetch_news():
    import random
    source = random.choice(["real", "fake"])
    if source == "real":
        # Fetch from news API
        for url in RSS_SOURCES:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                return {
                    "title": entry.title,
                    "summary": entry.get("summary", ""),
                    "published": entry.get("published", "Unknown"),
                    "link": entry.link,
                    "text": strip_html(entry.title) + " " + strip_html(entry.get("summary", ""))
                }
    else:
        row = fake_data.sample(1).iloc[0]
        return {
            "title": row["title"],
            "summary": row["text"],
            "published": "",
            "link": "",
            "text": row["title"] + " " + row["text"]
        }

def get_random_news():
    global fetch_cycle_index, fetch_cycle, shown_real_articles, last_real_article
    source = fetch_cycle[fetch_cycle_index]
    fetch_cycle_index = (fetch_cycle_index + 1) % len(fetch_cycle)

    if source == "real":
        # Try to avoid repeating the same article
        all_articles = []
        for url in RSS_SOURCES:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = strip_html(entry.get("title") or "")
                summary = strip_html(entry.get("summary") or "")
                if title and summary:
                    key = (title.strip(), summary.strip())
                    all_articles.append((key, title, summary))

        # Shuffle to randomize selection
        random.shuffle(all_articles)
        for key, title, summary in all_articles:
            if key not in shown_real_articles and key != last_real_article:
                shown_real_articles.add(key)
                last_real_article = key
                # Limit memory to last 20 articles
                if len(shown_real_articles) > 20:
                    shown_real_articles.pop()
                news_text = f"{title} {summary}".strip()
                return news_text
        # If all articles have been shown, reset memory and pick any
        shown_real_articles.clear()
        if all_articles:
            key, title, summary = all_articles[0]
            last_real_article = key
            news_text = f"{title} {summary}".strip()
            return news_text
        return "Unable to fetch a valid article. Please click Fetch News again."
    else:
        valid_rows = fake_data.dropna(subset=["title", "text"])
        if not valid_rows.empty:
            row = valid_rows.sample(1).iloc[0]
            title = row["title"] or ""
            text = row["text"] or ""
            news_text = f"{title} {text}".strip()
            if not news_text:
                news_text = "Unable to fetch a valid article. Please click Fetch News again."
            return news_text
        else:
            return "Unable to fetch a valid article. Please click Fetch News again."

@app.route("/fetch_news")
def fetch_news_route():
    news_text = get_random_news()
    return {"news": news_text}

MAX_IMAGE_BYTES = 10 * 1024 * 1024


@app.route("/predict_image", methods=["POST"])
def predict_image_route():
    """Classify an uploaded image as a real photo or AI-generated."""
    if not IMAGE_MODEL_READY:
        return jsonify({
            "error": "Image model not trained. Run: python train_image_model.py"
        }), 503

    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"error": "No image uploaded"}), 400

    data = upload.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        return jsonify({"error": "Image too large (max 10 MB)"}), 413

    try:
        label, css, conf, p_fake, p_real, info = image_predictor.predict(data, IMAGE_DEVICE)
    except Exception as exc:
        print(f"Image prediction error: {exc}")
        return jsonify({"error": "Could not read that image file"}), 400

    return jsonify({
        "prediction": label,
        "result_class": css,
        "confidence": float(conf),
        "fake_prob": float(p_fake),
        "real_prob": float(p_real),
        **info,
    })


@app.route("/image_status")
def image_status():
    return jsonify({"ready": IMAGE_MODEL_READY, **image_predictor.metadata()})


@app.route("/", methods=["GET", "POST"])
def index():

    result_text = None
    result_class = None
    confidence = None
    real_prob = None
    fake_prob = None
    top_features = []
    entered_text = ""

    article_title = None
    article_summary = None
    article_date = None
    article_link = None

    if request.method == "POST":
        input_text = request.form.get("news_text", "").strip()
        entered_text = input_text
        if input_text:
            try:
                label, css, conf, p_fake, p_real = predict_news(input_text)
                result_text = label
                result_class = css
                confidence = round(conf * 100, 1)
                real_prob = round(p_real * 100, 1)
                fake_prob = round(p_fake * 100, 1)
                top_features = get_top_features(input_text, n=5)
            except Exception as exc:
                print(f"Route prediction error: {exc}")
                result_text = "UNCERTAIN"
                result_class = "result-warning"
        else:
            result_text = "Please enter some news text."
            result_class = "result-warning"

    return render_template(
        "index.html",
        result_text=result_text,
        result_class=result_class,
        confidence=confidence,
        real_prob=real_prob,
        fake_prob=fake_prob,
        top_features=top_features,
        entered_text=entered_text,
        article_title=article_title,
        article_summary=article_summary,
        article_date=article_date,
        article_link=article_link
    )



if __name__ == "__main__":
    print("Starting Flask server...")
    try:
        app.run(host="127.0.0.1", port=5000, debug=True)
    except Exception as e:
        print(f"Flask app failed to start: {e}")