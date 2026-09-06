# TruthLens

Fake news detection and AI-generated image detection in a single Flask app.

## Setup

Requires Python 3.10+ (developed on 3.12). A CUDA GPU is optional; the app runs
inference on CPU.

```bash
git clone https://github.com/Pritesh-30/TruthLens.git
cd TruthLens

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Download the news model weights

The DistilBERT weights (256 MB) exceed GitHub's 100 MB file limit, so they are
attached to a Release instead. Download `model.safetensors` and place it in
`fake_news_model_v2/`:

```bash
curl -L -o fake_news_model_v2/model.safetensors \
  https://github.com/Pritesh-30/TruthLens/releases/download/v1.0/model.safetensors
```

The directory should then contain:

```
fake_news_model_v2/
├── config.json
├── model.safetensors      <- downloaded
├── tokenizer.json
└── tokenizer_config.json
```

## Run

```bash
python app.py
```

Open <http://127.0.0.1:5000>. Use the tabs to switch between the two detectors.

- **Fake News** — paste article text, or click *Fetch News* to pull a live
  article from NDTV / Times of India / The Hindu RSS feeds.
- **AI Image** — upload or drag in an image to check whether it was generated.

---

