# TruthLens

Fake news detection and AI-generated image detection in a single Flask app.

Two independent models behind one page:

| Detector | Model | Accuracy |
|---|---|---|
| Fake news | DistilBERT (fine-tuned) | 99.96% on ISOT held-out · 97.35% on live RSS news |
| AI-generated image | ResNet-18 (CIFAR stem) | 97.91% on 20,000 held-out CIFAKE images |

Both models are already trained — you do **not** need to retrain to run the app.

---

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

The image model (`ai_image_model/image_model.pt`, 43 MB) is committed to the
repo, so nothing extra is needed for the image detector.

## Run

```bash
python app.py
```

Open <http://127.0.0.1:5000>. Use the tabs to switch between the two detectors.

- **Fake News** — paste article text, or click *Fetch News* to pull a live
  article from NDTV / Times of India / The Hindu RSS feeds.
- **AI Image** — upload or drag in an image to check whether it was generated.

---

## Retraining (optional)

Only needed if you want to rebuild the models from scratch. Both require
datasets that are not in this repo.

### News model

Download [ISOT Fake News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
(`Fake.csv`, `True.csv`) into the project root, then:

```bash
python harvest_real_news.py   # collects ~3,000 current articles from 50+ RSS feeds
python retrain_model.py       # ~15 min on a GPU
python calibrate_model.py     # fits the confidence temperature
```

`harvest_real_news.py` matters: the ISOT real-news corpus is entirely Reuters
wire copy, so a model trained on it alone learns "Reuters house style = real"
and flags legitimate articles from other outlets as fake. The harvested
multi-outlet articles correct that.

### Image model

Download [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
and extract it to `archive/` (giving `archive/train/REAL`, `archive/train/FAKE`,
`archive/test/REAL`, `archive/test/FAKE`), then:

```bash
python train_image_model.py --benchmark   # prints a projected training time
python train_image_model.py               # ~15 min on a GPU
```

---

## Project layout

```
app.py                  Flask server and routes
news_cleaning.py        Shared text cleaning for training and serving
retrain_model.py        Fine-tunes DistilBERT on ISOT + harvested news
harvest_real_news.py    Collects real articles from RSS feeds
calibrate_model.py      Temperature scaling for calibrated confidence
image_data.py           Decodes CIFAKE into a cached array
image_model.py          CIFAR-stem ResNet-18 and a small CNN baseline
train_image_model.py    Trains the image detector
image_predictor.py      Image inference used by the server
templates/index.html    Single-page UI for both detectors
```

## Known limitations

- The news model classifies **writing style**, not factual truth. A fabricated
  claim written in neutral news style can still be labelled REAL.
- The image model was trained on 32×32 Stable Diffusion images. High-resolution
  uploads are downscaled to 32×32 before classification, and images from other
  generators are outside what it has seen, so accuracy is lower than the
  headline figure in both cases.
