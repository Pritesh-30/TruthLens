"""
news_cleaning.py
================
Shared text normalisation for training (retrain_model.py) and serving (app.py).

Both paths must scrub identically or the model sees a different distribution at
inference than it trained on. Two jobs:

  strip_html()             - RSS summaries carry markup and HTML entities.
  scrub_source_artifacts() - Removes tells that identify the *corpus* rather
                             than the *truthfulness* of an article. ISOT's
                             True.csv is 100% Reuters wire copy, so "(Reuters)"
                             and its datelines are a perfect giveaway a model
                             will happily learn instead of the actual task.
                             Fake.csv has its own tells (image credits, embedded
                             tweets, "READ MORE:" trails).
"""

import html
import re

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+")

# Reuters tells: the leading "WASHINGTON (Reuters) - " dateline, plus any
# later mention of the agency.
DATELINE_RE = re.compile(r"^\s*[A-Z][A-Za-z.\s/,-]{0,40}\s*\(Reuters\)\s*[-–—]\s*")
GENERIC_DATELINE_RE = re.compile(r"^\s*([A-Z][A-Z\s.,/-]{2,40})\s*[-–—]\s+")
AGENCY_RE = re.compile(
    r"\(\s*Reuters\s*\)|\bReuters\b|\(\s*AP\s*\)|\bAssociated Press\b|"
    r"\(\s*AFP\s*\)|\bAgence France-Presse\b|\bPTI\b|\bANI\b|\bIANS\b",
    re.IGNORECASE,
)

# Fake.csv boilerplate.
CREDIT_RE = re.compile(
    r"Featured image via[^.]*\.?|Image via[^.]*\.?|Photo(?: credit)? by[^.]*\.?|"
    r"via:?\s*[A-Z][\w\s]{0,30}(?:\.com|\.org|\b)|Getty Images|"
    r"READ MORE:?|WATCH:?|\[VIDEO\]|\[WATCH\]|21st Century Wire|"
    r"Follow us on Twitter[^.]*\.?|@[A-Za-z0-9_]{2,15}",
    re.IGNORECASE,
)


def strip_html(raw):
    """Remove tags and unescape entities from RSS-supplied text.

    Unescape first: feeds often double-encode, so "&lt;p&gt;" only becomes a
    strippable tag after decoding. Decode again afterwards for entities that
    were nested inside the markup.
    """
    text = html.unescape(str(raw or ""))
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return WS_RE.sub(" ", text).strip()


def scrub_source_artifacts(text):
    """Strip corpus-identifying tells so the model can't shortcut on them."""
    text = str(text)
    text = DATELINE_RE.sub("", text)

    m = GENERIC_DATELINE_RE.match(text)
    if m and len(m.group(1).split()) <= 4:
        text = text[m.end():]

    text = AGENCY_RE.sub(" ", text)
    # URLs before credits: the credit patterns would otherwise chew a URL in
    # half and leave a fragment like "com/a" behind.
    text = URL_RE.sub(" ", text)
    text = CREDIT_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def clean_text(raw_text):
    """Full normalisation applied to every article, training and serving."""
    return scrub_source_artifacts(strip_html(raw_text))
