"""
harvest_real_news.py
====================
Collects genuine, recently-published articles from a broad set of RSS feeds
and writes them to news_dataset.csv (label 1 = real).

The ISOT True.csv corpus is 100% Reuters wire copy, so a model trained on it
alone learns "Reuters house style = real" and flags every other outlet as fake.
This adds the outlet diversity that Fake.csv/True.csv cannot provide, in the
same short title+summary shape the app actually sees at /fetch_news.
"""

import html
import os
import re
import sys
import time

import feedparser
import pandas as pd

FEEDS = [
    # ── Indian outlets ────────────────────────────────────────────
    "https://feeds.feedburner.com/ndtvnews-top-stories",
    "https://feeds.feedburner.com/ndtvnews-india-news",
    "https://feeds.feedburner.com/ndtvnews-world-news",
    "https://feeds.feedburner.com/ndtvnews-latest",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
    "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
    "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",
    "https://www.thehindu.com/news/national/feeder/default.rss",
    "https://www.thehindu.com/news/international/feeder/default.rss",
    "https://www.thehindu.com/business/feeder/default.rss",
    "https://www.thehindu.com/sport/feeder/default.rss",
    "https://indianexpress.com/feed/",
    "https://indianexpress.com/section/india/feed/",
    "https://indianexpress.com/section/world/feed/",
    "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
    "https://www.hindustantimes.com/feeds/rss/world-news/rssfeed.xml",
    "https://www.news18.com/rss/india.xml",
    "https://www.news18.com/rss/world.xml",
    "https://www.firstpost.com/commonfeeds/v1/mfp/rss/india.xml",
    "https://www.livemint.com/rss/news",
    "https://www.business-standard.com/rss/home_page_top_stories.rss",
    "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
    "https://www.deccanherald.com/rss/national.rss",
    "https://scroll.in/feed",
    # ── International outlets ─────────────────────────────────────
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/uk/business/rss",
    "https://feeds.npr.org/1001/rss.xml",
    "https://feeds.npr.org/1004/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://rss.dw.com/rdf/rss-en-all",
    "https://www.france24.com/en/rss",
    "https://feeds.skynews.com/feeds/rss/world.xml",
    "https://www.cbsnews.com/latest/rss/world",
    "https://abcnews.go.com/abcnews/topstories",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml",
    # ── Google News topic feeds (broad multi-outlet coverage) ─────
    "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/headlines/section/topic/NATION?hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/headlines/section/topic/HEALTH?hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/headlines/section/topic/SPORTS?hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=india&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=world+news&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=economy&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=elections&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=cricket&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=climate&hl=en-IN&gl=IN&ceid=IN:en",
]

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def strip_html(raw):
    """RSS summaries routinely carry markup and entities; drop both."""
    text = TAG_RE.sub(" ", str(raw or ""))
    text = html.unescape(text)
    return WS_RE.sub(" ", text).strip()


def harvest(out_path="news_dataset.csv"):
    rows = []
    seen = set()

    for i, url in enumerate(FEEDS, 1):
        try:
            feed = feedparser.parse(url)
        except Exception as exc:
            print(f"[{i:>2}/{len(FEEDS)}] FAILED {url}: {exc}")
            continue

        n_before = len(rows)
        for entry in getattr(feed, "entries", []):
            title = strip_html(entry.get("title", ""))
            summary = strip_html(entry.get("summary", "") or entry.get("description", ""))

            # Google News summaries are just a list of outlet links; drop them
            # and keep the headline, which is the informative part.
            if "news.google.com" in url:
                summary = ""

            if len(title.split()) < 4:
                continue

            key = title.lower()
            if key in seen:
                continue
            seen.add(key)

            rows.append({"title": title, "text": summary, "label": 1,
                         "source": url.split("/")[2]})

        print(f"[{i:>2}/{len(FEEDS)}] {len(rows) - n_before:>4} new  {url.split('/')[2]}")
        time.sleep(0.3)

    df = pd.DataFrame(rows)

    # Merge with anything harvested on previous runs so repeated runs accumulate.
    if os.path.isfile(out_path):
        prev = pd.read_csv(out_path).fillna("")
        df = pd.concat([prev, df], ignore_index=True)
        df = df.drop_duplicates(subset=["title"], keep="first")

    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} real articles to {out_path}")
    print(df["source"].value_counts().head(20).to_string() if "source" in df else "")
    return df


if __name__ == "__main__":
    harvest(sys.argv[1] if len(sys.argv) > 1 else "news_dataset.csv")
