# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import os
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(title="SentimentOpenBB", version="0.0.1")

# CORS: allow OpenBB Pro to call your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pro.openbb.co", "https://app.openbb.co", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = SentimentIntensityAnalyzer()


@app.get("/")
def root():
    return {"info": "SentimentOpenBB backend is running"}


@app.get("/widgets.json")
def widgets():
    path = Path(__file__).parent / "widgets.json"
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))


@app.get("/apps.json")
def apps():
    path = Path(__file__).parent / "apps.json"
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))


@app.get("/sentiment")
def sentiment(q: str = "AAPL", page_size: int = 20):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
    }
    headers = {"X-Api-Key": api_key}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    if data.get("status") != "ok":
        return []

    articles = data.get("articles", [])
    df = pd.DataFrame(articles)
    if df.empty:
        return []

    # Flatten + clean
    df["publisher"] = df["source"].apply(
        lambda s: (s or {}).get("name") if isinstance(s, dict) else None
    )
    df["title"] = df["title"].fillna("").astype(str)

    # Sentiment
    df["sentiment"] = df["title"].apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )

    out = pd.DataFrame(
        {
            "published_at": df.get("publishedAt"),
            "title": df.get("title"),
            "publisher": df.get("publisher"),
            "sentiment": df.get("sentiment"),
        }
    )

    # Ensure JSON-safe nulls (no NaN)
    out = out.where(pd.notna(out), None)

    return out.to_dict(orient="records")
