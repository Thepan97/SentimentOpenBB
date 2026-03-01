from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = SentimentIntensityAnalyzer()

@app.get("/sentiment")
def sentiment(q: str = "AAPL", page_size: int = 20):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return {"error": "Missing NEWSAPI_KEY"}

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
    }
    headers = {"X-Api-Key": api_key}

    r = requests.get(url, params=params, headers=headers, timeout=30)
    data = r.json()
    articles = data.get("articles", [])

    df = pd.DataFrame(articles)
    if df.empty:
        return []

    df["sentiment"] = df["title"].fillna("").apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )

    out = df[["publishedAt", "title", "sentiment"]].copy()
    out["source"] = df["source"].apply(lambda s: (s or {}).get("name"))

    return out.to_dict(orient="records")
