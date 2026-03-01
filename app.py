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

def _empty_table(message: str | None = None):
    payload = {
        "table_data": {
            "name": "Sentiment Results",
            "data": [],
        }
    }
    if message:
        payload["error"] = message
    return payload

@app.get("/sentiment")
def sentiment(q: str = "AAPL", page_size: int = 20):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return _empty_table("Missing NEWSAPI_KEY")

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
    except Exception as e:
        return _empty_table(f"NewsAPI request failed: {str(e)}")

    if data.get("status") != "ok":
        return _empty_table(f"NewsAPI error: {data.get('message', 'unknown error')}")

    articles = data.get("articles", [])
    df = pd.DataFrame(articles)

    if df.empty:
        return _empty_table("No articles returned for this query")

    df["sentiment"] = df["title"].fillna("").apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )

    out = df[["publishedAt", "title", "sentiment"]].copy()
    out["source"] = df["source"].apply(lambda s: (s or {}).get("name"))

    return {
        "table_data": {
            "name": "Sentiment Results",
            "data": out.to_dict(orient="records"),
        }
    }
