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

def _table(rows, message: str | None = None):
    payload = {
        "table_data": {
            "name": "Sentiment Results",
            "data": rows,
        }
    }
    if message:
        payload["error"] = message
    return payload

@app.get("/sentiment")
def sentiment(q: str = "AAPL", page_size: int = 20):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return _table([], "Missing NEWSAPI_KEY")

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
        return _table([], f"NewsAPI request failed: {str(e)}")

    if data.get("status") != "ok":
        return _table([], f"NewsAPI error: {data.get('message', 'unknown error')}")

    articles = data.get("articles", [])
    df = pd.DataFrame(articles)

    if df.empty:
        return _table([], "No articles returned for this query")

    # Safe publisher extraction
    df["publisher"] = df["source"].apply(lambda s: (s or {}).get("name") if isinstance(s, dict) else None)

    # Sentiment
    df["sentiment"] = df["title"].fillna("").astype(str).apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )

    out = pd.DataFrame({
        "published_at": df.get("publishedAt"),
        "title": df.get("title"),
        "publisher": df.get("publisher"),
        "sentiment": df.get("sentiment"),
    })

    # Replace NaN with None so JSON is clean
    out = out.where(pd.notna(out), None)

    return _table(out.to_dict(orient="records"))
