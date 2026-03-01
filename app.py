# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(title="SentimentOpenBB", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pro.openbb.co", "https://app.openbb.co", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = SentimentIntensityAnalyzer()

WIDGETS = {
    "news_sentiment_table": {
        "name": "News Sentiment",
        "description": "NewsAPI headlines with VADER sentiment",
        "category": "Custom",
        "type": "table",
        "endpoint": "sentiment",
        "gridData": {"w": 20, "h": 10},
        "params": [
            {"name": "q", "label": "Query", "type": "text", "default": "AAPL"},
            {"name": "page_size", "label": "Page Size", "type": "number", "default": 20},
        ],
        "data": {
            "table": {
                "showAll": True,
                "enableCharts": False,
                "enableAdvanced": True,
                "enableFormulas": True,
                "columnsDefs": [
                    {"field": "published_at", "headerName": "Published", "cellDataType": "dateString"},
                    {"field": "title", "headerName": "Title", "cellDataType": "text"},
                    {"field": "publisher", "headerName": "Publisher", "cellDataType": "text"},
                    {"field": "sentiment", "headerName": "Sentiment", "cellDataType": "number"},
                ],
            }
        },
    }
}

@app.get("/")
def root():
    return {"ok": True}

@app.get("/widgets.json")
def widgets():
    return JSONResponse(content=WIDGETS)

@app.get("/apps.json")
def apps():
    return JSONResponse(content=[])

@app.get("/sentiment")
def sentiment(q: str = "AAPL", page_size: int = 20):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {"q": q, "language": "en", "sortBy": "publishedAt", "pageSize": page_size}
    headers = {"X-Api-Key": api_key}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    if data.get("status") != "ok":
        return []

    df = pd.DataFrame(data.get("articles", []))
    if df.empty:
        return []

    df["publisher"] = df["source"].apply(
        lambda s: (s or {}).get("name") if isinstance(s, dict) else None
    )
    df["title"] = df["title"].fillna("").astype(str)
    df["sentiment"] = df["title"].apply(lambda t: analyzer.polarity_scores(t)["compound"])

    out = pd.DataFrame(
        {
            "published_at": df.get("publishedAt"),
            "title": df.get("title"),
            "publisher": df.get("publisher"),
            "sentiment": df.get("sentiment"),
        }
    )

    out = out.where(pd.notna(out), None)
    return out.to_dict(orient="records")
