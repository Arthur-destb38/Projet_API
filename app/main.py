"""
API Crypto Sentiment - FastAPI
Projet MoSEF 2024-2025
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from datetime import datetime
import time

from app.scrapers import HttpScraper, SeleniumScraper
from app.sentiment import SentimentAnalyzer
from app.prices import CryptoPrices
from app.utils import clean_text

app = FastAPI(title="Crypto Sentiment API")
templates = Jinja2Templates(directory="templates")

sentiment_analyzer = None
prices_client = CryptoPrices()


def get_analyzer():
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = SentimentAnalyzer()
    return sentiment_analyzer


class AnalyzeRequest(BaseModel):
    crypto: str = Field(default="bitcoin")
    subreddit: str = Field(default="Bitcoin")
    limit: int = Field(default=50, ge=10, le=200)
    method: str = Field(default="selenium")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    prices = prices_client.get_multiple_prices(["bitcoin", "ethereum", "solana"])
    return templates.TemplateResponse("index.html", {"request": request, "prices": prices})


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/analyze")
async def full_analysis(req: AnalyzeRequest):
    start = time.time()

    if req.method == "http":
        scraper = HttpScraper()
        posts = scraper.scrape_subreddit(req.subreddit, limit=req.limit)
        scraper.close()
        method_name = "HTTP/JSON"
    else:
        scraper = SeleniumScraper(headless=True)
        posts = scraper.scrape_subreddit(req.subreddit, limit=req.limit)
        scraper.close()
        method_name = "Selenium"

    scrape_time = time.time() - start

    analyzer = get_analyzer()
    texts = [clean_text(p["title"] + " " + p.get("text", "")) for p in posts]
    texts = [t for t in texts if t and len(t) > 10]

    results = analyzer.analyze_batch(texts)

    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0

    labels = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        labels[r["label"]] += 1

    price_data = prices_client.get_price(req.crypto)

    return {
        "method": method_name,
        "crypto": req.crypto,
        "subreddit": req.subreddit,
        "posts_analyzed": len(results),
        "total_time": round(time.time() - start, 2),
        "sentiment": {"average": round(avg_score, 4), "distribution": labels},
        "price": price_data,
        "posts": [
            {"title": posts[i]["title"], "score": results[i]["score"], "label": results[i]["label"]}
            for i in range(min(len(posts), len(results)))
        ]
    }


@app.post("/compare")
async def compare_methods(subreddit: str = "Bitcoin", limit: int = 30):
    results = {}

    start = time.time()
    http = HttpScraper()
    http_posts = http.scrape_subreddit(subreddit, limit=limit)
    http.close()
    results["http"] = {"time": round(time.time() - start, 2), "posts": len(http_posts)}

    start = time.time()
    sel = SeleniumScraper(headless=True)
    sel_posts = sel.scrape_subreddit(subreddit, limit=limit)
    sel.close()
    results["selenium"] = {"time": round(time.time() - start, 2), "posts": len(sel_posts)}

    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)