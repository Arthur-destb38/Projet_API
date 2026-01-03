"""
API Crypto Sentiment - FastAPI
Projet MoSEF 2024-2025
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from datetime import datetime
import time

from app.scrapers import HttpScraper, SeleniumScraper
from app.sentiment import SentimentAnalyzer
from app.prices import CryptoPrices
from app.utils import clean_text

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(
    title="Crypto Sentiment API",
    description="Analyse de sentiment crypto - MoSEF 2024-2025",
    version="1.0"
)

templates = Jinja2Templates(directory="templates")

# Variables globales
sentiment_analyzer = None
prices_client = CryptoPrices()
http_scraper = HttpScraper()


def get_analyzer():
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = SentimentAnalyzer()
    return sentiment_analyzer


# ============ MODELS ============

class AnalyzeRequest(BaseModel):
    crypto: str = Field(default="bitcoin")
    subreddit: str = Field(default="Bitcoin")
    limit: int = Field(default=50, ge=10, le=1000)  # Max 1000!
    method: str = Field(default="http")


class MultiAnalyzeRequest(BaseModel):
    cryptos: list[str] = Field(default=["bitcoin", "ethereum"])
    limit_per_crypto: int = Field(default=30, ge=10, le=200)


# ============ PAGES ============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Page principale"""
    prices = prices_client.get_multiple_prices(["bitcoin", "ethereum", "solana"])
    cryptos = http_scraper.list_cryptos()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prices": prices,
        "cryptos": cryptos
    })


# ============ API ============

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/cryptos")
async def list_cryptos():
    """Liste des cryptos disponibles"""
    return {
        "cryptos": http_scraper.list_cryptos(),
        "subreddits": http_scraper.SUBREDDITS
    }


@app.get("/prices/{crypto}")
async def get_price(crypto: str):
    """Prix d'une crypto"""
    price = prices_client.get_price(crypto)
    if price:
        return price
    return {"error": f"Crypto {crypto} non trouvee"}


@app.get("/prices")
async def get_all_prices():
    """Prix des principales cryptos"""
    cryptos = ["bitcoin", "ethereum", "solana", "cardano", "dogecoin"]
    return prices_client.get_multiple_prices(cryptos)


@app.post("/scrape")
async def scrape(req: AnalyzeRequest):
    """Scrape sans analyse sentiment"""
    start = time.time()

    if req.method == "selenium":
        scraper = SeleniumScraper(headless=True)
        posts = scraper.scrape_subreddit(req.subreddit, limit=req.limit)
        scraper.close()
        method_name = "Selenium"
    else:
        posts = http_scraper.scrape_subreddit(req.subreddit, limit=req.limit)
        method_name = "HTTP"

    return {
        "method": method_name,
        "subreddit": req.subreddit,
        "posts_count": len(posts),
        "time": round(time.time() - start, 2),
        "posts": posts
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Scrape + analyse sentiment"""
    start = time.time()

    # Scraping
    if req.method == "selenium":
        scraper = SeleniumScraper(headless=True)
        posts = scraper.scrape_subreddit(req.subreddit, limit=req.limit)
        scraper.close()
        method_name = "Selenium"
    else:
        sub = http_scraper.get_subreddit(req.crypto)
        posts = http_scraper.scrape_subreddit(sub, limit=req.limit)
        method_name = "HTTP"

    scrape_time = time.time() - start

    if not posts:
        return {"error": "Aucun post recupere"}

    # Sentiment
    analyzer = get_analyzer()
    texts = [clean_text(p["title"] + " " + p.get("text", "")) for p in posts]
    texts = [t for t in texts if t and len(t) > 10]

    sent_start = time.time()
    results = analyzer.analyze_batch(texts)
    sent_time = time.time() - sent_start

    # Stats
    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0

    labels = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        labels[r["label"]] += 1

    # Prix
    price = prices_client.get_price(req.crypto)

    return {
        "method": method_name,
        "crypto": req.crypto,
        "subreddit": req.subreddit,
        "posts_scraped": len(posts),
        "posts_analyzed": len(results),
        "scrape_time": round(scrape_time, 2),
        "sentiment_time": round(sent_time, 2),
        "total_time": round(time.time() - start, 2),
        "sentiment": {
            "average": round(avg, 4),
            "distribution": labels
        },
        "price": price,
        "posts": [
            {
                "title": posts[i]["title"],
                "score": results[i]["score"],
                "label": results[i]["label"]
            }
            for i in range(min(len(posts), len(results)))
        ]
    }


@app.post("/analyze/multi")
async def analyze_multi(req: MultiAnalyzeRequest):
    """Analyse plusieurs cryptos"""
    results = {}
    analyzer = get_analyzer()

    for crypto in req.cryptos:
        sub = http_scraper.get_subreddit(crypto)
        posts = http_scraper.scrape_subreddit(sub, limit=req.limit_per_crypto)

        if posts:
            texts = [clean_text(p["title"] + " " + p.get("text", "")) for p in posts]
            texts = [t for t in texts if t and len(t) > 10]
            sentiments = analyzer.analyze_batch(texts)

            scores = [s["score"] for s in sentiments]
            avg = sum(scores) / len(scores) if scores else 0

            labels = {"positive": 0, "negative": 0, "neutral": 0}
            for s in sentiments:
                labels[s["label"]] += 1

            results[crypto] = {
                "posts": len(sentiments),
                "sentiment_avg": round(avg, 4),
                "distribution": labels
            }

    return results


@app.post("/compare")
async def compare(subreddit: str = "Bitcoin", limit: int = 30):
    """Compare HTTP vs Selenium"""

    # HTTP
    start = time.time()
    http_posts = http_scraper.scrape_subreddit(subreddit, limit=limit)
    http_time = time.time() - start

    # Selenium
    start = time.time()
    sel = SeleniumScraper(headless=True)
    sel_posts = sel.scrape_subreddit(subreddit, limit=limit)
    sel.close()
    sel_time = time.time() - start

    return {
        "http": {
            "time": round(http_time, 2),
            "posts": len(http_posts)
        },
        "selenium": {
            "time": round(sel_time, 2),
            "posts": len(sel_posts)
        },
        "speedup": round(sel_time / http_time, 1) if http_time > 0 else 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)