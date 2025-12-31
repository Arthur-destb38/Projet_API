"""
Crypto Sentiment Analysis API
Author: MoSEF Student
Description: FastAPI application for cryptocurrency sentiment analysis.
Scrapes Reddit posts, analyzes sentiment with FinBERT, and fetches price data from CoinGecko.
Master 2 MoSEF Data Science 2024-2025
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from app.scraper import RedditScraper
from app.scraper_http import RedditHTTPScraper
from app.sentiment import SentimentAnalyzer
from app.prices import CryptoPrices
from app.utils import clean_text
from app.econometrics import EconometricAnalyzer

# -------------------- FastAPI App Init --------------------
app = FastAPI(
    title="Crypto Sentiment API",
    description="API d'analyse de sentiment pour cryptomonnaies via Reddit et FinBERT",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

# -------------------- Pydantic Models --------------------
class ScrapeRequest(BaseModel):
    """Request model for scraping endpoint"""
    subreddit: str = Field(default="Bitcoin", min_length=1, max_length=50)
    crypto: str = Field(default="BTC", min_length=2, max_length=10)
    limit: int = Field(default=100, ge=1, le=1000)  # 100 par defaut, 1000 max
    method: str = Field(default="http", description="http (old.reddit JSON) ou selenium")


class SentimentRequest(BaseModel):
    """Request model for sentiment analysis endpoint"""
    texts: list[str] = Field(min_length=1)


class AnalyzeRequest(BaseModel):
    """Request model for full pipeline analysis"""
    crypto: str = Field(default="bitcoin", min_length=2, max_length=20)
    subreddit: str = Field(default="Bitcoin", min_length=1, max_length=50)
    limit: int = Field(default=100, ge=1, le=1000)  # 100 par defaut, 1000 max


class EconometricsRequest(BaseModel):
    """Request model for econometric analysis"""
    crypto: str = Field(default="bitcoin", min_length=2, max_length=20)
    subreddit: str = Field(default="Bitcoin", min_length=1, max_length=50)
    limit: int = Field(default=200, ge=50, le=1000)  # 200 par defaut, 1000 max
    days: int = Field(default=30, ge=7, le=90)
    maxlag: int = Field(default=14, ge=1, le=21)


# -------------------- Endpoints --------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    GET / : Page d'accueil HTML de présentation
    Status: 200 OK
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Crypto Sentiment API"}
    )


@app.get("/health")
async def health_check():
    """
    GET /health : Health check endpoint
    Status: 200 OK
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "crypto-sentiment-api"
    }


@app.post("/scrape", status_code=status.HTTP_201_CREATED)
async def scrape_reddit(request: ScrapeRequest):
    """
    POST /scrape : Webscraping Reddit posts
    Status: 201 Created (new data scraped)
    Status: 400 Bad Request (invalid parameters)
    Status: 500 Internal Server Error
    """
    try:
        if request.method.lower() == "selenium":
            scraper = RedditScraper()
            posts = scraper.scrape_subreddit(
                subreddit=request.subreddit,
                crypto=request.crypto,
                limit=request.limit
            )
            scraper.close()
        else:
            # Méthode HTTP par défaut (old.reddit JSON)
            http_scraper = RedditHTTPScraper()
            # Utilise la crypto comme requête par défaut
            query = request.crypto or request.subreddit
            posts = http_scraper.scrape(
                subreddit=request.subreddit,
                query=query,
                limit=request.limit
            )
        
        if not posts:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No posts found for {request.crypto} in r/{request.subreddit}"
            )
        
        return {
            "status": "success",
            "subreddit": request.subreddit,
            "crypto": request.crypto,
            "method": request.method.lower(),
            "count": len(posts),
            "posts": posts
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scraping error: {str(e)}"
        )


@app.post("/sentiment", status_code=status.HTTP_200_OK)
async def analyze_sentiment(request: SentimentRequest):
    """
    POST /sentiment : Analyse de sentiment avec FinBERT
    Status: 200 OK
    Status: 400 Bad Request (invalid input)
    Status: 500 Internal Server Error
    """
    try:
        # Clean texts before analysis
        cleaned_texts = [clean_text(text) for text in request.texts]
        cleaned_texts = [t for t in cleaned_texts if t]  # Remove empty
        
        if not cleaned_texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid text to analyze after cleaning"
            )
        
        analyzer = SentimentAnalyzer()
        results = analyzer.analyze_batch(cleaned_texts)
        
        # Calculate aggregate score
        avg_score = sum(r["score"] for r in results) / len(results)
        
        return {
            "status": "success",
            "count": len(results),
            "average_score": round(avg_score, 4),
            "results": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis error: {str(e)}"
        )


@app.get("/prices/{crypto}")
async def get_crypto_price(crypto: str):
    """
    GET /prices/{crypto} : Récupère le prix actuel d'une crypto
    Status: 200 OK
    Status: 404 Not Found (crypto not found)
    """
    try:
        prices = CryptoPrices()
        data = prices.get_current_price(crypto.lower())
        
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Crypto '{crypto}' not found"
            )
        
        return {
            "status": "success",
            "crypto": crypto,
            "data": data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Price fetch error: {str(e)}"
        )


@app.post("/analyze", status_code=status.HTTP_201_CREATED)
async def full_analysis(request: AnalyzeRequest):
    """
    POST /analyze : Pipeline complet (scrape + sentiment + price)
    Status: 201 Created
    Status: 500 Internal Server Error
    """
    try:
        # Step 1: Scrape Reddit
        scraper = RedditScraper()
        posts = scraper.scrape_subreddit(
            subreddit=request.subreddit,
            crypto=request.crypto,
            limit=request.limit
        )
        
        if not posts:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No posts found"
            )
        
        # Step 2: Clean and analyze sentiment
        texts = [clean_text(p["title"] + " " + p.get("text", "")) for p in posts]
        texts = [t for t in texts if t]
        
        analyzer = SentimentAnalyzer()
        sentiments = analyzer.analyze_batch(texts)
        avg_sentiment = sum(s["score"] for s in sentiments) / len(sentiments)
        
        # Step 3: Get current price
        prices = CryptoPrices()
        price_data = prices.get_current_price(request.crypto.lower())
        
        return {
            "status": "success",
            "crypto": request.crypto,
            "subreddit": request.subreddit,
            "analysis": {
                "posts_analyzed": len(texts),
                "average_sentiment": round(avg_sentiment, 4),
                "sentiment_label": "positive" if avg_sentiment > 0 else "negative",
                "current_price": price_data
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis error: {str(e)}"
        )


@app.post("/econometrics", status_code=status.HTTP_201_CREATED)
async def econometric_analysis(request: EconometricsRequest):
    """
    POST /econometrics : Analyse économétrique complète (VAR, Granger causality)
    
    Valide la relation sentiment → returns selon la méthodologie du papier académique:
    - Test de stationnarité (ADF)
    - Modèle VAR avec sélection automatique du lag optimal
    - Tests de causalité de Granger bidirectionnels
    - Analyse des corrélations croisées
    
    Status: 201 Created
    Status: 400 Bad Request (insufficient data)
    Status: 500 Internal Server Error
    """
    try:
        # Step 1: Scrape Reddit posts
        scraper = RedditScraper()
        posts = scraper.scrape_subreddit(
            subreddit=request.subreddit,
            crypto=request.crypto,
            limit=request.limit
        )
        
        if len(posts) < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient posts: {len(posts)} (minimum 50 required)"
            )
        
        # Step 2: Analyze sentiment
        texts = [clean_text(p["title"] + " " + p.get("text", "")) for p in posts]
        texts = [t for t in texts if t]
        
        analyzer = SentimentAnalyzer()
        sentiments = analyzer.analyze_batch(texts)
        
        # Step 3: Get historical prices
        prices_client = CryptoPrices()
        prices = prices_client.get_historical_prices(
            crypto=request.crypto,
            days=request.days
        )
        
        if not prices or len(prices) < 7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Insufficient price data"
            )
        
        # Step 4: Run econometric analysis
        econ_analyzer = EconometricAnalyzer()
        results = econ_analyzer.full_analysis(
            posts=posts,
            sentiments=sentiments,
            prices=prices,
            maxlag=request.maxlag
        )
        
        return {
            "status": "success",
            "crypto": request.crypto,
            "methodology": {
                "source": "Adapted from 'From Tweets to Returns' paper",
                "sentiment_model": "FinBERT (ProsusAI)",
                "econometric_tests": ["ADF", "VAR", "Granger Causality", "Cross-Correlation"]
            },
            "results": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Econometric analysis error: {str(e)}"
        )


# -------------------- Run with Uvicorn --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
