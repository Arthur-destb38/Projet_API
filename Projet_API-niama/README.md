# Crypto Sentiment API 🪙

API d'analyse de sentiment pour cryptomonnaies via webscraping Reddit et FinBERT.

**Master 2 MoSEF Data Science 2024-2025**  
Cours: Webscraping & API - José Ángel García Sánchez

---

## 🎯 Objectif

Adaptation de la méthodologie de l'article "From Tweets to Returns: Validating LLM-Based Sentiment Signals in Energy Stocks" au marché des cryptomonnaies:

- **Source originale**: Twitter → **Adaptation**: Reddit (API gratuite)
- **Marché original**: Actions énergétiques → **Adaptation**: Cryptomonnaies (BTC, ETH, SOL)
- **Modèle NLP**: FinBERT (sentiment financier)

---

## 🚀 Installation

```bash
# Cloner le projet
cd crypto_sentiment_api

# Installer Poetry (si nécessaire)
pip install poetry

# Installer les dépendances
poetry install

# Lancer l'API
poetry run uvicorn app.main:app --reload
```

L'API sera disponible sur `http://localhost:8000`

**Note:** Chrome doit être installé pour le webscraping Selenium.

---

## 🔌 Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Page d'accueil HTML |
| GET | `/health` | Health check |
| POST | `/scrape` | Webscraping Reddit |
| POST | `/sentiment` | Analyse FinBERT |
| GET | `/prices/{crypto}` | Prix CoinGecko |
| POST | `/analyze` | Pipeline complet |
| POST | `/econometrics` | Analyse VAR & Granger |

---

## 📊 Exemples d'utilisation

### Scraper Reddit
```bash
curl -X POST "http://localhost:8000/scrape" \
  -H "Content-Type: application/json" \
  -d '{"subreddit": "Bitcoin", "crypto": "BTC", "limit": 10}'
```

### Analyser le sentiment
```bash
curl -X POST "http://localhost:8000/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Bitcoin is pumping!", "Market is crashing"]}'
```

### Prix d'une crypto
```bash
curl "http://localhost:8000/prices/bitcoin"
```

### Pipeline complet
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"crypto": "bitcoin", "subreddit": "Bitcoin", "limit": 50}'
```

### Analyse économétrique (VAR, Granger)
```bash
curl -X POST "http://localhost:8000/econometrics" \
  -H "Content-Type: application/json" \
  -d '{"crypto": "bitcoin", "subreddit": "Bitcoin", "limit": 100, "days": 30, "maxlag": 14}'
```

La réponse inclut:
- Tests de stationnarité (ADF)
- Modèle VAR avec lag optimal
- Tests de causalité de Granger bidirectionnels
- Corrélations croisées avec lag optimal
- Conclusion sur la relation sentiment → returns

---

## 🛠️ Stack Technique

- **FastAPI**: Framework API async
- **Selenium**: Webscraping avec simulation comportement humain
- **BeautifulSoup**: Parsing HTML
- **FinBERT**: Modèle de sentiment financier (HuggingFace)
- **CoinGecko**: API prix crypto (gratuite)
- **Statsmodels**: VAR, tests de Granger, ADF
- **Pandas/NumPy**: Data manipulation
- **Pydantic**: Validation des données
- **Uvicorn**: Serveur ASGI

---

## 📁 Structure du Projet

```
crypto_sentiment_api/
├── pyproject.toml      # Dépendances Poetry
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py         # Endpoints FastAPI
│   ├── scraper.py      # Reddit scraper (PRAW)
│   ├── sentiment.py    # FinBERT analyzer
│   ├── prices.py       # CoinGecko client
│   ├── utils.py        # Text cleaning
│   └── econometrics.py # VAR, Granger causality
└── templates/
    └── index.html      # Page d'accueil
```

---

## 📋 Status Codes HTTP

- `200 OK`: Requête réussie
- `201 Created`: Nouvelle ressource créée (scraping)
- `400 Bad Request`: Paramètres invalides
- `404 Not Found`: Ressource non trouvée
- `500 Internal Server Error`: Erreur serveur

---

## 🔗 Prérequis Selenium

1. Installer Chrome ou Chromium
2. ChromeDriver sera géré automatiquement par `webdriver-manager`

Le scraper utilise `old.reddit.com` (plus facile à parser) et simule un comportement humain:
- Random delays entre actions
- User-Agent rotation
- Scrolling naturel
- Anti-detection flags

---

## 📚 Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI: `http://localhost:8000/openapi.json`
