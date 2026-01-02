# Projet_API

Analyse de sentiment crypto : scraping (HTTP/Selenium), normalisation, FastAPI et interface Streamlit.

## Structure
- `api/` : service FastAPI (scraping, sentiment, prix, économétrie)
- `scripts/` : scrapers/normalisation batch
- `data/raw/` : sorties brutes (JSONL/CSV)
- `data/clean/` : données normalisées
- `notebooks/` : démos et explorations (voir `getting_started.ipynb`)

## Prérequis
- Python 3.11 recommandé (Poetry) ; Chrome installé si Selenium.
- Installer via Poetry :
  ```
  export PATH="$HOME/Library/Python/3.14/bin:$PATH"   # si poetry est ici
  POETRY_VIRTUALENVS_IN_PROJECT=1 poetry install --no-root
  ```

## Lancer l’API
```
cd api
POETRY_VIRTUALENVS_IN_PROJECT=1 poetry run uvicorn app.main:app --reload
```
Endpoints clés : `/health`, `/cryptos`, `/prices`, `/scrape`, `/analyze`, `/analyze/multi`.

## Streamlit
Interface locale :
```
POETRY_VIRTUALENVS_IN_PROJECT=1 poetry run streamlit run streamlit_app.py
```

## Scraper Reddit (HTTP)
- Exemple :
  ```
  python scripts/scrape_reddit_http.py --query "bitcoin OR ethereum" --subreddit CryptoCurrency --limit 100 --output data/raw/reddit_http.jsonl
  ```
- Pourquoi HTTP : old.reddit.com expose un JSON public, évite cookies/captcha et le JS lourd.

## Batch et normalisation
- Batch multi-subreddits : `python scripts/run_reddit_batch.py`
- Fusion/déduplication : `python scripts/normalize_reddit.py --inputs "data/raw/*.jsonl" --output data/clean/reddit_clean.jsonl`

## Notes Selenium
- Optionnel : `method=selenium` sur `/scrape` ou `/analyze`.
- Nécessite un driver Chrome/Firefox accessible (headless par défaut).
