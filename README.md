# Projet_API

Objectif: scripts Selenium pour scraper des posts publics (Twitter/X, Reddit, Bluesky) sur les cryptomonnaies, normaliser les données et préparer l’analyse (sentiment finance).

Arborescence:
- `scripts/` : scrapers et helpers.
- `data/raw/` : sorties brutes (JSONL/CSV) par plateforme.
- `data/clean/` : données normalisées prêtes pour l’analyse.

Prérequis rapides:
- Python 3.9+, Chrome installé.
- (Optionnel) `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -r ../requirements.txt`

## Chronologie et solutions
- Tentative initiale Selenium sur reddit.com (new UI) → bloqué par pop-up cookies/captcha et résolution DNS pour télécharger ChromeDriver.
- Ajout des dépendances `selenium` + `webdriver_manager`, détection du binaire Chrome local et possibilité de forcer la version de ChromeDriver (env `CHROME_BINARY`, `CHROMEDRIVER_VERSION` ou `CHROMEDRIVER`).
- Bascule vers old.reddit.com pour réduire le JS, mais pop-up + anti-bot restaient un frein à l’automatisation.
- Résolution: passage à l’API JSON publique de old.reddit.com (pas de Selenium), récupération directe des posts au format JSON → succès avec 100 posts sur r/CryptoCurrency.

## Scraper Reddit (HTTP, sans Selenium)
- Commande exemple :\
  `python Projet_API/scripts/scrape_reddit_http.py --query "bitcoin OR ethereum" --subreddit CryptoCurrency --limit 100 --output Projet_API/data/raw/reddit_http.jsonl`
- Pourquoi ça marche : old.reddit.com expose un endpoint JSON public qui évite cookies/captcha; les données (titre+texte, auteur, score, commentaires, lien, timestamp) sont directement accessibles.
- Fichier généré (exécution réussie) : `Projet_API/data/raw/reddit_http.jsonl` (100 posts).
