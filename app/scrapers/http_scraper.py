"""
Scraper Reddit via API JSON (methode rapide)
Pas de Selenium, juste requests
"""

import requests
import time
from urllib.parse import urlencode


class HttpScraper:
    BASE_URL = "https://old.reddit.com"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def scrape_subreddit(self, subreddit: str, query: str = "", limit: int = 50) -> list[dict]:
        posts = []
        after = None

        while len(posts) < limit:
            params = {"limit": min(100, limit - len(posts)), "sort": "new"}

            if query:
                params["q"] = query
                params["restrict_sr"] = "on"
                url = f"{self.BASE_URL}/r/{subreddit}/search.json"
            else:
                url = f"{self.BASE_URL}/r/{subreddit}/new.json"

            if after:
                params["after"] = after

            try:
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"Erreur HTTP: {e}")
                break

            children = data.get("data", {}).get("children", [])
            if not children:
                break

            for child in children:
                d = child.get("data", {})
                posts.append({
                    "id": d.get("id"),
                    "title": d.get("title", ""),
                    "text": d.get("selftext", ""),
                    "score": d.get("score", 0),
                    "num_comments": d.get("num_comments", 0),
                    "created_utc": d.get("created_utc"),
                    "author": d.get("author"),
                    "url": f"https://reddit.com{d.get('permalink', '')}",
                    "subreddit": d.get("subreddit")
                })

                if len(posts) >= limit:
                    break

            after = data.get("data", {}).get("after")
            if not after:
                break

            time.sleep(0.5)

        return posts

    def close(self):
        self.session.close()