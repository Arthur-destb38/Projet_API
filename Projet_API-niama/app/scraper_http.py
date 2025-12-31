"""
Reddit Scraper Module - HTTP Version
Scrapes posts via old.reddit.com JSON search (pas d'UI Selenium).
"""

import random
import time
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


class RedditHTTPScraper:
    """
    Scraper HTTP pour old.reddit.com/search.json
    Pas de Selenium, pagination via le token 'after'.
    """

    def __init__(self):
        pass

    def _build_url(self, subreddit: Optional[str]) -> str:
        base = "https://old.reddit.com"
        if subreddit:
            return f"{base}/r/{subreddit}/search.json"
        return f"{base}/search.json"

    def _fetch_page(self, query: str, subreddit: Optional[str], limit: int, after: Optional[str]) -> Dict:
        params = {
            "q": query,
            "sort": "new",
            "restrict_sr": "on" if subreddit else "off",
            "limit": min(limit, 100),
        }
        if after:
            params["after"] = after
        url = self._build_url(subreddit) + "?" + urlencode(params)
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        # Petite pause pour être poli
        time.sleep(random.uniform(0.5, 1.5))
        return resp.json()

    def _map_item(self, child: Dict) -> Dict:
        d = child.get("data", {})
        title = d.get("title") or ""
        text_body = d.get("selftext") or ""
        full_text = (title + " " + text_body).strip()
        url = d.get("url") or ""
        if d.get("permalink"):
            url = f"https://www.reddit.com{d.get('permalink')}"
        return {
            "id": d.get("id"),
            "title": title,
            "text": full_text,
            "score": d.get("score"),
            "num_comments": d.get("num_comments"),
            "created_utc": d.get("created_utc"),
            "url": url,
            "author": d.get("author"),
            "subreddit": d.get("subreddit"),
        }

    def scrape(self, subreddit: str, query: str, limit: int = 100) -> List[Dict]:
        collected: List[Dict] = []
        after = None
        while len(collected) < limit:
            remaining = limit - len(collected)
            data = self._fetch_page(query, subreddit, remaining, after)
            children = data.get("data", {}).get("children", [])
            if not children:
                break
            for child in children:
                collected.append(self._map_item(child))
                if len(collected) >= limit:
                    break
            after = data.get("data", {}).get("after")
            if not after:
                break
        return collected
