import argparse
import json
from pathlib import Path
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


def build_search_url(query: str, subreddit: Optional[str]) -> str:
    base = "https://old.reddit.com"
    if subreddit:
        return f"{base}/r/{subreddit}/search.json"
    return f"{base}/search.json"


def fetch_page(
    query: str,
    subreddit: Optional[str],
    limit: int,
    after: Optional[str] = None,
) -> Dict:
    params = {
        "q": query,
        "sort": "new",
        "restrict_sr": "on" if subreddit else "off",
        "limit": min(limit, 100),
    }
    if after:
        params["after"] = after
    url = build_search_url(query, subreddit) + "?" + urlencode(params)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def map_item(child: Dict) -> Dict:
    d = child.get("data", {})
    text = d.get("title") or ""
    if d.get("selftext"):
        text = (text + " " + d["selftext"]).strip()
    return {
        "platform": "reddit",
        "post_id": d.get("id"),
        "author": d.get("author"),
        "handle": d.get("author"),
        "created_utc": d.get("created_utc"),
        "text": text,
        "lang": d.get("lang"),
        "likes": d.get("score"),
        "replies": d.get("num_comments"),
        "shares": None,
        "url": f"https://www.reddit.com{d.get('permalink')}" if d.get("permalink") else d.get("url"),
        "subreddit": d.get("subreddit"),
    }


def save_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scrape(query: str, subreddit: Optional[str], limit: int, output: Path) -> int:
    collected: List[Dict] = []
    after = None
    while len(collected) < limit:
        remaining = limit - len(collected)
        data = fetch_page(query, subreddit, remaining, after)
        children = data.get("data", {}).get("children", [])
        if not children:
            break
        for child in children:
            collected.append(map_item(child))
            if len(collected) >= limit:
                break
        after = data.get("data", {}).get("after")
        if not after:
            break
    save_jsonl(collected, output)
    return len(collected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Reddit via old.reddit.com JSON search (no Selenium).")
    parser.add_argument("--query", required=True, help="Mots-clés (ex: bitcoin OR ethereum)")
    parser.add_argument("--subreddit", help="Subreddit ciblé (ex: CryptoCurrency)")
    parser.add_argument("--limit", type=int, default=50, help="Nombre de posts à collecter")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Projet_API/data/raw/reddit_http.jsonl"),
        help="Chemin de sortie JSONL",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count = scrape(args.query, args.subreddit, args.limit, args.output)
    print(f"Scraped {count} posts -> {args.output}")
