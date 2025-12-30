import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urlparse

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from selenium_helpers import build_chrome, human_delay, scroll_incremental


def build_search_url(query: str, subreddit: Optional[str]) -> str:
    q = quote_plus(query)
    if subreddit:
        return f"https://old.reddit.com/r/{subreddit}/search?q={q}&sort=new&restrict_sr=on"
    return f"https://old.reddit.com/search?q={q}&sort=new"


def safe_text(element, selector: str) -> Optional[str]:
    try:
        return element.find_element(By.CSS_SELECTOR, selector).text
    except Exception:
        return None


def safe_attr(element, selector: str, attr: str) -> Optional[str]:
    try:
        return element.find_element(By.CSS_SELECTOR, selector).get_attribute(attr)
    except Exception:
        return None


def extract_post(card) -> Dict:
    url = safe_attr(card, "p.title a.title", "href")
    timestamp = safe_attr(card, "time", "datetime")
    post_id = None
    if url:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if parts:
            post_id = parts[-1]
    return {
        "platform": "reddit",
        "post_id": post_id,
        "author": card.get_attribute("data-author"),
        "handle": card.get_attribute("data-author"),
        "created_utc": timestamp,
        "text": safe_text(card, "p.title"),
        "lang": None,
        "likes": safe_attr(card, "div.score.unvoted", "title") or safe_text(card, "div.score.unvoted"),
        "replies": safe_text(card, "a.comments"),
        "shares": None,
        "url": url,
        "subreddit": safe_text(card, "a.subreddit"),
    }


def save_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def accept_consent_if_present(driver, wait: WebDriverWait) -> None:
    """Handle Reddit consent pop-up if it shows up."""
    selectors = [
        (By.XPATH, "//button[contains(translate(., 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 'CONTINUE')]"),
        (By.XPATH, "//button[contains(translate(., 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 'ACCEPT')]"),
        (By.CSS_SELECTOR, "button[aria-label*='Accept']"),
    ]
    for by, sel in selectors:
        try:
            consent = wait.until(EC.element_to_be_clickable((by, sel)))
            consent.click()
            human_delay(1.0, 2.0)
            return
        except Exception:
            continue


def scrape(query: str, subreddit: Optional[str], limit: int, headless: bool, output: Path) -> int:
    driver = build_chrome(headless=headless)
    driver.get(build_search_url(query, subreddit))
    wait = WebDriverWait(driver, 20)
    collected: List[Dict] = []
    seen_urls = set()

    try:
        accept_consent_if_present(driver, wait)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.thing")))
    except Exception:
        print("Aucun post détecté (échec d'attente initiale).")
        driver.quit()
        return 0

    iterations = 0
    while len(collected) < limit and iterations < 20:
        iterations += 1
        cards = driver.find_elements(By.CSS_SELECTOR, "div.thing")
        print(f"[it={iterations}] cartes visibles: {len(cards)} / collectés: {len(collected)}")
        for card in cards:
            post = extract_post(card)
            if not post["url"] or post["url"] in seen_urls:
                continue
            seen_urls.add(post["url"])
            collected.append(post)
            if len(collected) >= limit:
                break
        if len(collected) >= limit:
            break
        scroll_incremental(driver, steps=3, pause_range=(1.0, 3.0))
        human_delay(1.0, 3.0)

    driver.quit()
    save_jsonl(collected, output)
    return len(collected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Reddit posts with Selenium (crypto focus).")
    parser.add_argument("--query", required=True, help="Mots-clés à rechercher (ex: bitcoin OR ethereum).")
    parser.add_argument("--subreddit", help="Subreddit ciblé (ex: CryptoCurrency).")
    parser.add_argument("--limit", type=int, default=50, help="Nombre de posts à collecter (approx).")
    parser.add_argument("--headless", action="store_true", help="Activer le mode headless.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/reddit_crypto.jsonl"),
        help="Chemin de sortie JSONL.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count = scrape(args.query, args.subreddit, args.limit, args.headless, args.output)
    print(f"Scraped {count} posts -> {args.output}")
