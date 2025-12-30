"""
Reddit Scraper Module - Selenium Version
Author: MoSEF Student
Description: Scrapes cryptocurrency-related posts from Reddit using Selenium.
Simulates human behavior with random delays and scrolling.
Master 2 MoSEF Data Science 2024-2025
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime
from typing import Optional
import re


class RedditScraper:
    """
    Reddit scraper using Selenium + BeautifulSoup.
    Simulates human behavior to avoid detection.
    """

    # User agents to rotate (simulate different browsers)
    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]

    # Subreddit URLs (old.reddit.com = easier to scrape)
    SUBREDDIT_URLS = {
        # Generaux
        "CryptoCurrency": "https://old.reddit.com/r/CryptoCurrency/",
        "CryptoMarkets": "https://old.reddit.com/r/CryptoMarkets/",
        # Top cryptos
        "Bitcoin": "https://old.reddit.com/r/Bitcoin/",
        "btc": "https://old.reddit.com/r/Bitcoin/",
        "ethereum": "https://old.reddit.com/r/ethereum/",
        "eth": "https://old.reddit.com/r/ethereum/",
        "solana": "https://old.reddit.com/r/solana/",
        "sol": "https://old.reddit.com/r/solana/",
        "cardano": "https://old.reddit.com/r/cardano/",
        "ada": "https://old.reddit.com/r/cardano/",
        "ripple": "https://old.reddit.com/r/Ripple/",
        "xrp": "https://old.reddit.com/r/Ripple/",
        "dogecoin": "https://old.reddit.com/r/dogecoin/",
        "doge": "https://old.reddit.com/r/dogecoin/",
        "polkadot": "https://old.reddit.com/r/Polkadot/",
        "dot": "https://old.reddit.com/r/Polkadot/",
        "avalanche": "https://old.reddit.com/r/Avax/",
        "avax": "https://old.reddit.com/r/Avax/",
        "polygon": "https://old.reddit.com/r/maticnetwork/",
        "matic": "https://old.reddit.com/r/maticnetwork/",
        "chainlink": "https://old.reddit.com/r/Chainlink/",
        "link": "https://old.reddit.com/r/Chainlink/",
        "litecoin": "https://old.reddit.com/r/litecoin/",
        "ltc": "https://old.reddit.com/r/litecoin/",
        "cosmos": "https://old.reddit.com/r/cosmosnetwork/",
        "atom": "https://old.reddit.com/r/cosmosnetwork/",
        "uniswap": "https://old.reddit.com/r/UniSwap/",
        "uni": "https://old.reddit.com/r/UniSwap/",
        "arbitrum": "https://old.reddit.com/r/arbitrum/",
        "arb": "https://old.reddit.com/r/arbitrum/",
        "optimism": "https://old.reddit.com/r/optimism/",
        "op": "https://old.reddit.com/r/optimism/",
        # Memecoins
        "shib": "https://old.reddit.com/r/SHIBArmy/",
        "shibainu": "https://old.reddit.com/r/SHIBArmy/",
        "pepe": "https://old.reddit.com/r/pepecoin/",
        "floki": "https://old.reddit.com/r/Floki/",
        "bonk": "https://old.reddit.com/r/BONK/",
    }

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver = None
        self.demo_mode = False

    def _setup_driver(self):
        """Configure Chrome WebDriver with anti-detection settings"""
        options = Options()

        if self.headless:
            options.add_argument("--headless=new")

        # Anti-detection settings
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        options.add_argument(f"--user-agent={random.choice(self.USER_AGENTS)}")
        options.add_argument("--window-size=1920,1080")

        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                """
            })
            print("Selenium WebDriver initialized")
        except Exception as e:
            print(f"Selenium failed: {e}")
            self.demo_mode = True

    def _random_delay(self, min_sec: float = 1.0, max_sec: float = 3.0):
        """Human-like random delay between actions"""
        delay = random.uniform(min_sec, max_sec)
        time.sleep(delay)

    def _scroll_page(self, scrolls: int = 3):
        """Simulate human scrolling behavior"""
        for i in range(scrolls):
            scroll_distance = random.randint(300, 700)
            self.driver.execute_script(f"window.scrollBy(0, {scroll_distance});")
            self._random_delay(0.5, 1.5)

    def _parse_post_old_reddit(self, post_element) -> Optional[dict]:
        """Parse a single post from old.reddit.com HTML"""
        try:
            post_id = post_element.get("data-fullname", "")

            title_elem = post_element.select_one("a.title")
            title = title_elem.get_text(strip=True) if title_elem else ""

            score_elem = post_element.select_one("div.score.unvoted")
            score_text = score_elem.get_text(strip=True) if score_elem else "0"
            try:
                score = int(score_text) if score_text != "•" else 0
            except ValueError:
                score = 0

            comments_elem = post_element.select_one("a.comments")
            comments_text = comments_elem.get_text(strip=True) if comments_elem else "0"
            num_match = re.search(r"(\d+)", comments_text)
            num_comments = int(num_match.group(1)) if num_match else 0

            time_elem = post_element.select_one("time")
            timestamp = time_elem.get("datetime", "") if time_elem else datetime.now().isoformat()

            url = title_elem.get("href", "") if title_elem else ""
            if url.startswith("/"):
                url = f"https://reddit.com{url}"

            author_elem = post_element.select_one("a.author")
            author = author_elem.get_text(strip=True) if author_elem else "[deleted]"

            selftext_elem = post_element.select_one("div.md")
            text = selftext_elem.get_text(strip=True)[:500] if selftext_elem else ""

            return {
                "id": post_id,
                "title": title,
                "text": text,
                "score": score,
                "num_comments": num_comments,
                "created_utc": timestamp,
                "url": url,
                "author": author
            }

        except Exception as e:
            print(f"Error parsing post: {e}")
            return None

    def scrape_subreddit(
        self,
        subreddit: str,
        crypto: str = "BTC",
        limit: int = 50,
        sort: str = "hot"
    ) -> list[dict]:
        """Scrape posts from a subreddit using Selenium + BeautifulSoup"""

        if self.driver is None and not self.demo_mode:
            self._setup_driver()

        if self.demo_mode:
            print(f"DEMO MODE: Generating {limit} simulated posts")
            return self._generate_demo_posts(crypto, limit)

        posts = []

        try:
            base_url = self.SUBREDDIT_URLS.get(subreddit, f"https://old.reddit.com/r/{subreddit}/")
            if sort == "new":
                url = f"{base_url}new/"
            elif sort == "top":
                url = f"{base_url}top/?t=week"
            else:
                url = base_url

            print(f"Scraping {url}")

            self.driver.get(url)
            self._random_delay(2, 4)

            pages_scraped = 0
            max_pages = (limit // 25) + 1

            while len(posts) < limit and pages_scraped < max_pages:
                self._scroll_page(2)

                soup = BeautifulSoup(self.driver.page_source, "lxml")
                post_elements = soup.select("div.thing.link")

                for post_elem in post_elements:
                    if "stickied" in post_elem.get("class", []):
                        continue
                    if "promoted" in post_elem.get("class", []):
                        continue

                    post_data = self._parse_post_old_reddit(post_elem)

                    if post_data and post_data["title"]:
                        posts.append(post_data)

                        if len(posts) >= limit:
                            break

                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, "span.next-button a")
                    next_button.click()
                    self._random_delay(2, 4)
                    pages_scraped += 1
                except NoSuchElementException:
                    break

            print(f"Scraped {len(posts)} posts from r/{subreddit}")
            return posts[:limit]

        except Exception as e:
            print(f"Scraping error: {e}")
            return self._generate_demo_posts(crypto, limit)

    def _generate_demo_posts(self, crypto: str, limit: int) -> list[dict]:
        """Generate demo posts for testing"""
        from datetime import timedelta

        positive_templates = [
            f"{crypto} is pumping! Great time to accumulate",
            f"Just bought more {crypto}, this dip is a gift",
            f"{crypto} breaking out! Next stop moon",
            f"Institutional adoption of {crypto} is accelerating",
            f"{crypto} fundamentals stronger than ever",
        ]

        negative_templates = [
            f"{crypto} looking weak, might dump further",
            f"Sold all my {crypto}, this market is dead",
            f"{crypto} bear market not over yet",
            f"Regulation fears hitting {crypto} hard",
            f"{crypto} volume declining, not a good sign",
        ]

        neutral_templates = [
            f"What do you think about {crypto} at current levels?",
            f"{crypto} moving sideways, waiting for direction",
            f"Anyone else DCAing into {crypto}?",
            f"Interesting {crypto} analysis I found today",
            f"{crypto} holding support for now",
        ]

        posts = []
        base_date = datetime.now()
        posts_per_day = max(limit // 30, 3)

        for day in range(30):
            for j in range(posts_per_day):
                if len(posts) >= limit:
                    break

                rand = random.random()
                if rand < 0.4:
                    title = random.choice(positive_templates)
                elif rand < 0.7:
                    title = random.choice(negative_templates)
                else:
                    title = random.choice(neutral_templates)

                post_date = base_date - timedelta(days=day, hours=random.randint(0, 23))

                posts.append({
                    "id": f"demo_{len(posts)}_{random.randint(1000, 9999)}",
                    "title": title,
                    "text": "",
                    "score": random.randint(1, 500),
                    "num_comments": random.randint(0, 100),
                    "created_utc": post_date.strftime("%Y-%m-%dT%H:%M:%S"),
                    "url": f"https://reddit.com/r/{crypto}/comments/demo{len(posts)}",
                    "author": f"demo_user_{random.randint(1, 100)}"
                })

            if len(posts) >= limit:
                break

        return posts

    def close(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None


if __name__ == "__main__":
    scraper = RedditScraper(headless=True)
    posts = scraper.scrape_subreddit("Bitcoin", "BTC", limit=10)

    for p in posts:
        print(f"- {p['title'][:60]}... (score: {p['score']})")

    scraper.close()