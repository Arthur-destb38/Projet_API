"""
Orchestrateur simple pour lancer plusieurs scrapes Reddit HTTP.
Adapté aux combos utilisés plus haut (queries/subreddits) pour augmenter la couverture.
"""
import subprocess
import sys
from pathlib import Path


SCRAPES = [
    {
        "query": "bitcoin OR ethereum",
        "subreddit": "CryptoCurrency",
        "limit": "1000",
        "output": "Projet_API/data/raw/reddit_cc_btc_eth.jsonl",
    },
    {
        "query": "bitcoin OR ethereum",
        "subreddit": "CryptoMarkets",
        "limit": "1000",
        "output": "Projet_API/data/raw/reddit_cm_btc_eth.jsonl",
    },
    {
        "query": "defi OR solana OR cardano",
        "subreddit": "CryptoCurrency",
        "limit": "800",
        "output": "Projet_API/data/raw/reddit_cc_defi.jsonl",
    },
    {
        "query": "btc OR eth",
        "subreddit": "Bitcoin",
        "limit": "800",
        "output": "Projet_API/data/raw/reddit_btc_sub.jsonl",
    },
]


def run(cmd: list):
    print(">>", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    script_path = Path(__file__).resolve().parent / "scrape_reddit_http.py"
    for job in SCRAPES:
        cmd = [
            sys.executable,
            str(script_path),
            "--query",
            job["query"],
            "--subreddit",
            job["subreddit"],
            "--limit",
            job["limit"],
            "--output",
            job["output"],
        ]
        run(cmd)
