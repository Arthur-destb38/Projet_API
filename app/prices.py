"""
Cryptocurrency Price Module
Author: MoSEF Student
Description: Fetches cryptocurrency prices from CoinGecko API.
Free API, no authentication required.
Master 2 MoSEF Data Science 2024-2025
"""

import requests
from datetime import datetime, timedelta
from typing import Optional
import time


class CryptoPrices:
    """
    Fetches crypto prices from CoinGecko API.
    Free tier: 10-30 calls/minute depending on endpoint.
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    # Mapping from ticker to CoinGecko ID
    # Liste complete des cryptos supportees
    CRYPTO_IDS = {
        # Top 10
        "btc": "bitcoin",
        "bitcoin": "bitcoin",
        "eth": "ethereum",
        "ethereum": "ethereum",
        "bnb": "binancecoin",
        "xrp": "ripple",
        "ripple": "ripple",
        "sol": "solana",
        "solana": "solana",
        "ada": "cardano",
        "cardano": "cardano",
        "doge": "dogecoin",
        "dogecoin": "dogecoin",
        "avax": "avalanche-2",
        "avalanche": "avalanche-2",
        "dot": "polkadot",
        "polkadot": "polkadot",
        # Top 20
        "matic": "matic-network",
        "polygon": "matic-network",
        "link": "chainlink",
        "chainlink": "chainlink",
        "atom": "cosmos",
        "cosmos": "cosmos",
        "ltc": "litecoin",
        "litecoin": "litecoin",
        "shib": "shiba-inu",
        "uni": "uniswap",
        "uniswap": "uniswap",
        "xlm": "stellar",
        "stellar": "stellar",
        "etc": "ethereum-classic",
        "near": "near",
        "apt": "aptos",
        "aptos": "aptos",
        "arb": "arbitrum",
        "arbitrum": "arbitrum",
        "op": "optimism",
        "optimism": "optimism",
        # Memecoins
        "pepe": "pepe",
        "floki": "floki",
        "bonk": "bonk",
    }

    # User-Agent header to avoid blocks
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json"
    }

    def __init__(self):
        """Initialize with session for connection pooling"""
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def _get_coin_id(self, crypto: str) -> str:
        """Convert ticker/name to CoinGecko ID"""
        return self.CRYPTO_IDS.get(crypto.lower(), crypto.lower())

    def get_current_price(self, crypto: str, currency: str = "usd") -> Optional[dict]:
        """
        Get current price and basic market data.
        """
        coin_id = self._get_coin_id(crypto)

        url = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": currency,
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true"
        }

        try:
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if coin_id not in data:
                    return None

                coin_data = data[coin_id]

                return {
                    "crypto": crypto,
                    "coin_id": coin_id,
                    "currency": currency,
                    "price": coin_data.get(currency),
                    "market_cap": coin_data.get(f"{currency}_market_cap"),
                    "volume_24h": coin_data.get(f"{currency}_24h_vol"),
                    "change_24h": round(coin_data.get(f"{currency}_24h_change", 0), 2),
                    "last_updated": datetime.fromtimestamp(
                        coin_data.get("last_updated_at", 0)
                    ).isoformat()
                }

            elif response.status_code == 429:
                print("Rate limited by CoinGecko, waiting...")
                time.sleep(60)
                return None

            else:
                print(f"CoinGecko API error: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None

    def get_historical_prices(
        self,
        crypto: str,
        days: int = 30,
        currency: str = "usd"
    ) -> Optional[list[dict]]:
        """
        Get historical daily prices.
        """
        coin_id = self._get_coin_id(crypto)

        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": currency,
            "days": days,
            "interval": "daily"
        }

        try:
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                prices = data.get("prices", [])

                result = []
                for timestamp, price in prices:
                    result.append({
                        "date": datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d"),
                        "timestamp": timestamp,
                        "price": round(price, 2)
                    })

                return result

            elif response.status_code == 429:
                print("Rate limited by CoinGecko")
                return None

            else:
                print(f"CoinGecko API error: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None

    def get_multiple_prices(
        self,
        cryptos: list[str],
        currency: str = "usd"
    ) -> dict:
        """
        Get prices for multiple cryptocurrencies in one call.
        """
        coin_ids = [self._get_coin_id(c) for c in cryptos]
        ids_string = ",".join(coin_ids)

        url = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": ids_string,
            "vs_currencies": currency,
            "include_24hr_change": "true"
        }

        try:
            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                result = {}
                for crypto, coin_id in zip(cryptos, coin_ids):
                    if coin_id in data:
                        result[crypto] = {
                            "price": data[coin_id].get(currency),
                            "change_24h": round(data[coin_id].get(f"{currency}_24h_change", 0), 2)
                        }

                return result

            else:
                return {}

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return {}


# -------------------- Test --------------------
if __name__ == "__main__":
    prices = CryptoPrices()

    btc = prices.get_current_price("bitcoin")
    if btc:
        print(f"Bitcoin: ${btc['price']:,.2f} ({btc['change_24h']:+.2f}%)")

    multi = prices.get_multiple_prices(["btc", "eth", "sol"])
    for crypto, data in multi.items():
        print(f"{crypto.upper()}: ${data['price']:,.2f}")