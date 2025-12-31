"""
Text Cleaning Utilities
Author: MoSEF Student
Description: Text preprocessing functions for sentiment analysis.
Cleans Reddit posts by removing URLs, mentions, emojis, and normalizing text.
Master 2 MoSEF Data Science 2024-2025
"""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean text for sentiment analysis.
    
    Steps:
    1. Remove URLs
    2. Remove Reddit mentions (u/username)
    3. Remove subreddit links (r/subreddit)
    4. Remove emojis
    5. Remove special characters
    6. Normalize whitespace
    7. Convert to lowercase
    
    Args:
        text: Raw text from Reddit
    
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove Reddit-specific formatting
    text = re.sub(r'u/\w+', '', text)  # u/username
    text = re.sub(r'r/\w+', '', text)  # r/subreddit
    text = re.sub(r'\[deleted\]', '', text)
    text = re.sub(r'\[removed\]', '', text)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)  # Italic
    text = re.sub(r'~~(.+?)~~', r'\1', text)  # Strikethrough
    text = re.sub(r'`(.+?)`', r'\1', text)  # Code
    text = re.sub(r'#+\s*', '', text)  # Headers
    
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip and lowercase
    text = text.strip().lower()
    
    return text


def extract_crypto_mentions(text: str) -> list[str]:
    """
    Extract cryptocurrency mentions from text.
    
    Args:
        text: Input text
    
    Returns:
        List of mentioned cryptocurrencies
    """
    # Common crypto patterns
    patterns = {
        "BTC": [r'\bbitcoin\b', r'\bbtc\b', r'\$btc\b'],
        "ETH": [r'\bethereum\b', r'\beth\b', r'\$eth\b', r'\bether\b'],
        "SOL": [r'\bsolana\b', r'\bsol\b', r'\$sol\b'],
        "XRP": [r'\bripple\b', r'\bxrp\b', r'\$xrp\b'],
        "ADA": [r'\bcardano\b', r'\bada\b', r'\$ada\b'],
        "DOGE": [r'\bdogecoin\b', r'\bdoge\b', r'\$doge\b'],
    }
    
    text_lower = text.lower()
    mentioned = []
    
    for crypto, crypto_patterns in patterns.items():
        for pattern in crypto_patterns:
            if re.search(pattern, text_lower):
                if crypto not in mentioned:
                    mentioned.append(crypto)
                break
    
    return mentioned


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to max length, preserving word boundaries.
    
    Args:
        text: Input text
        max_length: Maximum character length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    
    # Find last space to avoid cutting words
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # Keep at least 80% of text
        truncated = truncated[:last_space]
    
    return truncated + "..."


def is_valid_text(text: str, min_length: int = 10) -> bool:
    """
    Check if text is valid for sentiment analysis.
    
    Args:
        text: Input text
        min_length: Minimum character length
    
    Returns:
        True if text is valid
    """
    if not text:
        return False
    
    cleaned = clean_text(text)
    
    # Check minimum length
    if len(cleaned) < min_length:
        return False
    
    # Check if it's mostly noise
    words = cleaned.split()
    if len(words) < 3:
        return False
    
    return True


def batch_clean(texts: list[str]) -> list[str]:
    """
    Clean multiple texts at once.
    
    Args:
        texts: List of raw texts
    
    Returns:
        List of cleaned texts (empty strings for invalid texts)
    """
    return [clean_text(t) if is_valid_text(t) else "" for t in texts]


# -------------------- Test --------------------
if __name__ == "__main__":
    test_texts = [
        "🚀🚀 BITCOIN TO THE MOON!!! $BTC is pumping hard https://example.com/link u/someone",
        "Just bought some **ETH** at $2000, r/ethereum is bullish!",
        "Market is crashing 📉📉📉 everyone selling their SOL bags",
    ]
    
    for text in test_texts:
        print(f"Original: {text}")
        print(f"Cleaned:  {clean_text(text)}")
        print(f"Cryptos:  {extract_crypto_mentions(text)}")
        print()
