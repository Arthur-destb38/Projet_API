"""
Crypto Sentiment - Streamlit
Projet MoSEF 2024-2025
Deux methodes: Selenium (cours) et HTTP (bonus)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import re
import random
from datetime import datetime, timedelta

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Selenium - import conditionnel
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import NoSuchElementException
    from bs4 import BeautifulSoup
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


# ============ CONFIG ============

st.set_page_config(page_title="Crypto Sentiment", layout="wide")

CRYPTO_LIST = {
    "Bitcoin (BTC)": {"id": "bitcoin", "sub": "Bitcoin"},
    "Ethereum (ETH)": {"id": "ethereum", "sub": "ethereum"},
    "Solana (SOL)": {"id": "solana", "sub": "solana"},
    "Cardano (ADA)": {"id": "cardano", "sub": "cardano"},
    "Dogecoin (DOGE)": {"id": "dogecoin", "sub": "dogecoin"},
    "Ripple (XRP)": {"id": "ripple", "sub": "xrp"},
    "Polkadot (DOT)": {"id": "polkadot", "sub": "polkadot"},
    "Chainlink (LINK)": {"id": "chainlink", "sub": "chainlink"},
    "Litecoin (LTC)": {"id": "litecoin", "sub": "litecoin"},
    "Avalanche (AVAX)": {"id": "avalanche-2", "sub": "avax"},
    "Polygon (MATIC)": {"id": "matic-network", "sub": "maticnetwork"},
    "Cosmos (ATOM)": {"id": "cosmos", "sub": "cosmosnetwork"},
    "Shiba Inu (SHIB)": {"id": "shiba-inu", "sub": "SHIBArmy"},
    "Pepe (PEPE)": {"id": "pepe", "sub": "pepecoin"},
    "Arbitrum (ARB)": {"id": "arbitrum", "sub": "arbitrum"},
}


# ============ UTILS ============

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"u/\w+", "", text)
    text = re.sub(r"r/\w+", "", text)
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# ============ SCRAPER HTTP ============

def scrape_http(subreddit, limit=50):
    """Scrape via API JSON Reddit"""
    posts = []
    after = None
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}

    while len(posts) < limit:
        url = f"https://old.reddit.com/r/{subreddit}/new.json"
        params = {"limit": min(100, limit - len(posts))}

        if after:
            params["after"] = after

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            data = resp.json()
        except:
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
                "author": d.get("author"),
            })

            if len(posts) >= limit:
                break

        after = data.get("data", {}).get("after")
        if not after:
            break

        time.sleep(0.3)

    return posts


# ============ SCRAPER SELENIUM ============

def scrape_selenium(subreddit, limit=50):
    """Scrape avec Selenium (methode cours)"""

    if not SELENIUM_AVAILABLE:
        st.error("Selenium non installe. Utilise: pip install selenium beautifulsoup4 lxml")
        return []

    posts = []

    # Setup Chrome
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0")
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    try:
        driver = webdriver.Chrome(options=options)
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })
    except Exception as e:
        st.error(f"Erreur Chrome: {e}")
        return []

    try:
        url = f"https://old.reddit.com/r/{subreddit}/new/"
        driver.get(url)
        time.sleep(random.uniform(2, 4))

        pages = 0
        max_pages = (limit // 25) + 2

        while len(posts) < limit and pages < max_pages:
            # Scroll
            for _ in range(2):
                driver.execute_script(f"window.scrollBy(0, {random.randint(300, 600)});")
                time.sleep(random.uniform(0.5, 1.0))

            # Parse
            soup = BeautifulSoup(driver.page_source, "lxml")
            elements = soup.select("div.thing.link")

            for elem in elements:
                if "stickied" in elem.get("class", []):
                    continue
                if "promoted" in elem.get("class", []):
                    continue

                try:
                    title_el = elem.select_one("a.title")
                    title = title_el.get_text(strip=True) if title_el else ""

                    score_el = elem.select_one("div.score.unvoted")
                    score_txt = score_el.get_text(strip=True) if score_el else "0"
                    try:
                        score = int(score_txt) if score_txt != "•" else 0
                    except:
                        score = 0

                    comments_el = elem.select_one("a.comments")
                    comments_txt = comments_el.get_text(strip=True) if comments_el else "0"
                    match = re.search(r"(\d+)", comments_txt)
                    num_comments = int(match.group(1)) if match else 0

                    author_el = elem.select_one("a.author")
                    author = author_el.get_text(strip=True) if author_el else "[deleted]"

                    if title:
                        posts.append({
                            "id": elem.get("data-fullname", ""),
                            "title": title,
                            "text": "",
                            "score": score,
                            "num_comments": num_comments,
                            "author": author,
                        })
                except:
                    continue

                if len(posts) >= limit:
                    break

            # Next page
            try:
                next_btn = driver.find_element(By.CSS_SELECTOR, "span.next-button a")
                next_btn.click()
                time.sleep(random.uniform(2, 3))
                pages += 1
            except:
                break

    except Exception as e:
        st.error(f"Erreur scraping: {e}")
    finally:
        driver.quit()

    return posts[:limit]


# ============ SENTIMENT ============

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model


def analyze_sentiment(text, tokenizer, model):
    if not text or len(text) < 5:
        return {"score": 0.0, "label": "neutral"}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]

    pos, neg, neu = probs[0], probs[1], probs[2]
    score = float(pos - neg)

    if score > 0.05:
        label = "positive"
    elif score < -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {"score": round(score, 4), "label": label}


# ============ PRIX ============

@st.cache_data(ttl=300)
def get_prices(cryptos):
    try:
        ids = ",".join(cryptos)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true"
        resp = requests.get(url, timeout=10)
        return resp.json()
    except:
        return {}


# ============ PAGES ============

def page_single():
    """Analyse simple avec choix de methode"""
    st.title("Crypto Sentiment Analyzer")
    st.caption("Projet MoSEF 2024-2025 - Webscraping et NLP")

    # Sidebar
    with st.sidebar:
        st.header("Parametres")

        # Choix methode
        st.subheader("Methode de scraping")
        method = st.radio(
            "Choisir",
            ["HTTP/JSON (rapide)", "Selenium (cours)"],
            help="HTTP: appel API direct. Selenium: vrai webscraping avec navigateur."
        )

        if "Selenium" in method:
            if not SELENIUM_AVAILABLE:
                st.warning("Selenium non disponible")
            else:
                st.info("Selenium: plus lent mais conforme au cours")

        st.divider()

        crypto_name = st.selectbox("Crypto", list(CRYPTO_LIST.keys()))
        config = CRYPTO_LIST[crypto_name]

        # Limite differente selon methode
        if "HTTP" in method:
            limit = st.slider("Nombre de posts", 20, 1000, 50, 10)
        else:
            limit = st.slider("Nombre de posts", 20, 200, 50, 10)
            st.caption("Selenium limite a 200 posts (temps)")

        st.divider()
        st.caption(f"Subreddit: r/{config['sub']}")

        run = st.button("Analyser", type="primary", use_container_width=True)

    # Prix
    st.subheader("Prix en temps reel")
    prices = get_prices(["bitcoin", "ethereum", "solana"])

    c1, c2, c3 = st.columns(3)
    if "bitcoin" in prices:
        c1.metric("Bitcoin", f"${prices['bitcoin']['usd']:,.0f}", f"{prices['bitcoin'].get('usd_24h_change', 0):.1f}%")
    if "ethereum" in prices:
        c2.metric("Ethereum", f"${prices['ethereum']['usd']:,.0f}", f"{prices['ethereum'].get('usd_24h_change', 0):.1f}%")
    if "solana" in prices:
        c3.metric("Solana", f"${prices['solana']['usd']:,.0f}", f"{prices['solana'].get('usd_24h_change', 0):.1f}%")

    st.divider()

    # Info methodes
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Selenium (cours)**
        - Ouvre un vrai navigateur Chrome
        - Simule comportement humain
        - Parse le HTML avec BeautifulSoup
        - Plus lent mais conforme au cours
        """)
    with col2:
        st.markdown("""
        **HTTP/JSON (bonus)**
        - Appel direct API Reddit
        - Pas de navigateur
        - Reponse JSON structuree
        - Rapide mais pas du "vrai" scraping
        """)

    st.divider()

    if run:
        method_key = "selenium" if "Selenium" in method else "http"

        # Scraping
        with st.spinner(f"Scraping r/{config['sub']} avec {method_key.upper()}..."):
            start = time.time()

            if method_key == "selenium":
                posts = scrape_selenium(config['sub'], limit)
            else:
                posts = scrape_http(config['sub'], limit)

            scrape_time = time.time() - start

        if not posts:
            st.error("Aucun post recupere")
            return

        st.success(f"{len(posts)} posts recuperes en {scrape_time:.1f}s ({method_key.upper()})")

        # Sentiment
        with st.spinner("Analyse sentiment (FinBERT)..."):
            tokenizer, model = load_model()

            results = []
            progress = st.progress(0)

            for i, post in enumerate(posts):
                text = clean_text(post["title"] + " " + post.get("text", ""))
                if text and len(text) > 10:
                    sentiment = analyze_sentiment(text, tokenizer, model)
                    results.append(sentiment)
                else:
                    results.append({"score": 0, "label": "neutral"})

                progress.progress((i + 1) / len(posts))

        # Stats
        scores = [r["score"] for r in results]
        avg = sum(scores) / len(scores) if scores else 0

        labels = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            labels[r["label"]] += 1

        # Metriques
        st.subheader("Resultats")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Methode", method_key.upper())
        c2.metric("Temps", f"{scrape_time:.1f}s")
        c3.metric("Posts", len(results))
        c4.metric("Sentiment", f"{avg:+.3f}")
        c5.metric("Positifs", f"{labels['positive']}/{len(results)}")

        # Graphiques
        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                values=list(labels.values()),
                names=["Positif", "Negatif", "Neutre"],
                color_discrete_sequence=["#28a745", "#dc3545", "#6c757d"],
                title="Distribution des sentiments"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(x=scores, nbins=20, title="Distribution des scores")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=avg, line_color="red", annotation_text=f"Moy: {avg:.3f}")
            st.plotly_chart(fig, use_container_width=True)

        # Tableau
        st.subheader("Posts analyses")

        df = pd.DataFrame([
            {
                "Titre": posts[i]["title"][:70],
                "Score": round(results[i]["score"], 3),
                "Label": results[i]["label"],
                "Upvotes": posts[i].get("score", 0)
            }
            for i in range(len(results))
        ])

        tab1, tab2, tab3 = st.tabs(["Tous", "Positifs", "Negatifs"])

        with tab1:
            st.dataframe(df, use_container_width=True, height=400)
        with tab2:
            st.dataframe(df[df["Label"] == "positive"].sort_values("Score", ascending=False), use_container_width=True)
        with tab3:
            st.dataframe(df[df["Label"] == "negative"].sort_values("Score"), use_container_width=True)

        st.download_button("Telecharger CSV", df.to_csv(index=False), f"sentiment_{config['id']}_{method_key}.csv")


def page_compare():
    """Comparaison des methodes"""
    st.title("Comparaison des methodes")
    st.caption("Selenium (cours) vs HTTP/JSON (bonus)")

    with st.sidebar:
        crypto_name = st.selectbox("Crypto", list(CRYPTO_LIST.keys()))
        config = CRYPTO_LIST[crypto_name]

        limit = st.slider("Posts", 10, 100, 30, 10)

        run = st.button("Comparer", type="primary", use_container_width=True)

    # Tableau comparatif
    st.subheader("Differences")

    st.markdown("""
    | Aspect | Selenium | HTTP/JSON |
    |--------|----------|-----------|
    | Technique | Webscraping HTML | Appel API |
    | Navigateur | Oui (Chrome) | Non |
    | BeautifulSoup | Oui | Non |
    | Vitesse | Lent (~1s/post) | Rapide (~0.01s/post) |
    | Limite posts | ~200 | ~1000 |
    | Conforme cours | **OUI** | NON |
    """)

    st.divider()

    if run:
        if not SELENIUM_AVAILABLE:
            st.error("Selenium non disponible pour la comparaison")
            return

        col1, col2 = st.columns(2)

        # HTTP
        with col1:
            st.subheader("HTTP/JSON")
            with st.spinner("Scraping HTTP..."):
                start = time.time()
                http_posts = scrape_http(config['sub'], limit)
                http_time = time.time() - start

            st.metric("Temps", f"{http_time:.2f}s")
            st.metric("Posts", len(http_posts))
            if http_posts:
                st.write("Exemple:", http_posts[0]["title"][:50] + "...")

        # Selenium
        with col2:
            st.subheader("Selenium")
            with st.spinner("Scraping Selenium..."):
                start = time.time()
                sel_posts = scrape_selenium(config['sub'], limit)
                sel_time = time.time() - start

            st.metric("Temps", f"{sel_time:.2f}s")
            st.metric("Posts", len(sel_posts))
            if sel_posts:
                st.write("Exemple:", sel_posts[0]["title"][:50] + "...")

        # Resume
        st.divider()
        st.subheader("Resume")

        df = pd.DataFrame({
            "Methode": ["HTTP/JSON", "Selenium"],
            "Temps (s)": [round(http_time, 2), round(sel_time, 2)],
            "Posts": [len(http_posts), len(sel_posts)],
            "Conforme cours": ["Non", "Oui"]
        })
        st.dataframe(df, use_container_width=True)

        if http_time > 0:
            ratio = sel_time / http_time
            st.info(f"HTTP est **{ratio:.1f}x plus rapide** que Selenium")
            st.warning("Mais Selenium est la methode demandee par le cours!")


def page_multi():
    """Multi-crypto"""
    st.title("Analyse Multi-Crypto")
    st.caption("Analyse plusieurs cryptos en une fois")

    with st.sidebar:
        selected = st.multiselect(
            "Cryptos",
            list(CRYPTO_LIST.keys()),
            default=["Bitcoin (BTC)", "Ethereum (ETH)", "Solana (SOL)"]
        )

        method = st.radio("Methode", ["HTTP/JSON", "Selenium"])

        limit = st.slider("Posts par crypto", 20, 100, 30)

        run = st.button("Analyser tout", type="primary", use_container_width=True)

    if run and selected:
        tokenizer, model = load_model()
        all_results = []

        progress = st.progress(0)
        status = st.empty()

        for i, crypto_name in enumerate(selected):
            config = CRYPTO_LIST[crypto_name]
            status.text(f"Scraping {crypto_name}...")

            if "HTTP" in method:
                posts = scrape_http(config['sub'], limit)
            else:
                posts = scrape_selenium(config['sub'], limit)

            if posts:
                status.text(f"Analyse {crypto_name}...")
                scores = []
                labs = {"positive": 0, "negative": 0, "neutral": 0}

                for post in posts:
                    text = clean_text(post["title"] + " " + post.get("text", ""))
                    if text and len(text) > 10:
                        sent = analyze_sentiment(text, tokenizer, model)
                        scores.append(sent["score"])
                        labs[sent["label"]] += 1

                avg = sum(scores) / len(scores) if scores else 0

                all_results.append({
                    "Crypto": crypto_name,
                    "Posts": len(scores),
                    "Sentiment": round(avg, 4),
                    "Positifs": labs["positive"],
                    "Negatifs": labs["negative"],
                    "Neutres": labs["neutral"]
                })

            progress.progress((i + 1) / len(selected))

        status.text("Termine!")

        # Resultats
        st.subheader("Resultats")

        df = pd.DataFrame(all_results)
        st.dataframe(df, use_container_width=True)

        # Graphique
        fig = px.bar(
            df, x="Crypto", y="Sentiment", color="Sentiment",
            color_continuous_scale=["red", "gray", "green"],
            title="Sentiment par crypto"
        )
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("Telecharger CSV", df.to_csv(index=False), "multi_crypto_sentiment.csv")


def page_about():
    """Methodologie"""
    st.title("Methodologie")

    st.markdown("""
    ## Objectif
    
    Analyser le sentiment des discussions Reddit sur les cryptomonnaies 
    et reproduire les methodologies des papiers academiques.
    
    ## Deux methodes de scraping
    
    ### Selenium (conforme au cours)
```python
    driver = webdriver.Chrome()
    driver.get("https://old.reddit.com/r/Bitcoin")
    soup = BeautifulSoup(driver.page_source, "lxml")
    posts = soup.select("div.thing.link")
```
    
    ### HTTP/JSON (bonus)
```python
    resp = requests.get("https://old.reddit.com/r/Bitcoin/new.json")
    data = resp.json()
```
    
    ## Pipeline NLP
    
    1. Nettoyage (URLs, mentions, caracteres speciaux)
    2. FinBERT (modele specialise finance)
    3. Score: P(positif) - P(negatif)
    
    ## References
    
    - Kraaijeveld & De Smedt (2020) - Twitter sentiment for crypto
    - ProsusAI/FinBERT - Financial sentiment model
    """)


# ============ MAIN ============

def main():
    page = st.sidebar.radio(
        "Navigation",
        ["Analyse", "Comparaison", "Multi-crypto", "Methodologie"]
    )

    if page == "Analyse":
        page_single()
    elif page == "Comparaison":
        page_compare()
    elif page == "Multi-crypto":
        page_multi()
    else:
        page_about()


if __name__ == "__main__":
    main()