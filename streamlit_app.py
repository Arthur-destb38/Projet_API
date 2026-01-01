"""
Crypto Sentiment - Streamlit
Projet MoSEF 2024-2025
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.scrapers import HttpScraper, SeleniumScraper
from app.sentiment import SentimentAnalyzer
from app.prices import CryptoPrices
from app.utils import clean_text

st.set_page_config(page_title="Crypto Sentiment", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .method-box { padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    .selenium-box { background: #e3f2fd; border-left: 4px solid #1976d2; }
    .http-box { background: #e8f5e9; border-left: 4px solid #388e3c; }
</style>
""", unsafe_allow_html=True)

CRYPTOS = {
    "Bitcoin (BTC)": {"crypto": "bitcoin", "subreddit": "Bitcoin"},
    "Ethereum (ETH)": {"crypto": "ethereum", "subreddit": "ethereum"},
    "Solana (SOL)": {"crypto": "solana", "subreddit": "solana"},
    "Dogecoin (DOGE)": {"crypto": "dogecoin", "subreddit": "dogecoin"},
    "Cardano (ADA)": {"crypto": "cardano", "subreddit": "cardano"},
}


@st.cache_resource
def load_model():
    return SentimentAnalyzer()


@st.cache_data(ttl=300)
def get_prices():
    client = CryptoPrices()
    return client.get_multiple_prices(["bitcoin", "ethereum", "solana", "dogecoin"])


def run_analysis(method, subreddit, crypto, limit):
    start = time.time()

    if method == "selenium":
        scraper = SeleniumScraper(headless=True)
        posts = scraper.scrape_subreddit(subreddit, crypto, limit)
        scraper.close()
    else:
        scraper = HttpScraper()
        posts = scraper.scrape_subreddit(subreddit, limit=limit)
        scraper.close()

    scrape_time = time.time() - start

    analyzer = load_model()
    texts = [clean_text(p["title"] + " " + p.get("text", "")) for p in posts]
    texts = [t for t in texts if t and len(t) > 10]

    start = time.time()
    results = analyzer.analyze_batch(texts)
    sentiment_time = time.time() - start

    return posts, results, scrape_time, sentiment_time


def page_home():
    st.title("Crypto Sentiment Analyzer")
    st.caption("Projet MoSEF 2024-2025 - Webscraping & NLP")

    # Prix
    st.subheader("Prix en temps reel")
    prices = get_prices()

    cols = st.columns(4)
    for i, (name, data) in enumerate(prices.items()):
        with cols[i]:
            st.metric(name.capitalize(), f"${data['price']:,.0f}", f"{data['change_24h']:+.2f}%")

    st.divider()

    # Methodes
    st.subheader("Methode de scraping")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="method-box selenium-box">
        <strong>Selenium (methode cours)</strong><br>
        Navigateur, simulation humaine, parse HTML
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="method-box http-box">
        <strong>HTTP/JSON (bonus)</strong><br>
        Appel API direct, plus rapide
        </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Parametres")

        method = st.radio("Methode", ["Selenium (cours)", "HTTP/JSON (bonus)"])
        method_key = "selenium" if "Selenium" in method else "http"

        selected = st.selectbox("Crypto", list(CRYPTOS.keys()))
        config = CRYPTOS[selected]

        limit = st.slider("Posts", 20, 100, 30, 10)
        run = st.button("Lancer l'analyse", type="primary", use_container_width=True)

    if run:
        with st.spinner(f"Scraping avec {method_key}..."):
            posts, results, scrape_time, sentiment_time = run_analysis(
                method_key, config["subreddit"], config["crypto"], limit
            )

        if not posts:
            st.error("Erreur scraping")
            return

        st.subheader("Resultats")

        scores = [r["score"] for r in results]
        avg = sum(scores) / len(scores) if scores else 0

        labels = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            labels[r["label"]] += 1

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Methode", method_key.upper())
        with col2:
            st.metric("Temps", f"{scrape_time:.1f}s")
        with col3:
            st.metric("Posts", len(results))
        with col4:
            st.metric("Sentiment", f"{avg:+.3f}")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                values=list(labels.values()),
                names=["Positif", "Negatif", "Neutre"],
                color_discrete_sequence=["#28a745", "#dc3545", "#6c757d"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(x=scores, nbins=15)
            fig.add_vline(x=avg, line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame([
            {"Titre": posts[i]["title"][:60], "Score": round(results[i]["score"], 3), "Label": results[i]["label"]}
            for i in range(min(len(posts), len(results)))
        ])
        st.dataframe(df, use_container_width=True)


def page_compare():
    st.title("Comparaison des methodes")

    with st.sidebar:
        subreddit = st.selectbox("Subreddit", ["Bitcoin", "ethereum", "CryptoCurrency"])
        limit = st.slider("Posts", 10, 50, 20)
        run = st.button("Comparer", type="primary", use_container_width=True)

    if run:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("HTTP/JSON")
            with st.spinner("..."):
                start = time.time()
                http = HttpScraper()
                http_posts = http.scrape_subreddit(subreddit, limit=limit)
                http.close()
                http_time = time.time() - start
            st.metric("Temps", f"{http_time:.2f}s")
            st.metric("Posts", len(http_posts))

        with col2:
            st.subheader("Selenium")
            with st.spinner("..."):
                start = time.time()
                sel = SeleniumScraper(headless=True)
                sel_posts = sel.scrape_subreddit(subreddit, limit=limit)
                sel.close()
                sel_time = time.time() - start
            st.metric("Temps", f"{sel_time:.2f}s")
            st.metric("Posts", len(sel_posts))

        st.divider()
        if http_time > 0:
            st.info(f"HTTP est {sel_time / http_time:.1f}x plus rapide")


def page_method():
    st.title("Methodologie")

    st.markdown("""
    ### Ce que le prof demande
    > "Pour le webscraping vous devez utiliser **Selenium** et **simuler un comportement humain**."

    ### Differences
    | Aspect | Selenium | HTTP/JSON |
    |--------|----------|-----------|
    | Navigateur | Oui | Non |
    | Parsing HTML | Oui | Non |
    | Vitesse | Lent | Rapide |
    | Cours | Oui | Non |
    """)


def main():
    page = st.sidebar.radio("Navigation", ["Analyse", "Comparaison", "Methodo"], label_visibility="collapsed")

    if page == "Analyse":
        page_home()
    elif page == "Comparaison":
        page_compare()
    else:
        page_method()


if __name__ == "__main__":
    main()