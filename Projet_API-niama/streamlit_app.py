"""
Crypto Sentiment Analyzer - Interface Streamlit
Master 2 MoSEF Data Science 2024-2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.scraper import RedditScraper
from app.sentiment import SentimentAnalyzer
from app.prices import CryptoPrices
from app.utils import clean_text
from app.econometrics import EconometricAnalyzer


st.set_page_config(page_title="Crypto Sentiment", page_icon="📊", layout="wide")

CRYPTO_OPTIONS = {
    "Bitcoin (BTC)": {"crypto": "bitcoin", "subreddit": "Bitcoin"},
    "Ethereum (ETH)": {"crypto": "ethereum", "subreddit": "ethereum"},
    "Solana (SOL)": {"crypto": "solana", "subreddit": "solana"},
    "Cardano (ADA)": {"crypto": "cardano", "subreddit": "cardano"},
    "Dogecoin (DOGE)": {"crypto": "dogecoin", "subreddit": "dogecoin"},
}

@st.cache_resource
def load_sentiment_model():
    return SentimentAnalyzer()

@st.cache_data(ttl=300)
def get_crypto_prices(cryptos):
    return CryptoPrices().get_multiple_prices(cryptos)


def page_sentiment():
    st.title("Analyse de Sentiment")

    with st.sidebar:
        selected = st.selectbox("Crypto", list(CRYPTO_OPTIONS.keys()))
        config = CRYPTO_OPTIONS[selected]
        num_posts = st.slider("Posts", 20, 200, 50)
        run = st.button("Analyser", type="primary")

    if run:
        with st.spinner("Scraping..."):
            scraper = RedditScraper(headless=True)
            posts = scraper.scrape_subreddit(config['subreddit'], config['crypto'], num_posts)
            scraper.close()

        with st.spinner("Analyse NLP..."):
            analyzer = load_sentiment_model()
            texts = [clean_text(p["title"]) for p in posts]
            results = analyzer.analyze_batch([t for t in texts if t])

        scores = [r["score"] for r in results]
        avg = sum(scores)/len(scores)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment moyen", f"{avg:+.3f}")
        with col2:
            st.metric("Posts", len(results))

        fig = px.histogram(x=scores, nbins=20, title="Distribution")
        st.plotly_chart(fig)

        st.session_state['posts'] = posts
        st.session_state['sentiments'] = results
        st.session_state['crypto'] = config['crypto']


def page_econometrics():
    st.title("Analyse Econometrique")

    if 'posts' not in st.session_state:
        st.warning("Lancez d'abord l'analyse sentiment")
        return

    posts = st.session_state['posts']
    sentiments = st.session_state['sentiments']
    crypto = st.session_state['crypto']

    days = st.slider("Jours historique", 7, 60, 30)
    maxlag = st.slider("Lag max", 3, 14, 7)

    if st.button("Lancer econometrie", type="primary"):
        prices = CryptoPrices().get_historical_prices(crypto, days)

        econ = EconometricAnalyzer()
        results = econ.full_analysis(posts, sentiments, prices, maxlag)

        if 'error' in results:
            st.error(results['error'])
            return

        # ADF
        st.subheader("Test ADF (stationnarite)")
        adf = results.get('stationarity_tests', {})
        for k, v in adf.items():
            st.write(f"{v['series']}: p={v['p_value']:.4f} → {'Stationnaire' if v['is_stationary'] else 'Non-stat'}")

        # Granger
        st.subheader("Causalite de Granger")
        granger = results.get('granger_causality', {}).get('summary', {})
        if granger.get('sentiment_granger_causes_returns'):
            st.success("Sentiment PREDIT les returns")
        else:
            st.error("Pas de relation predictive")

        # Cross-corr
        st.subheader("Correlation croisee")
        cc = results.get('cross_correlation', {})
        if cc:
            corrs = cc.get('correlations', {})
            lags = list(range(-maxlag, maxlag+1))
            vals = [corrs.get(f'lag_{l}', 0) for l in lags]
            fig = go.Figure(go.Bar(x=lags, y=vals))
            st.plotly_chart(fig)

        # Conclusion
        st.subheader("Conclusion")
        concl = results.get('conclusion', {})
        st.write(concl.get('summary', ''))


def page_method():
    st.title("Methodologie")
    st.markdown("""
    ### Papier vs Notre projet
    | Aspect | Papier | Nous |
    |--------|--------|------|
    | Source | Twitter | Reddit |
    | NLP | VADER/BERT | FinBERT |
    | Stats | VAR, Granger, GARCH | VAR, Granger |
    """)


def main():
    page = st.sidebar.radio("Page", ["Sentiment", "Econometrie", "Methodo"])
    if page == "Sentiment":
        page_sentiment()
    elif page == "Econometrie":
        page_econometrics()
    else:
        page_method()

if __name__ == "__main__":
    main()