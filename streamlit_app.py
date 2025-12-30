"""
Crypto Sentiment Analyzer - Interface Streamlit
Master 2 MoSEF Data Science 2024-2025
Projet Webscraping & NLP
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Ajouter le dossier app au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.scraper import RedditScraper
from app.sentiment import SentimentAnalyzer
from app.prices import CryptoPrices
from app.utils import clean_text

# Configuration de la page
st.set_page_config(
    page_title="Crypto Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# CSS custom
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .sentiment-positive { color: #28a745; }
    .sentiment-negative { color: #dc3545; }
    .sentiment-neutral { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# Liste des cryptos disponibles
CRYPTO_OPTIONS = {
    "Bitcoin (BTC)": {"crypto": "bitcoin", "subreddit": "Bitcoin"},
    "Ethereum (ETH)": {"crypto": "ethereum", "subreddit": "ethereum"},
    "Solana (SOL)": {"crypto": "solana", "subreddit": "solana"},
    "Cardano (ADA)": {"crypto": "cardano", "subreddit": "cardano"},
    "Ripple (XRP)": {"crypto": "ripple", "subreddit": "xrp"},
    "Dogecoin (DOGE)": {"crypto": "dogecoin", "subreddit": "dogecoin"},
    "Polkadot (DOT)": {"crypto": "polkadot", "subreddit": "polkadot"},
    "Avalanche (AVAX)": {"crypto": "avalanche", "subreddit": "avax"},
    "Polygon (MATIC)": {"crypto": "polygon", "subreddit": "matic"},
    "Chainlink (LINK)": {"crypto": "chainlink", "subreddit": "chainlink"},
    "Litecoin (LTC)": {"crypto": "litecoin", "subreddit": "litecoin"},
    "Cosmos (ATOM)": {"crypto": "cosmos", "subreddit": "atom"},
}


@st.cache_resource
def load_sentiment_model():
    """Charge le modele FinBERT (cache pour eviter de recharger)"""
    return SentimentAnalyzer()


@st.cache_data(ttl=300)
def get_crypto_prices(cryptos):
    """Recupere les prix des cryptos"""
    prices_client = CryptoPrices()
    return prices_client.get_multiple_prices(cryptos)


def get_sentiment_label(score):
    """Retourne le label selon le score"""
    if score > 0.05:
        return "Positif", "sentiment-positive"
    elif score < -0.05:
        return "Negatif", "sentiment-negative"
    else:
        return "Neutre", "sentiment-neutral"


def main():
    # Header
    st.markdown('<p class="main-header">Crypto Sentiment Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyse de sentiment Reddit avec FinBERT - Projet MoSEF 2024-2025</p>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Parametres")

        selected_crypto = st.selectbox(
            "Cryptomonnaie",
            options=list(CRYPTO_OPTIONS.keys()),
            index=0
        )

        crypto_config = CRYPTO_OPTIONS[selected_crypto]

        num_posts = st.slider(
            "Nombre de posts",
            min_value=20,
            max_value=200,
            value=50,
            step=10
        )

        analyze_button = st.button("Lancer l'analyse", type="primary", use_container_width=True)

        st.divider()
        st.caption("Technologies: Selenium, FinBERT, CoinGecko")

    # Prix en temps reel
    st.subheader("Prix en temps reel")

    col1, col2, col3, col4 = st.columns(4)
    prices = get_crypto_prices(["bitcoin", "ethereum", "solana", "dogecoin"])

    with col1:
        if "bitcoin" in prices:
            st.metric("Bitcoin", f"${prices['bitcoin']['price']:,.0f}", f"{prices['bitcoin']['change_24h']:+.2f}%")

    with col2:
        if "ethereum" in prices:
            st.metric("Ethereum", f"${prices['ethereum']['price']:,.0f}", f"{prices['ethereum']['change_24h']:+.2f}%")

    with col3:
        if "solana" in prices:
            st.metric("Solana", f"${prices['solana']['price']:,.0f}", f"{prices['solana']['change_24h']:+.2f}%")

    with col4:
        if "dogecoin" in prices:
            st.metric("Dogecoin", f"${prices['dogecoin']['price']:,.4f}", f"{prices['dogecoin']['change_24h']:+.2f}%")

    st.divider()

    # Analyse
    if analyze_button:
        with st.spinner(f"Scraping r/{crypto_config['subreddit']}..."):
            scraper = RedditScraper(headless=True)
            posts = scraper.scrape_subreddit(
                subreddit=crypto_config['subreddit'],
                crypto=crypto_config['crypto'],
                limit=num_posts
            )
            scraper.close()

        if not posts:
            st.error("Erreur lors du scraping.")
            return

        st.success(f"{len(posts)} posts recuperes")

        with st.spinner("Analyse de sentiment..."):
            analyzer = load_sentiment_model()
            texts = [clean_text(p["title"] + " " + p.get("text", "")) for p in posts]
            texts = [t for t in texts if t and len(t) > 10]
            results = analyzer.analyze_batch(texts)

        # Metriques
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores)

        label_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            label_counts[r["label"]] += 1

        sentiment_label, sentiment_class = get_sentiment_label(avg_score)

        # Resultats
        st.subheader(f"Resultats - {selected_crypto}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Sentiment moyen", f"{avg_score:+.3f}")
            st.markdown(
                f'<p class="{sentiment_class}" style="font-size: 1.5rem; font-weight: bold;">{sentiment_label}</p>',
                unsafe_allow_html=True)

        with col2:
            crypto_price = get_crypto_prices([crypto_config['crypto']])
            if crypto_config['crypto'] in crypto_price:
                p = crypto_price[crypto_config['crypto']]
                st.metric("Prix actuel", f"${p['price']:,.2f}", f"{p['change_24h']:+.2f}%")

        with col3:
            st.metric("Posts analyses", len(results))
            st.caption(
                f"Positifs: {label_counts['positive']} | Negatifs: {label_counts['negative']} | Neutres: {label_counts['neutral']}")

        # Graphiques
        col1, col2 = st.columns(2)

        with col1:
            fig_pie = px.pie(
                values=[label_counts['positive'], label_counts['negative'], label_counts['neutral']],
                names=['Positif', 'Negatif', 'Neutre'],
                color_discrete_sequence=['#28a745', '#dc3545', '#6c757d'],
                title="Distribution des sentiments"
            )
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_hist = px.histogram(
                x=scores,
                nbins=20,
                title="Distribution des scores",
                labels={'x': 'Score', 'y': 'Nombre'},
                color_discrete_sequence=['#007bff']
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_hist.add_vline(x=avg_score, line_dash="solid", line_color="red")
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Tableau
        st.subheader("Details des posts")

        df_posts = pd.DataFrame([
            {
                "Titre": posts[i]["title"][:80] + "..." if len(posts[i]["title"]) > 80 else posts[i]["title"],
                "Score": round(results[i]["score"], 3),
                "Label": results[i]["label"],
                "Upvotes": posts[i]["score"],
                "Commentaires": posts[i]["num_comments"]
            }
            for i in range(min(len(posts), len(results)))
        ])

        tab1, tab2, tab3 = st.tabs(["Tous", "Top positifs", "Top negatifs"])

        with tab1:
            st.dataframe(df_posts, use_container_width=True, height=400)

        with tab2:
            df_pos = df_posts[df_posts["Label"] == "positive"].sort_values("Score", ascending=False).head(10)
            st.dataframe(df_pos, use_container_width=True)

        with tab3:
            df_neg = df_posts[df_posts["Label"] == "negative"].sort_values("Score").head(10)
            st.dataframe(df_neg, use_container_width=True)

        # Export
        st.download_button(
            label="Telecharger CSV",
            data=df_posts.to_csv(index=False).encode('utf-8'),
            file_name=f"sentiment_{crypto_config['crypto']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    else:
        st.info("Selectionnez une crypto et cliquez sur 'Lancer l'analyse'")

        with st.expander("A propos du projet"):
            st.markdown("""
            **Pipeline:**
            1. Webscraping Reddit avec Selenium
            2. Nettoyage du texte (preprocessing)
            3. Analyse de sentiment avec FinBERT
            4. Visualisation des resultats

            **Technologies:** Python, FastAPI, Streamlit, Selenium, Transformers, CoinGecko API
            """)


if __name__ == "__main__":
    main()