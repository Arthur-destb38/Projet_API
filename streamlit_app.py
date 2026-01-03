"""
Crypto Sentiment - Streamlit
Projet MoSEF 2024-2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import re
import random
from datetime import datetime
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Selenium
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from bs4 import BeautifulSoup
    SELENIUM_OK = True
except ImportError:
    SELENIUM_OK = False

# Econometrie
try:
    from econometrics import run_full_analysis
    ECONO_OK = True
except ImportError:
    ECONO_OK = False


# ============ CONFIG ============

st.set_page_config(page_title="Crypto Sentiment", layout="wide")

# Limites
LIMIT_HTTP = 1000
LIMIT_SELENIUM = 200
MIN_DAYS_ECONO = 10

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


def get_top_words(texts, n=20):
    """Mots les plus frequents"""
    all_words = []
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'to', 'of', 'in', 'for', 'on', 'with',
                  'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after',
                  'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                  'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'can',
                  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
                  'this', 'that', 'these', 'those', 'am', 'if', 'my', 'your', 'his', 'her',
                  'its', 'our', 'their', 'me', 'him', 'us', 'them', 'get', 'got', 'like',
                  'dont', 'im', 'ive', 'cant', 'wont', 'about', 'up', 'out', 'all', 'just'}

    for text in texts:
        words = text.lower().split()
        words = [w for w in words if len(w) > 2 and w not in stop_words and w.isalpha()]
        all_words.extend(words)

    return Counter(all_words).most_common(n)


# ============ SCRAPER HTTP ============

def scrape_http(subreddit, limit=50):
    posts = []
    after = None
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}

    limit = min(limit, LIMIT_HTTP)

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
                "created_utc": d.get("created_utc"),
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
    if not SELENIUM_OK:
        st.error("Selenium non installe")
        return []

    limit = min(limit, LIMIT_SELENIUM)
    posts = []

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    try:
        driver = webdriver.Chrome(options=options)
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
            for _ in range(2):
                driver.execute_script(f"window.scrollBy(0, {random.randint(300, 600)});")
                time.sleep(random.uniform(0.5, 1.0))

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

                    time_el = elem.select_one("time")
                    timestamp = None
                    if time_el and time_el.get("datetime"):
                        try:
                            dt = datetime.fromisoformat(time_el.get("datetime").replace("Z", "+00:00"))
                            timestamp = dt.timestamp()
                        except:
                            pass

                    if title:
                        posts.append({
                            "id": elem.get("data-fullname", ""),
                            "title": title,
                            "text": "",
                            "score": score,
                            "num_comments": 0,
                            "author": "",
                            "created_utc": timestamp,
                        })
                except:
                    continue

                if len(posts) >= limit:
                    break

            try:
                next_btn = driver.find_element(By.CSS_SELECTOR, "span.next-button a")
                next_btn.click()
                time.sleep(random.uniform(2, 3))
                pages += 1
            except:
                break

    except Exception as e:
        st.error(f"Erreur: {e}")
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

def page_analyse():
    st.title("Crypto Sentiment Analyzer")
    st.caption("Projet MoSEF 2024-2025")

    # Sidebar
    with st.sidebar:
        st.header("Parametres")

        method = st.radio("Methode", ["HTTP/JSON", "Selenium"])

        # Affiche la limite clairement
        if method == "HTTP/JSON":
            max_limit = LIMIT_HTTP
            st.info(f"Limite HTTP: {LIMIT_HTTP} posts max")
        else:
            max_limit = LIMIT_SELENIUM
            st.warning(f"Limite Selenium: {LIMIT_SELENIUM} posts max (evite ban)")

        crypto_name = st.selectbox("Crypto", list(CRYPTO_LIST.keys()))
        config = CRYPTO_LIST[crypto_name]

        limit = st.slider("Nombre de posts", 20, max_limit, min(100, max_limit), 10)

        st.divider()
        st.caption(f"Subreddit: r/{config['sub']}")
        st.caption(f"Min econometrie: {MIN_DAYS_ECONO} jours")

        run = st.button("Analyser", type="primary", use_container_width=True)

    # Prix
    st.subheader("Prix")
    prices = get_prices(["bitcoin", "ethereum", "solana"])

    c1, c2, c3 = st.columns(3)
    if "bitcoin" in prices:
        c1.metric("Bitcoin", f"${prices['bitcoin']['usd']:,.0f}", f"{prices['bitcoin'].get('usd_24h_change', 0):.1f}%")
    if "ethereum" in prices:
        c2.metric("Ethereum", f"${prices['ethereum']['usd']:,.0f}", f"{prices['ethereum'].get('usd_24h_change', 0):.1f}%")
    if "solana" in prices:
        c3.metric("Solana", f"${prices['solana']['usd']:,.0f}", f"{prices['solana'].get('usd_24h_change', 0):.1f}%")

    st.divider()

    if run:
        method_key = "selenium" if method == "Selenium" else "http"

        # Scraping
        with st.spinner(f"Scraping r/{config['sub']}..."):
            start = time.time()

            if method_key == "selenium":
                posts = scrape_selenium(config['sub'], limit)
            else:
                posts = scrape_http(config['sub'], limit)

            scrape_time = time.time() - start

        if not posts:
            st.error("Aucun post")
            return

        st.success(f"{len(posts)} posts en {scrape_time:.1f}s ({method_key})")

        # Sentiment
        with st.spinner("Analyse FinBERT..."):
            tokenizer, model = load_model()

            results = []
            progress = st.progress(0)

            for i, post in enumerate(posts):
                text = clean_text(post["title"] + " " + post.get("text", ""))
                if text and len(text) > 10:
                    sent = analyze_sentiment(text, tokenizer, model)
                    results.append(sent)
                else:
                    results.append({"score": 0, "label": "neutral"})

                progress.progress((i + 1) / len(posts))

        # Sauvegarde pour econometrie
        st.session_state['last_posts'] = posts
        st.session_state['last_results'] = results
        st.session_state['last_crypto_id'] = config['id']
        st.session_state['last_crypto_name'] = crypto_name

        # Stats
        scores = [r["score"] for r in results]
        avg = sum(scores) / len(scores) if scores else 0
        std = np.std(scores) if scores else 0

        labels = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            labels[r["label"]] += 1

        # Metriques
        st.subheader("Resultats")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Posts", len(results))
        c2.metric("Sentiment moyen", f"{avg:+.3f}")
        c3.metric("Ecart-type", f"{std:.3f}")
        c4.metric("Positifs", labels['positive'])
        c5.metric("Negatifs", labels['negative'])

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
            fig = px.histogram(x=scores, nbins=30, title="Distribution des scores")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=avg, line_color="red", annotation_text=f"Moy={avg:.3f}")
            st.plotly_chart(fig, use_container_width=True)

        # Top mots
        st.subheader("Mots frequents")
        texts = [clean_text(p["title"]) for p in posts]
        top_words = get_top_words(texts, 15)

        if top_words:
            fig = px.bar(
                x=[w[1] for w in top_words],
                y=[w[0] for w in top_words],
                orientation='h',
                title="Top 15 mots"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        # Tableau
        st.subheader("Posts")

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
            st.dataframe(df, use_container_width=True, height=300)
        with tab2:
            st.dataframe(df[df["Label"] == "positive"].sort_values("Score", ascending=False), use_container_width=True)
        with tab3:
            st.dataframe(df[df["Label"] == "negative"].sort_values("Score"), use_container_width=True)

        st.download_button("Telecharger CSV", df.to_csv(index=False), f"sentiment_{config['id']}.csv")


def page_econometrie():
    st.title("Analyse Econometrique")

    if not ECONO_OK:
        st.error("Module econometrics.py non trouve")
        st.code("pip install statsmodels scipy")
        return

    if 'last_posts' not in st.session_state:
        st.warning("Lance d'abord une analyse dans 'Analyse'")
        return

    posts = st.session_state['last_posts']
    results = st.session_state['last_results']
    crypto_id = st.session_state.get('last_crypto_id', 'bitcoin')
    crypto_name = st.session_state.get('last_crypto_name', 'Bitcoin')

    # Compte les jours uniques
    dates = set()
    for p in posts:
        ts = p.get("created_utc")
        if ts:
            try:
                dt = datetime.fromtimestamp(ts)
                dates.add(dt.strftime("%Y-%m-%d"))
            except:
                pass

    n_days = len(dates)

    st.info(f"Donnees: {len(posts)} posts sur {n_days} jours ({crypto_name})")

    if n_days < MIN_DAYS_ECONO:
        st.error(f"Minimum {MIN_DAYS_ECONO} jours requis. Tu as {n_days} jours.")
        st.warning("Augmente le nombre de posts ou change de crypto.")
        return

    with st.sidebar:
        st.header("Parametres")
        days = st.slider("Jours historiques (prix)", 30, 90, 60)
        max_lag = st.slider("Lag max", 3, 10, 5)
        run = st.button("Lancer tests", type="primary", use_container_width=True)

    if run:
        with st.spinner("Tests en cours..."):
            output = run_full_analysis(posts, results, crypto_id, days, max_lag)

        if output["status"] == "error":
            st.error(f"Erreur: {output.get('error')}")
            return

        # ===== 1. STATS DESCRIPTIVES =====
        st.subheader("1. Statistiques descriptives")

        info = output["data_info"]
        merged = output.get("merged_data")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Jours sentiment", info["jours_sentiment"])
        c2.metric("Jours prix", info["jours_prix"])
        c3.metric("Jours fusionnes", info["jours_merged"])
        c4.metric("Debut", info['date_debut'])
        c5.metric("Fin", info['date_fin'])

        # Dates sur une ligne separee
        st.markdown(f"**Periode:** {info['date_debut']} → {info['date_fin']}")

        if merged is not None and not merged.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Sentiment**")
                sent_stats = merged['sentiment_mean'].describe()
                st.dataframe(pd.DataFrame({
                    "Stat": ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
                    "Valeur": [round(sent_stats['mean'], 4), round(sent_stats['std'], 4),
                              round(sent_stats['min'], 4), round(sent_stats['25%'], 4),
                              round(sent_stats['50%'], 4), round(sent_stats['75%'], 4),
                              round(sent_stats['max'], 4)]
                }), hide_index=True)

            with col2:
                st.markdown("**Log Returns**")
                ret_stats = merged['log_return'].describe()
                st.dataframe(pd.DataFrame({
                    "Stat": ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
                    "Valeur": [round(ret_stats['mean'], 4), round(ret_stats['std'], 4),
                              round(ret_stats['min'], 4), round(ret_stats['25%'], 4),
                              round(ret_stats['50%'], 4), round(ret_stats['75%'], 4),
                              round(ret_stats['max'], 4)]
                }), hide_index=True)

        st.divider()

        # ===== 2. VISUALISATION =====
        st.subheader("2. Series temporelles")

        if merged is not None and not merged.empty:
            # Sentiment over time
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged['date'], y=merged['sentiment_mean'],
                name="Sentiment", line=dict(color='blue')
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(title="Sentiment journalier", xaxis_title="Date", yaxis_title="Sentiment")
            st.plotly_chart(fig, use_container_width=True)

            # Returns over time
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=merged['date'], y=merged['log_return'],
                name="Return", line=dict(color='green')
            ))
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.update_layout(title="Log returns journaliers", xaxis_title="Date", yaxis_title="Log Return")
            st.plotly_chart(fig2, use_container_width=True)

            # Dual axis
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=merged['date'], y=merged['sentiment_mean'], name="Sentiment", yaxis="y1"))
            fig3.add_trace(go.Scatter(x=merged['date'], y=merged['log_return'], name="Return", yaxis="y2"))
            fig3.update_layout(
                title="Sentiment vs Returns",
                yaxis=dict(title="Sentiment", side="left", color="blue"),
                yaxis2=dict(title="Return", side="right", overlaying="y", color="green")
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        # ===== 3. CORRELATION =====
        st.subheader("3. Correlation")

        if merged is not None and not merged.empty:
            pearson = merged['sentiment_mean'].corr(merged['log_return'])
            spearman = merged['sentiment_mean'].corr(merged['log_return'], method='spearman')

            c1, c2 = st.columns(2)
            c1.metric("Pearson", f"{pearson:.4f}")
            c2.metric("Spearman", f"{spearman:.4f}")

            fig = px.scatter(
                merged, x='sentiment_mean', y='log_return',
                trendline="ols",
                title="Scatter Sentiment vs Returns"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ===== 4. ADF =====
        st.subheader("4. Stationnarite (ADF)")

        adf = output["adf_tests"]
        c1, c2 = st.columns(2)

        with c1:
            s = adf.get("sentiment", {})
            if "error" not in s:
                stationary = s.get("stationary")
                st.markdown(f"**Sentiment:** p = {s.get('pvalue')}")
                if stationary:
                    st.success("Stationnaire")
                else:
                    st.warning("Non stationnaire")

        with c2:
            r = adf.get("returns", {})
            if "error" not in r:
                stationary = r.get("stationary")
                st.markdown(f"**Returns:** p = {r.get('pvalue')}")
                if stationary:
                    st.success("Stationnaire")
                else:
                    st.warning("Non stationnaire")

        st.divider()

        # ===== 5. GRANGER =====
        st.subheader("5. Causalite de Granger")

        granger = output["granger"]

        if "error" not in granger:
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Sentiment → Returns**")
                s2r = granger.get("sentiment_to_returns", {})
                if s2r.get("significant"):
                    st.success(f"Significatif (lag={s2r.get('best_lag')})")
                else:
                    st.warning("Non significatif")

                if s2r.get("pvalues"):
                    df_p = pd.DataFrame({"Lag": list(s2r["pvalues"].keys()), "P-value": list(s2r["pvalues"].values())})
                    st.dataframe(df_p, hide_index=True)

            with c2:
                st.markdown("**Returns → Sentiment**")
                r2s = granger.get("returns_to_sentiment", {})
                if r2s.get("significant"):
                    st.success(f"Significatif (lag={r2s.get('best_lag')})")
                else:
                    st.warning("Non significatif")

                if r2s.get("pvalues"):
                    df_p = pd.DataFrame({"Lag": list(r2s["pvalues"].keys()), "P-value": list(r2s["pvalues"].values())})
                    st.dataframe(df_p, hide_index=True)
        else:
            st.error(granger.get("error"))

        st.divider()

        # ===== 6. VAR =====
        st.subheader("6. VAR")

        var = output["var"]
        if "error" not in var:
            c1, c2, c3 = st.columns(3)
            c1.metric("Lag optimal (AIC)", var.get("optimal_lag"))
            c2.metric("AIC", var.get("aic"))
            c3.metric("BIC", var.get("bic"))
        else:
            st.error(var.get("error"))

        st.divider()

        # ===== 7. CROSS-CORRELATION =====
        st.subheader("7. Cross-correlation")

        cross = output["cross_corr"]

        if cross.get("correlations"):
            corrs = cross["correlations"]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(corrs.keys()),
                y=list(corrs.values()),
                marker_color=['#dc3545' if v < 0 else '#28a745' for v in corrs.values()]
            ))
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(
                title="Cross-correlation (lag positif = sentiment precede)",
                xaxis_title="Lag (jours)",
                yaxis_title="Correlation"
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Meilleur lag", cross.get("best_lag"))
            c2.metric("Correlation max", cross.get("best_correlation"))

        st.divider()

        # ===== 8. CONCLUSION =====
        st.subheader("8. Synthese")

        st.text(output.get("conclusion", ""))

        if merged is not None:
            st.download_button(
                "Telecharger donnees",
                merged.to_csv(index=False),
                f"merged_{crypto_id}.csv"
            )


def page_multi():
    st.title("Multi-Crypto")

    with st.sidebar:
        st.header("Selection")

        selected = st.multiselect(
            "Cryptos",
            list(CRYPTO_LIST.keys()),
            default=["Bitcoin (BTC)", "Ethereum (ETH)", "Solana (SOL)"]
        )

        method = st.radio("Methode", ["HTTP/JSON", "Selenium"])

        # Affiche la limite clairement
        if method == "HTTP/JSON":
            st.info(f"Limite HTTP: {LIMIT_HTTP} posts/crypto")
            max_limit = 200  # on limite a 200 pour multi
        else:
            st.warning(f"Limite Selenium: {LIMIT_SELENIUM} posts/crypto")
            max_limit = 100

        limit = st.slider("Posts par crypto", 20, max_limit, 50)

        run = st.button("Analyser", type="primary", use_container_width=True)

    if run and selected:
        tokenizer, model = load_model()
        all_results = []

        progress = st.progress(0)
        status = st.empty()

        for i, name in enumerate(selected):
            config = CRYPTO_LIST[name]
            status.text(f"Scraping {name}...")

            if method == "Selenium":
                posts = scrape_selenium(config['sub'], limit)
            else:
                posts = scrape_http(config['sub'], limit)

            if posts:
                status.text(f"Analyse {name}...")
                scores = []
                labs = {"positive": 0, "negative": 0, "neutral": 0}

                for post in posts:
                    text = clean_text(post["title"])
                    if text:
                        s = analyze_sentiment(text, tokenizer, model)
                        scores.append(s["score"])
                        labs[s["label"]] += 1

                avg = sum(scores) / len(scores) if scores else 0
                std = np.std(scores) if scores else 0

                all_results.append({
                    "Crypto": name,
                    "Posts": len(scores),
                    "Sentiment": round(avg, 4),
                    "Std": round(std, 4),
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

        # Graphique sentiment
        fig = px.bar(
            df, x="Crypto", y="Sentiment", color="Sentiment",
            color_continuous_scale=["red", "gray", "green"],
            title="Sentiment moyen par crypto"
        )
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        # Graphique distribution
        fig2 = go.Figure()
        for _, row in df.iterrows():
            fig2.add_trace(go.Bar(
                name=row["Crypto"],
                x=["Positif", "Negatif", "Neutre"],
                y=[row["Positifs"], row["Negatifs"], row["Neutres"]]
            ))
        fig2.update_layout(barmode='group', title="Distribution par crypto")
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button("CSV", df.to_csv(index=False), "multi_crypto.csv")


def page_methodo():
    st.title("Methodologie")

    st.markdown(f"""
    ## Pipeline
    
    1. **Scraping Reddit** (Selenium ou HTTP)
    2. **Nettoyage texte** (regex)
    3. **Sentiment FinBERT** (ProsusAI/finbert)
    4. **Econometrie** (ADF, Granger, VAR)
    
    ## Limites techniques
    
    | Methode | Max posts | Vitesse |
    |---------|-----------|---------|
    | HTTP/JSON | {LIMIT_HTTP} | Rapide |
    | Selenium | {LIMIT_SELENIUM} | Lent (evite ban) |
    
    **Minimum pour econometrie:** {MIN_DAYS_ECONO} jours de donnees
    
    ## References
    
    - Kraaijeveld & De Smedt (2020) - Twitter sentiment
    - ProsusAI/FinBERT - Financial sentiment model
    """)


# ============ MAIN ============

def main():
    page = st.sidebar.radio(
        "Navigation",
        ["Analyse", "Econometrie", "Multi-crypto", "Methodologie"]
    )

    if page == "Analyse":
        page_analyse()
    elif page == "Econometrie":
        page_econometrie()
    elif page == "Multi-crypto":
        page_multi()
    else:
        page_methodo()


if __name__ == "__main__":
    main()