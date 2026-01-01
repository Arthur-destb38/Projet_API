"""
Crypto Sentiment - Streamlit
Projet MoSEF 2024-2025
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

# Selenium
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import NoSuchElementException
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

                    comments_el = elem.select_one("a.comments")
                    comments_txt = comments_el.get_text(strip=True) if comments_el else "0"
                    match = re.search(r"(\d+)", comments_txt)
                    num_comments = int(match.group(1)) if match else 0

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
                            "num_comments": num_comments,
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

        method = st.radio("Methode", ["HTTP/JSON (rapide)", "Selenium (cours)"])

        crypto_name = st.selectbox("Crypto", list(CRYPTO_LIST.keys()))
        config = CRYPTO_LIST[crypto_name]

        if "HTTP" in method:
            limit = st.slider("Nombre de posts", 20, 1000, 50, 10)
        else:
            limit = st.slider("Nombre de posts", 20, 200, 50, 10)

        st.divider()
        st.caption(f"Subreddit: r/{config['sub']}")

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

    # Info methodes
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Selenium (cours):** navigateur, simulation humaine, parse HTML")
    with c2:
        st.markdown("**HTTP/JSON (bonus):** appel API direct, rapide")

    st.divider()

    if run:
        method_key = "selenium" if "Selenium" in method else "http"

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

        labels = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            labels[r["label"]] += 1

        # Metriques
        st.subheader("Resultats")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Methode", method_key.upper())
        c2.metric("Posts", len(results))
        c3.metric("Sentiment", f"{avg:+.3f}")
        c4.metric("Positifs", f"{labels['positive']}/{len(results)}")

        # Graphiques
        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                values=list(labels.values()),
                names=["Positif", "Negatif", "Neutre"],
                color_discrete_sequence=["#28a745", "#dc3545", "#6c757d"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(x=scores, nbins=20)
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=avg, line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        # Tableau
        st.subheader("Posts")

        df = pd.DataFrame([
            {
                "Titre": posts[i]["title"][:60],
                "Score": round(results[i]["score"], 3),
                "Label": results[i]["label"],
                "Upvotes": posts[i].get("score", 0)
            }
            for i in range(len(results))
        ])

        st.dataframe(df, use_container_width=True, height=400)
        st.download_button("CSV", df.to_csv(index=False), f"sentiment_{config['id']}.csv")

        st.info("Donnees sauvegardees! Va dans 'Econometrie' pour les tests statistiques.")


def page_econometrie():
    st.title("Analyse Econometrique")
    st.caption("ADF, Granger, VAR, Cross-correlation")

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

    st.success(f"Donnees: {len(posts)} posts ({crypto_name})")

    with st.sidebar:
        st.header("Parametres")
        days = st.slider("Jours historiques", 30, 90, 60)
        max_lag = st.slider("Lag max", 3, 10, 5)
        run = st.button("Lancer tests", type="primary", use_container_width=True)

    # Explication
    with st.expander("Explication des tests"):
        st.markdown("""
        **ADF:** teste la stationnarite (p < 0.05 = stationnaire)
        
        **Granger:** teste si X predit Y (p < 0.05 = causalite)
        
        **VAR:** modele vectoriel autoregressif
        
        **Cross-corr:** correlation a differents lags
        """)

    if run:
        with st.spinner("Tests en cours..."):
            output = run_full_analysis(posts, results, crypto_id, days, max_lag)

        if output["status"] == "error":
            st.error(output.get("error"))
            return

        # Donnees
        st.subheader("1. Donnees")
        info = output["data_info"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Jours sentiment", info["jours_sentiment"])
        c2.metric("Jours prix", info["jours_prix"])
        c3.metric("Jours fusionnes", info["jours_merged"])
        st.caption(f"{info['date_debut']} -> {info['date_fin']}")

        st.divider()

        # ADF
        st.subheader("2. Stationnarite (ADF)")
        adf = output["adf_tests"]
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Sentiment**")
            s = adf.get("sentiment", {})
            if "error" not in s:
                st.write(f"p-value: {s.get('pvalue')}")
                if s.get("stationary"):
                    st.success("Stationnaire")
                else:
                    st.warning("Non stationnaire")

        with c2:
            st.markdown("**Returns**")
            r = adf.get("returns", {})
            if "error" not in r:
                st.write(f"p-value: {r.get('pvalue')}")
                if r.get("stationary"):
                    st.success("Stationnaire")
                else:
                    st.warning("Non stationnaire")

        st.divider()

        # Granger
        st.subheader("3. Granger Causality")
        granger = output["granger"]

        if "error" not in granger:
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Sentiment -> Returns**")
                s2r = granger.get("sentiment_to_returns", {})
                if s2r.get("significant"):
                    st.success(f"SIGNIFICATIF (lag={s2r.get('best_lag')})")
                else:
                    st.warning("Non significatif")

                if s2r.get("pvalues"):
                    df_p = pd.DataFrame({"Lag": list(s2r["pvalues"].keys()), "P-val": list(s2r["pvalues"].values())})
                    st.dataframe(df_p, hide_index=True)

            with c2:
                st.markdown("**Returns -> Sentiment**")
                r2s = granger.get("returns_to_sentiment", {})
                if r2s.get("significant"):
                    st.success(f"SIGNIFICATIF (lag={r2s.get('best_lag')})")
                else:
                    st.warning("Non significatif")

                if r2s.get("pvalues"):
                    df_p = pd.DataFrame({"Lag": list(r2s["pvalues"].keys()), "P-val": list(r2s["pvalues"].values())})
                    st.dataframe(df_p, hide_index=True)

        st.divider()

        # VAR
        st.subheader("4. VAR")
        var = output["var"]
        if "error" not in var:
            c1, c2 = st.columns(2)
            c1.metric("Lag optimal", var.get("optimal_lag"))
            c2.metric("AIC", var.get("aic"))

        st.divider()

        # Cross-corr
        st.subheader("5. Cross-correlation")
        cross = output["cross_corr"]

        if cross.get("correlations"):
            corrs = cross["correlations"]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(corrs.keys()),
                y=list(corrs.values()),
                marker_color=['red' if v < 0 else 'green' for v in corrs.values()]
            ))
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(xaxis_title="Lag", yaxis_title="Correlation")
            st.plotly_chart(fig, use_container_width=True)

            st.metric("Meilleur lag", cross.get("best_lag"))
            st.metric("Correlation", cross.get("best_correlation"))

        st.divider()

        # Graphique
        st.subheader("6. Visualisation")
        merged = output.get("merged_data")

        if merged is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=merged['date'], y=merged['sentiment_mean'], name="Sentiment", yaxis="y1"))
            fig.add_trace(go.Scatter(x=merged['date'], y=merged['log_return'], name="Return", yaxis="y2"))
            fig.update_layout(
                yaxis=dict(title="Sentiment", side="left"),
                yaxis2=dict(title="Return", side="right", overlaying="y")
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Conclusion
        st.subheader("7. Conclusion")
        st.text(output.get("conclusion", ""))


def page_compare():
    st.title("Comparaison des methodes")

    with st.sidebar:
        crypto_name = st.selectbox("Crypto", list(CRYPTO_LIST.keys()))
        config = CRYPTO_LIST[crypto_name]
        limit = st.slider("Posts", 10, 100, 30)
        run = st.button("Comparer", type="primary", use_container_width=True)

    st.markdown("""
    | Aspect | Selenium | HTTP |
    |--------|----------|------|
    | Navigateur | Oui | Non |
    | Vitesse | Lent | Rapide |
    | Cours | OUI | NON |
    """)

    if run:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("HTTP")
            start = time.time()
            http_posts = scrape_http(config['sub'], limit)
            http_time = time.time() - start
            st.metric("Temps", f"{http_time:.2f}s")
            st.metric("Posts", len(http_posts))

        with c2:
            st.subheader("Selenium")
            if SELENIUM_OK:
                start = time.time()
                sel_posts = scrape_selenium(config['sub'], limit)
                sel_time = time.time() - start
                st.metric("Temps", f"{sel_time:.2f}s")
                st.metric("Posts", len(sel_posts))
            else:
                st.error("Selenium non dispo")
                sel_time = 1

        if http_time > 0 and SELENIUM_OK:
            st.info(f"HTTP est {sel_time/http_time:.1f}x plus rapide")


def page_multi():
    st.title("Multi-Crypto")

    with st.sidebar:
        selected = st.multiselect("Cryptos", list(CRYPTO_LIST.keys()),
                                  default=["Bitcoin (BTC)", "Ethereum (ETH)", "Solana (SOL)"])
        limit = st.slider("Posts/crypto", 20, 100, 30)
        run = st.button("Analyser", type="primary", use_container_width=True)

    if run and selected:
        tokenizer, model = load_model()
        all_results = []

        progress = st.progress(0)

        for i, name in enumerate(selected):
            config = CRYPTO_LIST[name]
            posts = scrape_http(config['sub'], limit)

            if posts:
                scores = []
                for post in posts:
                    text = clean_text(post["title"])
                    if text:
                        s = analyze_sentiment(text, tokenizer, model)
                        scores.append(s["score"])

                avg = sum(scores) / len(scores) if scores else 0
                all_results.append({"Crypto": name, "Posts": len(scores), "Sentiment": round(avg, 4)})

            progress.progress((i + 1) / len(selected))

        df = pd.DataFrame(all_results)
        st.dataframe(df, use_container_width=True)

        fig = px.bar(df, x="Crypto", y="Sentiment", color="Sentiment",
                     color_continuous_scale=["red", "gray", "green"])
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)


def page_methodo():
    st.title("Methodologie")

    st.markdown("""
    ## Pipeline
    
    1. Scraping Reddit (Selenium ou HTTP)
    2. Nettoyage texte
    3. FinBERT sentiment
    4. Tests econometriques (Granger, VAR)
    
    ## References
    
    - Kraaijeveld & De Smedt (2020) - Twitter sentiment
    - ProsusAI/FinBERT
    """)


# ============ MAIN ============

def main():
    page = st.sidebar.radio(
        "Navigation",
        ["Analyse", "Econometrie", "Comparaison", "Multi-crypto", "Methodologie"]
    )

    if page == "Analyse":
        page_analyse()
    elif page == "Econometrie":
        page_econometrie()
    elif page == "Comparaison":
        page_compare()
    elif page == "Multi-crypto":
        page_multi()
    else:
        page_methodo()


if __name__ == "__main__":
    main()