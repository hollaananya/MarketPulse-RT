# =======================
# app.py  (MarketPulse RT)
# Privacy-safe: no API keys or model names displayed
# =======================

# Quiet ALL noisy libs BEFORE any other imports
import os
import warnings

# Suppress ChromaDB telemetry completely
os.environ.setdefault("CHROMADB_TELEMETRY_ENABLED", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Suppress various warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import logging

# Set logging levels to suppress noisy libraries
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Redirect stderr temporarily during imports to catch telemetry errors
class SuppressStderr:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.original_stderr

# Now do the main imports with error suppression
import time
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import openai

# Import our modules
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from fetchers import DEFAULT_FEEDS, fetch_rss, load_prices
from utils_sentiment import headline_sentiment

# Import RAG with suppressed output
with SuppressStderr():
    from rag import LiveRAG, format_context

# ---------- (NEW) HTML stripping helpers for news rendering ----------
import re, html
_TAG_RE = re.compile(r"<[^>]+>")
_SPACES_RE = re.compile(r"\s+")
def _clean_html_text(s: str) -> str:
    """Remove tags/entities and collapse whitespace. Safe to call on None."""
    if not s:
        return ""
    t = html.unescape(s)
    t = _TAG_RE.sub(" ", t)
    t = _SPACES_RE.sub(" ", t).strip()
    # kill stray "image:" artifacts some feeds add
    return t.replace("image:", "").strip()

# --- Load env (keys stay in environment, never rendered) ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()

# Configure OpenAI client if key exists
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="MarketPulse RT", page_icon="📈", layout="wide")
st.title("📈 MarketPulse RT — Real-time LLM + RAG for Stocks & News")
st.caption("Information only. Not investment advice.")

# --- Minimal, privacy-safe status (no secrets shown) ---
with st.sidebar:
    st.header("Settings")
    st.info("Keys are loaded securely from environment. Nothing sensitive is shown here.")

    # RAG params
    persist = st.checkbox("Persist Vector DB", value=False)
    refresh_mins = st.slider("Refresh interval (minutes)", 1, 30, 3)
    top_k = st.slider("RAG top_k", 3, 20, 10)

    # Feed toggles (safe to show)
    st.markdown("### Feeds")
    enabled = {}
    default_on = ["Reuters Business","Economic Times Markets","Moneycontrol Markets","Mint Markets","WSJ Markets"]
    for name, url in DEFAULT_FEEDS.items():
        enabled[name] = st.checkbox(name, value=(name in default_on))

    st.markdown("### Market Focus & Themes")
    market_focus = st.selectbox("Focus", ["Both","India","US"])
    theme_str = st.text_input("Themes (comma separated): e.g., IT,Banks,Energy", value="")

    st.markdown("### Charts")
    show_charts = st.checkbox("Show price charts", value=True)

# Non-PII inputs
colA, colB = st.columns([2,1])
with colA:
    tickers = st.text_input(
        "Tickers (comma-separated, e.g., INFY.NS, TCS.NS, AAPL)",
        value="INFY.NS,TCS.NS,AAPL"
    )
with colB:
    period = st.selectbox("Chart Period", ["1d","5d","1mo"])

st.markdown("---")

# --- Security/Key checks (generic messages, no names/values) ---
def warn_missing_secrets():
    if not OPENAI_API_KEY:
        st.warning("Language model key not detected. Q&A will be disabled.")
    if not ALPHA_VANTAGE_API_KEY and show_charts:
        st.info("Price data key not detected. Using alternative data sources for Indian stocks.")

warn_missing_secrets()

# --- RAG init (Chroma with suppressed output) ---
if "rag" not in st.session_state:
    with st.spinner("Initializing vector database..."):
        with SuppressStderr():
            st.session_state["rag"] = LiveRAG(persist_dir="./chroma" if persist else None)
rag = st.session_state["rag"]

# --- Fetch & rebuild index ---
def refresh_index():
    feeds = [url for name, url in DEFAULT_FEEDS.items() if enabled.get(name, False)]
    docs = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, url in enumerate(feeds):
        status_text.text(f"Fetching from feed {i+1}/{len(feeds)}...")
        progress_bar.progress((i) / max(len(feeds), 1))

        try:
            feed_docs = fetch_rss(url, limit=40)

            # ------- (NEW) Clean HTML for titles & summaries at ingest -------
            for d in feed_docs:
                d["title"] = _clean_html_text(d.get("title", ""))
                d["summary"] = _clean_html_text(d.get("summary", ""))
            # -----------------------------------------------------------------

            docs.extend(feed_docs)
        except Exception as e:
            st.warning(f"Failed to fetch from one feed: {str(e)[:100]}")

    status_text.text("Processing sentiment...")
    progress_bar.progress(0.8)

    for d in docs:
        d["sentiment"] = headline_sentiment(f"{d['title']} {d['summary']}")

    status_text.text("Building vector index...")
    progress_bar.progress(0.9)

    with SuppressStderr():
        rag.rebuild(docs)  # atomic clear + add (implemented in rag.py)

    st.session_state["docs"] = docs
    st.session_state["last_refresh"] = time.time()

    progress_bar.progress(1.0)
    status_text.empty()
    progress_bar.empty()

    st.success(f"✅ Indexed {len(docs)} items from {len(feeds)} feeds.")

do_refresh = st.button("🔄 Refresh News & Data")
ts = st.session_state.get("last_refresh", 0)
if do_refresh or (time.time() - ts) > (refresh_mins*60):
    refresh_index()

docs = st.session_state.get("docs", [])

# --- Market Snapshot ---
st.subheader("📊 Market Snapshot (latest headlines sentiment)")
def snap(region):
    d = [x for x in docs if x.get("region")==region or (region=="Both" and x.get("region") in ["India","US","Both"])]
    if not d:
        return 0.0, 0
    sc = sum(x.get("sentiment", 0.0) for x in d) / len(d)
    return sc, len(d)

ind_sc, ind_n = snap("India")
us_sc, us_n = snap("US")
col1, col2 = st.columns(2)
with col1:
    color = "normal" if abs(ind_sc) < 0.1 else ("inverse" if ind_sc < 0 else "off")
    st.metric("🇮🇳 India sentiment (VADER)", f"{ind_sc:+.2f}", help=f"Based on {ind_n} headlines", delta_color=color)
with col2:
    color = "normal" if abs(us_sc) < 0.1 else ("inverse" if us_sc < 0 else "off")
    st.metric("🇺🇸 US sentiment (VADER)", f"{us_sc:+.2f}", help=f"Based on {us_n} headlines", delta_color=color)

# --- Event Radar ---
st.subheader("🎯 Event Radar")
def pick(docs, key):
    def get_events(x):
        e = x.get("events", [])
        if isinstance(e, list):
            return ",".join(e)
        return str(e)
    return [x for x in docs if key in get_events(x)]

ev_cols = st.columns(4)
for i, ev in enumerate(["earnings","macro","dividend","mna"]):
    with ev_cols[i]:
        count = len(pick(docs, ev))
        emoji = {"earnings": "📈", "macro": "🏛️", "dividend": "💰", "mna": "🤝"}
        st.metric(f"{emoji[ev]} {ev.capitalize()}", count)

# --- Price charts (Enhanced with better error handling) ---
if show_charts and tickers.strip():
    st.subheader("📉 Live / Recent Price Charts")

    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]

    if len(ticker_list) > 6:
        st.warning("Showing first 6 tickers to avoid rate limits")
        ticker_list = ticker_list[:6]

    chart_progress = st.progress(0)
    chart_status = st.empty()

    charts_shown = 0
    for i, tk in enumerate(ticker_list):
        chart_status.text(f"Loading {tk}...")
        chart_progress.progress((i) / max(len(ticker_list), 1))

        try:
            dfp = load_prices(tk, period=period, interval="5m" if period!="1mo" else "60m")
            if not dfp.empty:
                # Create a nice chart with title
                st.write(f"**{tk}** - Last {period}")
                st.line_chart(dfp.set_index("time")["close"], height=200)
                charts_shown += 1
            else:
                st.info(f"⚠️ No price data available for {tk}")
        except Exception as e:
            st.error(f"❌ Failed to load {tk}: {str(e)[:100]}")

        # Small delay to avoid overwhelming APIs
        time.sleep(0.1)

    chart_progress.progress(1.0)
    chart_status.empty()
    chart_progress.empty()

    if charts_shown == 0:
        st.warning("No price data could be loaded. This can be due to:")
        st.write("• Market hours (Indian markets: 9:15 AM - 3:30 PM IST)")
        st.write("• Rate limits on free APIs")
        st.write("• Symbol mapping issues")
        st.write("• Network connectivity")

# --- Q&A grounded in latest headlines ---
st.subheader("🤖 Ask about India/US markets (grounded in latest headlines)")
question = st.text_input(
    "Example: Compare Infosys vs TCS outlook this week and mention any macro watchpoints.",
    placeholder="Ask about market trends, specific companies, or economic events..."
)

if st.button("🎯 Get AI Analysis", type="primary"):
    if not OPENAI_API_KEY:
        st.error("❌ Language model key is not configured.")
    elif not question.strip():
        st.warning("⚠️ Please enter a question first.")
    else:
        region = None if market_focus == "Both" else market_focus
        themes = [t.strip() for t in theme_str.split(",") if t.strip()] if theme_str else None

        with st.spinner("🧠 Analyzing latest market data..."):
            try:
                # Suppress any RAG-related stderr during query
                with SuppressStderr():
                    ctx_items = rag.query(question + " " + tickers, k=top_k, region=region, sectors=themes, events=None)

                context = format_context(ctx_items)

                if not context.strip():
                    st.warning("⚠️ No relevant market data found for your question. Try refreshing the news or asking about different topics.")
                else:
                    msgs = [
                        {"role":"system","content": SYSTEM_PROMPT},
                        {"role":"user","content": USER_PROMPT_TEMPLATE.format(
                            question=question,
                            tickers=tickers,
                            market_focus=market_focus,
                            themes=theme_str or "None",
                            context=context
                        )}
                    ]

                    try:
                        resp = openai.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=msgs,
                            temperature=0.2,
                            max_tokens=1000
                        )

                        # Display the response with nice formatting
                        st.markdown("### 📝 AI Analysis")
                        st.markdown(resp.choices[0].message.content)

                        # Show context sources used
                        with st.expander("📚 Sources used in analysis", expanded=False):
                            for i, item in enumerate(ctx_items[:5]):  # Show top 5 sources
                                st.write(f"**{i+1}.** {item.get('title', 'Untitled')}")
                                st.write(f"*{item.get('source', 'Unknown source')}* - {item.get('published', 'No date')}")
                                if item.get('summary'):
                                    sm = item['summary']
                                    st.write(sm[:200] + "..." if len(sm) > 200 else sm)
                                st.write("---")

                    except Exception as e:
                        st.error("❌ LLM request failed. Please try again later.")
                        if "rate limit" in str(e).lower():
                            st.info("💡 You may have hit API rate limits. Try again in a few minutes.")

            except Exception as e:
                st.error("❌ Analysis failed. Please try refreshing the data.")

# --- Recent Headlines Preview (CLEAN RENDER) ---
if docs:
    st.subheader("📰 Recent Headlines Preview")

    # Filter by market focus if set
    filtered_docs = docs
    if market_focus != "Both":
        filtered_docs = [d for d in docs if d.get("region") == market_focus or d.get("region") == "Both"]

    # Show latest 10 headlines
    recent_docs = sorted(filtered_docs, key=lambda x: x.get("published", ""), reverse=True)[:10]

    def _sent_emoji(score: float) -> str:
        if score >= 0.35: return "😄"
        if score >= 0.1:  return "🙂"
        if score > -0.1:  return "😐"
        if score > -0.35: return "🙁"
        return "😠"

    for i, doc in enumerate(recent_docs):
        col1, col2 = st.columns([6, 1.5], gap="large")

        # Ensure clean text again in case old docs exist in session
        title_clean = _clean_html_text(doc.get("title", "Untitled"))
        summ_clean = _clean_html_text(doc.get("summary", ""))

        with col1:
            emoji = _sent_emoji(doc.get("sentiment", 0))
            title_md = f"**{emoji} [{title_clean}]({doc.get('link','')})**" if doc.get("link") else f"**{emoji} {title_clean}**"
            st.markdown(title_md)
            if summ_clean:
                st.write(summ_clean if len(summ_clean) <= 300 else (summ_clean[:297] + "…"))

        with col2:
            st.markdown(f"*{doc.get('source', 'Unknown')}*")
            region_emoji = "🇮🇳" if doc.get("region") == "India" else "🇺🇸" if doc.get("region") == "US" else "🌍"
            st.markdown(f"{region_emoji} {doc.get('region', 'Unknown')}")

        st.markdown("---")

st.markdown("---")
st.caption("📊 Feeds via RSS; 📈 Indian prices via Yahoo Finance + NSE; 🇺🇸 US prices via Alpha Vantage. For education only — not investment advice.")

# Add some helpful tips in the sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    **Indian Tickers**: Use .NS suffix (e.g., INFY.NS, TCS.NS, RELIANCE.NS)

    **US Tickers**: Use standard symbols (e.g., AAPL, MSFT, GOOGL)

    **Best Questions**: Ask about specific companies, compare stocks, or inquire about recent market events.
    """)

    if docs:
        st.markdown("### 📊 Data Summary")
        st.write(f"📰 {len(docs)} total headlines")
        india_count = len([d for d in docs if d.get("region") == "India"])
        us_count = len([d for d in docs if d.get("region") == "US"])
        st.write(f"🇮🇳 {india_count} India-related")
        st.write(f"🇺🇸 {us_count} US-related")
