# fetchers.py — India via Yahoo Finance (JSON v8, off-hours OK) + US via Alpha Vantage
import os
import logging
import datetime as dt
from functools import lru_cache
from typing import List

import requests
import pandas as pd
import feedparser
from io import StringIO  # kept in case you later want CSV, unused otherwise
import random

# ------------------------------
# Logging hygiene (quiet noisy libs)
# ------------------------------
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ------------------------------
# NEWS (unchanged)
# ------------------------------
DEFAULT_FEEDS = {
    # US/Global
    "Reuters Business": "http://feeds.reuters.com/reuters/businessNews",
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    # India
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol Markets": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "Mint Markets": "https://www.livemint.com/rss/markets",
    "Business Standard Markets": "https://www.business-standard.com/rss/markets-106.rss",
    "Investing.com India": "https://in.investing.com/rss/news.rss",
}

INDIA_HINTS = ["india","nse","bse",".ns",".nse",".bo",".bse","rbi","sensex","nifty",
               "hdfc","icici","infosys","tcs","reliance","sebi"]
US_HINTS = ["us","u.s.","nasdaq","dow","s&p","sec","fed","apple","microsoft","google","amazon"]

SECTOR_KEYWORDS = {
    "IT": ["it services","software","cloud","saas","ai","semiconductor","chip","infy","tcs","wipro","hcl",
           "oracle","microsoft","google","amazon","nvidia"],
    "Banks/NBFC": ["bank","nbfc","lender","credit","loan","hdfc","icici","axis","sbi","federal","kotak",
                   "boa","wells fargo","citi"],
    "Energy": ["oil","gas","refinery","opec","energy","ongc","reliance","bpcl","hpcl","exxon","chevron"],
    "Auto": ["auto","automobile","ev","car","suv","two-wheeler","tesla","maruti","tata motors","m&m"],
    "Pharma/Healthcare": ["pharma","drug","vaccine","healthcare","biotech","sun pharma","dr reddy","pfizer","moderna"],
    "Metals/Mining": ["steel","aluminium","aluminum","copper","mining","coal","tata steel","jsw"],
    "Telecom": ["telecom","5g","airtel","jio","verizon","att"],
    "FMCG/Retail": ["fmcg","retail","consumer","unilever","itc","hul","walmart","costco","dmart"],
    "Capital Goods/Infra": ["infra","infrastructure","capex","cap goods","l&t","construction"],
    "Finance/Brokerage": ["brokerage","asset mgmt","mutual fund","insurance","amc","hdfc amc","sbi mf"],
}

EVENT_KEYWORDS = {
    "earnings": ["earnings","results","q1","q2","q3","q4","quarter","profit","revenue","guidance"],
    "dividend": ["dividend","buyback","payout"],
    "mna": ["merger","acquisition","deal","takeover"],
    "macro": ["cpi","inflation","gdp","jobs","payrolls","ppi","trade deficit","fomc","rate hike","rate cut","rbi","policy"],
}

def fetch_rss(feed_url: str, limit: int = 50):
    try:
        d = feedparser.parse(feed_url)
        items = []
        for e in d.entries[:limit]:
            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or ""
            link = getattr(e, "link", "") or ""
            published = getattr(e, "published", "") or ""
            source = d.feed.get("title", "RSS")
            text = f"{title} {summary}".lower()

            is_india = any(k in text for k in INDIA_HINTS) or (".ns" in text or ".nse" in text)
            is_us = any(k in text for k in US_HINTS)
            region = "Both" if (is_india and is_us) else ("India" if is_india else ("US" if is_us else "Unknown"))

            sectors = [sec for sec, kws in SECTOR_KEYWORDS.items() if any(k in text for k in kws)] or ["General"]
            events = [ev for ev, kws in EVENT_KEYWORDS.items() if any(k in text for k in kws)] or ["news"]

            items.append({
                "title": title,
                "summary": summary,
                "link": link,
                "published": published,
                "source": source,
                "region": region,
                "sectors": sectors,
                "events": events,
            })
        return items
    except Exception as e:
        print(f"RSS fetch failed for {feed_url}: {e}")
        return []

# ------------------------------
# PRICE ROUTING
#   • INDIA: Yahoo Finance Chart JSON (primary) → Alpha Vantage daily (fallback)
#   • US:    Alpha Vantage intraday → Alpha Vantage daily
# ------------------------------

# --- Helpers: identify Indian symbols and map to candidates ---
def _is_india_symbol(raw: str) -> bool:
    s = (raw or "").upper().strip()
    return s.endswith(".NS") or s.endswith(".NSE") or s.endswith(".BO") or s.endswith(".BSE")

def _to_yahoo_candidates(raw: str) -> List[str]:
    """
    Build an ordered list of Yahoo symbols to try for Indian tickers.
    Accepts .NS/.NSE/.BO/.BSE; tries the provided exchange first, then the alternate.
    """
    s = (raw or "").upper().strip()
    if s.endswith(".NSE"):
        s = s[:-4] + ".NS"
    if s.endswith(".BSE"):
        s = s[:-4] + ".BO"
    if s.endswith(".NS"):
        return [s, s.replace(".NS", ".BO")]
    if s.endswith(".BO"):
        return [s, s.replace(".BO", ".NS")]
    # No suffix → try NSE then BSE
    return [f"{s}.NS", f"{s}.BO"]

# --- Yahoo Finance Chart JSON (v8) for India ---
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]
def _ua() -> str:
    return random.choice(_USER_AGENTS)

@lru_cache(maxsize=256)
def _yahoo_chart_json_df(symbol: str, rng: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Programmatic chart endpoint that works without crumb/cookies.
    Returns DataFrame with columns: time, close (NaNs dropped), sorted ascending.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "range": rng,          # e.g., '1d','5d','1mo','3mo','6mo','1y'
        "interval": interval,  # '1d' (stable off-hours); intraday like '15m' also possible
        "includePrePost": "false",
        "events": "div,splits",
    }
    headers = {"User-Agent": _ua(), "Accept": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    if r.status_code != 200:
        return pd.DataFrame(columns=["time","close"])

    try:
        j = r.json()
        res = j["chart"]["result"][0]
        ts = res.get("timestamp", [])
        quotes = res["indicators"]["quote"][0]
        closes = quotes.get("close", [])
        if not ts or not closes:
            return pd.DataFrame(columns=["time","close"])
        df = pd.DataFrame({"time": pd.to_datetime(ts, unit="s"), "close": closes})
        df = df.dropna().sort_values("time")
        return df
    except Exception:
        return pd.DataFrame(columns=["time","close"])

# --- Alpha Vantage (US + fallback for India) ---
AV_BASE = "https://www.alphavantage.co/query"

def _av_symbol_for(raw: str) -> List[str]:
    """
    Map Yahoo-style Indian suffixes to AV equivalents; leave US tickers unchanged.
    """
    s = (raw or "").strip().upper()
    if s.endswith(".NS"):
        return [s.replace(".NS", ".NSE")]
    if s.endswith(".BO"):
        return [s.replace(".BO", ".BSE")]
    if s.endswith(".NSE") or s.endswith(".BSE"):
        return [s]
    return [s]  # US or others unchanged

@lru_cache(maxsize=256)
def _av_intraday(symbol: str, api_key: str, interval: str = "5min", output_size: str = "compact") -> dict:
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": output_size,
        "datatype": "json",
        "apikey": api_key,
    }
    r = requests.get(AV_BASE, params=params, timeout=15)
    return r.json()

@lru_cache(maxsize=256)
def _av_daily(symbol: str, api_key: str, output_size: str = "compact") -> dict:
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": output_size,
        "datatype": "json",
        "apikey": api_key,
    }
    r = requests.get(AV_BASE, params=params, timeout=15)
    return r.json()

def _intraday_to_df(j: dict) -> pd.DataFrame:
    if not isinstance(j, dict):
        return pd.DataFrame(columns=["time","close"])
    key = next((k for k in j.keys() if k.lower().startswith("time series")), None)
    if not key:
        return pd.DataFrame(columns=["time","close"])
    ts = j.get(key, {})
    if not isinstance(ts, dict) or not ts:
        return pd.DataFrame(columns=["time","close"])
    rows = []
    for t, v in ts.items():
        try:
            rows.append({"time": pd.to_datetime(t), "close": float(v.get("4. close", "nan"))})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("time")

def _daily_to_df(j: dict) -> pd.DataFrame:
    ts = j.get("Time Series (Daily)", {})
    if not isinstance(ts, dict) or not ts:
        return pd.DataFrame(columns=["time","close"])
    rows = []
    for d, v in ts.items():
        try:
            rows.append({"time": pd.to_datetime(d), "close": float(v.get("4. close", "nan"))})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("time")

# ------------------------------
# Unified loader (used by app.py)
# ------------------------------
def load_prices(ticker: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
    """
    Routing:
      - Indian tickers (.NS/.NSE/.BO/.BSE): Yahoo JSON daily (primary) → AV daily fallback
      - Others (US): Alpha Vantage intraday → AV daily
    Returns DataFrame with ['time','close'] or empty if unavailable.
    """
    if not ticker:
        return pd.DataFrame(columns=["time","close"])

    t = ticker.strip().upper()

    # INDIA PATH (Yahoo JSON primary)
    if _is_india_symbol(t):
        for yahoo_sym in _to_yahoo_candidates(t):
            try:
                df = _yahoo_chart_json_df(yahoo_sym, rng="6mo", interval="1d")
                if not df.empty:
                    return df
            except Exception:
                continue

        # Fallback: Alpha Vantage daily (if key present)
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
        if api_key:
            for av_sym in _av_symbol_for(t):
                try:
                    j = _av_daily(av_sym, api_key, output_size="compact")
                    if isinstance(j, dict) and ("Note" in j or "Information" in j):
                        return pd.DataFrame(columns=["time","close"])
                    df = _daily_to_df(j)
                    if not df.empty:
                        return df
                except Exception:
                    continue

        return pd.DataFrame(columns=["time","close"])

    # US / OTHERS PATH (Alpha Vantage)
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if not api_key:
        return pd.DataFrame(columns=["time","close"])

    # Try intraday first
    try:
        j = _av_intraday(t, api_key, interval="5min", output_size="compact")
        if isinstance(j, dict) and ("Note" in j or "Information" in j):
            raise RuntimeError("AV throttled")
        df = _intraday_to_df(j)
        if not df.empty:
            return df
    except Exception:
        pass

    # Daily fallback
    try:
        j = _av_daily(t, api_key, output_size="compact")
        if isinstance(j, dict) and ("Note" in j or "Information" in j):
            return pd.DataFrame(columns=["time","close"])
        df = _daily_to_df(j)
        return df
    except Exception:
        return pd.DataFrame(columns=["time","close"])
