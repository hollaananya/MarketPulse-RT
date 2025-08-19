# 📈 MarketPulse RT

> Real-time AI-powered dashboard for **India and US stock markets**.  
> Combines **live market data, financial news feeds, sentiment analysis, and RAG-based Q&A** into one Streamlit app.

---

## 🌟 Features

- **Dual-Market Support**
  - 🇮🇳 **Indian stocks** via NSE/Yahoo Finance feeds  
  - 🇺🇸 **US stocks** via Alpha Vantage API  

- **Live Market Snapshots**
  - Fetches **real-time stock prices** (configurable tickers)  
  - Updates charts at regular intervals  

- **News & Event Radar**
  - Aggregates headlines from **Reuters, Economic Times, Mint, Moneycontrol, WSJ**  
  - Cleans and deduplicates summaries  
  - Tags events (earnings, macro, dividends, M&A)  
  - Sentiment scores with emojis (😄 🙂 😐 🙁 😠)  

- **Vector Search + RAG**
  - Stores latest news in a local **ChromaDB**  
  - Query headlines with a **retrieval-augmented LLM**  
  - Ask contextual questions like  
    > “Compare Infosys vs TCS outlook this week”  
    > “What are key macro watchpoints for US tech?”  

- **Interactive Dashboard (Streamlit)**
  - Sidebar toggles for feeds, tickers, refresh interval, chart settings  
  - Real-time metrics & analysis panels  
  - Clickable news headlines with sentiment indicators  

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – Frontend dashboard  
- [OpenAI](https://platform.openai.com/) – LLM for Q&A  
- [ChromaDB](https://www.trychroma.com/) – Vector store for news indexing  
- [Alpha Vantage](https://www.alphavantage.co/) – US stock prices  
- [Yahoo Finance / NSE APIs](https://pypi.org/project/yfinance/) – Indian stock prices  
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) – Headline sentiment scoring  
- Python: `pandas`, `feedparser`, `requests`, `dotenv`  

---

## ⚡ Installation

```bash
# Clone the repo
git clone https://github.com/your-username/marketpulse-rt.git
cd marketpulse-rt

# Create venv (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py



