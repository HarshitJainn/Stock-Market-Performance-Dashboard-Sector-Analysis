# Stock Market Sector Analytics Dashboard for Indian Markets (NIFTY 50)

This repository contains an end-to-end interactive analytics platform for Indian stock market sector analysis using NIFTY 50 data. The project includes automated data extraction, sector-level analytics, risk modeling, correlation studies, trend tracking, event-based analysis, and a complete interactive Streamlit dashboard.

The goal is to provide a comprehensive research-grade tool for understanding sector performance, rotation behavior, and risk-return dynamics within the Indian equity market.

---

## Features

### 1. Automated Data Pipeline (India Market)
- Fetches historical OHLCV data for all NIFTY 50 tickers using `yfinance`.
- Supports `.NS` and `.BO` fallback for robust data coverage.
- Saves data in efficient Parquet format (one file per ticker).
- Computes market caps and stores them in `market_caps.csv`.
- Validation script checks for missing dates, ticker inconsistencies, and data gaps.
- Prepares the entire dataset needed for the dashboard.

---

### 2. Sector Classification (NSE-Based)
- Clean mapping of each NIFTY 50 stock to its official NSE sector.
- Supports sectors such as Financials, IT, FMCG, Pharma, Automobile, Metals, Cement, Telecom, Power, Energy, Logistics and more.
- Enables equal-weighted and market-cap-weighted sector returns.

---

### 3. Sector Returns & Risk Metrics Engine
The analytics engine computes:

- Daily returns  
- Monthly and annual returns  
- Cumulative returns  
- Rolling returns  
- Annualized volatility  
- Rolling volatility (60-day)  
- Sharpe ratio (risk-adjusted return)  
- Maximum drawdown  
- Rolling correlations  
- Sector-to-sector and sector-to-index correlation  
- Stock-level contribution analysis  

All key metrics are validated using separate scripts.

---

### 4. Streamlit Dashboard

The application is built using Streamlit and includes multiple interactive views:

#### A. Market Overview
- Cumulative returns for chosen sectors.
- Annual return bar charts.
- Best and worst performing sector (YTD).
- Sharpe ratio summary.
- Volatility summary.
- Fully interactive zoom and hover functionality.

#### B. Sector Deep Dive
- Sector vs NIFTY cumulative return comparison.
- Rolling volatility.
- Top stock contributors within a given sector.
- Concentration and dispersion analysis.

#### C. Monthly Heatmap
- Heatmap of monthly returns for each sector.
- Useful for seasonality and pattern analysis.

#### D. Risk–Return Scatter Plot
- Annualized return vs annualized volatility.
- Bubble size reflects cumulative return.
- Bubble color reflects Sharpe ratio.
- Helps identify efficient, defensive, and high-risk sectors.

#### E. Correlation & Network Graph
- Correlation heatmap for all sectors.
- Rolling correlation between any two sectors.
- Interactive network graph:
  - Nodes represent sectors
  - Node size shows cumulative return
  - Edges show correlation above threshold
- Useful for clustering and diversification analysis.

#### F. Sector Leaderboard with Trends
- Ranks sectors monthly or quarterly based on returns.
- Trend arrows show rank improvement or decline.
- Animated bar-chart race illustrating sector rotation over time.

#### G. Event Marker Engine
- Preloaded major events: COVID crash, RBI rate decisions, Union Budgets, global risk events.
- Ability to add custom market events via UI.
- Events saved to `data/events.csv`.
- Charts automatically annotate vertical event markers.
- Helps visualize market reactions to key events.

#### H. Download & Audit Panel
- Lists backend files.
- Allows selective downloading for inspection.
- Useful for transparency and debugging.

---

## Technology Stack

**Language:** Python 3  
**Libraries:**  
- pandas  
- numpy  
- yfinance  
- plotly  
- streamlit  
- networkx  
- pyarrow  

**Storage:**  
- Parquet files for historical price data  
- CSV files for mapping and events

---
