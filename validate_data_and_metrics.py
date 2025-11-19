# validate_data_and_metrics.py
"""
Run quick validation of the downloaded data and compute sample metrics.
Usage:
    python validate_data_and_metrics.py
Outputs:
 - number of parquet files found
 - date ranges per ticker (first/last)
 - sample cumulative returns for sectors (equal-weight)
 - Sharpe ratios per sector (equal-weight)
 - Compare constructed NIFTY (equal-weight aggregate) vs ^NSEI from yfinance
"""

import glob
import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from metrics_india import compute_returns_df, cumulative_returns_from_returns, sharpe_ratio, max_drawdown
from sector_aggregation import compute_sector_returns

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

def count_parquets():
    files = glob.glob(str(DATA_DIR / "*.parquet"))
    return files

def show_date_ranges():
    files = count_parquets()
    rows = []
    for f in files:
        name = Path(f).stem
        try:
            df = pd.read_parquet(f)
            idx = pd.to_datetime(df.index)
            rows.append({"Ticker": name, "start": str(idx.min().date()), "end": str(idx.max().date()), "rows": len(idx)})
        except Exception as e:
            rows.append({"Ticker": name, "start": None, "end": None, "rows": 0, "error": str(e)})
    return pd.DataFrame(rows).sort_values("Ticker")

def compare_nifty_index(const_df, returns_panel):
    """
    Build simple equal-weighted index from components and compare with ^NSEI (Yahoo).
    returns_panel: per-ticker price panel (Adj Close)
    """
    # compute daily returns per ticker
    returns = returns_panel.pct_change().dropna(how='all')
    # equal-weight sector-agnostic index: mean across tickers
    ew_returns = returns.mean(axis=1)
    ew_cum = (1 + ew_returns).cumprod() - 1

    # fetch ^NSEI
    idx = yf.download("^NSEI", start=ew_returns.index.min().strftime("%Y-%m-%d"), end=ew_returns.index.max().strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
    if idx is None or idx.empty:
        print("Could not fetch ^NSEI for comparison.")
        return
    idx_adj = idx['Adj Close'].pct_change().dropna()
    idx_cum = (1 + idx_adj).cumprod() - 1

    # join on dates
    df = pd.concat([ew_cum.rename("constructed"), idx_cum.rename("^NSEI")], axis=1).dropna()
    # compute correlation and last-day difference
    corr = df.corr().iloc[0,1]
    last_diff = df.iloc[-1,0] - df.iloc[-1,1]
    print(f"Constructed EW index vs ^NSEI correlation: {corr:.4f}")
    print(f"Constructed EW index last cumulative - ^NSEI last cumulative = {last_diff:.4%}")

def main():
    print("=== Data & metrics validation ===")
    const_path = DATA_DIR / "nifty50_constituents.csv"
    if not const_path.exists():
        raise FileNotFoundError(f"Missing constituents CSV at {const_path}")
    const = pd.read_csv(const_path)
    print(f"Loaded constituents: {len(const)} rows")

    ranges_df = show_date_ranges()
    print("\nTicker date ranges (sample):")
    print(ranges_df.head(8).to_string(index=False))

    # load price panel via sector_aggregation helper
    tickers = const['Ticker'].tolist()
    print("\nBuilding price panel and computing sector returns (equal-weight)")
    raw_returns, sector_returns = compute_sector_returns(equal_weight=True, constituents_df=const, market_caps_df=None)

    # show cumulative returns for each sector over entire span
    cum = (1 + sector_returns).cumprod() - 1
    last = cum.iloc[-1].sort_values(ascending=False)
    print("\nCumulative returns by sector (last available date):")
    print(last.to_string())

    # Sharpe per sector (daily returns -> annualized)
    sr = sector_returns.apply(lambda s: sharpe_ratio(s.dropna(), risk_free_rate=0.06), axis=0)
    print("\nSharpe ratios by sector (annualized) with 6% RF (may be NaN if insufficient data):")
    print(sr.sort_values(ascending=False).to_string())

    # Max drawdown for each sector computed from price panel (rebuild price series)
    # reconstruct sector price-series by converting cumulative returns back to price-like series (start=1)
    sector_prices = (1 + sector_returns).cumprod()
    mdds = sector_prices.apply(lambda s: max_drawdown(s), axis=0)
    print("\nMax drawdown by sector (negative value):")
    print(mdds.sort_values().to_string())

    # Compare to ^NSEI
    try:
        # build price panel for tickers (adj close) using sector_aggregation loader
        # This module loads per-ticker adj close from data/*.parquet
        # compute simple comparison
        # For convenience reuse raw_returns to build price panel by cumulative product
        # But we can also load price_panel via sector_aggregation.load_price_panel directly if needed.
        import importlib
        sa = importlib.import_module("sector_aggregation")
        price_panel = sa.load_price_panel(tickers)
        compare_nifty_index(const, price_panel)
    except Exception as e:
        print("Index comparison skipped due to error:", e)

    print("\nValidation complete. If numbers look plausible, proceed to build dashboard.")

if __name__ == "__main__":
    main()
