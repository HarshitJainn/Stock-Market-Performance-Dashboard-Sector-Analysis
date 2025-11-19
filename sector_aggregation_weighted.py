# sector_aggregation_weighted.py
"""
Market-cap weighted sector aggregation with monthly rebalancing (or custom frequency).

Usage:
    from sector_aggregation_weighted import (
        load_price_panel,
        compute_marketcap_weighted_sector_returns
    )

Notes:
- Expects per-ticker parquet files in data/<TICKER>.parquet with 'Adj Close' or 'Close'.
- Expects data/market_caps.csv with columns: Ticker, yahoo_ticker, market_cap, market_cap_date
  If you have historical market caps, pass market_caps_history (DataFrame indexed by date with columns for tickers).
- If a ticker misses market_cap at rebalance date, that sector-month falls back to equal-weight across available tickers.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

def load_price_panel(tickers):
    """
    Load per-ticker parquet files and return a DataFrame of Adj Close prices (columns = tickers).
    """
    adj = {}
    for t in tickers:
        p = DATA_DIR / f"{t}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        # Prefer 'Adj Close' if present
        if 'Adj Close' in df.columns:
            s = df['Adj Close'].rename(t)
        elif 'Adj_Close' in df.columns:
            s = df['Adj_Close'].rename(t)
        elif 'Close' in df.columns:
            s = df['Close'].rename(t)
        else:
            # try to guess first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                continue
            s = df[numeric_cols[0]].rename(t)
        s.index = pd.to_datetime(s.index)
        adj[t] = s
    if not adj:
        raise ValueError("No parquet price files found in data/ for provided tickers.")
    panel = pd.concat(adj.values(), axis=1).sort_index()
    return panel

def _build_rebalance_dates(price_index, freq='M'):
    """
    Given a DatetimeIndex of trading days, return the list of rebalance dates (the first trading day of each freq period).
    freq='M' -> month end -> we use first trading day of next period to apply weights for that month;
    Implementation: get unique year-months and pick the first date in that month present in price_index.
    """
    df = pd.Series(1, index=price_index)
    # group by year-month
    groups = df.groupby([price_index.year, price_index.month])
    rebalance_dates = [g.index[0] for _, g in groups]
    return pd.DatetimeIndex(rebalance_dates)

def compute_marketcap_weighted_sector_returns(
    constituents_df,
    market_caps_df=None,
    market_caps_history=None,
    rebalance_freq='M'
):
    """
    Compute market-cap weighted sector returns with periodic rebalancing.

    Args:
      constituents_df : DataFrame with columns ['Ticker','Sector'] (Ticker = base ticker e.g., 'TCS')
      market_caps_df  : DataFrame single snapshot with columns ['Ticker','market_cap','market_cap_date',...]
                        (used if market_caps_history not provided)
      market_caps_history: DataFrame indexed by date with columns = tickers, containing historical market caps (optional)
      rebalance_freq : 'M' for monthly rebalancing (default), supports 'Q' quarterly or 'Y' yearly (but implemented with month grouping)

    Returns:
      price_panel : DataFrame of adjusted close prices (columns=tickers)
      ticker_returns : DataFrame of daily returns per ticker
      sector_returns_mcap : DataFrame of daily market-cap weighted sector returns
    """

    # 1) build price panel
    tickers = constituents_df['Ticker'].astype(str).tolist()
    price_panel = load_price_panel(tickers)

    # 2) compute daily returns (pct_change)
    # Do not fill gaps automatically; drop rows entirely empty
    ticker_returns = price_panel.pct_change().dropna(how='all')

    # 3) prepare mapping ticker -> sector
    ticker_to_sector = constituents_df.set_index('Ticker')['Sector'].to_dict()

    # 4) get rebalance dates (first trading day of each month in price_panel)
    rebalance_dates = _build_rebalance_dates(ticker_returns.index, freq=rebalance_freq)

    # initialize sector returns DataFrame
    sectors = sorted(list(set(ticker_to_sector.values())))
    sector_returns_mcap = pd.DataFrame(index=ticker_returns.index, columns=sectors, dtype=float)

    # 5) iterate over rebalance windows (from rebalance_date[i] to rebalance_date[i+1]-1)
    # We'll find for each rebalance date the weights per ticker to apply until next rebalance.
    # Weight source priority: market_caps_history (if provided, pick value at rebalance date) -> market_caps_df snapshot (single)
    for i, start_date in enumerate(rebalance_dates):
        # determine end of this window (inclusive)
        if i+1 < len(rebalance_dates):
            end_date = rebalance_dates[i+1] - pd.Timedelta(days=1)
            # but ensure end_date is within index
            end_date = ticker_returns.index[ticker_returns.index.get_indexer([end_date], method='ffill')[0]]
        else:
            end_date = ticker_returns.index[-1]

        # get available trading dates for this window
        window_idx = ticker_returns.loc[(ticker_returns.index >= start_date) & (ticker_returns.index <= end_date)].index
        if len(window_idx) == 0:
            continue

        # Obtain weights for this rebalance moment
        weights = pd.Series(index=tickers, dtype=float)

        if market_caps_history is not None:
            # market_caps_history expected indexed by date, columns = tickers
            # find nearest earlier or equal date in history for start_date
            try:
                mc_on_date = market_caps_history.loc[market_caps_history.index.asof(start_date)]
            except Exception:
                # asof may not work if not DatetimeIndex; fallback to nearest
                mc_on_date = market_caps_history.reindex(market_caps_history.index.union([start_date])).fillna(method='ffill').loc[start_date]
            # if any tickers missing, leave NaN
            weights = mc_on_date.reindex(tickers)
        elif market_caps_df is not None:
            # single snapshot; use these same market caps for all windows (approximate)
            mc_map = market_caps_df.set_index('Ticker')['market_cap'].to_dict()
            weights = pd.Series({t: mc_map.get(t, np.nan) for t in tickers})
        else:
            weights = pd.Series({t: np.nan for t in tickers})

        # For each sector compute normalized weights; if all NaNs in sector, fallback to equal-weight
        sector_weights = {}
        for sec in sectors:
            tick_in_sec = [t for t in tickers if ticker_to_sector.get(t) == sec and t in ticker_returns.columns]
            if not tick_in_sec:
                continue
            w_sub = weights.reindex(tick_in_sec).astype(float)
            # if all NaN or sum 0 -> fallback to equal-weight
            if w_sub.isna().all() or w_sub.sum(skipna=True) == 0:
                w_norm = pd.Series(1.0/len(tick_in_sec), index=tick_in_sec)
            else:
                # fill NaN with 0 (so missing marketcaps treated as zero weight)
                w_norm = w_sub.fillna(0)
                total = w_norm.sum()
                if total == 0:
                    w_norm = pd.Series(1.0/len(tick_in_sec), index=tick_in_sec)
                else:
                    w_norm = w_norm / total
            sector_weights[sec] = w_norm

        # Now compute weighted sector returns for every day in window
        # sector_return(t) = sum_i weight_i * return_i (for tickers in sector)
        for sec, w in sector_weights.items():
            # ensure ticker_returns has all tickers; missing columns will be ignored by .mul
            sec_returns = ticker_returns.loc[window_idx, w.index].copy()
            # fill missing ticker returns with 0 for multiplication (if a ticker absent that day, treat as 0 return)
            sec_returns = sec_returns.fillna(0.0)
            # multiply each column by the weight, then sum across columns
            weighted = sec_returns.mul(w, axis=1).sum(axis=1)
            sector_returns_mcap.loc[window_idx, sec] = weighted.values

    # final cleanup: fill any remaining NaNs with 0 (no return)
    sector_returns_mcap = sector_returns_mcap.fillna(0.0)
    return price_panel, ticker_returns, sector_returns_mcap

if __name__ == "__main__":
    # quick demo when run directly: loads constituents & market_caps and computes returns
    import pandas as pd
    const_path = DATA_DIR / "nifty50_constituents.csv"
    mcap_path = DATA_DIR / "market_caps.csv"
    if not const_path.exists():
        raise FileNotFoundError(f"Missing constituents file at {const_path}")
    constituents = pd.read_csv(const_path)
    market_caps = None
    if mcap_path.exists():
        market_caps = pd.read_csv(mcap_path)
        # ensure numeric
        market_caps['market_cap'] = pd.to_numeric(market_caps['market_cap'], errors='coerce')
    price_panel, ticker_returns, sector_returns_mcap = compute_marketcap_weighted_sector_returns(
        constituents_df=constituents,
        market_caps_df=market_caps,
        market_caps_history=None,
        rebalance_freq='M'
    )
    print("Computed market-cap weighted sector returns. Sectors:", list(sector_returns_mcap.columns))
    print("Sample (last 5 rows):")
    print(sector_returns_mcap.tail().round(6))
