# sector_aggregation.py
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

def load_price_panel(tickers):
    """Load parquet files for tickers (list of base tickers) and return adj close DataFrame"""
    adj = {}
    for t in tickers:
        p = DATA_DIR / f"{t}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        # prefer Adj Close if present else Close
        col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        s = df[col].rename(t)
        adj[t] = s
    panel = pd.concat(adj.values(), axis=1).sort_index()
    return panel

def compute_sector_returns(equal_weight=True, constituents_df=None, market_caps_df=None):
    """
    constituents_df: DataFrame with columns ['Ticker', 'Sector']
    market_caps_df: DataFrame with columns ['Ticker','market_cap'] - latest snapshot
    returns: daily sector returns DataFrame
    """
    tickers = constituents_df['Ticker'].tolist()
    price_panel = load_price_panel(tickers)
    returns = price_panel.pct_change().dropna(how='all')
    # map tickers->sector
    mapping = constituents_df.set_index('Ticker')['Sector'].to_dict()
    # create MultiIndex columns: (ticker, sector)
    returns.columns = pd.MultiIndex.from_tuples([(c, mapping.get(c, "Unknown")) for c in returns.columns])
    if equal_weight:
        # group by sector and mean across tickers per day
        sector_returns = returns.groupby(level=1, axis=1).mean()
    else:
        # market-cap weighted: need market_caps_df
        if market_caps_df is None:
            raise ValueError("market_caps_df required for market-cap weighted aggregation")
        mc_map = market_caps_df.set_index('Ticker')['market_cap'].to_dict()
        # construct weights aligned to tickers
        weights = []
        for t in returns.columns.get_level_values(0):
            w = mc_map.get(t, np.nan)
            weights.append(w)
        weight_series = pd.Series(weights, index=returns.columns.get_level_values(0))
        # for each sector, compute weighted returns across tickers with available weights
        sector_returns = pd.DataFrame(index=returns.index)
        sectors = returns.columns.get_level_values(1).unique()
        for s in sectors:
            tick_in_sector = [t for (t, sec) in returns.columns if sec == s]
            if not tick_in_sector:
                continue
            sector_ret = returns[tick_in_sector]
            w = weight_series.loc[tick_in_sector]
            # normalize weights (ignore NaN)
            w = w.fillna(0)
            if w.sum() == 0:
                # fallback to equal weight
                sector_returns[s] = sector_ret.mean(axis=1)
            else:
                w_norm = w / w.sum()
                sector_returns[s] = sector_ret.mul(w_norm, axis=1).sum(axis=1)
    return returns, sector_returns
