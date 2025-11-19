# metrics_india.py
"""
Financial metrics utilities for Indian equities project.

Functions:
- compute_returns_df: DataFrame of daily returns from price DataFrame
- cumulative_returns_from_returns: cumulative returns from returns series/df
- rolling_volatility: rolling vol (window days), annualized
- annualized_volatility: annualized vol from daily returns
- sharpe_ratio: annualized Sharpe (use risk_free_rate as annual decimal, e.g., 0.04)
- max_drawdown: maximum drawdown given a price series (returns-based)
- rolling_correlation: rolling correlation between two series
- beta: beta of a stock vs market
- value_at_risk: historical VaR (alpha)
- top_contributors: concentration analysis for sector
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # standard

def compute_returns_df(price_df, price_col=None):
    """
    price_df: DataFrame where columns are tickers and rows are dates, containing price series.
              If passed a multi-column per-ticker DataFrame (OHLCV), try to pick 'Adj Close'.
    price_col: optional explicit column name to use (e.g., 'Adj Close'); if None, function
               will try to handle simple DataFrame (columns=tickers) or multi-column (per ticker).
    Returns: DataFrame of daily returns (pct change).
    """
    # if price_col specified and price_df is multi-column (e.g., dict-of-dfs concat), select it
    if price_col and price_col in price_df.columns:
        price_series = price_df[price_col]
        return price_series.pct_change().dropna(how='all')
    # if columns look like tickers (strings) and values are prices -> assume already price panel
    # If DataFrame columns are not unique or are multiindex, try to find 'Adj Close'
    if isinstance(price_df.columns, pd.MultiIndex):
        # try to locate 'Adj Close' column level
        if 'Adj Close' in price_df.columns.get_level_values(1):
            adj = price_df.xs('Adj Close', axis=1, level=1, drop_level=False)
            # after xs, columns might still be MultiIndex; flatten to tickers
            adj = adj.loc[:, adj.columns.get_level_values(1) == 'Adj Close']
            adj.columns = adj.columns.get_level_values(0)
            return adj.pct_change().dropna(how='all')
        # else try 'Close'
        if 'Close' in price_df.columns.get_level_values(1):
            close = price_df.xs('Close', axis=1, level=1)
            close.columns = close.columns.get_level_values(0)
            return close.pct_change().dropna(how='all')
        raise ValueError("Price DataFrame has MultiIndex columns but no 'Adj Close' or 'Close' found.")
    else:
        # assume price_df columns are tickers and values are prices
        return price_df.pct_change().dropna(how='all')

def cumulative_returns_from_returns(returns):
    """(1 + returns).cumprod() - 1 ; supports Series or DataFrame."""
    return (1 + returns).cumprod() - 1

def rolling_volatility(returns, window=30, annualize=TRADING_DAYS):
    """Compute rolling standard deviation of returns and annualize it."""
    return returns.rolling(window).std() * np.sqrt(annualize)

def annualized_volatility(returns):
    """Single-series or DataFrame: std_dev * sqrt(TRADING_DAYS)"""
    return returns.std() * np.sqrt(TRADING_DAYS)

def sharpe_ratio(returns, risk_free_rate=0.0, freq=TRADING_DAYS):
    """
    Annualized Sharpe ratio.
    returns: daily returns Series (or DataFrame -> returns per column)
    risk_free_rate: annual decimal (e.g., 0.04)
    freq: periods per year (252)
    Returns: float (or Series if DataFrame input)
    """
    rf_per_period = (1 + risk_free_rate)**(1/freq) - 1
    excess = returns - rf_per_period
    mean_excess_annual = excess.mean() * freq
    sd_annual = returns.std() * np.sqrt(freq)
    # if DataFrame, do elementwise
    if isinstance(mean_excess_annual, pd.Series):
        return mean_excess_annual / sd_annual.replace(0, np.nan)
    if sd_annual == 0:
        return np.nan
    return float(mean_excess_annual / sd_annual)

def max_drawdown(price_series):
    """
    Compute max drawdown for a price series.
    price_series: pd.Series of prices (not returns). If given returns, user should pass wealth curve.
    Returns most negative drawdown as a negative float (e.g., -0.35)
    """
    # convert to wealth series if input appears to be returns (values between -1 and large), but we assume price series
    # compute cumulative wealth
    returns = price_series.pct_change().fillna(0)
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    return float(drawdown.min())

def rolling_correlation(series_a, series_b, window=60):
    """Rolling Pearson correlation between two series of returns."""
    return series_a.rolling(window).corr(series_b)

def beta(stock_returns, market_returns):
    """Sample beta = cov(stock, market) / var(market). Expects returns aligned."""
    cov = stock_returns.cov(market_returns)
    var = market_returns.var()
    if var == 0:
        return np.nan
    return float(cov / var)

def value_at_risk(returns, alpha=0.05):
    """Historical VaR at alpha (e.g., 0.05). Returns negative number (loss)."""
    return float(returns.quantile(alpha))

def top_contributors(price_panel, start=None, end=None, top_n=5):
    """
    Determine top contributors to cumulative returns per ticker over given period.
    price_panel: DataFrame of price series (columns=tickers)
    Returns DataFrame of cumulative return per ticker in descending order.
    """
    p = price_panel.copy()
    if start:
        p = p.loc[p.index >= pd.to_datetime(start)]
    if end:
        p = p.loc[p.index <= pd.to_datetime(end)]
    # compute cumulative returns by ticker
    cum = (1 + p.pct_change().fillna(0)).cumprod().iloc[-1] - 1
    out = cum.sort_values(ascending=False).head(top_n)
    return out

# small helper: ensure returns alignment
def align_series(a, b):
    """Align two series/dataframes on index intersection and drop NaNs."""
    df = pd.concat([a, b], axis=1).dropna()
    return df.iloc[:,0], df.iloc[:,1]
