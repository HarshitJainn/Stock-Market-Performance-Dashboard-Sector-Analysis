# compute_mcap_sector_metrics.py
import pandas as pd
import numpy as np
from pathlib import Path
from sector_aggregation_weighted import compute_marketcap_weighted_sector_returns, load_price_panel
from metrics_india import rolling_volatility, annualized_volatility, sharpe_ratio, max_drawdown, cumulative_returns_from_returns

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

def main():
    const_path = DATA_DIR / "nifty50_constituents.csv"
    mcap_path = DATA_DIR / "market_caps.csv"
    if not const_path.exists():
        raise FileNotFoundError("Missing constituents CSV")
    constituents = pd.read_csv(const_path)
    market_caps = None
    if mcap_path.exists():
        market_caps = pd.read_csv(mcap_path)
        market_caps['market_cap'] = pd.to_numeric(market_caps['market_cap'], errors='coerce')

    # Compute MCAP-weighted sector returns
    price_panel, ticker_returns, sector_mcap_returns = compute_marketcap_weighted_sector_returns(
        constituents_df=constituents,
        market_caps_df=market_caps,
        market_caps_history=None,
        rebalance_freq='M'
    )

    # 1) cumulative returns
    cum = cumulative_returns_from_returns(sector_mcap_returns)

    # 2) annualized vol and Sharpe (use RF = 0.06 as example)
    ann_vol = sector_mcap_returns.apply(lambda s: annualized_volatility(s.dropna()))
    sharpe = sector_mcap_returns.apply(lambda s: sharpe_ratio(s.dropna(), risk_free_rate=0.06))

    # 3) max drawdown (we compute from implied price series)
    sector_prices = (1 + sector_mcap_returns).cumprod()
    mdds = sector_prices.apply(lambda s: max_drawdown(s))

    # 4) save results
    outdir = DATA_DIR
    cum.iloc[-1].to_csv(outdir / "sector_mcap_cum_last.csv", header=["cumulative_return"])
    ann_vol.to_csv(outdir / "sector_mcap_ann_vol.csv", header=["annual_vol"])
    sharpe.to_csv(outdir / "sector_mcap_sharpe.csv", header=["sharpe"])
    mdds.to_csv(outdir / "sector_mcap_max_drawdown.csv", header=["max_drawdown"])
    sector_mcap_returns.to_parquet(outdir / "sector_mcap_daily_returns.parquet")

    # 5) Print a concise summary
    print("Top 5 sectors by cumulative return (market-cap weighted):")
    print(cum.iloc[-1].sort_values(ascending=False).head(5).to_string())
    print("\nBottom 5 sectors by cumulative return:")
    print(cum.iloc[-1].sort_values(ascending=True).head(5).to_string())
    print("\nTop 5 sectors by Sharpe (RF=6%):")
    print(sharpe.sort_values(ascending=False).head(5).to_string())
    print("\nMax drawdown (worst 5):")
    print(mdds.sort_values().head(5).to_string())

    print("\nSaved files to data/: sector_mcap_daily_returns.parquet, sector_mcap_cum_last.csv, sector_mcap_ann_vol.csv, sector_mcap_sharpe.csv, sector_mcap_max_drawdown.csv")

if __name__ == "__main__":
    main()
