# data_pipeline_india.py
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# CONFIG
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CONSTITUENTS_CSV = DATA_DIR / "nifty50_constituents.csv"
MARKET_CAP_CSV = DATA_DIR / "market_caps.csv"
ERROR_LOG = DATA_DIR / "download_errors.log"

START_DATE = (datetime.today() - timedelta(days=365*5)).strftime("%Y-%m-%d")  # 5 years
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Logging
logging.basicConfig(filename=str(ERROR_LOG), level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def fetch_nifty50_from_wiki():
    """Best-effort fetch of NIFTY50 constituents from Wikipedia (fallback)."""
    try:
        url_wiki = "https://en.wikipedia.org/wiki/NIFTY_50"
        tables = pd.read_html(url_wiki)
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if any("symbol" in c or "company" in c for c in cols):
                df = t.copy()
                df.columns = [str(c).strip() for c in df.columns]
                symbol_col = next((c for c in df.columns if 'symbol' in c.lower() or 'ticker' in c.lower()), None)
                name_col = next((c for c in df.columns if 'company' in c.lower() or 'name' in c.lower() or 'security' in c.lower()), None)
                sector_col = next((c for c in df.columns if 'sector' in c.lower()), None)
                if symbol_col is None or name_col is None:
                    continue
                df2 = df[[symbol_col, name_col]].copy()
                df2 = df2.rename(columns={symbol_col: "Ticker", name_col: "Company"})
                df2['Sector'] = df.get(sector_col, "")
                df2['Ticker'] = df2['Ticker'].astype(str).str.strip()
                df2 = df2.drop_duplicates(subset=['Ticker'])
                return df2[['Ticker', 'Company', 'Sector']].reset_index(drop=True)
    except Exception as e:
        logging.info(f"Wiki fetch failed: {e}")
    return None


def load_constituents():
    """Robust load: prefer local CSV if valid, else try wiki fallback."""
    if CONSTITUENTS_CSV.exists() and CONSTITUENTS_CSV.stat().st_size > 10:
        try:
            df = pd.read_csv(CONSTITUENTS_CSV)
            cols = [c.strip().lower() for c in df.columns.astype(str)]
            if "ticker" in cols and ("company" in cols or "security" in cols):
                # Normalize columns
                df = df.rename(columns={c: c.strip() for c in df.columns})
                if 'Ticker' not in df.columns:
                    for c in df.columns:
                        if c.strip().lower() in ('ticker', 'symbol'):
                            df = df.rename(columns={c: 'Ticker'})
                            break
                if 'Company' not in df.columns:
                    for c in df.columns:
                        if c.strip().lower() in ('company','security','name'):
                            df = df.rename(columns={c: 'Company'})
                            break
                if 'Sector' not in df.columns:
                    df['Sector'] = ""
                df = df[['Ticker','Company','Sector']].copy()
                df['Ticker'] = df['Ticker'].astype(str).str.strip()
                print(f"Loaded constituents from {CONSTITUENTS_CSV}")
                return df.reset_index(drop=True)
            else:
                print("Local CSV missing required columns; trying online fallback.")
        except Exception as e:
            print(f"Error reading {CONSTITUENTS_CSV}: {e}; trying online fallback.")

    print("Attempting to fetch constituents from Wikipedia (fallback).")
    df_online = fetch_nifty50_from_wiki()
    if df_online is not None and len(df_online) >= 40:
        df_online.to_csv(CONSTITUENTS_CSV, index=False)
        print(f"Fetched constituents online and saved to {CONSTITUENTS_CSV}")
        return df_online
    raise FileNotFoundError(
        f"Could not load valid constituents CSV from {CONSTITUENTS_CSV}. "
        "Please create a CSV there with columns: Ticker,Company,Sector"
    )


def download_batch(tickers, start=START_DATE, end=END_DATE):
    """Download batch from yfinance; returns dict yahoo_ticker->DataFrame."""
    out = {}
    if not tickers:
        return out
    try:
        raw = yf.download(tickers, start=start, end=end, group_by='ticker', threads=True, progress=False, auto_adjust=False)
    except Exception as e:
        logging.info(f"yfinance batch download failed: {e}")
        raw = None

    if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                df = raw[t].copy()
                if not df.empty:
                    out[t] = df
            except Exception:
                continue
    else:
        for t in tickers:
            try:
                df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
                if df is not None and not df.empty:
                    out[t] = df
            except Exception as e:
                logging.info(f"failed single download {t}: {e}")
    return out


def fetch_prices_for_constituents(const_df):
    """Download price data and market caps; save per-base-ticker parquet files."""
    price_dfs = {}
    market_caps = []
    errors = []
    base_tickers = const_df['Ticker'].astype(str).str.strip().tolist()
    yahoo_ns = [f"{t}.NS" for t in base_tickers]
    print("Attempting bulk download for NSE (.NS) tickers via yfinance...")
    ns_data = download_batch(yahoo_ns)
    for base in base_tickers:
        yahoo_ns_t = f"{base}.NS"
        yahoo_bo_t = f"{base}.BO"
        df = None
        used = None
        if yahoo_ns_t in ns_data and not ns_data[yahoo_ns_t].empty:
            df = ns_data[yahoo_ns_t]
            used = yahoo_ns_t
        else:
            try:
                df_bo = yf.download(yahoo_bo_t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
                if df_bo is not None and not df_bo.empty:
                    df = df_bo
                    used = yahoo_bo_t
            except Exception as e:
                logging.info(f"failed fetch {yahoo_bo_t}: {e}")

        if df is None or df.empty:
            logging.info(f"No data for {base} (.NS/.BO).")
            errors.append(base)
            continue

        out_path = DATA_DIR / f"{base}.parquet"
        df.index = pd.to_datetime(df.index)
        df.to_parquet(out_path)
        price_dfs[base] = df

        try:
            info = yf.Ticker(used).info
            mcap = info.get("marketCap", None)
        except Exception as e:
            logging.info(f"Failed market cap for {used}: {e}")
            mcap = None

        market_caps.append({
            "Ticker": base,
            "yahoo_ticker": used,
            "market_cap": mcap,
            "market_cap_date": datetime.today().strftime("%Y-%m-%d")
        })
        time.sleep(0.2)

    pd.DataFrame(market_caps).to_csv(DATA_DIR / MARKET_CAP_CSV, index=False)
    if errors:
        with open(DATA_DIR / ERROR_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()} - failed tickers: {errors}\n")
        print(f"Some tickers failed: {errors} (see {ERROR_LOG})")
    return price_dfs


def main():
    print("=== Running NIFTY50 data pipeline ===")
    constituents = load_constituents()
    # Ensure yahoo_ticker columns saved
    if 'yahoo_ticker' not in constituents.columns:
        constituents['yahoo_ticker'] = constituents['Ticker'].astype(str).str.strip() + ".NS"
        constituents.to_csv(CONSTITUENTS_CSV, index=False)

    price_dfs = fetch_prices_for_constituents(constituents)
    print(f"Downloaded price files for {len(price_dfs)} tickers.")
    # Save summary of date ranges
    ranges = []
    for t, df in price_dfs.items():
        ranges.append({"Ticker": t, "start": str(df.index.min().date()), "end": str(df.index.max().date()), "rows": len(df)})
    if ranges:
        pd.DataFrame(ranges).to_csv(DATA_DIR / "download_date_ranges.csv", index=False)
        print("Saved download date ranges.")

    print("Pipeline complete. Parquet price files are in the data/ folder.")


if __name__ == "__main__":
    main()
