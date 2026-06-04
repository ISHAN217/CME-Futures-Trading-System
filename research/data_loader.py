"""
data_loader.py — Data loading for the competition system.

By default reads from ../cme_execution_system/data/ to reuse cached downloads.
Use --fetch to re-download fresh data.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import config

logger = logging.getLogger(__name__)


def fetch_and_save(start: str = None, end: str = None) -> None:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    start = start or config.START_DATE

    for ticker in config.SYMBOLS:
        name = config.SYMBOL_NAMES[ticker]
        _fetch_daily(yf, ticker, name, start, end)
        _fetch_intraday(yf, ticker, name)


def load_daily() -> pd.DataFrame:
    dfs = []
    for ticker in config.SYMBOLS:
        name = config.SYMBOL_NAMES[ticker]
        path = Path(config.DATA_DIR) / f"{name}_daily.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Data not found: {path}\n"
                f"Run: python main.py --fetch   (or point DATA_DIR to existing data)"
            )
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True).sort_values(["date", "symbol"])
    logger.info("Loaded daily: %d rows | %s → %s",
                len(df), df["date"].min().date(), df["date"].max().date())
    return df.reset_index(drop=True)


def load_4h() -> Optional[pd.DataFrame]:
    dfs = []
    for ticker in config.SYMBOLS:
        name = config.SYMBOL_NAMES[ticker]
        path = Path(config.DATA_DIR) / f"{name}_4h.csv"
        if not path.exists():
            logger.warning("4H data missing for %s — daily-only signals will be used", name)
            return None
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["date"] = pd.to_datetime(df["date"])
        dfs.append(df)

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True).sort_values(["datetime", "symbol"])
    n_days = df["date"].nunique()
    logger.info("Loaded 4H: %d bars / %d days (yfinance ~730d limit)", len(df), n_days)
    return df.reset_index(drop=True)


def _fetch_daily(yf, ticker, name, start, end):
    try:
        tkr = yf.Ticker(ticker)
        df  = tkr.history(start=start, end=end, interval="1d", auto_adjust=True)
        if df.empty:
            logger.error("No data: %s", name); return
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c), None)
        df["date"] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
        df["symbol"] = name
        out = df[["date", "symbol", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        out.to_csv(Path(config.DATA_DIR) / f"{name}_daily.csv", index=False)
        logger.info("Saved %d daily bars for %s", len(out), name)
    except Exception as e:
        logger.error("Daily fetch failed %s: %s", name, e)


def _fetch_intraday(yf, ticker, name):
    try:
        tkr   = yf.Ticker(ticker)
        df_1h = tkr.history(period=config.INTRADAY_PERIOD,
                             interval=config.INTRADAY_INTERVAL, auto_adjust=True)
        if df_1h.empty:
            logger.warning("No intraday data for %s", name); return
        df_1h = df_1h.reset_index()
        df_1h.columns = [c.lower().replace(" ", "_") for c in df_1h.columns]
        dt_col = next((c for c in df_1h.columns if "datetime" in c or c == "date"), None)
        df_1h["datetime"] = pd.to_datetime(df_1h[dt_col], utc=True)
        df_1h = df_1h.set_index("datetime").sort_index()
        df_4h = _resample_4h(df_1h[["open", "high", "low", "close", "volume"]])
        df_4h["symbol"] = name
        df_4h["date"] = df_4h["datetime"].dt.tz_localize(None).dt.normalize()
        df_4h.to_csv(Path(config.DATA_DIR) / f"{name}_4h.csv", index=False)
        logger.info("Saved %d 4H bars for %s", len(df_4h), name)
    except Exception as e:
        logger.warning("Intraday fetch failed %s: %s", name, e)


def _resample_4h(df_1h):
    agg = df_1h.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    counts = df_1h["close"].resample("4h").count()
    agg = agg[counts >= 2].dropna(subset=["close"]).reset_index()
    agg.columns = ["datetime", "open", "high", "low", "close", "volume"]
    return agg
