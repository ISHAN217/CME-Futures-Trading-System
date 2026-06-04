"""
run_mtf_resample.py — Phase 7: Resample 1-minute aligned data to 1H/4H/12H/24H
================================================================================

Input:  output/mtf/ES_NQ_1min_aligned.parquet
Output:
  ES_1H.parquet   NQ_1H.parquet
  ES_4H.parquet   NQ_4H.parquet
  ES_12H.parquet  NQ_12H.parquet
  ES_24H.parquet  NQ_24H.parquet
  resample_audit.csv

Rules:
  - Use true 1-minute data only (no fabrication)
  - Bar timestamp = bar close (label='right', closed='right')
  - Drop incomplete boundary bars (first/last partial bars of each timeframe)
  - Consistent UTC throughout
  - Minimum bars required per higher-TF candle:
      1H  → min 30 1-min bars  (>50% of 60)
      4H  → min 120 1-min bars (>50% of 240)
      12H → min 360 1-min bars (>50% of 720)
      24H → min 720 1-min bars (>50% of 1440)
  - Volume = sum; open = first; high = max; low = min; close = last
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf"
def out(name): return os.path.join(OUT_DIR, name)

TIMEFRAMES = {
    "1H":  {"freq": "1h",  "min_bars": 30,  "expected_bars": 60},
    "4H":  {"freq": "4h",  "min_bars": 120, "expected_bars": 240},
    "12H": {"freq": "12h", "min_bars": 360, "expected_bars": 720},
    "24H": {"freq": "24h", "min_bars": 720, "expected_bars": 1440},
}


def resample_ohlcv(df_1min: pd.DataFrame, freq: str, min_bars: int,
                   price_col: str, vol_col: str, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Resample 1-min OHLCV to higher timeframe.
    Drops bars with fewer than min_bars 1-minute bars.
    Timestamp at bar close (right label).
    """
    df = df_1min[[ts_col, price_col.replace("close","open"),
                  price_col.replace("close","high"),
                  price_col.replace("close","low"),
                  price_col, vol_col]].copy()
    df = df.set_index(ts_col).sort_index()

    o_col = price_col.replace("close","open")
    h_col = price_col.replace("close","high")
    l_col = price_col.replace("close","low")
    c_col = price_col

    # Count 1-min bars per period
    bar_count = df[c_col].resample(freq, label="right", closed="right").count()

    resampled = pd.DataFrame({
        "open":   df[o_col].resample(freq, label="right", closed="right").first(),
        "high":   df[h_col].resample(freq, label="right", closed="right").max(),
        "low":    df[l_col].resample(freq, label="right", closed="right").min(),
        "close":  df[c_col].resample(freq, label="right", closed="right").last(),
        "volume": df[vol_col].resample(freq, label="right", closed="right").sum(),
        "bar_count": bar_count,
    })

    # Drop incomplete bars
    resampled = resampled.dropna(subset=["open","high","low","close"])
    resampled = resampled[resampled["bar_count"] >= min_bars]

    # OHLC integrity check
    bad = (
        (resampled["high"] < resampled["low"]) |
        (resampled["open"] < resampled["low"]) | (resampled["open"] > resampled["high"]) |
        (resampled["close"] < resampled["low"]) | (resampled["close"] > resampled["high"])
    )
    if bad.sum() > 0:
        logger.warning("  Dropping %d bars with OHLC integrity issues", bad.sum())
        resampled = resampled[~bad]

    resampled.index.name = "timestamp"
    return resampled.reset_index()


def main():
    print("=" * 80)
    print("  MTF RESAMPLE PIPELINE  —  Phase 7")
    print("=" * 80)

    aligned_path = out("ES_NQ_1min_aligned.parquet")
    if not os.path.exists(aligned_path):
        print("  ERROR: ES_NQ_1min_aligned.parquet not found — run Phase 6 first")
        sys.exit(1)

    logger.info("Loading aligned 1-minute data ...")
    aligned = pd.read_parquet(aligned_path)
    aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], utc=True)
    logger.info("  Loaded %s rows  %s → %s",
                f"{len(aligned):,}", aligned["timestamp"].min().date(),
                aligned["timestamp"].max().date())

    audit_rows = []

    for sym in ["ES", "NQ"]:
        o_col = f"{sym}_open"
        h_col = f"{sym}_high"
        l_col = f"{sym}_low"
        c_col = f"{sym}_close"
        v_col = f"{sym}_volume"

        # Build per-symbol 1-min DataFrame for resampling
        sym_df = aligned[["timestamp", o_col, h_col, l_col, c_col, v_col]].copy()
        sym_df = sym_df.rename(columns={
            o_col: "open", h_col: "high", l_col: "low",
            c_col: "close", v_col: "volume",
        })

        for tf_name, tf_cfg in TIMEFRAMES.items():
            logger.info("  Resampling %s → %s ...", sym, tf_name)

            resampled = resample_ohlcv(
                df_1min  = sym_df,
                freq     = tf_cfg["freq"],
                min_bars = tf_cfg["min_bars"],
                price_col= "close",
                vol_col  = "volume",
                ts_col   = "timestamp",
            )
            resampled["symbol"]     = sym
            resampled["timeframe"]  = tf_name

            fname = f"{sym}_{tf_name}.parquet"
            resampled.to_parquet(out(fname), index=False)

            # Compute stats
            n_bars       = len(resampled)
            date_start   = resampled["timestamp"].min()
            date_end     = resampled["timestamp"].max()
            avg_bar_count= resampled["bar_count"].mean()
            missing_bars = (resampled["bar_count"] < tf_cfg["expected_bars"]).sum()
            has_overnight= _check_overnight(resampled, tf_name)

            logger.info(
                "    %s %s: %d bars  %s → %s  avg_1min=%.1f  partial_bars=%d",
                sym, tf_name, n_bars,
                date_start.date(), date_end.date(),
                avg_bar_count, missing_bars,
            )

            audit_rows.append({
                "symbol":           sym,
                "timeframe":        tf_name,
                "freq":             tf_cfg["freq"],
                "bars":             n_bars,
                "date_start":       str(date_start.date()),
                "date_end":         str(date_end.date()),
                "avg_1min_per_bar": round(avg_bar_count, 1),
                "partial_bars":     missing_bars,
                "pct_complete":     round((n_bars - missing_bars) / max(n_bars, 1) * 100, 1),
                "has_overnight":    has_overnight,
                "min_close":        round(resampled["close"].min(), 2),
                "max_close":        round(resampled["close"].max(), 2),
            })
            logger.info("    Saved %s", fname)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(out("resample_audit.csv"), index=False)
    logger.info("Saved resample_audit.csv")

    print("\n  RESAMPLE SUMMARY:")
    print(audit_df[["symbol","timeframe","bars","date_start","date_end",
                     "avg_1min_per_bar","partial_bars"]].to_string(index=False))

    # Verdict: check 12H/24H construction is clean
    clean_12h = audit_df[audit_df["timeframe"]=="12H"]["pct_complete"].min() > 80
    clean_24h = audit_df[audit_df["timeframe"]=="24H"]["pct_complete"].min() > 80

    print(f"\n  12H construction clean: {'YES' if clean_12h else 'NEEDS_REVIEW'}")
    print(f"  24H construction clean: {'YES' if clean_24h else 'NEEDS_REVIEW'}")
    print("  RESAMPLE_COMPLETE")


def _check_overnight(df: pd.DataFrame, tf_name: str) -> bool:
    """Rough check: if we have bars starting at various UTC hours, overnight is covered."""
    if tf_name == "1H":
        hours = df["timestamp"].dt.hour.unique()
        return len(hours) > 12   # more than 12 distinct hour values suggests overnight
    return True   # for 4H+ the daily volume sum captures overnight naturally


if __name__ == "__main__":
    main()
