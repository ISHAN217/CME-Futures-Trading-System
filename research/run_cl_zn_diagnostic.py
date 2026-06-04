"""
run_cl_zn_diagnostic.py — Port the ES/NQ pipeline to CL (Crude Oil) and ZN (10Y Treasury).

Strategy:
  - CL mapped to "ES" slot, ZN mapped to "NQ" slot in data files
  - Full pipeline runs unchanged (all ES/NQ hardcoding still works)
  - STAT_ARB will compute CL/ZN spread — interesting risk-off signal
  - Thresholds are ES/NQ-tuned; report raw results then diagnose gaps

Key differences to watch:
  - CL avg daily range ~3.4%  (ES ~1%)   → stops too tight, vol sizing auto-adjusts
  - ZN avg daily range ~0.46% (ES ~1%)   → stops too wide (almost never fire)
  - CL is negatively correlated with ZN in risk-off events (different from ES/NQ)
  - Monthly rolls for CL (vs quarterly) — already handled by front-month volume selection
"""

import logging
import sys
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cl_zn_diag")

sys.path.insert(0, str(Path(__file__).parent))
import config

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Prepare data directory
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR_CL_ZN = "/tmp/cl_zn_data"

def prepare_data():
    """
    Convert CL/ZN daily+4H parquets into the format data_loader.py expects.
    CL → "ES_daily.csv" / "ES_4h.csv"
    ZN → "NQ_daily.csv" / "NQ_4h.csv"
    """
    Path(DATA_DIR_CL_ZN).mkdir(exist_ok=True)

    # ── Daily ─────────────────────────────────────────────────────────────────
    daily = pd.read_parquet("/tmp/cl_zn_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])

    # Drop weekend dates (pipeline expects trading days only)
    daily = daily[daily["date"].dt.dayofweek < 5].copy()

    # Sanity: drop rows where close is non-positive (shouldn't happen with CLM0 roll)
    daily = daily[daily["close"] > 0].copy()

    for root, sym_name in [("CL", "ES"), ("ZN", "NQ")]:
        sub = daily[daily["root"] == root][
            ["date", "symbol", "open", "high", "low", "close", "volume"]
        ].copy()
        sub["symbol"] = sym_name          # relabel to ES/NQ for pipeline compat
        sub = sub.sort_values("date").reset_index(drop=True)
        out_path = f"{DATA_DIR_CL_ZN}/{sym_name}_daily.csv"
        sub.to_csv(out_path, index=False)
        log.info("Written %s: %d rows  %s→%s  close=[%.2f, %.2f]",
                 out_path, len(sub),
                 sub["date"].min().date(), sub["date"].max().date(),
                 sub["close"].min(), sub["close"].max())

    # ── 4H ───────────────────────────────────────────────────────────────────
    h4 = pd.read_parquet("/tmp/cl_zn_4h.parquet")
    h4["ts_4h"] = pd.to_datetime(h4["ts_4h"], utc=True)
    h4["date"] = pd.to_datetime(h4["date"])

    # Drop dates that fall on weekends at the daily level
    h4 = h4[h4["date"].dt.dayofweek < 5].copy()
    h4 = h4[h4["close"] > 0].copy()

    for root, sym_name in [("CL", "ES"), ("ZN", "NQ")]:
        sub = h4[h4["root"] == root][
            ["ts_4h", "date", "open", "high", "low", "close", "volume"]
        ].copy()
        sub.rename(columns={"ts_4h": "datetime"}, inplace=True)
        sub["symbol"] = sym_name
        sub = sub.sort_values("datetime").reset_index(drop=True)
        out_path = f"{DATA_DIR_CL_ZN}/{sym_name}_4h.csv"
        sub.to_csv(out_path, index=False)
        log.info("Written %s: %d bars", out_path, len(sub))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Run pipeline with patched config
# ─────────────────────────────────────────────────────────────────────────────

def override_config(**kwargs):
    for k, v in kwargs.items():
        setattr(config, k, v)

def run_cl_zn_pipeline():
    """Run the full main.py pipeline with CL/ZN data."""
    # Patch config to point at CL/ZN data
    override_config(
        DATA_DIR           = DATA_DIR_CL_ZN,
        SYMBOLS            = ["ES=F", "NQ=F"],   # keys still ES=F / NQ=F (data_loader uses SYMBOL_NAMES)
        SYMBOL_NAMES       = {"ES=F": "ES", "NQ=F": "NQ"},
        # Disable ablation flags so strategies run
        DISABLE_MEAN_REVERT = True,
        DISABLE_PULLBACK    = True,
        # Keep STAT_ARB active — will compute CL/ZN spread (valid risk-off signal)
    )

    from data_loader        import load_daily, load_4h
    from features           import compute_features
    from momentum_engine    import compute_momentum
    from weekly_bias        import compute_weekly_bias
    from regime_engine      import compute_regime
    from volatility_engine  import compute_volatility_features
    from intraday_4h_engine import compute_4h_features, aggregate_4h_to_daily, merge_4h_into_daily
    from mean_reversion     import compute_mean_reversion
    from stat_arb           import compute_stat_arb
    from sentiment_layer    import compute_sentiment
    from signal_engine      import compute_signals
    from signal_quality_engine import compute_signal_quality
    from trend_continuation_edge_engine import compute_trend_continuation_edge
    from backtest           import run_backtest

    log.info("Step 1 — Loading CL/ZN daily data")
    df = load_daily()
    log.info("  Loaded %d rows  symbols=%s", len(df), df["symbol"].unique().tolist())

    log.info("Step 2 — Loading CL/ZN 4H data")
    df_4h_raw = load_4h()

    log.info("Step 3 — Daily features")
    df = compute_features(df)

    log.info("Step 4 — Momentum")
    df = compute_momentum(df)

    log.info("Step 5 — Weekly bias")
    df = compute_weekly_bias(df)

    log.info("Step 6 — Regime")
    df = compute_regime(df)

    log.info("Step 7 — Volatility features")
    df = compute_volatility_features(df)

    # CWT disabled
    _cwt_neutral = {
        "cwt_chop_label": "CWT_UNCLEAR", "cwt_trade_allowed": 1,
        "cwt_size_multiplier": 1.0, "cwt_confidence": 0.0,
        "cwt_total_energy": 0.0, "cwt_entropy": 0.5, "cwt_energy_z": 0.0,
        "cwt_compression_flag": 0, "cwt_expansion_flag": 0,
        "cwt_high_energy_ratio": 1/3, "cwt_mid_energy_ratio": 1/3, "cwt_slow_energy_ratio": 1/3,
    }
    for col, val in _cwt_neutral.items():
        if col not in df.columns:
            df[col] = val

    log.info("Step 8 — 4H features")
    if df_4h_raw is not None:
        df_4h_feat  = compute_4h_features(df_4h_raw)
        df_4h_daily = aggregate_4h_to_daily(df_4h_feat)
    else:
        df_4h_daily = None
    df = merge_4h_into_daily(df, df_4h_daily)

    log.info("Step 9 — Mean reversion")
    df = compute_mean_reversion(df)

    log.info("Step 10 — Stat arb (CL/ZN spread)")
    df = compute_stat_arb(df)

    log.info("Step 11 — Sentiment")
    df = compute_sentiment(df)

    log.info("Step 12 — Signal engine")
    df = compute_signals(df)

    log.info("Step 12b — Signal quality")
    df = compute_signal_quality(df)

    log.info("Step 12c — Trend continuation edge")
    df = compute_trend_continuation_edge(df)

    log.info("Step 13 — Backtest")
    result = run_backtest(df)

    return result, df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Diagnostic report
# ─────────────────────────────────────────────────────────────────────────────

def report(result, df):
    stats = result["stats"]
    tl    = result["trade_log"]
    pr    = result["portfolio_returns"]

    print()
    print("=" * 72)
    print("CL / ZN PIPELINE DIAGNOSTIC — RAW (ES/NQ thresholds, no retuning)")
    print("=" * 72)
    print(f"  Sharpe     : {stats.get('sharpe', float('nan')):.3f}")
    print(f"  Annual     : {stats.get('annual_return', float('nan'))*100:+.2f}%")
    print(f"  MaxDD      : {stats.get('max_drawdown', float('nan'))*100:.2f}%")
    print(f"  Win Rate   : {stats.get('win_rate', float('nan'))*100:.1f}%")
    print(f"  Trades     : {int(stats.get('num_trades', 0))}")
    print(f"  $25k→      : ${25000 * (1 + stats.get('total_return', 0)):,.0f}")
    print()

    # Year-by-year
    pr_s = pd.Series(pr).sort_index()
    pr_s.index = pd.to_datetime(pr_s.index)
    ann = pr_s.groupby(pr_s.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    print("  Annual by year:")
    for yr, ret in ann.items():
        print(f"    {yr}: {ret:+.2f}%")
    print()

    # Strategy breakdown
    if tl is not None and len(tl) > 0:
        tl_df = pd.DataFrame(tl)
        print("  Strategy breakdown:")
        for strat, grp in tl_df.groupby("strategy"):
            wr = (grp["return"] > 0).mean()
            avg_r = grp["return"].mean() * 100
            tot_r = grp["return"].sum() * 100
            print(f"    {strat:<28} n={len(grp):3d}  WR={wr*100:.1f}%  "
                  f"avg={avg_r:+.3f}%  tot={tot_r:+.2f}%")
    print()

    # Regime distribution
    if "portfolio_regime" in df.columns:
        reg = df[df["symbol"] == "ES"]["portfolio_regime"].value_counts()
        print("  Regime distribution (CL days):")
        for r, c in reg.items():
            print(f"    {r:<15} {c:4d} days  ({c/len(reg.index.unique())*100:.0f}%)")
    print()

    # Volatility check
    cl_daily = df[df["symbol"] == "ES"].copy()
    if "atr_pct" in cl_daily.columns:
        print(f"  CL avg ATR%  : {cl_daily['atr_pct'].mean()*100:.2f}%  (ES target: ~1.0%)")
    if "vol_scalar" in cl_daily.columns:
        print(f"  CL avg vol_scalar: {cl_daily['vol_scalar'].mean():.3f}  (ES avg ~0.25-0.30)")
    print()

    # Key diagnosis flags
    print("  DIAGNOSIS FLAGS:")
    if tl is not None and len(tl) > 0:
        tl_df = pd.DataFrame(tl)
        stops = tl_df[tl_df.get("exit_reason", pd.Series(dtype=str)).str.contains("STOP", na=False)] if "exit_reason" in tl_df.columns else pd.DataFrame()
        if "exit_reason" in tl_df.columns:
            stop_pct = (tl_df["exit_reason"].str.contains("STOP", na=False)).mean()
            print(f"    Stop-loss exit rate : {stop_pct*100:.1f}%  (ES: ~8-12% — CL may be higher)")
        avg_hold = tl_df.get("days_held", pd.Series([0])).mean() if "days_held" in tl_df.columns else 0
        print(f"    Avg hold (days)     : {avg_hold:.1f}")
    print()
    print("=" * 72)

    # Save pipeline df for further analysis
    df.to_parquet("/tmp/cl_zn_pipeline_df.parquet", index=False)
    log.info("Saved pipeline df → /tmp/cl_zn_pipeline_df.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== CL/ZN Diagnostic ===")

    log.info("Preparing CL/ZN data files...")
    prepare_data()

    log.info("Running pipeline...")
    result, df = run_cl_zn_pipeline()

    report(result, df)
