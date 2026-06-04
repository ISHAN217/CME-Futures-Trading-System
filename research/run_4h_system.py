"""
run_4h_system.py — 4H resolution system experiment.

Tests on CL (Crude Oil) 2019-2026 — full 7-year history from Databento.
Also runs on ES/NQ 2024-2026 (limited yfinance 4H data) as a proof-of-concept.

Experiment structure:
  1. CL Long-only (regime-gated): baseline 4H system
  2. CL Long + Short (shorts only in BEAR_TREND)
  3. CL with ADX threshold tuning
  4. CL scaling grid (vol targets)
  5. ES/NQ 4H (2024-2026, PoC only)

Compare against:
  - CL daily system from run_cl_zn_diagnostic.py context
  - Key metric: more trades, lower drawdown per trade, faster exits
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from features_4h import compute_4h_features
from backtest_4h import run_backtest_4h

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data prep
# ─────────────────────────────────────────────────────────────────────────────

def _load_cl_data():
    """Load CL 4H bars and daily data from Databento parquets."""
    # 4H
    h4_all = pd.read_parquet("/tmp/cl_zn_4h.parquet")
    cl_4h  = h4_all[h4_all["root"] == "CL"].copy()
    cl_4h  = cl_4h.rename(columns={"ts_4h": "datetime"})
    cl_4h["symbol"] = "CL"
    cl_4h["date"]   = pd.to_datetime(cl_4h["date"])

    # Daily
    daily_all = pd.read_parquet("/tmp/cl_zn_daily.parquet")
    cl_daily  = daily_all[daily_all["root"] == "CL"].copy()
    cl_daily["symbol"] = "CL"
    cl_daily["date"]   = pd.to_datetime(cl_daily["date"])

    logger.info(
        "CL 4H: %d bars  %s → %s  (%d trading days)",
        len(cl_4h),
        cl_4h["datetime"].min(), cl_4h["datetime"].max(),
        cl_daily["date"].nunique(),
    )
    return cl_4h, cl_daily


def _load_es_nq_data():
    """Load ES/NQ 4H bars (yfinance, 2024-2026 only)."""
    es = pd.read_parquet("/tmp/ES_4h_yf.parquet")
    nq = pd.read_parquet("/tmp/NQ_4h_yf.parquet")
    df_4h = pd.concat([es, nq], ignore_index=True)
    df_4h["date"] = pd.to_datetime(df_4h["date"])
    logger.info("ES/NQ 4H: %d bars  %s → %s", len(df_4h),
                df_4h["datetime"].min(), df_4h["datetime"].max())
    return df_4h


# ─────────────────────────────────────────────────────────────────────────────
# Report helpers
# ─────────────────────────────────────────────────────────────────────────────

def _report_result(label: str, result: dict, base_stats: dict | None = None):
    s   = result["stats"]
    tl  = pd.DataFrame(result["trade_log"]) if result["trade_log"] else pd.DataFrame()
    pr  = pd.Series(result["portfolio_returns"])
    pr.index = pd.to_datetime(pr.index)

    ann = pr.groupby(pr.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    n_years = (pr.index.max() - pr.index.min()).days / 365.25 if len(pr) > 1 else 1
    trades_pw = s["total_trades"] / (n_years * 52) if n_years > 0 else 0

    d_sh  = s["sharpe_ratio"] - base_stats["sharpe_ratio"] if base_stats else 0
    d_ann = s["annualized_return_pct"] - base_stats["annualized_return_pct"] if base_stats else 0

    final_val = 25000 * (1 + s["total_return_pct"] / 100)
    logger.info(
        "%-38s | Sh=%+.3f(%+.3f)  Ann=%+.2f%%(%+.2f%%)  DD=%.2f%%  "
        "Trades=%d(%.2f/wk)  WR=%.1f%%  $25k->$%s",
        label,
        s["sharpe_ratio"], d_sh,
        s["annualized_return_pct"], d_ann,
        s["max_drawdown_pct"],
        s["total_trades"], trades_pw,
        s["overall_win_rate"] * 100,
        f"{final_val:,.0f}",
    )

    if not tl.empty:
        by_dir = tl.groupby("direction").agg(
            n=("pnl", "count"),
            wr=("pnl", lambda x: (x > 0).mean()),
            avg=("pnl", "mean"),
        )
        for d, row in by_dir.iterrows():
            logger.info(
                "  %-10s  n=%d  WR=%.1f%%  avg=%.3f%%",
                d, int(row["n"]), row["wr"] * 100, row["avg"] * 100,
            )

        exit_cts = tl["exit_reason"].value_counts()
        logger.info("  Exits: %s", "  ".join(f"{k}={v}" for k, v in exit_cts.items()))

        avg_hold = tl["bars_held"].mean()
        logger.info("  Avg hold: %.1f bars (%.1f calendar days)", avg_hold, avg_hold / 6)

    return s, ann


def _print_table(results: list[tuple]):
    """Print comparison table."""
    print()
    print("=" * 90)
    print("4H SYSTEM EXPERIMENT — RESULTS")
    print("=" * 90)
    hdr = (f"{'label':<36} {'sharpe':>7} {'d_sh':>7} {'annual':>8} {'d_ann':>7} "
           f"{'max_dd':>8} {'trades':>7} {'wr':>6} {'$25k→':>12}")
    print(hdr)
    print("-" * 90)
    base_sh  = results[0][1]["sharpe_ratio"]
    base_ann = results[0][1]["annualized_return_pct"]
    for lbl, s, ann in results:
        d_sh  = s["sharpe_ratio"]         - base_sh
        d_ann = s["annualized_return_pct"] - base_ann
        final = 25000 * (1 + s["total_return_pct"] / 100)
        print(
            f"{lbl:<36} {s['sharpe_ratio']:>7.3f} {d_sh:>+7.3f} "
            f"{s['annualized_return_pct']:>+7.2f}% {d_ann:>+6.2f}% "
            f"{s['max_drawdown_pct']:>7.2f}% {s['total_trades']:>7d} "
            f"{s['overall_win_rate']*100:>5.1f}%  ${final:>10,.0f}"
        )

    print()
    print("=" * 90)
    print("ANNUAL RETURNS BY YEAR")
    print("=" * 90)
    years = sorted(results[0][2].index)
    print(f"{'label':<36} " + "  ".join(f"yr_{y}" for y in years))
    for lbl, s, ann in results:
        row = f"{lbl:<36} " + "  ".join(f"{ann.get(y, 0):+6.1f}%" for y in years)
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def _bt_with_signal(df: pd.DataFrame, signal_col: str, **kwargs) -> dict:
    """Run backtest substituting a different signal column for hb_signal_long."""
    df2 = df.copy()
    if signal_col != "hb_signal_long":
        df2["hb_signal_long"] = df2[signal_col]
    return run_backtest_4h(df2, **kwargs)


def run_cl_experiments():
    logger.info("Loading CL data...")
    cl_4h, cl_daily = _load_cl_data()

    logger.info("Computing 4H features...")
    df = compute_4h_features(cl_4h, cl_daily)

    # Signal distribution
    logger.info(
        "Signals — BRK_L=%d  BRK_S=%d  MOM_L=%d  MOM_S=%d  Liquid=%d/%d",
        df["hb_signal_long"].sum(), df["hb_signal_short"].sum(),
        df["hb_signal_mom_long"].sum(), df["hb_signal_mom_short"].sum(),
        df["hb_is_liquid"].sum(), len(df),
    )
    logger.info("Daily regime dist: %s", df.groupby("daily_regime").size().to_dict())

    results = []

    # Common base: BULL_TREND only for longs (stops losers from BEAR_RANGE/NEUTRAL)
    # NO shorts on CL at 4H — bear rallies make short breakdowns unprofitable
    _BASE = dict(
        ALLOW_LONG_REGIMES   = {"BULL_TREND"},
        ALLOW_SHORT_REGIMES  = set(),
        MIN_ADX_LONG         = 18.0,
    )

    # ── EXP 1: Breakout, hold 6 bars (1 day), standard stop ──────────────
    logger.info("\n--- EXP 1: Breakout, 6-bar hold (1 day), BULL_TREND only ---")
    r1 = run_backtest_4h(df, **_BASE,
                         MAX_BARS_HELD=6, STOP_ATR_MULT=2.0, TRAIL_ATR_MULT=2.5)
    s1, ann1 = _report_result("CL_BRK_6BAR", r1)
    results.append(("CL_BRK_6BAR", s1, ann1))

    # ── EXP 2: Breakout, hold 12 bars (2 days) ────────────────────────────
    logger.info("\n--- EXP 2: Breakout, 12-bar hold (2 days) ---")
    r2 = run_backtest_4h(df, **_BASE,
                         MAX_BARS_HELD=12, STOP_ATR_MULT=2.0, TRAIL_ATR_MULT=2.5)
    s2, ann2 = _report_result("CL_BRK_12BAR", r2, s1)
    results.append(("CL_BRK_12BAR", s2, ann2))

    # ── EXP 3: Breakout, tight stop (1.5x), 6 bars ───────────────────────
    logger.info("\n--- EXP 3: Breakout, tight stop 1.5x ATR, 6-bar hold ---")
    r3 = run_backtest_4h(df, **_BASE,
                         MAX_BARS_HELD=6, STOP_ATR_MULT=1.5, TRAIL_ATR_MULT=2.0)
    s3, ann3 = _report_result("CL_BRK_TIGHT_STOP", r3, s1)
    results.append(("CL_BRK_TIGHT_STOP", s3, ann3))

    # ── EXP 4: Momentum signal, 6-bar hold ───────────────────────────────
    logger.info("\n--- EXP 4: Momentum (EMA-align), 6-bar hold ---")
    df4 = df.copy()
    df4["hb_signal_long"] = df4["hb_signal_mom_long"]
    r4 = run_backtest_4h(df4, **_BASE,
                         MAX_BARS_HELD=6, STOP_ATR_MULT=2.0, TRAIL_ATR_MULT=2.5)
    s4, ann4 = _report_result("CL_MOM_6BAR", r4, s1)
    results.append(("CL_MOM_6BAR", s4, ann4))

    # ── EXP 5: Breakout + stronger vol confirmation ───────────────────────
    logger.info("\n--- EXP 5: Breakout + vol_zscore > 1.0 (high-conviction only) ---")
    df5 = df.copy()
    df5["hb_signal_long"] = (df5["hb_signal_long"] == 1) & (df5["hb_vol_zscore"] > 1.0)
    r5 = run_backtest_4h(df5, **_BASE,
                         MAX_BARS_HELD=6, STOP_ATR_MULT=2.0, TRAIL_ATR_MULT=2.5)
    s5, ann5 = _report_result("CL_BRK_HIGH_VOL", r5, s1)
    results.append(("CL_BRK_HIGH_VOL", s5, ann5))

    # ── EXP 6: Breakout + ADX > 25 (stronger trend required) ─────────────
    logger.info("\n--- EXP 6: Breakout + ADX > 25, 6-bar hold ---")
    _base_adx25 = {**_BASE, "MIN_ADX_LONG": 25.0}
    r6 = run_backtest_4h(df, **_base_adx25,
                         MAX_BARS_HELD=6, STOP_ATR_MULT=2.0, TRAIL_ATR_MULT=2.5)
    s6, ann6 = _report_result("CL_BRK_ADX25", r6, s1)
    results.append(("CL_BRK_ADX25", s6, ann6))

    _print_table(results)
    return results, df


def run_es_nq_poc():
    """ES/NQ 4H proof of concept (2024-2026 only)."""
    logger.info("\n" + "=" * 60)
    logger.info("ES/NQ 4H PROOF OF CONCEPT (2024-2026 only)")
    logger.info("=" * 60)

    df_4h = _load_es_nq_data()
    df = compute_4h_features(df_4h, df_daily=None)   # no daily context (raw 4H only)

    # Without daily context: use momentum signal, no regime filter
    df["hb_signal_long"]  = df["hb_signal_mom_long"]
    df["hb_signal_short"] = df["hb_signal_mom_short"]

    logger.info("ES/NQ signals — MOM_L=%d  MOM_S=%d  Liquid=%d/%d",
                df["hb_signal_long"].sum(), df["hb_signal_short"].sum(),
                df["hb_is_liquid"].sum(), len(df))

    r = run_backtest_4h(
        df,
        ALLOW_LONG_REGIMES  = {"BULL_TREND", "BULL_RANGE", "NEUTRAL", "BEAR_RANGE"},
        ALLOW_SHORT_REGIMES = set(),
        MIN_ADX_LONG        = 0.0,      # disabled (no daily ADX without daily data)
        MIN_TREND_SCORE_LONG = -1.0,    # disabled
        MAX_BARS_HELD        = 12,      # 2 days for ES (6 bars/day)
    )
    s, ann = _report_result("ES_NQ_4H_POC (2024-2026)", r)

    print()
    print("=" * 60)
    print("ES/NQ 4H POC RESULTS (2024-2026 only — NOT a backtest, only 1.5 years)")
    print(f"  Sharpe: {s['sharpe_ratio']:.3f}")
    print(f"  Annual: {s['annualized_return_pct']:+.2f}%")
    print(f"  MaxDD:  {s['max_drawdown_pct']:.2f}%")
    print(f"  Trades: {s['total_trades']}  WR: {s['overall_win_rate']*100:.1f}%")
    print(f"  $25k→   ${25000 * (1 + s['total_return_pct']/100):,.0f}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("4H RESOLUTION SYSTEM EXPERIMENT")
    logger.info("=" * 60)

    run_cl_experiments()
    run_es_nq_poc()

    print()
    print("=" * 60)
    print("CONTEXT: CL daily system baseline (from run_cl_zn_diagnostic)")
    print("  CL solo 4H daily: Sh ~1.0  Ann ~5-6%  MaxDD ~8%")
    print("  (4H should show higher trade count, lower per-trade drawdown)")
    print("=" * 60)
