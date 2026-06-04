"""
run_tier1_experiment.py — Test Tier 1 cross-asset sentiment on ES/NQ system.

Builds on the best current stack (F1+F2+F3 fixes + H4 frequency signals).
Tests sentiment in isolation, then combined, then scaled.

Honest hold-out note: we test on the full 2019-2026 period,
then separately on 2024-2026 as a pseudo-out-of-sample check.
"""

import contextlib
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Re-use run_frequency.py infrastructure ────────────────────────────────────
from run_frequency import (
    build_base_df,
    build_fixed_stack,
    inject_h4_brk_chop,
    inject_h4_mom_chop_nq,
    override_config,
    _BASE_HOLD_DAYS,
    _BASE_MAX_WT_PER_SYM,
)
from tier1_sentiment_engine import compute_tier1_sentiment, apply_sentiment_to_entries
from backtest import run_backtest

SENTIMENT_PATH = "/tmp/sentiment_raw.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_bt(df, scale=1.0, label=""):
    max_wt = min(_BASE_MAX_WT_PER_SYM * scale, 1.20)
    with override_config(
        MAX_HOLD_DAYS         = _BASE_HOLD_DAYS,
        MAX_EXTENDED_HOLD_DAYS= 8,
        EXTEND_ALLOW_H4_NONE  = True,
        VOL_SCALE_MAX         = 0.28 * scale,
        MAX_WEIGHT_PER_SYMBOL = max_wt,
        STAT_ARB_NQ_WEIGHT    = min(0.20 * scale, 0.60),
        MAX_COMBINED_EXPOSURE = min(0.50 * scale, 2.0),
    ):
        result = run_backtest(df)

    s  = result["stats"]
    pr = pd.Series(result["portfolio_returns"])
    pr.index = pd.to_datetime(pr.index)
    ann = pr.groupby(pr.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    tl  = result["trade_log"]

    tw = len(pr[pr != 0]) / (len(pr) / 5) if len(pr) > 0 else 0  # rough trades/week
    n_trades = s["total_trades"]
    tr_pw    = n_trades / (len(pr) / 252 * 52)

    logger.info(
        "%-42s | Sh=%.3f  Ann=%+.2f%%  MaxDD=%.2f%%  Trades=%d(%.2f/wk)  WR=%.1f%%  $25k→$%,.0f",
        label,
        s["sharpe_ratio"], s["annualized_return_pct"], s["max_drawdown_pct"],
        n_trades, tr_pw, s["overall_win_rate"] * 100,
        25000 * (1 + s["total_return_pct"] / 100),
    )
    return s, ann, tl


def _sentiment_stats(df):
    """Log how sentiment is shaping the signal set."""
    eq = df[df["symbol"] == "ES"].drop_duplicates("date")
    if "sent_eq_mult" not in eq.columns:
        return
    blocked = (eq["sent_eq_mult"] == 0).sum()
    boosted = (eq["sent_eq_mult"] > 1.0).sum()
    reduced = ((eq["sent_eq_mult"] > 0) & (eq["sent_eq_mult"] < 1.0)).sum()
    ts_dist = eq["sent_ts_regime"].value_counts()
    logger.info(
        "  Sentiment: blocked=%d  boosted=%d  reduced=%d  | TS: %s",
        blocked, boosted, reduced,
        "  ".join(f"{k}={v}" for k, v in ts_dist.items())
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("Building base pipeline …")
    df0 = build_base_df()

    results = []

    # ── BASELINE: best current system (no sentiment) ──────────────────────
    logger.info("\n--- BASELINE (no sentiment) ---")
    df_base = build_fixed_stack(df0)
    df_base = inject_h4_brk_chop(df_base)
    df_base = inject_h4_mom_chop_nq(df_base)
    # Drop H4_BRK_TRANS (shown to hurt in run_frequency.py)
    s, ann, _ = _run_bt(df_base, scale=1.0, label="BASELINE_NO_SENTIMENT")
    results.append(("BASELINE", s, ann))

    # ── Add Tier 1 sentiment to pipeline df ──────────────────────────────
    logger.info("\n--- Computing Tier 1 sentiment features ---")
    df_sent = compute_tier1_sentiment(df0, sentiment_path=SENTIMENT_PATH)

    # ── EXPERIMENT 1: Sentiment blocking only (mult=0 days, no boost) ────
    logger.info("\n--- EXP 1: Block-only (no boost) ---")
    df_e1 = build_fixed_stack(df_sent)
    df_e1 = inject_h4_brk_chop(df_e1)
    df_e1 = inject_h4_mom_chop_nq(df_e1)
    # Override: remove boost, keep only blocks
    df_e1["sent_eq_mult"] = df_e1["sent_eq_mult"].clip(upper=1.0)
    df_e1 = apply_sentiment_to_entries(df_e1)
    _sentiment_stats(df_e1)
    s, ann, _ = _run_bt(df_e1, scale=1.0, label="SENT_BLOCK_ONLY")
    results.append(("SENT_BLOCK_ONLY", s, ann))

    # ── EXPERIMENT 2: Full sentiment (block + boost) ──────────────────────
    logger.info("\n--- EXP 2: Full sentiment (block + boost) ---")
    df_e2 = build_fixed_stack(df_sent)
    df_e2 = inject_h4_brk_chop(df_e2)
    df_e2 = inject_h4_mom_chop_nq(df_e2)
    df_e2 = apply_sentiment_to_entries(df_e2)
    _sentiment_stats(df_e2)
    s, ann, _ = _run_bt(df_e2, scale=1.0, label="SENT_FULL_1x")
    results.append(("SENT_FULL_1x", s, ann))

    # ── EXPERIMENT 3: Scaling grid with full sentiment ────────────────────
    logger.info("\n--- Scaling grid ---")
    for scale in [1.5, 2.0, 2.5, 3.0]:
        df_sc = build_fixed_stack(df_sent, scale=scale)
        df_sc = inject_h4_brk_chop(df_sc, scale=scale)
        df_sc = inject_h4_mom_chop_nq(df_sc, scale=scale)
        df_sc = apply_sentiment_to_entries(df_sc)
        lbl = f"SENT_FULL_{scale}x"
        s, ann, _ = _run_bt(df_sc, scale=scale, label=lbl)
        results.append((lbl, s, ann))

    # ─── Results table ────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("TIER 1 SENTIMENT EXPERIMENT — RESULTS")
    print("=" * 80)
    hdr = f"{'label':<28} {'sharpe':>7} {'d_sh':>7} {'annual':>8} {'d_ann':>7} {'max_dd':>8} {'trades':>7} {'wr':>6}"
    print(hdr)
    print("-" * 80)
    base_sh  = results[0][1]["sharpe_ratio"]
    base_ann = results[0][1]["annualized_return_pct"]
    for lbl, s, ann in results:
        d_sh  = s["sharpe_ratio"]         - base_sh
        d_ann = s["annualized_return_pct"] - base_ann
        print(
            f"{lbl:<28} {s['sharpe_ratio']:>7.3f} {d_sh:>+7.3f} "
            f"{s['annualized_return_pct']:>+7.2f}% {d_ann:>+6.2f}% "
            f"{s['max_drawdown_pct']:>7.2f}% {s['total_trades']:>7d} "
            f"{s['overall_win_rate']*100:>5.1f}%"
        )

    print()
    print("=" * 80)
    print("ANNUAL RETURNS BY YEAR")
    print("=" * 80)
    years = sorted(results[0][2].index)
    hdr2  = f"{'label':<28} " + "  ".join(f"yr_{y}" for y in years)
    print(hdr2)
    for lbl, s, ann in results:
        row = f"{lbl:<28} " + "  ".join(f"{ann.get(y,0):+6.1f}%" for y in years)
        print(row)

    # ── Hold-out check: 2024-2026 only ───────────────────────────────────
    print()
    print("=" * 80)
    print("PSEUDO-HOLD-OUT: 2024-2026 ONLY (3 years, ~156 weeks)")
    print("=" * 80)
    cutoff = pd.Timestamp("2024-01-01")

    for lbl, s, ann in results:
        yr_ret = {y: v for y, v in ann.items() if y >= 2024}
        avg = np.mean(list(yr_ret.values())) if yr_ret else 0.0
        print(f"  {lbl:<28} " + "  ".join(f"{yr_ret.get(y,0):+.1f}%" for y in sorted(yr_ret)) +
              f"  avg={avg:+.1f}%")

    print()
    print("=" * 80)
    print("CONTEXT: ES/NQ BEST BEFORE SENTIMENT (for comparison)")
    print("=" * 80)
    print("  FULL_2.5x (no sentiment):  Sh=1.416  Ann=+12.2%  MaxDD=-7.9%  $25k→$55,926")
    print("  FULL_2.0x (no sentiment):  Sh=1.410  Ann=+9.7%   MaxDD=-6.3%  $25k→$47,795")
    print()


if __name__ == "__main__":
    main()
