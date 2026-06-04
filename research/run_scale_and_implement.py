"""
run_scale_and_implement.py — Scale up sizing + implement two validated improvements.

Two tasks:
  1. Scale sizing: test 1x → 1.5x → 2x → 2.5x → 3x of current positions.
     Scales VOL_SCALE_MAX, STAT_ARB_NQ_WEIGHT, TREND_PULLBACK_WEIGHT,
     MEAN_REVERT_WEIGHT, and MAX_COMBINED_EXPOSURE proportionally.
     All injected-strategy mtf_entry_size values also scaled.

  2. Implement two validated improvements from idle-day research:
     a. SA hold +1 day (STAT_ARB max hold 5 → 6): +0.038 Sharpe, no new trades
     b. ES late exception + NQ high conviction: +0.018 Sharpe, 16 trades at 81% WR

Output:
  - scaling_experiments.csv  (scale table across all multipliers)
  - final_system_results.csv (baseline vs final implemented system)
  - final_system_trade_log.csv

Experiments:
  BASELINE_CURRENT          — exact current best (for reference)
  SA_HOLD_PLUS_1D           — only SA hold improvement
  ES_LATE_NQ_HIGHCONV       — only ES late exception improvement
  IMPROVEMENTS_ONLY         — both improvements, no size change
  SCALE_1X                  — improvements + 1x size (same)
  SCALE_1_5X                — improvements + 1.5x size
  SCALE_2X                  — improvements + 2x size
  SCALE_2_5X                — improvements + 2.5x size
  SCALE_3X                  — improvements + 3x size
"""

import contextlib
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import config
from data_loader                    import load_daily, load_4h
from features                       import compute_features
from momentum_engine                import compute_momentum
from weekly_bias                    import compute_weekly_bias
from regime_engine                  import compute_regime
from volatility_engine              import compute_volatility_features
from intraday_4h_engine             import (compute_4h_features,
                                            aggregate_4h_to_daily,
                                            merge_4h_into_daily)
from mean_reversion                 import compute_mean_reversion
from stat_arb                       import compute_stat_arb
from sentiment_layer                import compute_sentiment
from signal_engine                  import compute_signals
from signal_quality_engine          import compute_signal_quality
from trend_continuation_edge_engine import compute_trend_continuation_edge
from regime_router                  import compute_regime_router
from short_sleeve_engine            import compute_short_signals
from backtest                       import run_backtest

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT_DIR = "output/scale_and_implement"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Base sizing (1x reference — matches best V2 run) ─────────────────────────
_BASE_VOL_SCALE_MAX       = 0.28
_BASE_STAT_ARB_WEIGHT     = 0.20
_BASE_PULLBACK_WEIGHT     = 0.15
_BASE_MR_WEIGHT           = 0.18
_BASE_MAX_COMBINED_EXP    = 0.50
_BASE_MAX_WEIGHT_PER_SYM  = 0.30   # must scale with everything else
_BASE_HOLD_DAYS           = {
    **config.MAX_HOLD_DAYS,
    "TREND":                config.MAX_HOLD_DAYS.get("TREND", 3) + 3,   # 6 days
    "BULL_RESET_CONFIRMED": config.MAX_HOLD_DAYS.get("TREND", 3) + 3,   # 6 days
    "MTF_BREAKOUT": 2, "MTF_RESET": 3,
    "BEAR_TREND": 4, "FAILED_RALLY": 3, "SHOCK_CONT": 2, "BEAR_RESET": 3,
    "ES_LATE_EXCEPTION": 4,
}

# Injected strategy sizes at 1x
_BASE_BULL_RESET_SIZE     = 0.08
_BASE_BEAR_TREND_SIZE     = 0.07
_BASE_ES_LATE_SIZE        = 0.05   # for ES late + NQ highconv exception


@contextlib.contextmanager
def override_config(**kwargs):
    orig = {k: getattr(config, k) for k in kwargs if hasattr(config, k)}
    for k, v in kwargs.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in orig.items():
            setattr(config, k, v)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_base_df() -> pd.DataFrame:
    logger.info("Building pipeline…")
    df = load_daily(); df4 = load_4h()
    df = compute_features(df); df = compute_momentum(df)
    df = compute_weekly_bias(df); df = compute_regime(df)
    df = compute_volatility_features(df)
    for col, val in [
        ("cwt_chop_label","CWT_UNCLEAR"),("cwt_trade_allowed",1),
        ("cwt_size_multiplier",1.0),("cwt_confidence",0.0),
        ("cwt_total_energy",0.0),("cwt_entropy",0.5),
        ("cwt_energy_z",0.0),("cwt_compression_flag",0),
        ("cwt_expansion_flag",0),("cwt_high_energy_ratio",1/3),
        ("cwt_mid_energy_ratio",1/3),("cwt_slow_energy_ratio",1/3),
    ]:
        if col not in df.columns: df[col] = val
    d4d = None
    if df4 is not None:
        d4d = aggregate_4h_to_daily(compute_4h_features(df4))
    df = merge_4h_into_daily(df, d4d)
    df = compute_mean_reversion(df); df = compute_stat_arb(df)
    df = compute_sentiment(df);      df = compute_signals(df)
    df = compute_signal_quality(df); df = compute_trend_continuation_edge(df)
    df = compute_regime_router(df)
    df = _add_entry_location(df)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Pipeline done: %d rows, %d cols", len(df), len(df.columns))
    return df


def _add_entry_location(df):
    if "entry_location_pct_20d" in df.columns:
        return df
    df = df.copy()
    df["entry_location_pct_20d"] = np.nan
    for sym, g in df.groupby("symbol", sort=False):
        idx = g.index; s = pd.Series(g["close"].values, index=idx)
        mn  = s.rolling(20, min_periods=5).min()
        mx  = s.rolling(20, min_periods=5).max()
        df.loc[idx, "entry_location_pct_20d"] = ((s - mn) / (mx - mn).replace(0, np.nan)).clip(0, 1).values
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Signal injections
# ─────────────────────────────────────────────────────────────────────────────

def apply_v2_long_core(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """ES late filter + regime gating + BULL_RESET injection."""
    df = df.copy()
    loc  = df["entry_location_pct_20d"].fillna(0.5)

    # ES late filter
    late = ((df["symbol"] == "ES") & (df["strategy_used"] == "TREND") &
            (df["final_direction"] == "LONG") & (loc > 0.70))
    df.loc[late, "final_direction"] = "FLAT"; df.loc[late, "strategy_used"] = "NONE"

    # Regime router gating
    if "allow_trend_long" in df.columns:
        rg = ((df["strategy_used"] == "TREND") & (df["final_direction"] == "LONG") &
              (df["allow_trend_long"] == 0))
        df.loc[rg, "final_direction"] = "FLAT"; df.loc[rg, "strategy_used"] = "NONE"

    # BULL_RESET
    slope = df["slope_20d"].fillna(0); ret20 = df["ret_20d"].fillna(0)
    ret5  = df["ret_5d"].fillna(0)
    bk    = df.get("structure_break_count", pd.Series(0, index=df.index)).fillna(0)
    loc2  = df["entry_location_pct_20d"].fillna(0.5)
    reg   = df["portfolio_regime"].fillna("CHOP")
    sent  = df["portfolio_sentiment_flag"].fillna("NORMAL")
    br    = ((df["final_direction"] == "FLAT") & reg.isin(["TREND","TRANSITION"]) &
             (slope > 0) & (ret20 > 0) & (ret5 < 0) & (bk == 0) & (loc2 < 0.80) &
             (sent != "HIGH_RISK"))
    df.loc[br, "final_direction"] = "LONG"
    df.loc[br, "strategy_used"]   = "BULL_RESET_CONFIRMED"
    df.loc[br, "mtf_entry_size"]  = _BASE_BULL_RESET_SIZE * scale
    df.loc[br, "entry_quality"]   = 4
    return df


def apply_bear_trend_short(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """BEAR_TREND short only (accepted in V2)."""
    df = compute_short_signals(df)
    not_bear = (df["short_signal_type"] != "BEAR_TREND") & (df["short_signal_type"] != "NONE")
    df.loc[not_bear, "final_direction"]   = "FLAT"
    df.loc[not_bear, "strategy_used"]     = "NONE"
    df.loc[not_bear, "short_signal_type"] = "NONE"
    df.loc[not_bear, "short_signal_size"] = 0.0
    # Rescale BEAR_TREND size
    bear_fired = df["short_signal_type"] == "BEAR_TREND"
    df.loc[bear_fired, "mtf_entry_size"] = _BASE_BEAR_TREND_SIZE * scale
    return df


def apply_es_late_nq_highconv(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """
    Improvement #2: Allow ES late entries ONLY when NQ is LONG + high_conviction.
    16 trades at 81.3% WR in backtest. Very selective gate.
    """
    df = df.copy()
    loc  = df["entry_location_pct_20d"].fillna(0.5)

    # Build NQ state on each date
    nq_rows = df[df["symbol"] == "NQ"][["date", "final_direction", "high_conviction"]].copy()
    nq_rows = nq_rows.rename(columns={"final_direction": "nq_dir",
                                       "high_conviction": "nq_hc"})
    df = df.merge(nq_rows, on="date", how="left")

    nq_dir = df["nq_dir"].fillna("FLAT")
    nq_hc  = df["nq_hc"].fillna(False).astype(bool)

    allow = (
        (df["symbol"] == "ES") &
        (df["final_direction"] == "FLAT") &
        (df["portfolio_regime"].fillna("CHOP") == "TREND") &
        (df["momentum_direction"].fillna("NEUTRAL") == "LONG") &
        (df["slope_20d"].fillna(0) > 0) &
        (loc > 0.70) &
        (nq_dir == "LONG") &
        nq_hc &
        (df["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )
    df.loc[allow, "final_direction"] = "LONG"
    df.loc[allow, "strategy_used"]   = "ES_LATE_EXCEPTION"
    df.loc[allow, "mtf_entry_size"]  = _BASE_ES_LATE_SIZE * scale
    df.loc[allow, "entry_quality"]   = 4

    df = df.drop(columns=["nq_dir", "nq_hc"], errors="ignore")
    n = int(allow.sum())
    logger.info("  ES late + NQ highconv: %d exceptions allowed", n)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Backtest runner
# ─────────────────────────────────────────────────────────────────────────────

def run_bt(label: str, df: pd.DataFrame, vol_max: float, stat_arb_w: float,
           pb_w: float, mr_w: float, combined_exp: float,
           hold_days: dict, max_wt_per_sym: float = None,
           extra_label_info: str = "") -> dict:

    cfg = dict(
        VOL_SCALE_MAX        = vol_max,
        STAT_ARB_NQ_WEIGHT   = stat_arb_w,
        TREND_PULLBACK_WEIGHT= pb_w,
        MEAN_REVERT_WEIGHT   = mr_w,
        MAX_COMBINED_EXPOSURE= combined_exp,
        ADD_WINNER_SIZE_FRAC = 0.50,
        MAX_HOLD_DAYS        = hold_days,
    )
    if max_wt_per_sym is not None:
        cfg["MAX_WEIGHT_PER_SYMBOL"] = max_wt_per_sym
    with override_config(**cfg):
        res = run_backtest(df)

    st = res["stats"]; tl = res["trade_log"]; pr = res["portfolio_returns"]

    # Annual returns
    yr = {}
    for y in range(2019, 2027):
        m = pr.index.year == y
        if m.any():
            yr[y] = round(float((1 + pr[m]).prod() - 1) * 100, 2)

    # Sortino
    down    = pr[pr < 0]
    ds_std  = float(down.std()) * np.sqrt(252) if len(down) > 5 else np.nan
    sortino = float(pr.mean()) * 252 / ds_std if (ds_std and ds_std > 1e-9) else np.nan

    # Worst week / month
    wk_ret = pr.resample("W").sum()
    mo_ret = pr.resample("ME").sum()

    # Avg drawdown
    cumret  = (1 + pr).cumprod(); roll_max = cumret.cummax()
    dd      = (cumret - roll_max) / roll_max
    avg_dd  = float(dd[dd < 0].mean()) * 100 if (dd < 0).any() else 0.0

    # PnL by strategy
    strat_pnl = {}
    if not tl.empty and "strategy" in tl.columns:
        strat_pnl = {s: round(float(g["pnl"].sum()), 4) for s, g in tl.groupby("strategy")}

    # Final account value ($25k starting)
    final_25k = int(25000 * (1 + st["annualized_return_pct"] / 100) ** 7)

    row = dict(
        experiment      = label,
        scale           = extra_label_info,
        sharpe          = round(st["sharpe_ratio"], 4),
        sortino         = round(sortino, 4),
        annual_ret_pct  = round(st["annualized_return_pct"], 2),
        total_ret_pct   = round(st["total_return_pct"], 2),
        final_25k       = final_25k,
        gain_25k        = final_25k - 25000,
        max_dd_pct      = round(st["max_drawdown_pct"], 2),
        avg_dd_pct      = round(avg_dd, 2),
        worst_week_pct  = round(float(wk_ret.min()) * 100, 2),
        worst_month_pct = round(float(mo_ret.min()) * 100, 2),
        trades          = st["total_trades"],
        trades_per_wk   = round(st["total_trades"] / (len(pr) / 5), 2),
        win_rate        = round(st.get("overall_win_rate", 0), 4),
        profit_factor   = round(st.get("profit_factor", 0), 4),
        avg_trade_pnl   = round(float(tl["pnl"].mean()), 5) if not tl.empty else 0,
        long_wr         = round(st.get("long_win_rate", 0), 4),
        short_wr        = round(st.get("short_win_rate", 0), 4),
        pnl_ES          = round(float(tl[tl["symbol"]=="ES"]["pnl"].sum()), 4) if not tl.empty else 0,
        pnl_NQ          = round(float(tl[tl["symbol"]=="NQ"]["pnl"].sum()), 4) if not tl.empty else 0,
        **{f"yr_{k}": v for k, v in yr.items()},
        **{f"s_{k}": v for k, v in strat_pnl.items()},
    )

    logger.info(
        "%-28s | Sharpe=%.3f  Srt=%.3f  Ann=%+.1f%%  MaxDD=%.1f%%  "
        "$25k→$%d  Trades=%d  WR=%.1f%%",
        label, row["sharpe"], row["sortino"] or 0, row["annual_ret_pct"],
        row["max_dd_pct"], final_25k, row["trades"], row["win_rate"] * 100,
    )
    return row, tl


# ─────────────────────────────────────────────────────────────────────────────
# Scale variants
# ─────────────────────────────────────────────────────────────────────────────

_SCALE_VARIANTS = [
    # (name, scale_factor)
    ("1.0x (current)",  1.0),
    ("1.5x",            1.5),
    ("2.0x",            2.0),
    ("2.5x",            2.5),
    ("3.0x",            3.0),
]


def _cfg_for_scale(scale: float) -> dict:
    """Return all config overrides for a given scale factor."""
    hold = {**_BASE_HOLD_DAYS}
    # MAX_WEIGHT_PER_SYMBOL is the critical per-position cap; must scale with vol_max.
    # Without this, positions clip at 0.30 and the scaling has no effect.
    # Hard cap at 1.20 (120% per symbol = ~1.2 E-mini contracts per $25k NAV, fine for futures).
    max_wt = min(_BASE_MAX_WEIGHT_PER_SYM * scale, 1.20)
    return dict(
        vol_max             = _BASE_VOL_SCALE_MAX    * scale,
        stat_arb_w          = min(_BASE_STAT_ARB_WEIGHT  * scale, 0.60),
        pb_w                = min(_BASE_PULLBACK_WEIGHT   * scale, 0.50),
        mr_w                = min(_BASE_MR_WEIGHT         * scale, 0.50),
        combined_exp        = _BASE_MAX_COMBINED_EXP  * scale,
        max_wt_per_sym      = max_wt,
        hold_days           = hold,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df0 = build_base_df()

    results = []
    trade_logs = {}

    # ── Build the base V2 signal stack (no size change, no improvements yet) ──
    df_v2_base = apply_bear_trend_short(
        apply_v2_long_core(df0, scale=1.0), scale=1.0
    )

    cfg1x_full = _cfg_for_scale(1.0)
    _1x_max_wt = cfg1x_full.pop("max_wt_per_sym")
    cfg1x = cfg1x_full   # now safe to **-spread into run_bt

    # ── BASELINE_CURRENT: exact V2 best (reproduces L_LONG_PLUS_BEST_SHORT) ──
    logger.info("=== BASELINE_CURRENT ===")
    row, tl = run_bt("BASELINE_CURRENT", df_v2_base, **cfg1x,
                     max_wt_per_sym=_1x_max_wt, extra_label_info="1x, no improvements")
    results.append(row); trade_logs["BASELINE"] = tl
    base_sharpe = row["sharpe"]
    base_ann    = row["annual_ret_pct"]
    base_trades = row["trades"]

    # ── Improvement A: SA hold +1 day only ────────────────────────────────────
    logger.info("=== SA_HOLD_PLUS_1D ===")
    hold_sa_plus1 = {**_BASE_HOLD_DAYS, "STAT_ARB": _BASE_HOLD_DAYS.get("STAT_ARB", config.MAX_HOLD_DAYS.get("STAT_ARB", 5)) + 1}
    cfg_sa = {**cfg1x, "hold_days": hold_sa_plus1}
    row, _ = run_bt("SA_HOLD_PLUS_1D", df_v2_base, **cfg_sa,
                    max_wt_per_sym=_1x_max_wt, extra_label_info="+1d SA hold")
    results.append(row)

    # ── Improvement B: ES late + NQ highconv only ─────────────────────────────
    logger.info("=== ES_LATE_NQ_HIGHCONV ===")
    df_with_es = apply_es_late_nq_highconv(df_v2_base, scale=1.0)
    row, _ = run_bt("ES_LATE_NQ_HIGHCONV", df_with_es, **cfg1x,
                    max_wt_per_sym=_1x_max_wt, extra_label_info="ES except")
    results.append(row)

    # ── Both improvements, no size change ─────────────────────────────────────
    logger.info("=== IMPROVEMENTS_ONLY (1x) ===")
    df_improved = apply_es_late_nq_highconv(df_v2_base, scale=1.0)
    cfg_both = {**cfg1x, "hold_days": hold_sa_plus1}
    row, tl_improved = run_bt("IMPROVEMENTS_ONLY", df_improved, **cfg_both,
                               max_wt_per_sym=_1x_max_wt,
                               extra_label_info="both improvements, 1x size")
    results.append(row); trade_logs["IMPROVED_1X"] = tl_improved
    improved_1x_sharpe = row["sharpe"]

    # ── Scaling experiments (improvements applied at all scales) ─────────────
    logger.info("=== SCALING EXPERIMENTS ===")
    for scale_name, scale in _SCALE_VARIANTS:
        label = f"SCALE_{scale_name.replace(' ', '_').replace('(current)', '').strip()}"

        # Build df with scaled injection sizes
        df_scaled = apply_bear_trend_short(
            apply_v2_long_core(df0, scale=scale), scale=scale
        )
        df_scaled = apply_es_late_nq_highconv(df_scaled, scale=scale)

        # Config for this scale
        cfg = _cfg_for_scale(scale)
        cfg["hold_days"] = hold_sa_plus1   # always use SA +1d hold
        max_wt = cfg.pop("max_wt_per_sym")  # extracted separately, not in **cfg spread

        row, tl = run_bt(
            label, df_scaled, **cfg,
            max_wt_per_sym=max_wt,
            extra_label_info=f"all improvements + {scale_name}",
        )
        results.append(row)
        if abs(scale - 2.0) < 0.01:
            trade_logs["2X"] = tl
        if abs(scale - 3.0) < 0.01:
            trade_logs["3X"] = tl

    # ── Save and report ───────────────────────────────────────────────────────
    res = pd.DataFrame(results)
    res["delta_sharpe"]  = (res["sharpe"] - base_sharpe).round(4)
    res["delta_ann_ret"] = (res["annual_ret_pct"] - base_ann).round(2)
    res["delta_trades"]  = res["trades"] - base_trades
    res.to_csv(f"{OUT_DIR}/scaling_experiments.csv", index=False)

    if trade_logs.get("IMPROVED_1X") is not None:
        trade_logs["IMPROVED_1X"].to_csv(f"{OUT_DIR}/final_system_trade_log.csv", index=False)

    _print_report(res, base_sharpe, base_ann, base_trades, trade_logs)


def _print_report(res: pd.DataFrame, base_sh: float, base_ann: float,
                  base_trades: int, trade_logs: dict):
    w = "=" * 80

    print(f"\n{w}")
    print("SCALE + IMPLEMENT — FINAL RESULTS")
    print(w)

    # Main table
    core_cols = ["experiment", "scale", "sharpe", "delta_sharpe", "sortino",
                 "annual_ret_pct", "delta_ann_ret", "final_25k", "gain_25k",
                 "max_dd_pct", "avg_dd_pct", "worst_week_pct",
                 "trades", "delta_trades", "trades_per_wk",
                 "win_rate", "profit_factor"]
    print(res[[c for c in core_cols if c in res.columns]].to_string(index=False))

    # Annual returns
    yr_cols = sorted([c for c in res.columns if c.startswith("yr_")])
    if yr_cols:
        print(f"\n{w}")
        print("ANNUAL RETURNS % BY YEAR")
        print(w)
        print(res[["experiment"] + yr_cols].to_string(index=False))

    # Strategy PnL on improvements-only
    strat_cols = [c for c in res.columns if c.startswith("s_")]
    if strat_cols:
        print(f"\n{w}")
        print("PnL BY STRATEGY (IMPROVEMENTS_ONLY at 1x)")
        print(w)
        sub = res[res["experiment"] == "IMPROVEMENTS_ONLY"]
        if not sub.empty:
            for c in strat_cols:
                val = sub[c].values[0]
                if not pd.isna(val) and val != 0:
                    print(f"  {c.replace('s_',''):<28s}: {val:+.4f}")

    # ── Key comparison box ────────────────────────────────────────────────────
    print(f"\n{w}")
    print("IMPLEMENTATION IMPACT SUMMARY")
    print(w)

    def _g(exp):
        r = res[res["experiment"] == exp]
        return r.iloc[0] if not r.empty else None

    base  = _g("BASELINE_CURRENT")
    sa1d  = _g("SA_HOLD_PLUS_1D")
    eslnq = _g("ES_LATE_NQ_HIGHCONV")
    both  = _g("IMPROVEMENTS_ONLY")

    for label, row in [("Baseline (current best)", base),
                        ("+ SA hold +1d only", sa1d),
                        ("+ ES late NQ highconv only", eslnq),
                        ("Both improvements (1x size)", both)]:
        if row is not None:
            print(f"  {label:<35s}  Sharpe={row['sharpe']:.4f}  "
                  f"Ann={row['annual_ret_pct']:+.1f}%  MaxDD={row['max_dd_pct']:.1f}%  "
                  f"Trades={row['trades']}  $25k→${row['final_25k']:,}")

    print(f"\n{w}")
    print("SCALING IMPACT")
    print(w)
    print(f"  {'Scale':<12} {'Sharpe':>8} {'ΔSharpe':>9} {'Annual%':>8} "
          f"{'MaxDD%':>8} {'$25k→':>10} {'Gain':>10}")
    print(f"  {'-'*65}")
    for _, row in res[res["experiment"].str.startswith("SCALE_")].iterrows():
        flag = ""
        if row["max_dd_pct"] < -10:
            flag = " ← MaxDD warning"
        elif row["max_dd_pct"] < -7:
            flag = " ← approaching limit"
        print(f"  {row['experiment'].replace('SCALE_',''):<12} "
              f"{row['sharpe']:>8.3f} {row['delta_sharpe']:>+9.3f} "
              f"{row['annual_ret_pct']:>+7.1f}% {row['max_dd_pct']:>7.1f}% "
              f"${row['final_25k']:>9,} +${row['gain_25k']:>8,}{flag}")

    # Find the sweet spot: best risk-adjusted scale
    scale_rows = res[res["experiment"].str.startswith("SCALE_")].copy()
    if not scale_rows.empty:
        # Sweet spot: highest annual return where MaxDD > -9%
        acceptable = scale_rows[scale_rows["max_dd_pct"] >= -9.0]
        if not acceptable.empty:
            sweet = acceptable.loc[acceptable["annual_ret_pct"].idxmax()]
            print(f"\n  SWEET SPOT (best return within -9% MaxDD limit):")
            print(f"    {sweet['experiment'].replace('SCALE_','')} — "
                  f"Ann={sweet['annual_ret_pct']:+.1f}%  Sharpe={sweet['sharpe']:.3f}  "
                  f"MaxDD={sweet['max_dd_pct']:.1f}%  $25k→${sweet['final_25k']:,}")

    # SPY comparison
    spy_ann = 15.0
    print(f"\n{w}")
    print("CONTEXT: SPY CAGR ~15% (2019-2026)")
    print(w)
    for _, row in res[res["experiment"].str.startswith("SCALE_")].iterrows():
        gap = row["annual_ret_pct"] - spy_ann
        print(f"  {row['experiment'].replace('SCALE_',''):<12} "
              f"System={row['annual_ret_pct']:+.1f}%  SPY≈+15.0%  "
              f"Gap={gap:+.1f}%  MaxDD={row['max_dd_pct']:.1f}%")

    print(w)


if __name__ == "__main__":
    main()
