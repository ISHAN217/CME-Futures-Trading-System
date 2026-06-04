"""
run_fixes.py — Five targeted structural fixes for the ES/NQ system.

Diagnostic root causes → fixes:
  Fix 1  BULL_RESET sizing 0.08→0.20   108 trades at 70.4% WR but 4x undersized
  Fix 2  SA in TRANSITION               74 NQ SA signals blocked in TRANSITION regime
  Fix 3  TREND max-hold extension       71 winning TREND trades force-exited at 73% WR
  Fix 4  TRANSITION momentum long       159 days, momentum=LONG, avg +0.362%/day, flat
  Fix 5  Quality=4 TREND downgrade      30 NQ trades at loc avg 0.91 upsized into losses

Each fix is tested in isolation, then all combined.
Runs at 1x then 2.5x to show the realistic operating range.

Target: 10%+ annual at 1x without new instruments.
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

OUT_DIR = "output/fixes"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Sizing constants ─────────────────────────────────────────────────────────
_BASE_VOL_SCALE_MAX    = 0.28
_BASE_STAT_ARB_WEIGHT  = 0.20
_BASE_PULLBACK_WEIGHT  = 0.15
_BASE_MR_WEIGHT        = 0.18
_BASE_MAX_COMBINED_EXP = 0.50
_BASE_MAX_WT_PER_SYM   = 0.30

# Fix 1: raised from 0.08
_BR_SIZE_OLD = 0.08
_BR_SIZE_NEW = 0.20   # match approximate TREND sizing

_BASE_BEAR_TREND_SIZE  = 0.07
_BASE_ES_LATE_SIZE     = 0.05

SA_TRANSITION_SIZE     = 0.15   # Fix 2: SA in TRANSITION at reduced size
TRANS_LONG_SIZE        = 0.12   # Fix 4: TRANSITION momentum long

_BASE_HOLD_DAYS = {
    **config.MAX_HOLD_DAYS,
    "TREND":                config.MAX_HOLD_DAYS.get("TREND", 3) + 1,   # Fix 3: 4 (was 6) — extension fires sooner
    "BULL_RESET_CONFIRMED": config.MAX_HOLD_DAYS.get("TREND", 3) + 1,
    "MTF_BREAKOUT": 2, "MTF_RESET": 3,
    "BEAR_TREND": 4, "FAILED_RALLY": 3, "SHOCK_CONT": 2, "BEAR_RESET": 3,
    "ES_LATE_EXCEPTION": 4,
    "STAT_ARB": config.MAX_HOLD_DAYS.get("STAT_ARB", 5) + 1,  # SA hold +1d (existing improvement)
    "TRANS_LONG": 4,
}

# ─── Config context manager ───────────────────────────────────────────────────
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


# ─── Pipeline (cached) ────────────────────────────────────────────────────────
_DF0 = None

def build_base_df() -> pd.DataFrame:
    global _DF0
    if _DF0 is not None:
        return _DF0

    logger.info("Building base pipeline …")
    df = load_daily(); df4 = load_4h()
    df = compute_features(df); df = compute_momentum(df)
    df = compute_weekly_bias(df); df = compute_regime(df)
    df = compute_volatility_features(df)
    for col, val in [
        ("cwt_chop_label","CWT_UNCLEAR"), ("cwt_trade_allowed",1),
        ("cwt_size_multiplier",1.0), ("cwt_confidence",0.0),
        ("cwt_total_energy",0.0), ("cwt_entropy",0.5),
        ("cwt_energy_z",0.0), ("cwt_compression_flag",0),
        ("cwt_expansion_flag",0), ("cwt_high_energy_ratio",1/3),
        ("cwt_mid_energy_ratio",1/3), ("cwt_slow_energy_ratio",1/3),
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
    _DF0 = df
    logger.info("Pipeline done: %d rows, %d cols", len(df), len(df.columns))
    return df


def _add_entry_location(df):
    if "entry_location_pct_20d" in df.columns:
        return df
    df = df.copy()
    for sym, g in df.groupby("symbol", sort=False):
        idx = g.index; s = pd.Series(g["close"].values, index=idx)
        mn = s.rolling(20, min_periods=5).min()
        mx = s.rolling(20, min_periods=5).max()
        df.loc[idx, "entry_location_pct_20d"] = ((s - mn) / (mx - mn).replace(0, np.nan)).clip(0, 1).values
    return df


# ─── Base V2 signal stack ─────────────────────────────────────────────────────

def apply_v2_long_core(df: pd.DataFrame, br_size: float = _BR_SIZE_OLD) -> pd.DataFrame:
    """ES late filter + regime gating + BULL_RESET injection."""
    df = df.copy()
    loc = df["entry_location_pct_20d"].fillna(0.5)

    # ES late filter (unchanged)
    late = ((df["symbol"] == "ES") & (df["strategy_used"] == "TREND") &
            (df["final_direction"] == "LONG") & (loc > 0.70))
    df.loc[late, "final_direction"] = "FLAT"
    df.loc[late, "strategy_used"]   = "NONE"

    # Regime router gating
    if "allow_trend_long" in df.columns:
        rg = ((df["strategy_used"] == "TREND") & (df["final_direction"] == "LONG") &
              (df["allow_trend_long"] == 0))
        df.loc[rg, "final_direction"] = "FLAT"
        df.loc[rg, "strategy_used"]   = "NONE"

    # BULL_RESET injection
    slope  = df["slope_20d"].fillna(0)
    ret20  = df["ret_20d"].fillna(0)
    ret5   = df["ret_5d"].fillna(0)
    bk     = df.get("structure_break_count", pd.Series(0, index=df.index)).fillna(0)
    loc2   = df["entry_location_pct_20d"].fillna(0.5)
    reg    = df["portfolio_regime"].fillna("CHOP")
    sent   = df["portfolio_sentiment_flag"].fillna("NORMAL")

    br = ((df["final_direction"] == "FLAT") & reg.isin(["TREND","TRANSITION"]) &
          (slope > 0) & (ret20 > 0) & (ret5 < 0) & (bk == 0) & (loc2 < 0.80) &
          (sent != "HIGH_RISK"))
    df.loc[br, "final_direction"] = "LONG"
    df.loc[br, "strategy_used"]   = "BULL_RESET_CONFIRMED"
    df.loc[br, "mtf_entry_size"]  = br_size
    df.loc[br, "entry_quality"]   = 4
    return df


def apply_bear_trend_short(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """BEAR_TREND short only."""
    df = compute_short_signals(df)
    not_bear = (df["short_signal_type"] != "BEAR_TREND") & (df["short_signal_type"] != "NONE")
    df.loc[not_bear, "final_direction"]   = "FLAT"
    df.loc[not_bear, "strategy_used"]     = "NONE"
    df.loc[not_bear, "short_signal_type"] = "NONE"
    df.loc[not_bear, "short_signal_size"] = 0.0
    bear_fired = df["short_signal_type"] == "BEAR_TREND"
    df.loc[bear_fired, "mtf_entry_size"]  = _BASE_BEAR_TREND_SIZE * scale
    return df


def apply_es_late_nq_highconv(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """ES late exception when NQ high_conviction=True."""
    df = df.copy()
    loc = df["entry_location_pct_20d"].fillna(0.5)
    nq_rows = df[df["symbol"] == "NQ"][["date", "final_direction", "high_conviction"]].copy()
    nq_rows = nq_rows.rename(columns={"final_direction": "nq_dir", "high_conviction": "nq_hc"})
    df = df.merge(nq_rows, on="date", how="left")
    nq_dir = df["nq_dir"].fillna("FLAT")
    nq_hc  = df["nq_hc"].fillna(False).astype(bool)
    allow = ((df["symbol"] == "ES") & (df["final_direction"] == "FLAT") &
             (df["portfolio_regime"].fillna("CHOP") == "TREND") &
             (df["momentum_direction"].fillna("NEUTRAL") == "LONG") &
             (df["slope_20d"].fillna(0) > 0) & (loc > 0.70) &
             (nq_dir == "LONG") & nq_hc &
             (df["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK"))
    df.loc[allow, "final_direction"] = "LONG"
    df.loc[allow, "strategy_used"]   = "ES_LATE_EXCEPTION"
    df.loc[allow, "mtf_entry_size"]  = _BASE_ES_LATE_SIZE * scale
    df.loc[allow, "entry_quality"]   = 4
    df = df.drop(columns=["nq_dir", "nq_hc"], errors="ignore")
    logger.info("  ES late + NQ highconv: %d exceptions", int(allow.sum()))
    return df


# ─── Fix injections ───────────────────────────────────────────────────────────

def fix2_sa_transition(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """
    Fix 2: Allow STAT_ARB in TRANSITION regime (NQ only, hedge always disabled for ES).
    The signal_engine hard-blocks all new entries in TRANSITION; we restore SA here.
    """
    df = df.copy()
    sent = df["portfolio_sentiment_flag"].fillna("NORMAL")

    # Only NQ: sa_nq_direction=LONG, TRANSITION regime, currently flat
    allow = (
        (df["symbol"] == "NQ") &
        (df["sa_signal"] == "ENTRY") &
        (df["sa_nq_direction"] == "LONG") &   # spread says NQ is cheap
        (df["portfolio_regime"] == "TRANSITION") &
        (df["final_direction"] == "FLAT") &
        (sent != "HIGH_RISK")
    )

    # Also handle case where sa_nq_direction is not populated but sa_signal=ENTRY
    # (stat_arb.py always sets sa_nq_direction=LONG when sa_signal=ENTRY for NQ)
    allow2 = (
        (df["symbol"] == "NQ") &
        (df["sa_signal"] == "ENTRY") &
        (df["sa_nq_direction"] != "FLAT") &
        (df["portfolio_regime"] == "TRANSITION") &
        (df["final_direction"] == "FLAT") &
        (sent != "HIGH_RISK")
    )
    combined = allow | allow2

    df.loc[combined, "final_direction"] = "LONG"
    df.loc[combined, "strategy_used"]   = "STAT_ARB"
    df.loc[combined, "mtf_entry_size"]  = SA_TRANSITION_SIZE * scale
    df.loc[combined, "entry_quality"]   = 3

    n = int(combined.sum())
    logger.info("  Fix2 SA_TRANSITION: %d NQ entries injected", n)
    return df


def fix4_transition_momentum_long(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """
    Fix 4: Allow TREND-style longs in TRANSITION when momentum is clearly bullish.
    Diagnostic: 159 days where TRANSITION + momentum=LONG averaged +0.362%/day.
    Use reduced size (TRANS_LONG_SIZE) to reflect regime uncertainty.
    """
    df = df.copy()
    reg   = df["portfolio_regime"].fillna("CHOP")
    mom   = df["momentum_direction"].fillna("NEUTRAL")
    slope = df["slope_20d"].fillna(0)
    bias  = df["weekly_bias"].fillna("NEUTRAL")
    sent  = df["portfolio_sentiment_flag"].fillna("NORMAL")
    loc   = df["entry_location_pct_20d"].fillna(0.5)

    allow = (
        (reg == "TRANSITION") &
        (df["final_direction"] == "FLAT") &
        (mom == "LONG") &
        (bias == "LONG") &
        (slope > 0) &
        (loc < 0.85) &             # not at extreme late entry
        (sent != "HIGH_RISK")
    )
    df.loc[allow, "final_direction"] = "LONG"
    df.loc[allow, "strategy_used"]   = "BULL_RESET_CONFIRMED"   # reuse sizing logic
    df.loc[allow, "mtf_entry_size"]  = TRANS_LONG_SIZE * scale
    df.loc[allow, "entry_quality"]   = 3

    n = int(allow.sum())
    logger.info("  Fix4 TRANS_LONG: %d entries injected", n)
    return df


def fix5_quality4_downgrade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix 5: TREND quality=4 at high entry locations loses money.
    Avg loc=0.895 (89th percentile of 20d range) with -0.0160 total PnL.
    Downgrade to quality=3 when strategy=TREND AND loc>0.80.
    This removes the 20% sizing bonus from entries that are overextended.
    """
    df = df.copy()
    loc = df["entry_location_pct_20d"].fillna(0.5)
    mask = (
        (df["strategy_used"] == "TREND") &
        (df["entry_quality"] == 4) &
        (df["final_direction"] == "LONG") &
        (loc > 0.80)
    )
    df.loc[mask, "entry_quality"] = 3
    logger.info("  Fix5 Q4_downgrade: %d TREND quality=4→3 at loc>0.80", int(mask.sum()))
    return df


# ─── Backtest runner ──────────────────────────────────────────────────────────

def _cfg_for_scale(scale: float) -> tuple:
    """Returns (cfg_dict, max_wt_per_sym) for a given scale."""
    max_wt = min(_BASE_MAX_WT_PER_SYM * scale, 1.20)
    cfg = dict(
        vol_max      = _BASE_VOL_SCALE_MAX   * scale,
        stat_arb_w   = min(_BASE_STAT_ARB_WEIGHT  * scale, 0.60),
        pb_w         = min(_BASE_PULLBACK_WEIGHT   * scale, 0.50),
        mr_w         = min(_BASE_MR_WEIGHT         * scale, 0.50),
        combined_exp = _BASE_MAX_COMBINED_EXP * scale,
        hold_days    = _BASE_HOLD_DAYS,
    )
    return cfg, max_wt


def run_bt(label: str, df: pd.DataFrame, scale: float = 1.0,
           extend_h4_none: bool = False,
           max_hold_ext_days: int = 14,
           extra_info: str = "") -> dict:

    cfg, max_wt = _cfg_for_scale(scale)

    config_overrides = dict(
        VOL_SCALE_MAX         = cfg["vol_max"],
        STAT_ARB_NQ_WEIGHT    = cfg["stat_arb_w"],
        TREND_PULLBACK_WEIGHT = cfg["pb_w"],
        MEAN_REVERT_WEIGHT    = cfg["mr_w"],
        MAX_COMBINED_EXPOSURE = cfg["combined_exp"],
        MAX_HOLD_DAYS         = cfg["hold_days"],
        MAX_WEIGHT_PER_SYMBOL = max_wt,
        ADD_WINNER_SIZE_FRAC  = 0.50,
        # Fix 3: extension cap
        MAX_EXTENDED_HOLD_DAYS = max_hold_ext_days,
        EXTEND_ALLOW_H4_NONE   = extend_h4_none,
    )

    with override_config(**config_overrides):
        res = run_backtest(df)

    st = res["stats"]; tl = res["trade_log"]; pr = res["portfolio_returns"]

    yr = {}
    for y in range(2019, 2027):
        m = pr.index.year == y
        if m.any():
            yr[y] = round(float((1 + pr[m]).prod() - 1) * 100, 2)

    down   = pr[pr < 0]
    ds_std = float(down.std()) * np.sqrt(252) if len(down) > 5 else np.nan
    sortino = float(pr.mean()) * 252 / ds_std if (ds_std and ds_std > 1e-9) else np.nan

    wk_ret = pr.resample("W").sum()
    cumret = (1 + pr).cumprod(); roll_max = cumret.cummax()
    dd = (cumret - roll_max) / roll_max
    avg_dd = float(dd[dd < 0].mean()) * 100 if (dd < 0).any() else 0.0

    strat_pnl = {}
    if not tl.empty and "strategy" in tl.columns:
        strat_pnl = {s: round(float(g["pnl"].sum()), 4) for s, g in tl.groupby("strategy")}

    final_25k = int(25000 * (1 + st["annualized_return_pct"] / 100) ** 7)

    row = dict(
        label           = label,
        info            = extra_info,
        sharpe          = round(st["sharpe_ratio"], 4),
        sortino         = round(sortino, 4),
        annual_ret_pct  = round(st["annualized_return_pct"], 2),
        final_25k       = final_25k,
        gain_25k        = final_25k - 25000,
        max_dd_pct      = round(st["max_drawdown_pct"], 2),
        avg_dd_pct      = round(avg_dd, 2),
        worst_week_pct  = round(float(wk_ret.min()) * 100, 2),
        trades          = st["total_trades"],
        win_rate        = round(st.get("overall_win_rate", 0), 4),
        profit_factor   = round(st.get("profit_factor", 0), 4),
        **{f"yr_{k}": v for k, v in yr.items()},
        **{f"s_{k}": v for k, v in strat_pnl.items()},
    )

    logger.info(
        "%-38s | Sh=%.3f  Ann=%+.1f%%  MaxDD=%.1f%%  Trades=%d  WR=%.1f%%  $25k→$%d",
        label, row["sharpe"], row["annual_ret_pct"], row["max_dd_pct"],
        row["trades"], row["win_rate"] * 100, final_25k,
    )
    return row, tl


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df0 = build_base_df()
    results = []

    # ── Build baseline V2 signal stack (old BULL_RESET size) ─────────────────
    df_base = apply_bear_trend_short(
        apply_es_late_nq_highconv(
            apply_v2_long_core(df0, br_size=_BR_SIZE_OLD), scale=1.0
        ), scale=1.0
    )

    logger.info("=" * 60)
    logger.info("BASELINE — reproduces best V2 system with SA +1d and ES except")
    logger.info("=" * 60)
    row, tl_base = run_bt("BASELINE", df_base, scale=1.0,
                          extend_h4_none=False, max_hold_ext_days=7,
                          extra_info="current best")
    results.append(row)
    base_sh  = row["sharpe"]
    base_ann = row["annual_ret_pct"]

    # ══════════════════════════════════════════════════════════════
    # INDIVIDUAL FIX TESTS (always vs baseline, at 1x)
    # ══════════════════════════════════════════════════════════════

    # Fix 1: BULL_RESET sizing 0.08 → 0.20
    logger.info("=" * 60 + "\nFix 1: BULL_RESET sizing")
    df_f1 = apply_bear_trend_short(
        apply_es_late_nq_highconv(
            apply_v2_long_core(df0, br_size=_BR_SIZE_NEW), scale=1.0
        ), scale=1.0
    )
    row, _ = run_bt("FIX1_BULL_RESET_SIZE", df_f1, scale=1.0,
                    extend_h4_none=False, max_hold_ext_days=7,
                    extra_info="BR 0.08→0.20")
    results.append(row)

    # Fix 2: SA in TRANSITION
    logger.info("=" * 60 + "\nFix 2: SA in TRANSITION")
    df_f2 = fix2_sa_transition(df_base, scale=1.0)
    row, _ = run_bt("FIX2_SA_TRANSITION", df_f2, scale=1.0,
                    extend_h4_none=False, max_hold_ext_days=7,
                    extra_info="SA allowed in TRANSITION")
    results.append(row)

    # Fix 3: TREND max-hold extension
    logger.info("=" * 60 + "\nFix 3: Max-hold extension")
    row, _ = run_bt("FIX3_MAXHOLD_EXT", df_base, scale=1.0,
                    extend_h4_none=True, max_hold_ext_days=14,
                    extra_info="hold cap 7→14, relax h4 req")
    results.append(row)

    # Fix 4: TRANSITION momentum long
    logger.info("=" * 60 + "\nFix 4: TRANSITION momentum long")
    df_f4 = fix4_transition_momentum_long(df_base, scale=1.0)
    row, _ = run_bt("FIX4_TRANS_LONG", df_f4, scale=1.0,
                    extend_h4_none=False, max_hold_ext_days=7,
                    extra_info="TRANSITION mom=LONG entries")
    results.append(row)

    # Fix 5: quality=4 downgrade
    logger.info("=" * 60 + "\nFix 5: Quality=4 downgrade")
    df_f5 = fix5_quality4_downgrade(df_base)
    row, _ = run_bt("FIX5_Q4_DOWNGRADE", df_f5, scale=1.0,
                    extend_h4_none=False, max_hold_ext_days=7,
                    extra_info="TREND Q4→Q3 at loc>0.80")
    results.append(row)

    # ══════════════════════════════════════════════════════════════
    # INCREMENTAL STACK (add fixes one by one)
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 60 + "\nINCREMENTAL STACK")

    # F1 + F3 (sizing + extension — pure sizing/holding changes)
    df_f1f3 = apply_bear_trend_short(
        apply_es_late_nq_highconv(
            apply_v2_long_core(df0, br_size=_BR_SIZE_NEW), scale=1.0
        ), scale=1.0
    )
    row, _ = run_bt("F1+F3 (size+ext)", df_f1f3, scale=1.0,
                    extend_h4_none=True, max_hold_ext_days=14,
                    extra_info="BR big + hold ext")
    results.append(row)

    # F1+F3+F2 (add SA in TRANSITION)
    df_f1f3f2 = fix2_sa_transition(df_f1f3, scale=1.0)
    row, _ = run_bt("F1+F3+F2 (+SA trans)", df_f1f3f2, scale=1.0,
                    extend_h4_none=True, max_hold_ext_days=14,
                    extra_info="+SA in TRANSITION")
    results.append(row)

    # F1+F3+F2+F4 (add TRANSITION momentum)
    df_f1f3f2f4 = fix4_transition_momentum_long(df_f1f3f2, scale=1.0)
    row, _ = run_bt("F1+F3+F2+F4 (+trans long)", df_f1f3f2f4, scale=1.0,
                    extend_h4_none=True, max_hold_ext_days=14,
                    extra_info="+TRANSITION mom long")
    results.append(row)

    # ALL FIVE FIXES (add Q4 downgrade)
    df_all = fix5_quality4_downgrade(df_f1f3f2f4)
    row, tl_all_1x = run_bt("ALL_FIXES_1X", df_all, scale=1.0,
                             extend_h4_none=True, max_hold_ext_days=14,
                             extra_info="all 5 fixes, 1x size")
    results.append(row)
    all_fixes_ann = row["annual_ret_pct"]
    all_fixes_sh  = row["sharpe"]

    # ══════════════════════════════════════════════════════════════
    # SCALING with all fixes
    # ══════════════════════════════════════════════════════════════
    logger.info("=" * 60 + "\nSCALING WITH ALL FIXES")

    for scale_name, scale in [("1.5x", 1.5), ("2.0x", 2.0), ("2.5x", 2.5), ("3.0x", 3.0)]:
        df_s = apply_bear_trend_short(
            apply_es_late_nq_highconv(
                apply_v2_long_core(df0, br_size=_BR_SIZE_NEW * scale), scale=scale
            ), scale=scale
        )
        df_s = fix2_sa_transition(df_s, scale=scale)
        df_s = fix4_transition_momentum_long(df_s, scale=scale)
        df_s = fix5_quality4_downgrade(df_s)
        row, _ = run_bt(f"ALL_FIXES_{scale_name}", df_s, scale=scale,
                        extend_h4_none=True, max_hold_ext_days=14,
                        extra_info=f"all fixes + {scale_name}")
        results.append(row)

    # ─── Save and report ──────────────────────────────────────────
    res = pd.DataFrame(results)
    res["delta_sh"]  = (res["sharpe"]         - base_sh ).round(4)
    res["delta_ann"] = (res["annual_ret_pct"]  - base_ann).round(2)
    res.to_csv(f"{OUT_DIR}/fix_results.csv", index=False)
    tl_all_1x.to_csv(f"{OUT_DIR}/all_fixes_1x_trade_log.csv", index=False)

    _print_report(res, base_sh, base_ann, tl_all_1x)


def _print_report(res, base_sh, base_ann, tl_best):
    W = "=" * 80
    print(f"\n{W}")
    print("FIX EXPERIMENT — RESULTS")
    print(W)

    core = ["label", "info", "sharpe", "delta_sh", "sortino",
            "annual_ret_pct", "delta_ann", "final_25k", "gain_25k",
            "max_dd_pct", "trades", "win_rate", "profit_factor"]
    print(res[[c for c in core if c in res.columns]].to_string(index=False))

    # Annual returns by fix
    yr_cols = sorted([c for c in res.columns if c.startswith("yr_")])
    if yr_cols:
        print(f"\n{W}")
        print("ANNUAL RETURNS % BY EXPERIMENT")
        print(W)
        print(res[["label"] + yr_cols].to_string(index=False))

    # Per-strategy PnL for all-fixes 1x
    print(f"\n{W}")
    print("PnL BY STRATEGY — ALL FIXES 1x")
    print(W)
    for strat, g in tl_best.groupby("strategy"):
        n = len(g); wr = g["win"].mean(); tot = g["pnl"].sum(); avg = g["pnl"].mean()
        wt = g["entry_weight"].abs().mean()
        print(f"  {strat:<28} n={n:3d}  WR={wr:.1%}  avg={avg:.5f}  tot={tot:.4f}  avgWt={wt:.4f}")

    # Impact summary
    print(f"\n{W}")
    print("FIX IMPACT SUMMARY — ISOLATED EFFECTS")
    print(W)
    for lbl in ["BASELINE","FIX1_BULL_RESET_SIZE","FIX2_SA_TRANSITION",
                "FIX3_MAXHOLD_EXT","FIX4_TRANS_LONG","FIX5_Q4_DOWNGRADE"]:
        r = res[res["label"] == lbl]
        if r.empty: continue
        r = r.iloc[0]
        print(f"  {lbl:<30} Sharpe={r['sharpe']:.4f} ({r['delta_sh']:+.4f})  "
              f"Ann={r['annual_ret_pct']:+.1f}% ({r['delta_ann']:+.1f}%)  "
              f"Trades={r['trades']}  MaxDD={r['max_dd_pct']:.1f}%")

    # Scaling table
    print(f"\n{W}")
    print("ALL FIXES — SCALING GRID")
    print(W)
    print(f"  {'Experiment':<26} {'Sharpe':>8} {'ΔSh':>7} {'Ann%':>7} {'MaxDD%':>8} "
          f"{'$25k→':>10} {'Trades':>7}")
    print(f"  {'-'*75}")
    for lbl in ["BASELINE","ALL_FIXES_1X","ALL_FIXES_1.5x","ALL_FIXES_2.0x",
                "ALL_FIXES_2.5x","ALL_FIXES_3.0x"]:
        r = res[res["label"] == lbl]
        if r.empty: continue
        r = r.iloc[0]
        flag = " ← TARGET" if r["annual_ret_pct"] >= 10.0 else ""
        print(f"  {lbl:<26} {r['sharpe']:>8.3f} {r['delta_sh']:>+7.3f} "
              f"{r['annual_ret_pct']:>+6.1f}% {r['max_dd_pct']:>7.1f}% "
              f"${r['final_25k']:>9,} {r['trades']:>7}{flag}")

    # SPY context
    print(f"\n{W}")
    print("CONTEXT: SPY CAGR ~15%")
    print(W)
    for lbl in ["ALL_FIXES_1X","ALL_FIXES_1.5x","ALL_FIXES_2.0x","ALL_FIXES_2.5x"]:
        r = res[res["label"] == lbl]
        if r.empty: continue
        r = r.iloc[0]
        gap = r["annual_ret_pct"] - 15.0
        print(f"  {lbl:<26} System={r['annual_ret_pct']:+.1f}%  SPY≈+15%  "
              f"Gap={gap:+.1f}%  MaxDD={r['max_dd_pct']:.1f}%")
    print(W)


if __name__ == "__main__":
    main()
