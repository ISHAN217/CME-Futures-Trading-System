"""
run_frequency.py — Trade frequency improvement experiments.

Diagnostic findings (forward-looking next-day returns):
────────────────────────────────────────────────────────────────────────
Signal                    Regime      n    fwd1    WR    Note
4H BREAKOUT               CHOP     ES:23  +0.29%  70%   edge dies day3
4H BREAKOUT               CHOP     NQ:23  +0.47%  78%   best new signal
4H BREAKOUT               TRANS    ES:37  +0.18%  62%   holds 2-3 days
4H BREAKOUT               TRANS    NQ:32  +0.14%  66%   day2 even better
4H MOMENTUM               CHOP     NQ:42  +0.25%  64%   skip ES (0.06%)
NEUTRAL weekly bias       TREND    ES/NQ  +0.12%  55%   too weak, skip
PULLBACK re-enable        TREND    ES/NQ  -0.65%   4%   clearly harmful
────────────────────────────────────────────────────────────────────────

The 4H BREAKOUT signal in CHOP/TRANSITION is regime-transition detection:
the 4H has broken out before the daily regime classifier catches up to TREND.
It's 1-2 days early, generating real alpha.

Key implementation insight: max-hold for these trades must be SHORT (2-3 days).
The edge decays by day 3 in CHOP, so overstaying kills it.

Expected frequency improvement at daily timeframe:
  Current:   ~429 trades / 7yr = 1.18/week
  +4H BRK:   ~500 trades / 7yr = 1.38/week  (+17%)
  The ceiling at daily bars is real — to go above 2x needs 4H resolution.

Experiments:
  BASELINE_FIXES    — F1+F2+F3 system (best current)
  FREQ_BRK_CHOP     — 4H BREAKOUT in CHOP only
  FREQ_BRK_TRANS    — 4H BREAKOUT in TRANSITION only
  FREQ_MOM_CHOP_NQ  — 4H MOMENTUM in CHOP (NQ only)
  FREQ_ALL          — all frequency signals combined
  FULL_1X           — fixes + frequency, 1x scale
  FULL_1.5X..3X     — scaled versions
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

OUT_DIR = "output/frequency"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Sizing ───────────────────────────────────────────────────────────────────
_BASE_VOL_SCALE_MAX    = 0.28
_BASE_STAT_ARB_WEIGHT  = 0.20
_BASE_PULLBACK_WEIGHT  = 0.15
_BASE_MR_WEIGHT        = 0.18
_BASE_MAX_COMBINED_EXP = 0.50
_BASE_MAX_WT_PER_SYM   = 0.30

_BR_SIZE               = 0.20   # Fix 1 (raised from 0.08)
_SA_TRANS_SIZE         = 0.15   # Fix 2
_BEAR_TREND_SIZE       = 0.07
_ES_LATE_SIZE          = 0.05

# 4H frequency signal sizes
_H4_BRK_CHOP_SIZE   = 0.12   # strong signal, CHOP regime
_H4_BRK_TRANS_SIZE  = 0.10   # slightly uncertain, TRANSITION regime
_H4_MOM_CHOP_SIZE   = 0.10   # weaker than breakout

# Max hold for 4H signals: SHORT — edge decays fast
_H4_HOLD_DAYS = {
    "H4_BRK_CHOP":  2,
    "H4_BRK_TRANS": 3,
    "H4_MOM_CHOP":  2,
}

_BASE_HOLD_DAYS = {
    **config.MAX_HOLD_DAYS,
    "TREND":                config.MAX_HOLD_DAYS.get("TREND", 3) + 1,   # base 4 for extension
    "BULL_RESET_CONFIRMED": config.MAX_HOLD_DAYS.get("TREND", 3) + 1,
    "STAT_ARB":             config.MAX_HOLD_DAYS.get("STAT_ARB", 5) + 1, # SA +1d
    "MTF_BREAKOUT": 2, "MTF_RESET": 3,
    "BEAR_TREND": 4, "FAILED_RALLY": 3, "SHOCK_CONT": 2, "BEAR_RESET": 3,
    "ES_LATE_EXCEPTION": 4,
    **_H4_HOLD_DAYS,   # inject short-hold 4H strategies
}


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


# ─── Pipeline ─────────────────────────────────────────────────────────────────
_DF0 = None

def build_base_df():
    global _DF0
    if _DF0 is not None:
        return _DF0
    logger.info("Building pipeline …")
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
        df.loc[idx, "entry_location_pct_20d"] = ((s-mn)/(mx-mn).replace(0,np.nan)).clip(0,1).values
    return df


# ─── Signal stack (fixes applied) ────────────────────────────────────────────

def build_fixed_stack(df0, scale=1.0):
    """Apply Fix 1+2+3 signal stack on top of base pipeline."""
    df = df0.copy()

    # ES late filter
    loc = df["entry_location_pct_20d"].fillna(0.5)
    late = ((df["symbol"]=="ES") & (df["strategy_used"]=="TREND") &
            (df["final_direction"]=="LONG") & (loc > 0.70))
    df.loc[late, "final_direction"] = "FLAT"
    df.loc[late, "strategy_used"]   = "NONE"

    # Regime router gating
    if "allow_trend_long" in df.columns:
        rg = ((df["strategy_used"]=="TREND") & (df["final_direction"]=="LONG") &
              (df["allow_trend_long"]==0))
        df.loc[rg, "final_direction"] = "FLAT"
        df.loc[rg, "strategy_used"]   = "NONE"

    # Fix 1: BULL_RESET at proper size
    slope = df["slope_20d"].fillna(0)
    ret20 = df["ret_20d"].fillna(0)
    ret5  = df["ret_5d"].fillna(0)
    bk    = df.get("structure_break_count", pd.Series(0, index=df.index)).fillna(0)
    loc2  = df["entry_location_pct_20d"].fillna(0.5)
    reg   = df["portfolio_regime"].fillna("CHOP")
    sent  = df["portfolio_sentiment_flag"].fillna("NORMAL")

    br = ((df["final_direction"]=="FLAT") & reg.isin(["TREND","TRANSITION"]) &
          (slope > 0) & (ret20 > 0) & (ret5 < 0) & (bk == 0) & (loc2 < 0.80) &
          (sent != "HIGH_RISK"))
    df.loc[br, "final_direction"] = "LONG"
    df.loc[br, "strategy_used"]   = "BULL_RESET_CONFIRMED"
    df.loc[br, "mtf_entry_size"]  = _BR_SIZE * scale
    df.loc[br, "entry_quality"]   = 4

    # Fix 2: SA in TRANSITION (NQ only)
    sa_trans = (
        (df["symbol"] == "NQ") &
        (df["sa_signal"] == "ENTRY") &
        (df["sa_nq_direction"] == "LONG") &
        (df["portfolio_regime"] == "TRANSITION") &
        (df["final_direction"] == "FLAT") &
        (sent != "HIGH_RISK")
    )
    df.loc[sa_trans, "final_direction"] = "LONG"
    df.loc[sa_trans, "strategy_used"]   = "STAT_ARB"
    df.loc[sa_trans, "mtf_entry_size"]  = _SA_TRANS_SIZE * scale
    df.loc[sa_trans, "entry_quality"]   = 3

    # ES late + NQ highconv
    nq_rows = df[df["symbol"]=="NQ"][["date","final_direction","high_conviction"]].copy()
    nq_rows = nq_rows.rename(columns={"final_direction":"nq_dir","high_conviction":"nq_hc"})
    df = df.merge(nq_rows, on="date", how="left")
    nq_dir = df["nq_dir"].fillna("FLAT")
    nq_hc  = df["nq_hc"].fillna(False).astype(bool)
    allow_es = ((df["symbol"]=="ES") & (df["final_direction"]=="FLAT") &
                (df["portfolio_regime"].fillna("CHOP")=="TREND") &
                (df["momentum_direction"].fillna("NEUTRAL")=="LONG") &
                (df["slope_20d"].fillna(0)>0) & (loc > 0.70) &
                (nq_dir=="LONG") & nq_hc &
                (df["portfolio_sentiment_flag"].fillna("NORMAL")!="HIGH_RISK"))
    df.loc[allow_es, "final_direction"] = "LONG"
    df.loc[allow_es, "strategy_used"]   = "ES_LATE_EXCEPTION"
    df.loc[allow_es, "mtf_entry_size"]  = _ES_LATE_SIZE * scale
    df.loc[allow_es, "entry_quality"]   = 4
    df = df.drop(columns=["nq_dir","nq_hc"], errors="ignore")

    # BEAR_TREND short
    df = compute_short_signals(df)
    not_bear = (df["short_signal_type"]!="BEAR_TREND") & (df["short_signal_type"]!="NONE")
    df.loc[not_bear, "final_direction"]   = "FLAT"
    df.loc[not_bear, "strategy_used"]     = "NONE"
    df.loc[not_bear, "short_signal_type"] = "NONE"
    bear_fired = df["short_signal_type"] == "BEAR_TREND"
    df.loc[bear_fired, "mtf_entry_size"]  = _BEAR_TREND_SIZE * scale

    logger.info("  Fixed stack: BULL_RESET=%d  SA_TRANS=%d  ES_LATE=%d  BEAR=%d",
                int(br.sum()), int(sa_trans.sum()), int(allow_es.sum()), int(bear_fired.sum()))
    return df


# ─── 4H frequency signal injections ──────────────────────────────────────────

def inject_h4_brk_chop(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """
    4H BREAKOUT in CHOP regime.
    Forward-looking edge: ES +0.29%/day, NQ +0.47%/day, 70-78% WR.
    Edge concentrated in Day 1 — hold max 2 days.
    Mechanism: 4H breaks out before daily regime classifier catches up.
    Filter: slope not strongly negative (no shorting against trend).
    """
    df = df.copy()
    sent  = df["portfolio_sentiment_flag"].fillna("NORMAL")
    slope = df["slope_20d"].fillna(0)

    mask = (
        (df["h4_exec_signal"] == "BREAKOUT") &
        (df["portfolio_regime"] == "CHOP") &
        (df["final_direction"] == "FLAT") &
        (slope > -0.0005) &        # not in clear downtrend
        (sent != "HIGH_RISK")
    )
    df.loc[mask, "final_direction"] = "LONG"
    df.loc[mask, "strategy_used"]   = "H4_BRK_CHOP"
    df.loc[mask, "mtf_entry_size"]  = _H4_BRK_CHOP_SIZE * scale
    df.loc[mask, "entry_quality"]   = 3

    n = int(mask.sum())
    logger.info("  inject_h4_brk_chop: %d entries (ES=%d, NQ=%d)",
                n,
                int((mask & (df["symbol"]=="ES")).sum()),
                int((mask & (df["symbol"]=="NQ")).sum()))
    return df


def inject_h4_brk_trans(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """
    4H BREAKOUT in TRANSITION regime.
    Forward-looking edge: ES +0.18%/day 62% WR, NQ +0.14-0.38% over 1-2 days.
    Hold up to 3 days — NQ edge continues into day 2.
    Filter: slope not strongly negative.
    """
    df = df.copy()
    sent  = df["portfolio_sentiment_flag"].fillna("NORMAL")
    slope = df["slope_20d"].fillna(0)

    mask = (
        (df["h4_exec_signal"] == "BREAKOUT") &
        (df["portfolio_regime"] == "TRANSITION") &
        (df["final_direction"] == "FLAT") &
        (slope > -0.0005) &
        (sent != "HIGH_RISK")
    )
    df.loc[mask, "final_direction"] = "LONG"
    df.loc[mask, "strategy_used"]   = "H4_BRK_TRANS"
    df.loc[mask, "mtf_entry_size"]  = _H4_BRK_TRANS_SIZE * scale
    df.loc[mask, "entry_quality"]   = 3

    n = int(mask.sum())
    logger.info("  inject_h4_brk_trans: %d entries (ES=%d, NQ=%d)",
                n,
                int((mask & (df["symbol"]=="ES")).sum()),
                int((mask & (df["symbol"]=="NQ")).sum()))
    return df


def inject_h4_mom_chop_nq(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """
    4H MOMENTUM in CHOP regime — NQ only.
    Forward-looking: NQ +0.25%/day 64% WR. ES version is too weak (+0.06%, 54%) — skip.
    Hold max 2 days.
    """
    df = df.copy()
    sent  = df["portfolio_sentiment_flag"].fillna("NORMAL")
    slope = df["slope_20d"].fillna(0)

    mask = (
        (df["symbol"] == "NQ") &      # NQ only — ES momentum in CHOP is 54% WR
        (df["h4_exec_signal"] == "MOMENTUM") &
        (df["portfolio_regime"] == "CHOP") &
        (df["final_direction"] == "FLAT") &
        (slope > 0) &                  # momentum requires upward slope
        (sent != "HIGH_RISK")
    )
    df.loc[mask, "final_direction"] = "LONG"
    df.loc[mask, "strategy_used"]   = "H4_MOM_CHOP"
    df.loc[mask, "mtf_entry_size"]  = _H4_MOM_CHOP_SIZE * scale
    df.loc[mask, "entry_quality"]   = 3

    n = int(mask.sum())
    logger.info("  inject_h4_mom_chop_nq: %d NQ entries", n)
    return df


# ─── Backtest runner ──────────────────────────────────────────────────────────

def _cfg_for_scale(scale):
    max_wt = min(_BASE_MAX_WT_PER_SYM * scale, 1.20)
    cfg = dict(
        vol_max      = _BASE_VOL_SCALE_MAX   * scale,
        stat_arb_w   = min(_BASE_STAT_ARB_WEIGHT * scale, 0.60),
        pb_w         = min(_BASE_PULLBACK_WEIGHT  * scale, 0.50),
        mr_w         = min(_BASE_MR_WEIGHT        * scale, 0.50),
        combined_exp = _BASE_MAX_COMBINED_EXP * scale,
    )
    return cfg, max_wt


def run_bt(label, df, scale=1.0, extend_h4_none=True, max_ext=14, extra=""):
    cfg, max_wt = _cfg_for_scale(scale)
    overrides = dict(
        VOL_SCALE_MAX          = cfg["vol_max"],
        STAT_ARB_NQ_WEIGHT     = cfg["stat_arb_w"],
        TREND_PULLBACK_WEIGHT  = cfg["pb_w"],
        MEAN_REVERT_WEIGHT     = cfg["mr_w"],
        MAX_COMBINED_EXPOSURE  = cfg["combined_exp"],
        MAX_HOLD_DAYS          = _BASE_HOLD_DAYS,
        MAX_WEIGHT_PER_SYMBOL  = max_wt,
        ADD_WINNER_SIZE_FRAC   = 0.50,
        MAX_EXTENDED_HOLD_DAYS = max_ext,
        EXTEND_ALLOW_H4_NONE   = extend_h4_none,
    )
    with override_config(**overrides):
        res = run_backtest(df)

    st = res["stats"]; tl = res["trade_log"]; pr = res["portfolio_returns"]

    yr = {}
    for y in range(2019, 2027):
        m = pr.index.year == y
        if m.any():
            yr[y] = round(float((1+pr[m]).prod()-1)*100, 2)

    down = pr[pr < 0]
    ds   = float(down.std())*np.sqrt(252) if len(down)>5 else np.nan
    srt  = float(pr.mean())*252/ds if (ds and ds>1e-9) else np.nan

    wk = pr.resample("W").sum()
    cu = (1+pr).cumprod(); mx = cu.cummax()
    dd = (cu-mx)/mx
    avg_dd = float(dd[dd<0].mean())*100 if (dd<0).any() else 0.0

    strat_pnl = {}
    if not tl.empty and "strategy" in tl.columns:
        strat_pnl = {s: round(float(g["pnl"].sum()),4) for s,g in tl.groupby("strategy")}

    trades_pw = round(st["total_trades"]/(len(pr)/5), 2)
    final_25k = int(25000*(1+st["annualized_return_pct"]/100)**7)

    row = dict(
        label=label, info=extra,
        sharpe=round(st["sharpe_ratio"],4),
        sortino=round(srt,4),
        annual=round(st["annualized_return_pct"],2),
        max_dd=round(st["max_drawdown_pct"],2),
        avg_dd=round(avg_dd,2),
        worst_wk=round(float(wk.min())*100,2),
        trades=st["total_trades"],
        trades_pw=trades_pw,
        wr=round(st.get("overall_win_rate",0),4),
        pf=round(st.get("profit_factor",0),4),
        final_25k=final_25k,
        **{f"yr_{k}": v for k,v in yr.items()},
        **{f"s_{k}": v for k,v in strat_pnl.items()},
    )
    logger.info(
        "%-34s | Sh=%.3f  Ann=%+.1f%%  MaxDD=%.1f%%  Trades=%d(%.2f/wk)  WR=%.1f%%  $25k→$%d",
        label, row["sharpe"], row["annual"], row["max_dd"],
        row["trades"], trades_pw, row["wr"]*100, final_25k,
    )
    return row, tl


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df0 = build_base_df()
    results = []

    # ── Baseline: F1+F2+F3 (best fixes without freq changes) ─────────────────
    logger.info("="*60 + "\nBASELINE_FIXES (F1+F2+F3)")
    df_base = build_fixed_stack(df0, scale=1.0)
    row, tl_base = run_bt("BASELINE_FIXES", df_base, scale=1.0,
                          extend_h4_none=True, max_ext=14,
                          extra="F1+F2+F3, no freq change")
    results.append(row)
    base_sh  = row["sharpe"]
    base_ann = row["annual"]
    base_tr  = row["trades"]
    base_pw  = row["trades_pw"]

    # ── Isolated frequency signals ────────────────────────────────────────────
    logger.info("="*60 + "\nFREQ_BRK_CHOP")
    df_f = inject_h4_brk_chop(df_base, scale=1.0)
    row, _ = run_bt("FREQ_BRK_CHOP", df_f, scale=1.0,
                    extend_h4_none=True, max_ext=14,
                    extra="4H BREAKOUT in CHOP only")
    results.append(row)

    logger.info("="*60 + "\nFREQ_BRK_TRANS")
    df_f = inject_h4_brk_trans(df_base, scale=1.0)
    row, _ = run_bt("FREQ_BRK_TRANS", df_f, scale=1.0,
                    extend_h4_none=True, max_ext=14,
                    extra="4H BREAKOUT in TRANSITION only")
    results.append(row)

    logger.info("="*60 + "\nFREQ_MOM_CHOP_NQ")
    df_f = inject_h4_mom_chop_nq(df_base, scale=1.0)
    row, _ = run_bt("FREQ_MOM_CHOP_NQ", df_f, scale=1.0,
                    extend_h4_none=True, max_ext=14,
                    extra="4H MOMENTUM CHOP NQ only")
    results.append(row)

    # ── All frequency signals combined ────────────────────────────────────────
    logger.info("="*60 + "\nFREQ_ALL")
    df_freq = inject_h4_mom_chop_nq(
        inject_h4_brk_trans(
            inject_h4_brk_chop(df_base, scale=1.0),
        scale=1.0),
    scale=1.0)
    row, tl_freq = run_bt("FREQ_ALL_1X", df_freq, scale=1.0,
                          extend_h4_none=True, max_ext=14,
                          extra="all freq signals, 1x")
    results.append(row)

    # ── Scaling: all fixes + all frequency signals ────────────────────────────
    logger.info("="*60 + "\nSCALING EXPERIMENTS")
    best_tl = None
    for scale_name, scale in [("1.5x",1.5), ("2.0x",2.0), ("2.5x",2.5), ("3.0x",3.0)]:
        df_s = build_fixed_stack(df0, scale=scale)
        df_s = inject_h4_brk_chop(df_s, scale=scale)
        df_s = inject_h4_brk_trans(df_s, scale=scale)
        df_s = inject_h4_mom_chop_nq(df_s, scale=scale)
        row, tl = run_bt(f"FULL_{scale_name}", df_s, scale=scale,
                         extend_h4_none=True, max_ext=14,
                         extra=f"all fixes+freq @ {scale_name}")
        results.append(row)
        if scale_name == "2.5x":
            best_tl = tl

    # ── Save ──────────────────────────────────────────────────────────────────
    res = pd.DataFrame(results)
    res["d_sh"]  = (res["sharpe"]  - base_sh ).round(4)
    res["d_ann"] = (res["annual"]  - base_ann).round(2)
    res["d_tr"]  = (res["trades"]  - base_tr)
    res["d_pw"]  = (res["trades_pw"] - base_pw).round(2)
    res.to_csv(f"{OUT_DIR}/frequency_results.csv", index=False)
    if best_tl is not None:
        best_tl.to_csv(f"{OUT_DIR}/full_2.5x_trade_log.csv", index=False)

    _print_report(res, base_sh, base_ann, base_tr, base_pw, tl_freq)


def _print_report(res, base_sh, base_ann, base_tr, base_pw, tl_all):
    W = "="*80

    print(f"\n{W}")
    print("FREQUENCY EXPERIMENT — RESULTS")
    print(W)
    cols = ["label","info","sharpe","d_sh","annual","d_ann","max_dd",
            "trades","d_tr","trades_pw","d_pw","wr","pf","final_25k"]
    print(res[[c for c in cols if c in res.columns]].to_string(index=False))

    # Annual breakdown
    yr_cols = sorted([c for c in res.columns if c.startswith("yr_")])
    if yr_cols:
        print(f"\n{W}")
        print("ANNUAL RETURNS BY YEAR")
        print(W)
        print(res[["label"]+yr_cols].to_string(index=False))

    # Per-strategy PnL for FREQ_ALL_1X
    print(f"\n{W}")
    print("PnL BY STRATEGY — FREQ_ALL_1X")
    print(W)
    for strat, g in tl_all.groupby("strategy"):
        n=len(g); wr=g["win"].mean(); tot=g["pnl"].sum(); avg=g["pnl"].mean()
        wt=g["entry_weight"].abs().mean(); hold=g["holding_days"].mean()
        print(f"  {strat:<20} n={n:3d}  WR={wr:.1%}  avg={avg:.5f}  tot={tot:.4f}  "
              f"avgWt={wt:.4f}  hold={hold:.1f}d")

    # Summary
    print(f"\n{W}")
    print("SIGNAL CONTRIBUTION — ISOLATED")
    print(W)
    print(f"  {'Label':<25} {'Sharpe':>7} {'ΔSh':>7} {'Annual':>7} {'Trades':>7} "
          f"{'Δtrades':>8} {'tr/wk':>7} {'MaxDD':>7}")
    print(f"  {'-'*73}")
    for lbl in ["BASELINE_FIXES","FREQ_BRK_CHOP","FREQ_BRK_TRANS",
                "FREQ_MOM_CHOP_NQ","FREQ_ALL_1X"]:
        r = res[res["label"]==lbl]
        if r.empty: continue
        r = r.iloc[0]
        flag = " ←" if r["d_sh"] > 0 else ""
        print(f"  {lbl:<25} {r['sharpe']:>7.3f} {r['d_sh']:>+7.3f} "
              f"{r['annual']:>+6.1f}% {r['trades']:>7}  "
              f"{r['d_tr']:>+7}  {r['trades_pw']:>6.2f}  {r['max_dd']:>6.1f}%{flag}")

    print(f"\n{W}")
    print("FULL SYSTEM — SCALING GRID  (fixes + all freq signals)")
    print(W)
    print(f"  {'Scale':<16} {'Sharpe':>8} {'ΔSh':>7} {'Annual':>8} {'MaxDD':>7} "
          f"{'$25k→':>10} {'tr/wk':>7}")
    print(f"  {'-'*68}")
    for lbl in ["BASELINE_FIXES","FREQ_ALL_1X",
                "FULL_1.5x","FULL_2.0x","FULL_2.5x","FULL_3.0x"]:
        r = res[res["label"]==lbl]
        if r.empty: continue
        r = r.iloc[0]
        target = " ← 10% target" if r["annual"] >= 10.0 else ""
        print(f"  {lbl:<16} {r['sharpe']:>8.3f} {r['d_sh']:>+7.3f} "
              f"{r['annual']:>+7.1f}% {r['max_dd']:>6.1f}% "
              f"${r['final_25k']:>9,} {r['trades_pw']:>6.2f}{target}")

    # SPY context
    print(f"\n{W}")
    print("CONTEXT: SPY CAGR ~15%")
    print(W)
    for lbl in ["FREQ_ALL_1X","FULL_1.5x","FULL_2.0x","FULL_2.5x"]:
        r = res[res["label"]==lbl]
        if r.empty: continue
        r = r.iloc[0]
        print(f"  {lbl:<18} System={r['annual']:+.1f}%  SPY≈+15%  "
              f"Gap={r['annual']-15:.1f}%  MaxDD={r['max_dd']:.1f}%  "
              f"{r['trades_pw']:.2f} trades/wk")
    print(W)


if __name__ == "__main__":
    main()
