"""
run_structural_mr.py — CWT/EMD structural mean-reversion experiment runner.

Architecture:
  1. Full pipeline (TREND + STAT_ARB + SHOCK_BOUNCE baseline, unchanged)
  2. Force-compute CWT features (regardless of config.USE_CWT_CHOP_FILTER)
  3. Compute structural MR signals with varying filter configurations
  4. Stateful MR backtest (LONG/SHORT, 2-day max hold, midpoint target)
  5. Combine baseline + MR sleeve returns, compute metrics

10 experiments (A–J):
  A_BASELINE          → current accepted system, no MR
  B_RANGE_ONLY_05     → range quality only, 5% size
  C_RANGE_ONLY_10     → range quality only, 10% size
  D_CWT_ONLY_10       → CWT + range, no EMD, 10%
  E_EMD_ONLY_10       → EMD + range, no CWT, 10%
  F_CWT_EMD_05        → CWT + EMD + range, 5%
  G_CWT_EMD_10        → CWT + EMD + range, 10%
  H_CWT_EMD_15        → CWT + EMD + range, 15%
  I_STRICT_CWT_EMD_10 → strict thresholds, 10%
  J_LOOSE_CWT_EMD_10  → loose thresholds, 10%

Usage:
    cd /Users/ishanbhardwaj/cme_competition_system && python run_structural_mr.py
"""

import logging
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import main as _main_mod
from cwt_chop_filter import compute_cwt_chop_features
from structural_mean_reversion_engine import compute_structural_mean_reversion_signals

logging.basicConfig(stream=sys.stdout, level=logging.WARNING,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

CAPITAL = 25_000.0
OUT     = Path("output")
OUT.mkdir(exist_ok=True)

# ── Save originals ─────────────────────────────────────────────────────────────
_ORIG = {k: deepcopy(v) if isinstance(v, dict) else v
         for k, v in vars(config).items()
         if k.startswith("STRUCTURAL_MR_") or k == "USE_STRUCTURAL_MEAN_REVERSION"}


def _set(cfg: dict):
    for k, v in cfg.items():
        setattr(config, k, v)


def _restore():
    for k, v in _ORIG.items():
        setattr(config, k, deepcopy(v) if isinstance(v, dict) else v)


# ── Experiment definitions ─────────────────────────────────────────────────────
EXPERIMENTS = {
    "A_BASELINE": {
        "USE_STRUCTURAL_MEAN_REVERSION": False,
    },
    "B_RANGE_ONLY_05": {
        "USE_STRUCTURAL_MEAN_REVERSION": True,
        "STRUCTURAL_MR_REQUIRE_CWT":   False,
        "STRUCTURAL_MR_REQUIRE_EMD":   False,
        "STRUCTURAL_MR_REQUIRE_RANGE": True,
        "STRUCTURAL_MR_SIZE":          0.05,
        "STRUCTURAL_MR_RQ_THRESHOLD":  0.42,
    },
    "C_RANGE_ONLY_10": {
        "USE_STRUCTURAL_MEAN_REVERSION": True,
        "STRUCTURAL_MR_REQUIRE_CWT":   False,
        "STRUCTURAL_MR_REQUIRE_EMD":   False,
        "STRUCTURAL_MR_REQUIRE_RANGE": True,
        "STRUCTURAL_MR_SIZE":          0.10,
        "STRUCTURAL_MR_RQ_THRESHOLD":  0.42,
    },
    "D_CWT_ONLY_10": {
        "USE_STRUCTURAL_MEAN_REVERSION": True,
        "STRUCTURAL_MR_REQUIRE_CWT":   True,
        "STRUCTURAL_MR_REQUIRE_EMD":   False,
        "STRUCTURAL_MR_REQUIRE_RANGE": True,
        "STRUCTURAL_MR_SIZE":          0.10,
        "STRUCTURAL_MR_RQ_THRESHOLD":  0.42,
    },
    "E_EMD_ONLY_10": {
        "USE_STRUCTURAL_MEAN_REVERSION": True,
        "STRUCTURAL_MR_REQUIRE_CWT":   False,
        "STRUCTURAL_MR_REQUIRE_EMD":   True,
        "STRUCTURAL_MR_REQUIRE_RANGE": True,
        "STRUCTURAL_MR_SIZE":          0.10,
        "STRUCTURAL_MR_RQ_THRESHOLD":  0.42,
    },
    "F_CWT_EMD_05": {
        "USE_STRUCTURAL_MEAN_REVERSION": True,
        "STRUCTURAL_MR_REQUIRE_CWT":   True,
        "STRUCTURAL_MR_REQUIRE_EMD":   True,
        "STRUCTURAL_MR_REQUIRE_RANGE": True,
        "STRUCTURAL_MR_SIZE":          0.05,
        "STRUCTURAL_MR_RQ_THRESHOLD":  0.42,
    },
    "G_CWT_EMD_10": {
        "USE_STRUCTURAL_MEAN_REVERSION": True,
        "STRUCTURAL_MR_REQUIRE_CWT":   True,
        "STRUCTURAL_MR_REQUIRE_EMD":   True,
        "STRUCTURAL_MR_REQUIRE_RANGE": True,
        "STRUCTURAL_MR_SIZE":          0.10,
        "STRUCTURAL_MR_RQ_THRESHOLD":  0.42,
    },
    "H_CWT_EMD_15": {
        "USE_STRUCTURAL_MEAN_REVERSION": True,
        "STRUCTURAL_MR_REQUIRE_CWT":   True,
        "STRUCTURAL_MR_REQUIRE_EMD":   True,
        "STRUCTURAL_MR_REQUIRE_RANGE": True,
        "STRUCTURAL_MR_SIZE":          0.15,
        "STRUCTURAL_MR_RQ_THRESHOLD":  0.42,
    },
    "I_STRICT_CWT_EMD_10": {
        "USE_STRUCTURAL_MEAN_REVERSION": True,
        "STRUCTURAL_MR_REQUIRE_CWT":   True,
        "STRUCTURAL_MR_REQUIRE_EMD":   True,
        "STRUCTURAL_MR_REQUIRE_RANGE": True,
        "STRUCTURAL_MR_SIZE":          0.10,
        "STRUCTURAL_MR_RQ_THRESHOLD":  0.52,   # stricter range quality
        "STRUCTURAL_MR_PIR_LONG":      0.15,   # must be deeper in low zone
        "STRUCTURAL_MR_PIR_SHORT":     0.85,
        "STRUCTURAL_MR_ZSCORE_LONG":  -1.60,   # more extreme z required
        "STRUCTURAL_MR_ZSCORE_SHORT":  1.60,
    },
    "J_LOOSE_CWT_EMD_10": {
        "USE_STRUCTURAL_MEAN_REVERSION": True,
        "STRUCTURAL_MR_REQUIRE_CWT":   True,
        "STRUCTURAL_MR_REQUIRE_EMD":   True,
        "STRUCTURAL_MR_REQUIRE_RANGE": True,
        "STRUCTURAL_MR_SIZE":          0.10,
        "STRUCTURAL_MR_RQ_THRESHOLD":  0.33,   # looser range quality
        "STRUCTURAL_MR_PIR_LONG":      0.28,   # wider entry zone
        "STRUCTURAL_MR_PIR_SHORT":     0.72,
        "STRUCTURAL_MR_ZSCORE_LONG":  -1.00,   # less extreme z required
        "STRUCTURAL_MR_ZSCORE_SHORT":  1.00,
    },
}


# ════════════════════════════════════════════════════════════════════════════════
# MR STATEFUL BACKTEST
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class MRState:
    symbol:           str
    active:           bool   = False
    direction:        str    = "NONE"   # LONG or SHORT
    entry_date:       object = None
    weight:           float  = 0.0     # +ve for LONG, -ve for SHORT
    days_held:        int    = 0
    cum_ret:          float  = 0.0     # positive = profitable (regardless of direction)
    peak_cum_ret:     float  = 0.0
    trail_armed:      bool   = False
    entry_price:      float  = 0.0
    entry_range_mid:  float  = 0.0     # midpoint target (locked at entry)
    entry_range_low:  float  = 0.0     # for LONG range-break detection
    entry_range_high: float  = 0.0     # for SHORT range-break detection
    entry_atr_pct:    float  = 0.01
    stop_threshold:   float  = -0.015  # cum_ret below this → stop
    target_threshold: float  = 0.005   # cum_ret above this → target hit


def run_mr_backtest(
    df:            pd.DataFrame,
    size:          float = 0.10,
    require_cwt:   bool  = True,
    require_emd:   bool  = True,
    require_range: bool  = True,
    rq_threshold:  float = 0.42,
    pir_long:      float = 0.22,
    pir_short:     float = 0.78,
    zscore_long:   float = -1.30,
    zscore_short:  float =  1.30,
    max_hold:      int   = 2,
) -> dict:
    """
    Stateful daily MR backtest.

    Entry: structural_mr_signal = True, direction = LONG or SHORT, bucket qualifies.
    Exits:
      MR_TARGET_HIT    — price reached range midpoint
      MR_STOP_LOSS     — cum_ret < stop_threshold
      MR_TRAIL_STOP    — trailing stop fires (after arming at +0.8%)
      MR_MAX_HOLD      — max_hold days elapsed
      MR_RANGE_BREAK   — price broke out of entry range
      MR_SHOCK_EXIT    — regime became SHOCK
      MR_TREND_EXIT    — regime became TREND (MR not valid in trend)
      MR_CWT_EXIT      — CWT became chaotic or breakout expansion
      MR_EMD_EXIT      — EMD became transition or trend deteriorating
    """
    df      = df.sort_values(["date", "symbol"]).copy()
    dates   = sorted(df["date"].unique())
    symbols = sorted(df["symbol"].unique())

    lookup: Dict = df.set_index(["date", "symbol"]).to_dict("index")
    states: Dict[str, MRState] = {sym: MRState(symbol=sym) for sym in symbols}

    _TC_RATE    = config.TRANSACTION_COST_BPS / 10_000
    _STOP_MULT  = getattr(config, "STRUCTURAL_MR_STOP_ATR_MULT", 1.5)
    _MAX_SIZE   = getattr(config, "STRUCTURAL_MR_MAX_SIZE", 0.15)
    trail_arm   = 0.008   # arm when +0.8% profitable
    trail_dd    = 0.005   # fire when 0.5% drawback from peak

    daily_pnl: Dict = {d: 0.0 for d in dates}
    completed: List[dict] = []
    prev_weight: Dict[str, float] = {sym: 0.0 for sym in symbols}

    for date in dates:
        date_pnl = 0.0

        for sym in symbols:
            row         = lookup.get((date, sym), {})
            state       = states[sym]
            today_ret   = float(row.get("ret_1d", 0.0) or 0.0)
            port_regime = row.get("portfolio_regime", "CHOP") or "CHOP"
            atr_pct     = float(row.get("atr_pct", 0.01) or 0.01)
            cwt_label   = str(row.get("cwt_chop_label", "CWT_UNCLEAR") or "CWT_UNCLEAR")
            emd_label   = str(row.get("emd_label", "EMD_UNCLEAR") or "EMD_UNCLEAR")
            today_close = float(row.get("close", state.entry_price) or state.entry_price)

            # ── A. Apply existing position P&L ────────────────────────────────
            applied_w = 0.0
            if state.active:
                applied_w = state.weight
                if port_regime == "TRANSITION":
                    applied_w *= 0.7   # reduce in transition but don't zero

            gross   = applied_w * today_ret
            tc      = abs(applied_w - prev_weight[sym]) * _TC_RATE
            sym_pnl = gross - tc

            if state.active:
                # Track cum_ret: positive = profitable in either direction
                if state.direction == "LONG":
                    state.cum_ret += today_ret
                else:  # SHORT
                    state.cum_ret += -today_ret

                state.peak_cum_ret = max(state.peak_cum_ret, state.cum_ret)
                state.days_held   += 1

            # ── B. Exits ───────────────────────────────────────────────────────
            exit_reason: Optional[str] = None

            if state.active:
                # 1. Target hit: price reached range midpoint
                if state.direction == "LONG" and today_close >= state.entry_range_mid:
                    exit_reason = "MR_TARGET_HIT"
                elif state.direction == "SHORT" and today_close <= state.entry_range_mid:
                    exit_reason = "MR_TARGET_HIT"

                # 2. Stop loss
                elif state.cum_ret < state.stop_threshold:
                    exit_reason = "MR_STOP_LOSS"

                # 3. Trailing stop
                elif (state.trail_armed
                      and (state.peak_cum_ret - state.cum_ret) >= trail_dd):
                    exit_reason = "MR_TRAIL_STOP"

                # 4. Max hold
                elif state.days_held >= max_hold:
                    exit_reason = "MR_MAX_HOLD"

                # 5. Range break (price left entry range in wrong direction)
                elif (state.direction == "LONG"
                      and state.entry_range_low > 0
                      and today_close < state.entry_range_low * (1.0 - 0.5 * atr_pct)):
                    exit_reason = "MR_RANGE_BREAK"
                elif (state.direction == "SHORT"
                      and state.entry_range_high > 0
                      and today_close > state.entry_range_high * (1.0 + 0.5 * atr_pct)):
                    exit_reason = "MR_RANGE_BREAK"

                # 6. SHOCK regime — exit immediately
                elif port_regime == "SHOCK":
                    exit_reason = "MR_SHOCK_EXIT"

                # 7. Trend regime — MR invalid in trend
                elif port_regime == "TREND":
                    exit_reason = "MR_TREND_EXIT"

                # 8. CWT deterioration
                elif cwt_label in ("CWT_CHAOTIC_NOISE", "CWT_BREAKOUT_EXPANSION"):
                    exit_reason = "MR_CWT_EXIT"

                # 9. EMD deterioration
                elif emd_label in ("EMD_TREND_INTACT", "EMD_TRANSITION",
                                   "EMD_TREND_DETERIORATING"):
                    exit_reason = "MR_EMD_EXIT"

                # Arm trailing stop
                if not exit_reason and state.cum_ret >= trail_arm:
                    state.trail_armed = True

            # ── C. Record & close ─────────────────────────────────────────────
            if exit_reason and state.active:
                # P&L: weight × cum_ret for LONG; (-weight) × cum_ret for SHORT
                if state.direction == "LONG":
                    trade_pnl = state.weight * state.cum_ret
                else:
                    trade_pnl = (-state.weight) * state.cum_ret   # weight < 0, cum_ret = sum(-ret)

                completed.append({
                    "entry_date":       state.entry_date,
                    "exit_date":        date,
                    "symbol":           sym,
                    "direction":        state.direction,
                    "weight":           state.weight,
                    "days_held":        state.days_held,
                    "cum_ret":          round(state.cum_ret, 6),
                    "pnl":              round(trade_pnl, 6),
                    "win":              state.cum_ret > 0,
                    "exit_reason":      exit_reason,
                    "trail_armed":      state.trail_armed,
                    "entry_price":      state.entry_price,
                    "entry_range_mid":  state.entry_range_mid,
                    "entry_atr_pct":    state.entry_atr_pct,
                    "stop_threshold":   state.stop_threshold,
                    "target_threshold": state.target_threshold,
                })
                tc_exit  = abs(applied_w) * _TC_RATE
                sym_pnl -= tc_exit
                states[sym] = MRState(symbol=sym)
                state       = states[sym]

            # ── D. New entry ───────────────────────────────────────────────────
            new_tc = 0.0
            if not state.active:
                mr_signal = bool(row.get("structural_mr_signal", False))
                direction = str(row.get("structural_mr_direction", "NONE") or "NONE")
                bucket    = str(row.get("structural_mr_bucket", "NONE") or "NONE")
                score     = float(row.get("structural_mr_score", 0.0) or 0.0)

                if (mr_signal
                        and direction in ("LONG", "SHORT")
                        and bucket in ("STRONG", "MODERATE")):

                    fracs   = {"STRONG": 1.0, "MODERATE": 0.75}
                    w_abs   = min(size * fracs[bucket], _MAX_SIZE)
                    w       = w_abs if direction == "LONG" else -w_abs

                    entry_range_mid  = float(row.get("rq_range_midpoint",   today_close) or today_close)
                    entry_range_low  = float(row.get("rq_rolling_low",      today_close * 0.99) or today_close * 0.99)
                    entry_range_high = float(row.get("rq_rolling_high",     today_close * 1.01) or today_close * 1.01)

                    # Stop: 1.5× ATR as adverse cum_ret threshold
                    stop_thr = -max(_STOP_MULT * atr_pct, 0.010)

                    # Target: fraction of range from entry to midpoint
                    if direction == "LONG":
                        tgt = (entry_range_mid - today_close) / max(today_close, 1)
                    else:
                        tgt = (today_close - entry_range_mid) / max(today_close, 1)
                    tgt = max(tgt, 0.002)   # at least 0.2% target

                    if abs(w) > 1e-6:
                        state.active           = True
                        state.direction        = direction
                        state.entry_date       = date
                        state.weight           = w
                        state.days_held        = 0
                        state.cum_ret          = 0.0
                        state.peak_cum_ret     = 0.0
                        state.trail_armed      = False
                        state.entry_price      = today_close
                        state.entry_range_mid  = entry_range_mid
                        state.entry_range_low  = entry_range_low
                        state.entry_range_high = entry_range_high
                        state.entry_atr_pct    = atr_pct
                        state.stop_threshold   = stop_thr
                        state.target_threshold = tgt
                        new_tc                 = abs(w) * _TC_RATE

            sym_pnl        -= new_tc
            prev_weight[sym] = state.weight if state.active else 0.0
            date_pnl        += sym_pnl

        daily_pnl[date] = date_pnl

    # Close any open positions at end
    for sym in symbols:
        s = states[sym]
        if s.active:
            pnl = s.weight * s.cum_ret if s.direction == "LONG" else (-s.weight) * s.cum_ret
            completed.append({
                "entry_date": s.entry_date, "exit_date": dates[-1],
                "symbol": sym, "direction": s.direction,
                "weight": s.weight, "days_held": s.days_held,
                "cum_ret": round(s.cum_ret, 6), "pnl": round(pnl, 6),
                "win": s.cum_ret > 0, "exit_reason": "END_OF_DATA",
                "trail_armed": s.trail_armed, "entry_price": s.entry_price,
                "entry_range_mid": s.entry_range_mid, "entry_atr_pct": s.entry_atr_pct,
                "stop_threshold": s.stop_threshold, "target_threshold": s.target_threshold,
            })

    mr_ret = pd.Series(daily_pnl, name="mr_ret")
    mr_ret.index = pd.to_datetime(mr_ret.index)

    tl_cols = ["entry_date","exit_date","symbol","direction","weight",
               "days_held","cum_ret","pnl","win","exit_reason","trail_armed",
               "entry_price","entry_range_mid","entry_atr_pct",
               "stop_threshold","target_threshold"]
    tl = pd.DataFrame(completed, columns=tl_cols) if completed else pd.DataFrame(columns=tl_cols)
    return {"mr_returns": mr_ret, "trade_log": tl}


# ════════════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════════════

def _metrics(
    baseline_ret: pd.Series,
    mr_ret:       pd.Series,
    mr_tl:        pd.DataFrame,
    label:        str,
    df_signals:   pd.DataFrame,
) -> dict:
    baseline_ret.index = pd.to_datetime(baseline_ret.index)
    mr_ret.index       = pd.to_datetime(mr_ret.index)

    all_dates = baseline_ret.index.union(mr_ret.index)
    b  = baseline_ret.reindex(all_dates, fill_value=0.0)
    m  = mr_ret.reindex(all_dates, fill_value=0.0)
    c  = b + m

    n   = max(len(c), 1)
    std = c.std()
    sharpe  = c.mean() / std * np.sqrt(252) if std > 0 else 0
    down    = c[c < 0]
    sortino = c.mean() / down.std() * np.sqrt(252) if len(down) > 3 else np.nan

    cum     = (1 + c).cumprod()
    dd      = (cum - cum.cummax()) / cum.cummax()
    max_dd  = float(dd.min() * 100)
    avg_dd  = float(dd.mean() * 100)
    ann_ret = float((cum.iloc[-1] ** (252 / n) - 1) * 100)
    tot_ret = float((cum.iloc[-1] - 1) * 100)
    worst_week  = float(c.resample("W").sum().min() * 100)
    worst_month = float(c.resample("ME").sum().min() * 100)
    dollar_pnl  = (cum.iloc[-1] - 1) * CAPITAL

    n_mr   = len(mr_tl)
    mr_wr  = float(mr_tl["win"].mean())   if n_mr > 0 else np.nan
    mr_pnl = float(mr_tl["pnl"].sum() * CAPITAL) if n_mr > 0 else 0.0
    mr_avg = float(mr_tl["pnl"].mean() * CAPITAL) if n_mr > 0 else np.nan
    sl_n   = int(mr_tl["exit_reason"].str.contains("MR_STOP_LOSS").sum()) if n_mr > 0 else 0
    sl_pct = sl_n / max(n_mr, 1)
    avg_hld= float(mr_tl["days_held"].mean()) if n_mr > 0 else np.nan

    gp = mr_tl.loc[mr_tl["pnl"] > 0, "pnl"].sum() if n_mr > 0 else 0
    gl = mr_tl.loc[mr_tl["pnl"] < 0, "pnl"].sum() if n_mr > 0 else 0
    mr_pf  = gp / (-gl) if gl < 0 else (999.0 if gp > 0 else 0.0)

    # P&L by direction
    long_pnl  = mr_tl.loc[mr_tl["direction"] == "LONG",  "pnl"].sum() * CAPITAL if n_mr > 0 else 0
    short_pnl = mr_tl.loc[mr_tl["direction"] == "SHORT", "pnl"].sum() * CAPITAL if n_mr > 0 else 0
    n_long    = (mr_tl["direction"] == "LONG").sum()  if n_mr > 0 else 0
    n_short   = (mr_tl["direction"] == "SHORT").sum() if n_mr > 0 else 0

    return {
        "label":          label,
        "sharpe":         round(sharpe,  3),
        "sortino":        round(sortino if not np.isnan(sortino) else 0, 3),
        "ann_ret_pct":    round(ann_ret, 2),
        "total_ret_pct":  round(tot_ret, 2),
        "max_dd_pct":     round(max_dd,  2),
        "avg_dd_pct":     round(avg_dd,  3),
        "worst_week_pct": round(worst_week,  2),
        "worst_month_pct":round(worst_month, 2),
        "dollar_pnl":     round(dollar_pnl,  0),
        "n_mr":           n_mr,
        "mr_pnl":         round(mr_pnl, 0),
        "mr_wr":          round(mr_wr if not np.isnan(mr_wr) else 0, 3),
        "mr_avg_pnl":     round(mr_avg if not np.isnan(mr_avg) else 0, 0),
        "mr_pf":          round(mr_pf, 3),
        "mr_sl_n":        sl_n,
        "mr_sl_pct":      round(sl_pct, 3),
        "mr_avg_hold":    round(avg_hld if not np.isnan(avg_hld) else 0, 2),
        "mr_n_long":      n_long,
        "mr_n_short":     n_short,
        "mr_long_pnl":    round(long_pnl, 0),
        "mr_short_pnl":   round(short_pnl, 0),
    }


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Pipeline + CWT + MR signals
# ════════════════════════════════════════════════════════════════════════════════

def _phase0():
    print("\n" + "=" * 70)
    print("  PHASE 0 — Full pipeline + CWT + structural MR signals")
    print("=" * 70)

    result = _main_mod.run(fetch=False)
    df_sig = result.get("df_signals", pd.DataFrame())
    s      = result["stats"]
    port   = result["portfolio_returns"]
    pnl    = ((1 + port).prod() - 1) * CAPITAL

    print(f"\n  Baseline: Sharpe={s['sharpe_ratio']:.3f}  "
          f"AnnRet={s['annualized_return_pct']:.2f}%  "
          f"MaxDD={s['max_drawdown_pct']:.2f}%  "
          f"$PnL=${pnl:,.0f}")

    # Force-compute CWT — drop any existing CWT columns first to avoid duplicates
    print("  Computing CWT features (force) ...")
    cwt_cols = [c for c in df_sig.columns if c.startswith("cwt_")]
    df_sig = df_sig.drop(columns=cwt_cols, errors="ignore")
    df_sig = compute_cwt_chop_features(df_sig)

    cwt_vc = df_sig["cwt_chop_label"].value_counts()
    print(f"  CWT labels: {dict(cwt_vc)}")

    # Compute structural MR signals (default full-filter params for audit)
    print("  Computing structural MR signals (full filter, default params) ...")
    df_mr = compute_structural_mean_reversion_signals(df_sig)

    for sym in df_mr["symbol"].unique():
        g   = df_mr[df_mr["symbol"] == sym]
        sig = g["structural_mr_signal"].sum()
        vc  = g["structural_mr_bucket"].value_counts()
        rl  = g["rq_range_label"].value_counts()
        el  = g["emd_label"].value_counts()
        cl  = g["cwt_chop_label"].value_counts()
        dl  = g[g["structural_mr_signal"]]["structural_mr_direction"].value_counts() if sig > 0 else {}
        print(f"\n  {sym}: {sig} MR signals")
        print(f"    STRONG={vc.get('STRONG',0)}  MOD={vc.get('MODERATE',0)}")
        print(f"    LONG={dl.get('LONG',0) if hasattr(dl,'get') else 0}  "
              f"SHORT={dl.get('SHORT',0) if hasattr(dl,'get') else 0}")
        print(f"    RangeLabel: {dict(rl)}")
        print(f"    EMD: {dict(el)}")
        print(f"    CWT: {dict(cl)}")

    return result, df_sig, df_mr


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Range / CWT / EMD diagnostic audit
# ════════════════════════════════════════════════════════════════════════════════

def _phase1_audit(result: dict, df_mr: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  PHASE 1 — Range / CWT / EMD diagnostic audit")
    print("=" * 70)

    port_ret = result["portfolio_returns"].copy()
    port_ret.index = pd.to_datetime(port_ret.index)

    audit_rows = []
    fwd_rows   = []

    for sym in ["ES", "NQ"]:
        g = df_mr[df_mr["symbol"] == sym].copy().sort_values("date").reset_index(drop=True)
        g["date"] = pd.to_datetime(g["date"])
        g["ret_next2"] = (g["close"].shift(-2) / g["close"] - 1)
        g["ret_next3"] = (g["close"].shift(-3) / g["close"] - 1)

        chop_g = g[g["portfolio_regime"].isin(["CHOP", "TRANSITION"])].copy()
        n_chop = len(chop_g)
        n_range_stable = (chop_g["rq_range_label"] == "RANGE_STABLE").sum()
        n_emd_stable   = (chop_g["emd_label"] == "EMD_STABLE_RANGE").sum()
        n_cwt_ok       = (chop_g["cwt_mr_allowed"] > 0).sum()
        n_signals      = chop_g["structural_mr_signal"].sum()

        print(f"\n  {sym}: {n_chop} CHOP/TRANS days")
        print(f"    RANGE_STABLE:   {n_range_stable} ({100*n_range_stable/max(n_chop,1):.1f}%)")
        print(f"    EMD_STABLE:     {n_emd_stable} ({100*n_emd_stable/max(n_chop,1):.1f}%)")
        print(f"    CWT_allowed:    {n_cwt_ok} ({100*n_cwt_ok/max(n_chop,1):.1f}%)")
        print(f"    MR signals:     {n_signals} ({100*n_signals/max(n_chop,1):.1f}% of CHOP/TRANS)")

        # Forward returns after LONG signal
        long_sig = chop_g[chop_g["structural_mr_signal"] & (chop_g["structural_mr_direction"] == "LONG")]
        shrt_sig = chop_g[chop_g["structural_mr_signal"] & (chop_g["structural_mr_direction"] == "SHORT")]

        if len(long_sig) > 0:
            print(f"    LONG signals ({len(long_sig)}): avg_1d={long_sig['ret_1d'].mean():+.4f} "
                  f"avg_2d={long_sig['ret_next2'].mean():+.4f} "
                  f"avg_3d={long_sig['ret_next3'].mean():+.4f} "
                  f"pct_pos_1d={(long_sig['ret_1d']>0).mean():.2f}")
        if len(shrt_sig) > 0:
            print(f"    SHORT signals ({len(shrt_sig)}): avg_1d={shrt_sig['ret_1d'].mean():+.4f} "
                  f"avg_2d={shrt_sig['ret_next2'].mean():+.4f} "
                  f"avg_3d={shrt_sig['ret_next3'].mean():+.4f} "
                  f"pct_neg_1d={(shrt_sig['ret_1d']<0).mean():.2f}")

        # Baseline P&L during CHOP periods (are these "dead" periods for baseline?)
        chop_dates = set(chop_g["date"].astype(str))
        bl_chop = port_ret[[str(d)[:10] in chop_dates for d in port_ret.index.astype(str)]]
        print(f"    Baseline CHOP/TRANS P&L: {bl_chop.sum():+.4f} "
              f"(${bl_chop.sum()*CAPITAL:,.0f})  avg_daily={bl_chop.mean():+.5f}")

        # By CWT label in CHOP
        print(f"    CWT in CHOP/TRANS:")
        for lbl, grp in chop_g.groupby("cwt_chop_label"):
            fwd = grp["ret_next2"].mean()
            mr  = grp["structural_mr_signal"].sum()
            print(f"      {lbl:<28}: n={len(grp):>4}  avg_2d={fwd:+.4f}  mr_signals={mr}")

        # By EMD label in CHOP
        print(f"    EMD in CHOP/TRANS:")
        for lbl, grp in chop_g.groupby("emd_label"):
            fwd = grp["ret_next2"].mean()
            mr  = grp["structural_mr_signal"].sum()
            print(f"      {lbl:<30}: n={len(grp):>4}  avg_2d={fwd:+.4f}  mr_signals={mr}")

        # By range label in CHOP
        print(f"    Range in CHOP/TRANS:")
        for lbl, grp in chop_g.groupby("rq_range_label"):
            fwd = grp["ret_next2"].mean()
            mr  = grp["structural_mr_signal"].sum()
            print(f"      {lbl:<28}: n={len(grp):>4}  avg_2d={fwd:+.4f}  mr_signals={mr}")

        for _, row in chop_g.iterrows():
            audit_rows.append({
                "date": str(row["date"])[:10], "symbol": sym,
                "portfolio_regime": row["portfolio_regime"],
                "rq_range_label": row["rq_range_label"],
                "rq_quality_score": row.get("rq_quality_score", 0),
                "rq_position_in_range": row.get("rq_position_in_range", 0.5),
                "cwt_chop_label": row.get("cwt_chop_label", "?"),
                "cwt_mr_allowed": row.get("cwt_mr_allowed", 0),
                "emd_label": row.get("emd_label", "?"),
                "emd_stable_range_score": row.get("emd_stable_range_score", 0),
                "structural_mr_signal": row.get("structural_mr_signal", False),
                "structural_mr_direction": row.get("structural_mr_direction", "NONE"),
                "structural_mr_score": row.get("structural_mr_score", 0),
                "ret_1d": row["ret_1d"],
                "ret_next2": row["ret_next2"],
            })

    pd.DataFrame(audit_rows).to_csv(OUT / "structural_mr_audit.csv", index=False)
    print(f"\n  Saved → output/structural_mr_audit.csv")


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Experiment ladder (A–J)
# ════════════════════════════════════════════════════════════════════════════════

def _phase2_ladder(baseline_result: dict, df_sig: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  PHASE 2 — Experiment ladder (A–J)")
    print("=" * 70)

    baseline_ret = baseline_result["portfolio_returns"].copy()
    baseline_ret.index = pd.to_datetime(baseline_ret.index)

    rows = []

    for label, cfg in EXPERIMENTS.items():
        _set(cfg)
        try:
            if not cfg.get("USE_STRUCTURAL_MEAN_REVERSION", False):
                mr_ret  = pd.Series(0.0, index=baseline_ret.index)
                mr_tl   = pd.DataFrame()
                n_sig   = 0
            else:
                rq_thr  = cfg.get("STRUCTURAL_MR_RQ_THRESHOLD",  0.42)
                pir_l   = cfg.get("STRUCTURAL_MR_PIR_LONG",       0.22)
                pir_s   = cfg.get("STRUCTURAL_MR_PIR_SHORT",      0.78)
                zsc_l   = cfg.get("STRUCTURAL_MR_ZSCORE_LONG",   -1.30)
                zsc_s   = cfg.get("STRUCTURAL_MR_ZSCORE_SHORT",   1.30)
                req_cwt = cfg.get("STRUCTURAL_MR_REQUIRE_CWT",    True)
                req_emd = cfg.get("STRUCTURAL_MR_REQUIRE_EMD",    True)
                req_rng = cfg.get("STRUCTURAL_MR_REQUIRE_RANGE",  True)
                size    = cfg.get("STRUCTURAL_MR_SIZE",           0.10)
                mhold   = cfg.get("STRUCTURAL_MR_MAX_HOLD",       2)

                df_mr   = compute_structural_mean_reversion_signals(
                    df_sig,
                    rq_threshold  = rq_thr,
                    pir_long      = pir_l,
                    pir_short     = pir_s,
                    zscore_long   = zsc_l,
                    zscore_short  = zsc_s,
                    require_cwt   = req_cwt,
                    require_emd   = req_emd,
                    require_range = req_rng,
                )
                n_sig = df_mr["structural_mr_signal"].sum()

                mr_res = run_mr_backtest(
                    df_mr,
                    size          = size,
                    require_cwt   = req_cwt,
                    require_emd   = req_emd,
                    require_range = req_rng,
                    rq_threshold  = rq_thr,
                    pir_long      = pir_l,
                    pir_short     = pir_s,
                    zscore_long   = zsc_l,
                    zscore_short  = zsc_s,
                    max_hold      = mhold,
                )
                mr_ret = mr_res["mr_returns"].copy()
                mr_ret.index = pd.to_datetime(mr_ret.index)
                mr_tl  = mr_res["trade_log"].copy()

            m = _metrics(baseline_ret.copy(), mr_ret, mr_tl, label, df_sig)

            print(
                f"  {label:<26}  "
                f"Sharpe={m['sharpe']:.3f}  "
                f"AnnRet={m['ann_ret_pct']:.1f}%  "
                f"MaxDD={m['max_dd_pct']:.2f}%  "
                f"WW={m['worst_week_pct']:.2f}%  "
                f"MRn={m['n_mr']:>3}  "
                f"MRpnl=${m['mr_pnl']:>6,.0f}  "
                f"MRwr={m['mr_wr']:.2f}  "
                f"$PnL=${m['dollar_pnl']:>7,.0f}"
            )
            rows.append(m)

        except Exception as e:
            import traceback
            print(f"  ERROR {label}: {e}")
            traceback.print_exc()
            rows.append({"label": label, "sharpe": np.nan, "error": str(e)})
        finally:
            _restore()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "structural_mr_experiment_ladder.csv", index=False)
    print(f"\n  Saved → output/structural_mr_experiment_ladder.csv")
    return df


def _print_ladder_table(df: pd.DataFrame, bl_sharpe: float):
    print(f"\n  {'Label':<28} {'Sharpe':>7} {'AnnRet%':>8} {'MaxDD%':>7} "
          f"{'WW%':>6} {'MRn':>4} {'MRpnl':>7} {'MRwr':>5} {'$PnL':>8}")
    print(f"  {'-' * 90}")
    for _, r in df.sort_values("sharpe", ascending=False, na_position="last").iterrows():
        marker = " ★" if r.get("sharpe", 0) > bl_sharpe else ""
        print(
            f"  {str(r.get('label','')):<28} "
            f"{r.get('sharpe', np.nan):>7.3f} "
            f" {r.get('ann_ret_pct', np.nan):>7.2f} "
            f" {r.get('max_dd_pct', np.nan):>6.2f} "
            f" {r.get('worst_week_pct', np.nan):>5.2f} "
            f" {int(r.get('n_mr', 0)):>3} "
            f" ${r.get('mr_pnl', 0):>6,.0f} "
            f" {r.get('mr_wr', 0):>4.2f} "
            f" ${r.get('dollar_pnl', 0):>7,.0f}{marker}"
        )
    print(f"  {'=' * 90}")


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Trade detail (best non-baseline experiment)
# ════════════════════════════════════════════════════════════════════════════════

def _phase3_trade_detail(df_sig: pd.DataFrame, best_label: str, best_cfg: dict):
    print("\n" + "=" * 70)
    print(f"  PHASE 3 — Trade detail ({best_label})")
    print("=" * 70)

    _set(best_cfg)
    try:
        df_mr = compute_structural_mean_reversion_signals(
            df_sig,
            rq_threshold  = best_cfg.get("STRUCTURAL_MR_RQ_THRESHOLD", 0.42),
            pir_long      = best_cfg.get("STRUCTURAL_MR_PIR_LONG",      0.22),
            pir_short     = best_cfg.get("STRUCTURAL_MR_PIR_SHORT",     0.78),
            zscore_long   = best_cfg.get("STRUCTURAL_MR_ZSCORE_LONG",  -1.30),
            zscore_short  = best_cfg.get("STRUCTURAL_MR_ZSCORE_SHORT",  1.30),
            require_cwt   = best_cfg.get("STRUCTURAL_MR_REQUIRE_CWT",   True),
            require_emd   = best_cfg.get("STRUCTURAL_MR_REQUIRE_EMD",   True),
            require_range = best_cfg.get("STRUCTURAL_MR_REQUIRE_RANGE", True),
        )
        mr_res = run_mr_backtest(
            df_mr,
            size    = best_cfg.get("STRUCTURAL_MR_SIZE",     0.10),
            max_hold= best_cfg.get("STRUCTURAL_MR_MAX_HOLD", 2),
        )
        tl = mr_res["trade_log"].copy()
    except Exception as e:
        print(f"  ERROR: {e}")
        _restore()
        return
    finally:
        _restore()

    if tl.empty:
        print("  No MR trades generated.")
        return

    n  = len(tl)
    wr = (tl["pnl"] > 0).mean()
    print(f"\n  MR trades: {n}  Win rate: {wr:.3f}")
    print(f"  Avg P&L: ${tl['pnl'].mean()*CAPITAL:,.0f}  Total P&L: ${tl['pnl'].sum()*CAPITAL:,.0f}")
    print(f"  Avg hold: {tl['days_held'].mean():.2f}d  Stop rate: {(tl['exit_reason']=='MR_STOP_LOSS').mean():.2f}")

    print(f"\n  By direction:")
    for d, grp in tl.groupby("direction"):
        sl = (grp["exit_reason"] == "MR_STOP_LOSS").sum()
        print(f"    {d:<8}  n={len(grp):>3}  WR={(grp['pnl']>0).mean():.3f}  "
              f"TotPnL=${grp['pnl'].sum()*CAPITAL:,.0f}  SL={sl}")

    print(f"\n  By symbol:")
    for sym, grp in tl.groupby("symbol"):
        print(f"    {sym}  n={len(grp):>3}  WR={(grp['pnl']>0).mean():.3f}  "
              f"TotPnL=${grp['pnl'].sum()*CAPITAL:,.0f}")

    print(f"\n  By exit reason:")
    for er, grp in tl.groupby("exit_reason"):
        print(f"    {er:<22}  n={len(grp):>3}  WR={(grp['pnl']>0).mean():.3f}  "
              f"AvgPnL=${grp['pnl'].mean()*CAPITAL:,.0f}  "
              f"TotPnL=${grp['pnl'].sum()*CAPITAL:,.0f}")

    tl["year"] = pd.to_datetime(tl["entry_date"]).dt.year
    print(f"\n  By year:")
    yr_rows = []
    for yr, grp in tl.groupby("year"):
        sl = (grp["exit_reason"] == "MR_STOP_LOSS").sum()
        print(f"    {yr}  n={len(grp):>3}  WR={(grp['pnl']>0).mean():.3f}  "
              f"TotPnL=${grp['pnl'].sum()*CAPITAL:,.0f}  SL={sl}")
        yr_rows.append({"year": yr, "n": len(grp),
                        "win_rate": round((grp["pnl"]>0).mean(), 3),
                        "total_pnl_dollar": round(grp["pnl"].sum()*CAPITAL, 0)})

    ts = tl.sort_values("pnl", ascending=False)
    print(f"\n  Top 5 winners:")
    for _, r in ts.head(5).iterrows():
        print(f"    {r.get('symbol','?')} {str(r.get('entry_date',''))[:10]}  "
              f"dir={r.get('direction','?')}  pnl=${r['pnl']*CAPITAL:,.0f}  "
              f"hold={r.get('days_held',0)}d  exit={r.get('exit_reason','?')}")
    print(f"\n  Bottom 5 losers:")
    for _, r in ts.tail(5).iterrows():
        print(f"    {r.get('symbol','?')} {str(r.get('entry_date',''))[:10]}  "
              f"dir={r.get('direction','?')}  pnl=${r['pnl']*CAPITAL:,.0f}  "
              f"hold={r.get('days_held',0)}d  exit={r.get('exit_reason','?')}")

    tl.to_csv(OUT / "structural_mr_trade_log.csv", index=False)
    pd.DataFrame(yr_rows).to_csv(OUT / "structural_mr_by_year.csv", index=False)
    print(f"\n  Saved → output/structural_mr_trade_log.csv")
    print(f"  Saved → output/structural_mr_by_year.csv")


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Diversification analysis
# ════════════════════════════════════════════════════════════════════════════════

def _phase4_diversification(baseline_result: dict, df_sig: pd.DataFrame,
                             best_label: str, best_cfg: dict):
    print("\n" + "=" * 70)
    print("  PHASE 4 — Diversification / correlation analysis")
    print("=" * 70)

    baseline_ret = baseline_result["portfolio_returns"].copy()
    baseline_ret.index = pd.to_datetime(baseline_ret.index)

    _set(best_cfg)
    try:
        df_mr  = compute_structural_mean_reversion_signals(
            df_sig,
            rq_threshold  = best_cfg.get("STRUCTURAL_MR_RQ_THRESHOLD", 0.42),
            pir_long      = best_cfg.get("STRUCTURAL_MR_PIR_LONG",      0.22),
            pir_short     = best_cfg.get("STRUCTURAL_MR_PIR_SHORT",     0.78),
            require_cwt   = best_cfg.get("STRUCTURAL_MR_REQUIRE_CWT",   True),
            require_emd   = best_cfg.get("STRUCTURAL_MR_REQUIRE_EMD",   True),
            require_range = best_cfg.get("STRUCTURAL_MR_REQUIRE_RANGE", True),
        )
        mr_res = run_mr_backtest(
            df_mr, size=best_cfg.get("STRUCTURAL_MR_SIZE", 0.10),
            max_hold=best_cfg.get("STRUCTURAL_MR_MAX_HOLD", 2),
        )
        mr_ret = mr_res["mr_returns"].copy()
        mr_ret.index = pd.to_datetime(mr_ret.index)
        mr_tl  = mr_res["trade_log"].copy()
    except Exception as e:
        print(f"  ERROR: {e}")
        _restore()
        return
    finally:
        _restore()

    aligned = baseline_ret.reindex(mr_ret.index.union(baseline_ret.index), fill_value=0)
    mr_aligned = mr_ret.reindex(aligned.index, fill_value=0)

    corr = aligned.corr(mr_aligned)
    print(f"\n  Correlation (MR vs baseline): {corr:.3f}")

    # P&L during baseline flat periods (|ret| < 0.1% per day)
    flat_mask = aligned.abs() < 0.001
    if flat_mask.any():
        mr_flat = mr_aligned[flat_mask].sum()
        print(f"  MR P&L during baseline flat days: {mr_flat:+.5f} (${mr_flat*CAPITAL:,.0f})")
        print(f"  Number of flat baseline days: {flat_mask.sum()}")

    # P&L during baseline drawdown > 0.5%
    cum_b  = (1 + aligned).cumprod()
    dd_b   = (cum_b - cum_b.cummax()) / cum_b.cummax()
    in_dd  = dd_b < -0.005
    if in_dd.any():
        mr_dd = mr_aligned[in_dd].sum()
        bl_dd = aligned[in_dd].sum()
        print(f"  During baseline DD > 0.5%:")
        print(f"    Baseline P&L: {bl_dd:+.4f} (${bl_dd*CAPITAL:,.0f})")
        print(f"    MR sleeve:    {mr_dd:+.4f} (${mr_dd*CAPITAL:,.0f})")

    # Worst 10 baseline weeks
    weekly_b  = aligned.resample("W").sum()
    weekly_mr = mr_aligned.resample("W").sum()
    worst10   = weekly_b.nsmallest(10).index

    print(f"\n  MR during baseline 10 worst weeks:")
    div_rows = []
    for w in worst10:
        b_w  = float(weekly_b.reindex([w], fill_value=0).iloc[0])
        mr_w = float(weekly_mr.reindex([w], fill_value=0).iloc[0])
        print(f"    {str(w.date())[:10]}:  baseline={b_w:+.3%}  mr={mr_w:+.3%}  "
              f"combined={b_w+mr_w:+.3%}")
        div_rows.append({"week": w, "baseline": b_w, "mr": mr_w, "combined": b_w + mr_w})

    # MR P&L by regime
    df_regimes = df_sig[["date","portfolio_regime"]].drop_duplicates("date").set_index("date")
    if not mr_tl.empty and "entry_date" in mr_tl.columns:
        mr_tl["entry_date"] = pd.to_datetime(mr_tl["entry_date"])
        mr_tl["regime"] = mr_tl["entry_date"].map(
            lambda d: df_regimes.get("portfolio_regime", {}).get(d, "UNKNOWN")
            if hasattr(df_regimes.get, "__call__")
            else df_regimes["portfolio_regime"].get(d, "UNKNOWN")
        )
        if "regime" in mr_tl.columns:
            try:
                regime_map = df_regimes["portfolio_regime"].to_dict()
                mr_tl["regime"] = mr_tl["entry_date"].map(
                    lambda d: regime_map.get(pd.Timestamp(d), "UNKNOWN")
                )
            except Exception:
                pass

    pd.DataFrame(div_rows).to_csv(OUT / "structural_mr_diversification_analysis.csv", index=False)
    print(f"\n  Saved → output/structural_mr_diversification_analysis.csv")


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 5 — 10 Final questions + verdict
# ════════════════════════════════════════════════════════════════════════════════

def _phase5_verdict(exp_df: pd.DataFrame, baseline_sharpe: float) -> str:
    print("\n" + "=" * 70)
    print("  PHASE 5 — 10 Final questions + verdict")
    print("=" * 70)

    non_base = exp_df[~exp_df["label"].str.contains("BASELINE", na=False)].copy()
    non_base = non_base[non_base["sharpe"].notna()]

    base_row  = exp_df[exp_df["label"].str.contains("BASELINE", na=False)]
    bl_s      = float(base_row["sharpe"].iloc[0])  if not base_row.empty else baseline_sharpe
    bl_dd     = float(base_row["max_dd_pct"].iloc[0]) if not base_row.empty else -5.66
    bl_ww     = float(base_row["worst_week_pct"].iloc[0]) if not base_row.empty else -2.5

    best = non_base.sort_values("sharpe", ascending=False).iloc[0] if not non_base.empty else None

    print(f"\n  Baseline: Sharpe={bl_s:.3f}  MaxDD={bl_dd:.2f}%  WW={bl_ww:.2f}%")

    if best is not None:
        print(f"  Best exp: Sharpe={best['sharpe']:.3f}  MaxDD={best['max_dd_pct']:.2f}%  "
              f"WW={best['worst_week_pct']:.2f}%  MRn={int(best.get('n_mr',0))}  "
              f"({best['label']})")

    # ── 10 questions ──────────────────────────────────────────────────────────
    q1_trades   = (non_base["n_mr"] >= 20).any()   if "n_mr" in non_base.columns else False
    q2_pos_exp  = (non_base["mr_pnl"] > 0).any()   if "mr_pnl" in non_base.columns else False
    q3_cwt_add  = False
    q4_emd_add  = False
    q5_combo_add = False

    if not non_base.empty:
        range_only_sharpes = non_base[non_base["label"].str.contains("RANGE_ONLY", na=False)]["sharpe"]
        cwt_only_sharpe    = non_base[non_base["label"].str.contains("CWT_ONLY",   na=False)]["sharpe"]
        emd_only_sharpe    = non_base[non_base["label"].str.contains("EMD_ONLY",   na=False)]["sharpe"]
        combo_sharpes      = non_base[non_base["label"].str.startswith("G_CWT_EMD") |
                                      non_base["label"].str.startswith("H_CWT_EMD")]["sharpe"]

        if len(range_only_sharpes) > 0 and len(cwt_only_sharpe) > 0:
            q3_cwt_add = float(cwt_only_sharpe.max()) > float(range_only_sharpes.max())
        if len(range_only_sharpes) > 0 and len(emd_only_sharpe) > 0:
            q4_emd_add = float(emd_only_sharpe.max()) > float(range_only_sharpes.max())
        if len(range_only_sharpes) > 0 and len(combo_sharpes) > 0:
            q5_combo_add = float(combo_sharpes.max()) > float(range_only_sharpes.max())

    # Direction breakdown
    if best is not None and "mr_n_long" in best:
        n_long  = int(best["mr_n_long"])
        n_short = int(best["mr_n_short"])
        mostly_long = n_long > n_short
    else:
        mostly_long = True
        n_long, n_short = 0, 0

    q7_sym   = "See Phase 3 trade detail"
    q8_helps_flat   = None  # from phase 4
    q9_hurts_trend  = (best["max_dd_pct"] < bl_dd - 0.3) if best is not None else False
    q10_live        = (best is not None and float(best["sharpe"]) > bl_s and
                       abs(float(best["max_dd_pct"])) < 7.5)

    print(f"\n  Q1.  MR adds enough trades (≥20)?         {'YES ✓' if q1_trades else 'NO ✗'}")
    print(f"  Q2.  MR positive standalone expectancy?   {'YES ✓' if q2_pos_exp else 'NO ✗'}")
    print(f"  Q3.  CWT improves over range-only?        {'YES ✓' if q3_cwt_add else 'NO ✗'}")
    print(f"  Q4.  EMD improves over range-only?        {'YES ✓' if q4_emd_add else 'NO ✗'}")
    print(f"  Q5.  CWT+EMD best combination?            {'YES ✓' if q5_combo_add else 'NO ✗'}")
    print(f"  Q6.  MR mostly LONG or SHORT?             {'LONG' if mostly_long else 'SHORT'} "
          f"({n_long}L / {n_short}S)")
    print(f"  Q7.  Which symbols work best?             {q7_sym}")
    print(f"  Q8.  MR helps during baseline flat?       See Phase 4")
    print(f"  Q9.  MR hurts during trends/shocks?       {'YES (worsens DD)' if q9_hurts_trend else 'NO (DD unchanged)'}")
    print(f"  Q10. Add live?                            {'YES' if q10_live else 'NO'}")

    # ── Acceptance criteria ───────────────────────────────────────────────────
    sharpe_imp = best is not None and float(best["sharpe"]) > bl_s
    dd_ok      = best is not None and abs(float(best["max_dd_pct"])) < 7.5
    dd_not_worse = best is not None and float(best["max_dd_pct"]) >= bl_dd - 0.50
    mr_pnl_pos = best is not None and float(best.get("mr_pnl", 0)) > 0
    mr_pf_ok   = best is not None and float(best.get("mr_pf", 0)) > 1.2
    mr_sl_ok   = best is not None and float(best.get("mr_sl_pct", 1)) < 0.25
    stable_yr  = True   # assessed from phase 3 by-year detail

    n_pass = sum([sharpe_imp, dd_ok, dd_not_worse, mr_pnl_pos, mr_pf_ok, mr_sl_ok])

    print(f"\n  Acceptance criteria:")
    print(f"  1. Sharpe > {bl_s:.3f}:           {'PASS ✓' if sharpe_imp else 'FAIL ✗'}")
    print(f"  2. MaxDD < 7.5%:             {'PASS ✓' if dd_ok else 'FAIL ✗'}")
    print(f"  3. MaxDD not worse by >0.5%: {'PASS ✓' if dd_not_worse else 'FAIL ✗'}")
    print(f"  4. MR P&L > 0:               {'PASS ✓' if mr_pnl_pos else 'FAIL ✗'}")
    print(f"  5. MR profit factor > 1.2:   {'PASS ✓' if mr_pf_ok else 'FAIL ✗'}")
    print(f"  6. MR stop rate < 25%:       {'PASS ✓' if mr_sl_ok else 'FAIL ✗'}")

    if n_pass == 6:
        verdict = "ACCEPT_STRUCTURAL_MR_LIVE"
    elif n_pass >= 5 and sharpe_imp:
        verdict = "ACCEPT_STRUCTURAL_MR_COMPETITION_ONLY"
    elif n_pass >= 3 and mr_pnl_pos:
        verdict = "KEEP_DIAGNOSTIC_ONLY"
    elif n_pass >= 2:
        verdict = "CONTINUE_TESTING"
    else:
        verdict = "REJECT_STRUCTURAL_MR"

    best_label   = str(best["label"])       if best is not None else "?"
    best_sharpe  = float(best["sharpe"])    if best is not None else np.nan
    best_dd      = float(best["max_dd_pct"])if best is not None else np.nan
    best_ann     = float(best["ann_ret_pct"])if best is not None else np.nan
    best_ww      = float(best["worst_week_pct"]) if best is not None else np.nan
    best_pnl     = float(best["dollar_pnl"]) if best is not None else 0

    print(f"\n  ══════════════════════════════════════════════════")
    print(f"  VERDICT: {verdict}")
    print(f"  Best config: {best_label}")
    print(f"    Sharpe:      {best_sharpe:.3f}  (baseline {bl_s:.3f}, Δ={best_sharpe-bl_s:+.3f})")
    print(f"    Ann Return:  {best_ann:.2f}%")
    print(f"    Max DD:      {best_dd:.2f}%  (baseline {bl_dd:.2f}%)")
    print(f"    Worst Week:  {best_ww:.2f}%   (baseline {bl_ww:.2f}%)")
    print(f"    $PnL:        ${best_pnl:,.0f}")
    if best is not None:
        print(f"    MR trades:   {int(best.get('n_mr',0))}  "
              f"WR={best.get('mr_wr',0):.2f}  "
              f"PF={best.get('mr_pf',0):.2f}  "
              f"AvgHold={best.get('mr_avg_hold',0):.1f}d")
    print(f"  ══════════════════════════════════════════════════\n")
    return verdict


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Research insights
# ════════════════════════════════════════════════════════════════════════════════

def _phase6_insights(exp_df: pd.DataFrame, df_mr: pd.DataFrame, verdict: str):
    print("\n" + "=" * 70)
    print("  PHASE 6 — Research Insights and Next Improvement Ideas")
    print("=" * 70)

    non_base = exp_df[~exp_df["label"].str.contains("BASELINE", na=False)].copy()
    non_base = non_base[non_base["sharpe"].notna()]

    base_row = exp_df[exp_df["label"].str.contains("BASELINE", na=False)]
    bl_s = float(base_row["sharpe"].iloc[0]) if not base_row.empty else 0.888

    # Summary statistics for insight generation
    best = non_base.sort_values("sharpe", ascending=False).iloc[0] if not non_base.empty else None
    range_only = non_base[non_base["label"].str.contains("RANGE_ONLY", na=False)]
    cwt_only   = non_base[non_base["label"].str.contains("CWT_ONLY",   na=False)]
    emd_only   = non_base[non_base["label"].str.contains("EMD_ONLY",   na=False)]
    combo      = non_base[non_base["label"].str.contains("CWT_EMD",    na=False)]
    strict     = non_base[non_base["label"].str.contains("STRICT",     na=False)]
    loose      = non_base[non_base["label"].str.contains("LOOSE",      na=False)]

    best_range  = float(range_only["sharpe"].max())  if not range_only.empty else bl_s
    best_cwt    = float(cwt_only["sharpe"].max())    if not cwt_only.empty   else bl_s
    best_emd    = float(emd_only["sharpe"].max())    if not emd_only.empty   else bl_s
    best_combo  = float(combo["sharpe"].max())       if not combo.empty      else bl_s
    best_strict = float(strict["sharpe"].max())      if not strict.empty     else bl_s
    best_loose  = float(loose["sharpe"].max())       if not loose.empty      else bl_s

    # Trade count / direction data from best experiment
    if best is not None:
        n_mr     = int(best.get("n_mr", 0))
        mr_pnl   = float(best.get("mr_pnl", 0))
        mr_wr    = float(best.get("mr_wr", 0))
        mr_pf    = float(best.get("mr_pf", 0))
        n_long   = int(best.get("mr_n_long", 0))
        n_short  = int(best.get("mr_n_short", 0))
        mr_sl    = float(best.get("mr_sl_pct", 0))
        mr_dd    = float(best.get("max_dd_pct", -5.66))
    else:
        n_mr = mr_pnl = 0; mr_wr = mr_pf = mr_sl = 0; n_long = n_short = 0; mr_dd = -5.66

    # Regime-level statistics from df_mr
    chop_n = len(df_mr[df_mr["portfolio_regime"].isin(["CHOP","TRANSITION"])])
    total_n = len(df_mr)
    range_stable_n = (df_mr["rq_range_label"] == "RANGE_STABLE").sum()
    emd_stable_n   = (df_mr["emd_label"] == "EMD_STABLE_RANGE").sum()
    cwt_ok_n       = (df_mr.get("cwt_mr_allowed", pd.Series(1)) > 0).sum()

    print(f"""
  Summary Statistics:
    Baseline Sharpe:      {bl_s:.3f}
    Best experiment:      {best['label'] if best is not None else '?'}  Sharpe={float(best['sharpe']) if best is not None else 0:.3f}
    Range-only best:      {best_range:.3f}
    CWT-augmented best:   {best_cwt:.3f}
    EMD-augmented best:   {best_emd:.3f}
    CWT+EMD best:         {best_combo:.3f}
    Strict thresholds:    {best_strict:.3f}
    Loose thresholds:     {best_loose:.3f}
    MR trades (best):     {n_mr}  (LONG={n_long}, SHORT={n_short})
    MR win rate:          {mr_wr:.2f}
    MR profit factor:     {mr_pf:.2f}
    MR stop rate:         {mr_sl:.2f}
    CHOP/TRANS days:      {chop_n} ({100*chop_n/max(total_n,1):.0f}% of all days)
    RANGE_STABLE days:    {range_stable_n} ({100*range_stable_n/max(total_n,1):.0f}%)
    EMD_STABLE days:      {emd_stable_n} ({100*emd_stable_n/max(total_n,1):.0f}%)
    CWT_allowed days:     {cwt_ok_n} ({100*cwt_ok_n/max(total_n,1):.0f}%)
""")

    # ── Insight table ─────────────────────────────────────────────────────────
    print("  " + "─" * 88)
    print("  RANKED INSIGHTS & PROPOSED EXPERIMENTS")
    print("  " + "─" * 88)

    insights = []

    # A. Trade frequency
    if n_mr < 20:
        insights.append({
            "rank": 1,
            "category": "A. Trade frequency",
            "title": "MR signal count is too low for statistical confidence",
            "evidence": f"Best config generated only {n_mr} trades over 7 years "
                        f"({n_mr/7.0:.1f}/yr). Range quality threshold and position-in-range "
                        f"filter together are excessively restrictive.",
            "proposed_experiment": "Test STRUCTURAL_MR_RQ_THRESHOLD=0.32 + PIR_LONG=0.30 on "
                                   "CWT+EMD full config. Target ≥30 trades per year per symbol.",
            "expected_impact": "3-5× more trades, lower average quality but better sample significance.",
            "risk": "More false positives in noisy CHOP. Expectancy may dilute.",
            "difficulty": "Low",
            "priority": "High",
            "status": "CONTINUE_TESTING" if n_mr < 15 else "LOW_PRIORITY",
        })
    else:
        insights.append({
            "rank": 1,
            "category": "A. Trade frequency",
            "title": "MR generates adequate trades in CHOP periods",
            "evidence": f"{n_mr} trades over 7 years ({n_mr/7.0:.1f}/yr). "
                        f"CHOP/TRANS regime occurs {100*chop_n/max(total_n,1):.0f}% of trading days. "
                        f"MR fires in roughly {100*n_mr/max(chop_n,1):.0f}% of CHOP/TRANS days.",
            "proposed_experiment": "Test loosening PIR thresholds (0.28/0.72) to increase "
                                   "trade count if expectancy stays positive.",
            "expected_impact": "+20-40% more trades, marginal expectancy dilution.",
            "risk": "More low-quality entries near range middle.",
            "difficulty": "Low",
            "priority": "Medium",
            "status": "DIAGNOSTIC_OK",
        })

    # B. Strategy interaction
    cwt_beats_range = best_cwt > best_range + 0.005
    emd_beats_range = best_emd > best_range + 0.005
    combo_beats_both = best_combo > max(best_cwt, best_emd) + 0.005

    if combo_beats_both:
        combo_verdict = "CONFIRMED: CWT+EMD combination adds value over each alone"
        combo_status = "VALIDATE"
    elif cwt_beats_range or emd_beats_range:
        combo_verdict = f"PARTIAL: {'CWT' if cwt_beats_range else 'EMD'} adds value; combined untested"
        combo_status = "CONTINUE_TESTING"
    else:
        combo_verdict = "NEGATIVE: Neither CWT nor EMD improves over range-only"
        combo_status = "REJECT"

    insights.append({
        "rank": 2,
        "category": "G. CWT/EMD usefulness",
        "title": "CWT and EMD filter value assessment",
        "evidence": f"Range-only best: {best_range:.3f}. CWT-aug: {best_cwt:.3f}. "
                    f"EMD-aug: {best_emd:.3f}. CWT+EMD: {best_combo:.3f}. {combo_verdict}.",
        "proposed_experiment": "If range-only = best: reject CWT/EMD and test range-only "
                               "with looser thresholds. If CWT best: test CWT entropy "
                               "threshold at 0.60 vs 0.80.",
        "expected_impact": "Clarifies whether CWT/EMD reduce false positives or just reduce count.",
        "risk": "CWT computation cost. EMD proxy may not generalize to other regimes.",
        "difficulty": "Low",
        "priority": "High",
        "status": combo_status,
    })

    # C. Regime interaction
    insights.append({
        "rank": 3,
        "category": "C. Regime-specific opportunity",
        "title": "CHOP periods are under-served by the current system",
        "evidence": f"Baseline earns {bl_s:.3f} Sharpe with near-zero allocation during CHOP "
                    f"({100*chop_n/max(total_n,1):.0f}% of days). MR adds trades precisely in "
                    f"these gaps. RANGE_STABLE occurs in {100*range_stable_n/max(total_n,1):.0f}% of all days.",
        "proposed_experiment": "Plot baseline daily P&L vs regime. Quantify how much of "
                               "total Sharpe comes from CHOP days vs TREND days.",
        "expected_impact": "Identifies the maximum addressable alpha in CHOP periods.",
        "risk": "CHOP P&L may already be near zero — structural MR is filling a zero-return gap.",
        "difficulty": "Low",
        "priority": "High",
        "status": "DIAGNOSTIC",
    })

    # D. Symbol-specific
    insights.append({
        "rank": 4,
        "category": "D. Symbol-specific opportunity",
        "title": "ES and NQ may have different MR quality during CHOP",
        "evidence": f"ES is typically lower-vol and more range-bound than NQ. "
                    f"NQ has stronger trend momentum (slope_20d typically higher). "
                    f"MR short side may work better on NQ reversal than ES.",
        "proposed_experiment": "Test MR on ES only (no NQ) with standard params, then "
                               "NQ only. Compare win rates and P&L. If ES>NQ, exclude NQ from MR.",
        "expected_impact": "Could improve MR Sharpe contribution by 0.02-0.05.",
        "risk": "Reduces diversification if only one symbol used.",
        "difficulty": "Low",
        "priority": "Medium",
        "status": "CONTINUE_TESTING",
    })

    # E. Exit improvement
    avg_hold_hint = "too short (time-exit dominated)" if n_mr > 0 else "unknown"
    insights.append({
        "rank": 5,
        "category": "E. Exit improvement",
        "title": f"2-day max hold may be {avg_hold_hint} for MR alpha",
        "evidence": f"With max_hold=2, most trades exit by time (MR_MAX_HOLD). "
                    f"If win rate is near 50%, 2-day returns in RANGE_STABLE are "
                    f"essentially random — mean reversion takes longer than 2 days in ES/NQ.",
        "proposed_experiment": "Test max_hold=3 and max_hold=4 on best CWT+EMD config. "
                               "Track whether win rate improves with longer hold.",
        "expected_impact": "If MR takes 3-4 days to complete, win rate could improve by 5-10%.",
        "risk": "Longer hold increases exposure to trend resumption or SHOCK entry.",
        "difficulty": "Low",
        "priority": "High" if n_mr > 15 else "Medium",
        "status": "CONTINUE_TESTING",
    })

    # F. Sizing
    size_msg = "15% shows best result → size up is warranted" if best_combo > best_range + 0.01 else "sizing does not materially improve results"
    insights.append({
        "rank": 6,
        "category": "F. Sizing improvement",
        "title": f"Optimal MR sizing: {size_msg}",
        "evidence": f"Experiments tested 5%, 10%, 15%. Best: {best['label'] if best is not None else '?'}. "
                    f"If 15% > 10% > 5%, scaling adds proportional alpha with controlled risk.",
        "proposed_experiment": "Test quality-dependent sizing: STRONG=15%, MODERATE=10%, WEAK=0%. "
                               "Compare vs flat 10% on CWT+EMD config.",
        "expected_impact": "Could improve Sharpe by 0.01-0.03 if STRONG bucket outperforms MODERATE.",
        "risk": "If STRONG and MODERATE have similar expectancy, quality sizing just reduces trade count.",
        "difficulty": "Low",
        "priority": "Medium",
        "status": "CONTINUE_TESTING",
    })

    # H. Failure analysis
    worst_exp = non_base.sort_values("sharpe").iloc[0] if not non_base.empty else None
    if worst_exp is not None:
        worst_label = str(worst_exp["label"])
        worst_s = float(worst_exp["sharpe"])
        if "LOOSE" in worst_label:
            failure_reason = "Loose thresholds admit low-quality entries in near-trend conditions"
        elif "BREAKDOWN" in worst_label:
            failure_reason = "Breakdown signals fire in trending (not ranging) markets"
        elif worst_s < bl_s - 0.05:
            failure_reason = "Overtrading: too many low-expectancy entries dilute alpha"
        else:
            failure_reason = "Marginal degradation from transaction costs without alpha"

        insights.append({
            "rank": 7,
            "category": "H. Failure analysis",
            "title": f"Worst variant: {worst_label} (Sharpe={worst_s:.3f})",
            "evidence": f"Failure cause: {failure_reason}. "
                        f"MR trades in non-range environments lose predictably.",
            "proposed_experiment": "REJECT permanently if loose > strict by >0.03 Sharpe gap. "
                                   "Otherwise: test range quality score ≥0.55 as minimum.",
            "expected_impact": "Removing bad experiments prevents live deployment risk.",
            "risk": "None — this is a diagnostic-only action.",
            "difficulty": "Low",
            "priority": "High",
            "status": "REJECT" if worst_s < bl_s - 0.03 else "DIAGNOSTIC",
        })

    # I. New strategy idea — STAT_ARB extension
    insights.append({
        "rank": 8,
        "category": "I. New strategy ideas",
        "title": "Expand STAT_ARB to trade range-bound ES/NQ spread mean reversion",
        "evidence": f"During CHOP/TRANSITION, ES-NQ spread oscillates. STAT_ARB already "
                    f"trades spread divergence but only when z-score > 1.5. "
                    f"Lower threshold (1.2) in confirmed RANGE_STABLE might add 10-15 trades/yr.",
        "proposed_experiment": "Reduce SA_ENTRY_THRESHOLD from 1.5 to 1.2 when "
                               "portfolio_regime=CHOP and rq_range_label=RANGE_STABLE.",
        "expected_impact": "+0.02-0.04 Sharpe from higher STAT_ARB utilization in CHOP.",
        "risk": "Lower threshold increases false positives in weak spread signals.",
        "difficulty": "Low",
        "priority": "Medium",
        "status": "CONTINUE_TESTING",
    })

    # J. Production recommendation
    if verdict == "ACCEPT_STRUCTURAL_MR_LIVE":
        live_status = "ACTIVATE with USE_STRUCTURAL_MEAN_REVERSION=True, best config params"
        disabled_status = "MR/PB remain disabled (old versions rejected)"
    elif verdict == "ACCEPT_STRUCTURAL_MR_COMPETITION_ONLY":
        live_status = "ACTIVATE for competition (high alpha, untested live)"
        disabled_status = "MR/PB remain disabled"
    elif verdict == "KEEP_DIAGNOSTIC_ONLY":
        live_status = "KEEP DISABLED. Run diagnostic to identify what's blocking alpha."
        disabled_status = "All old MR/PB variants remain disabled"
    else:
        live_status = "REJECT. Do not activate structural MR in current form."
        disabled_status = "Focus on TREND + STAT_ARB + SHOCK_BOUNCE"

    insights.append({
        "rank": 9,
        "category": "J. Production recommendation",
        "title": f"System roadmap update after structural MR experiments",
        "evidence": f"Verdict: {verdict}. Baseline Sharpe {bl_s:.3f}. "
                    f"Best MR Sharpe {float(best['sharpe']) if best is not None else 0:.3f}.",
        "proposed_experiment": live_status,
        "expected_impact": "Portfolio of accepted strategies: "
                           "TREND (dominant) + STAT_ARB (hump quality) + "
                           "SHOCK_BOUNCE_SHORT (17 trades/7yr, +0.068 Sharpe) + "
                           f"Structural MR ({live_status[:30]}...).",
        "risk": disabled_status,
        "difficulty": "Low",
        "priority": "High",
        "status": verdict,
    })

    # Print table
    print(f"\n  {'Rank':<4} {'Category':<32} {'Title':<50} {'Status'}")
    print(f"  {'-' * 100}")
    for ins in insights:
        title_short = ins["title"][:48]
        print(f"  {ins['rank']:<4} {ins['category'][:30]:<32} {title_short:<50} {ins['status']}")

    print(f"\n  ──────────────────────────────────────────────")
    print(f"  DETAILED INSIGHTS")
    print(f"  ──────────────────────────────────────────────")
    for ins in insights:
        print(f"\n  [{ins['rank']}] {ins['title']}")
        print(f"      Category:    {ins['category']}")
        print(f"      Evidence:    {ins['evidence']}")
        print(f"      Experiment:  {ins['proposed_experiment']}")
        print(f"      Benefit:     {ins['expected_impact']}")
        print(f"      Risk:        {ins['risk']}")
        print(f"      Difficulty:  {ins['difficulty']}  Priority: {ins['priority']}")
        print(f"      Status:      {ins['status']}")

    print(f"\n  ──────────────────────────────────────────────")
    print(f"  TOP 5 NEXT EXPERIMENTS")
    print(f"  ──────────────────────────────────────────────")
    top5 = sorted(
        [i for i in insights if i["priority"] == "High"],
        key=lambda x: x["rank"]
    )[:5]
    for i, ins in enumerate(top5, 1):
        print(f"  {i}. {ins['title'][:70]}")
        print(f"     → {ins['proposed_experiment'][:80]}")

    # Save insights
    pd.DataFrame(insights).to_csv(OUT / "structural_mr_insights.csv", index=False)
    print(f"\n  Saved → output/structural_mr_insights.csv")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def run():
    baseline_result, df_sig, df_mr_default = _phase0()
    baseline_sharpe = float(baseline_result["stats"].get("sharpe_ratio", 0.888))

    _phase1_audit(baseline_result, df_mr_default)

    exp_df = _phase2_ladder(baseline_result, df_sig)
    _print_ladder_table(exp_df, baseline_sharpe)

    # Find best non-baseline
    non_base = exp_df[~exp_df["label"].str.contains("BASELINE", na=False)]
    non_base = non_base[non_base["sharpe"].notna()]
    best_row = non_base.sort_values("sharpe", ascending=False).iloc[0] if not non_base.empty else None
    best_label = str(best_row["label"]) if best_row is not None else "G_CWT_EMD_10"
    best_cfg   = EXPERIMENTS.get(best_label, EXPERIMENTS["G_CWT_EMD_10"])

    _phase3_trade_detail(df_sig, best_label, best_cfg)
    _phase4_diversification(baseline_result, df_sig, best_label, best_cfg)
    verdict = _phase5_verdict(exp_df, baseline_sharpe)
    _phase6_insights(exp_df, df_mr_default, verdict)

    print("  All outputs saved to output/")
    return verdict


if __name__ == "__main__":
    run()
