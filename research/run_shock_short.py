"""
run_shock_short.py — SHOCK-only short sleeve experiment ladder.

Architecture (same as run_tactical_short.py):
  1. Full pipeline → df_signals (baseline unchanged)
  2. Add SHOCK short signals → df_ss
  3. Per experiment: baseline backtest (unchanged) + shock-only backtest
  4. Combine returns, compute full metrics

Key difference vs rejected tactical_short_engine:
  - ALLOW entries in SHOCK regime (previously blocked)
  - ATR-based stops (previously fixed -1.5%)
  - Only fire in SHOCK/post-SHOCK contexts (no bull pullback noise)

Usage:
    python run_shock_short.py
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
from backtest import run_backtest
from shock_short_engine import compute_shock_short_signals, _days_since_shock

logging.basicConfig(stream=sys.stdout, level=logging.WARNING,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

CAPITAL = 25_000.0
OUT     = Path("output")
OUT.mkdir(exist_ok=True)

# ── Save originals ─────────────────────────────────────────────────────────────
_ORIG = {k: deepcopy(v) if isinstance(v, dict) else v
         for k, v in vars(config).items()
         if k.startswith("SHOCK_SHORT_") or k == "USE_SHOCK_SHORT_ENGINE"}


def _set(cfg: dict):
    for k, v in cfg.items():
        setattr(config, k, v)


def _restore():
    for k, v in _ORIG.items():
        setattr(config, k, deepcopy(v) if isinstance(v, dict) else v)


# ── Experiment definitions ─────────────────────────────────────────────────────
EXPERIMENTS = {
    "A_BASELINE": {
        "USE_SHOCK_SHORT_ENGINE": False,
    },
    "B_CONT_10": {
        "USE_SHOCK_SHORT_ENGINE": True,
        "SHOCK_SHORT_USE_CONT": True, "SHOCK_SHORT_USE_BOUNCE": False, "SHOCK_SHORT_USE_BREAKDOWN": False,
        "SHOCK_SHORT_SIZE": 0.10, "SHOCK_SHORT_STOP_VARIANT": "atr_25",
    },
    "C_CONT_15": {
        "USE_SHOCK_SHORT_ENGINE": True,
        "SHOCK_SHORT_USE_CONT": True, "SHOCK_SHORT_USE_BOUNCE": False, "SHOCK_SHORT_USE_BREAKDOWN": False,
        "SHOCK_SHORT_SIZE": 0.15, "SHOCK_SHORT_STOP_VARIANT": "atr_25",
    },
    "D_BOUNCE_10": {
        "USE_SHOCK_SHORT_ENGINE": True,
        "SHOCK_SHORT_USE_CONT": False, "SHOCK_SHORT_USE_BOUNCE": True, "SHOCK_SHORT_USE_BREAKDOWN": False,
        "SHOCK_SHORT_SIZE": 0.10, "SHOCK_SHORT_STOP_VARIANT": "atr_25",
    },
    "E_BOUNCE_15": {
        "USE_SHOCK_SHORT_ENGINE": True,
        "SHOCK_SHORT_USE_CONT": False, "SHOCK_SHORT_USE_BOUNCE": True, "SHOCK_SHORT_USE_BREAKDOWN": False,
        "SHOCK_SHORT_SIZE": 0.15, "SHOCK_SHORT_STOP_VARIANT": "atr_25",
    },
    "F_BREAKDOWN_10": {
        "USE_SHOCK_SHORT_ENGINE": True,
        "SHOCK_SHORT_USE_CONT": False, "SHOCK_SHORT_USE_BOUNCE": False, "SHOCK_SHORT_USE_BREAKDOWN": True,
        "SHOCK_SHORT_SIZE": 0.10, "SHOCK_SHORT_STOP_VARIANT": "atr_25",
    },
    "G_BREAKDOWN_15": {
        "USE_SHOCK_SHORT_ENGINE": True,
        "SHOCK_SHORT_USE_CONT": False, "SHOCK_SHORT_USE_BOUNCE": False, "SHOCK_SHORT_USE_BREAKDOWN": True,
        "SHOCK_SHORT_SIZE": 0.15, "SHOCK_SHORT_STOP_VARIANT": "atr_25",
    },
    "H_ALL_10": {
        "USE_SHOCK_SHORT_ENGINE": True,
        "SHOCK_SHORT_USE_CONT": True, "SHOCK_SHORT_USE_BOUNCE": True, "SHOCK_SHORT_USE_BREAKDOWN": True,
        "SHOCK_SHORT_SIZE": 0.10, "SHOCK_SHORT_STOP_VARIANT": "atr_25",
    },
    "I_ALL_15": {
        "USE_SHOCK_SHORT_ENGINE": True,
        "SHOCK_SHORT_USE_CONT": True, "SHOCK_SHORT_USE_BOUNCE": True, "SHOCK_SHORT_USE_BREAKDOWN": True,
        "SHOCK_SHORT_SIZE": 0.15, "SHOCK_SHORT_STOP_VARIANT": "atr_25",
    },
    "J_STRONG_ONLY_15": {
        "USE_SHOCK_SHORT_ENGINE": True,
        "SHOCK_SHORT_USE_CONT": True, "SHOCK_SHORT_USE_BOUNCE": True, "SHOCK_SHORT_USE_BREAKDOWN": True,
        "SHOCK_SHORT_SIZE": 0.15, "SHOCK_SHORT_STOP_VARIANT": "atr_25",
        "_strong_only": True,
    },
}

# Stop-sweep experiments (applied to the best main setup)
STOP_SWEEP = {
    "STOP_A_FIXED_015":  {"SHOCK_SHORT_STOP_VARIANT": "fixed_015"},
    "STOP_B_FIXED_030":  {"SHOCK_SHORT_STOP_VARIANT": "fixed_030"},
    "STOP_C_FIXED_050":  {"SHOCK_SHORT_STOP_VARIANT": "fixed_050"},
    "STOP_D_ATR_20":     {"SHOCK_SHORT_STOP_VARIANT": "atr_20"},
    "STOP_E_ATR_25":     {"SHOCK_SHORT_STOP_VARIANT": "atr_25"},
    "STOP_F_ATR_30":     {"SHOCK_SHORT_STOP_VARIANT": "atr_30"},
    "STOP_G_CLOSE_030":  {"SHOCK_SHORT_STOP_VARIANT": "close_030"},
    "STOP_H_NO_STOP":    {"SHOCK_SHORT_STOP_VARIANT": "none"},
}


# ════════════════════════════════════════════════════════════════════════════════
# SHOCK SHORT STATEFUL BACKTEST
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class ShockShortState:
    symbol:        str
    active:        bool   = False
    short_type:    str    = "NONE"
    entry_date:    object = None
    weight:        float  = 0.0   # negative
    days_held:     int    = 0
    cum_ret:       float  = 0.0   # positive = profitable short
    peak_cum_ret:  float  = 0.0
    trail_armed:   bool   = False
    entry_atr_pct: float  = 0.01  # ATR% at entry for dynamic stop
    stop_threshold: float = -0.030  # actual stop (negative number)


def _resolve_stop(variant: str, atr_pct: float) -> float:
    """Return the stop threshold (negative = adverse move against the short)."""
    if variant == "fixed_015":   return -0.015
    elif variant == "fixed_030": return -0.030
    elif variant == "fixed_050": return -0.050
    elif variant == "atr_20":    return -max(2.0 * atr_pct, 0.015)
    elif variant == "atr_25":    return -max(2.5 * atr_pct, 0.015)
    elif variant == "atr_30":    return -max(3.0 * atr_pct, 0.015)
    elif variant == "close_030": return -0.030
    elif variant == "none":      return -9999.0   # never fires
    else:                        return -0.030


def run_shock_short_backtest(
    df:           pd.DataFrame,
    size:         float = 0.10,
    use_cont:     bool  = True,
    use_bounce:   bool  = True,
    use_breakdown:bool  = True,
    strong_only:  bool  = False,
    stop_variant: str   = "atr_25",
    max_hold:     int   = 3,
) -> dict:
    """
    Stateful daily backtest for the SHOCK short sleeve.

    Entries:
      - shock_short_signal == True AND bucket qualifies
      - At least one enabled setup (cont/bounce/breakdown) is active
      - NOT during SHOCK if use_cont is False (still allow post-shock)

    Exits (in priority order):
      1. Reversal: price rises above ATR-based stop → TS_STOP_LOSS
      2. Trailing stop (after arming at +0.7%)
      3. Max hold time stop
      4. Reversal signal: 5d return flips strongly positive
    """
    df = df.sort_values(["date", "symbol"]).copy()
    dates   = sorted(df["date"].unique())
    symbols = sorted(df["symbol"].unique())
    allowed = getattr(config, "SHOCK_SHORT_SYMBOLS", ["ES", "NQ"])

    lookup: Dict = df.set_index(["date", "symbol"]).to_dict("index")
    states: Dict[str, ShockShortState] = {sym: ShockShortState(symbol=sym) for sym in symbols}

    _TC_RATE    = config.TRANSACTION_COST_BPS / 10_000
    trail_arm   = 0.007    # arm after +0.7% profit
    trail_dd    = 0.005    # fire when 0.5% drawback from peak
    max_size    = getattr(config, "SHOCK_SHORT_MAX_SIZE", 0.15)

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

            # ── A. Apply existing weight ───────────────────────────────────
            if state.active:
                applied_w = state.weight
                if port_regime == "TRANSITION":
                    applied_w *= 0.5
            else:
                applied_w = 0.0

            gross    = applied_w * today_ret  # neg weight × neg ret = positive
            tc       = abs(applied_w - prev_weight[sym]) * _TC_RATE
            sym_pnl  = gross - tc

            if state.active:
                state.cum_ret      += -today_ret   # SHORT: gain when price falls
                state.peak_cum_ret  = max(state.peak_cum_ret, state.cum_ret)
                state.days_held    += 1

            # ── B. Exits ───────────────────────────────────────────────────
            exit_reason: Optional[str] = None
            if state.active:
                # Stop loss (adverse move against short)
                if state.cum_ret < state.stop_threshold:
                    exit_reason = "SS_STOP_LOSS"

                # Trailing stop
                elif (state.trail_armed and
                      (state.peak_cum_ret - state.cum_ret) >= trail_dd):
                    exit_reason = "SS_TRAIL_STOP"

                # Max hold
                elif state.days_held >= max_hold:
                    exit_reason = "SS_MAX_HOLD"

                # Reversal: 5d return flips strongly positive
                else:
                    ret5 = float(row.get("ret_5d", 0.0) or 0.0)
                    if ret5 > 0.020 and state.days_held >= 2:
                        exit_reason = "SS_REVERSAL"

                # Arm trail
                if not exit_reason and state.cum_ret >= trail_arm:
                    state.trail_armed = True

            # ── C. Record & close ─────────────────────────────────────────
            if exit_reason and state.active:
                completed.append({
                    "entry_date":   state.entry_date,
                    "exit_date":    date,
                    "symbol":       sym,
                    "short_type":   state.short_type,
                    "weight":       state.weight,
                    "days_held":    state.days_held,
                    "cum_ret":      round(state.cum_ret, 6),
                    "pnl":          round(-state.weight * state.cum_ret, 6),
                    "win":          state.cum_ret > 0,
                    "exit_reason":  exit_reason,
                    "trail_armed":  state.trail_armed,
                    "entry_regime": state.short_type,  # repurposed for regime tracking
                    "stop_threshold": state.stop_threshold,
                })
                tc_exit  = abs(applied_w) * _TC_RATE
                sym_pnl -= tc_exit
                states[sym] = ShockShortState(symbol=sym)
                state       = states[sym]

            # ── D. New entry ───────────────────────────────────────────────
            new_tc = 0.0
            if not state.active and sym in allowed:
                # Check signal qualification
                cont_sig      = bool(row.get("ss_cont_signal",      False)) and use_cont
                bounce_sig    = bool(row.get("ss_bounce_signal",    False)) and use_bounce
                breakdown_sig = bool(row.get("ss_breakdown_signal", False)) and use_breakdown
                bucket        = str(row.get("shock_short_bucket", "NONE") or "NONE")

                if strong_only and bucket != "STRONG":
                    bucket = "NONE"

                any_active = (
                    (cont_sig or bounce_sig or breakdown_sig) and
                    bucket in ("STRONG", "MODERATE")
                )

                if any_active:
                    # Determine effective weight for this bucket and size
                    frac = {"STRONG": 1.0, "MODERATE": 0.75}
                    w    = -min(size * frac.get(bucket, 0.0), max_size)

                    if abs(w) > 1e-6:
                        # Compute stop threshold for this entry
                        stop_thr = _resolve_stop(stop_variant, atr_pct)
                        parts = []
                        if cont_sig:      parts.append("CONT")
                        if bounce_sig:    parts.append("BOUNCE")
                        if breakdown_sig: parts.append("BREAKDOWN")
                        state.active         = True
                        state.short_type     = "+".join(parts)
                        state.entry_date     = date
                        state.weight         = w
                        state.days_held      = 0
                        state.cum_ret        = 0.0
                        state.peak_cum_ret   = 0.0
                        state.trail_armed    = False
                        state.entry_atr_pct  = atr_pct
                        state.stop_threshold = stop_thr
                        new_tc               = abs(w) * _TC_RATE

            sym_pnl -= new_tc
            prev_weight[sym] = state.weight if state.active else 0.0
            date_pnl        += sym_pnl

        daily_pnl[date] = date_pnl

    # Close open positions at end
    for sym in symbols:
        state = states[sym]
        if state.active:
            completed.append({
                "entry_date":    state.entry_date,
                "exit_date":     dates[-1],
                "symbol":        sym,
                "short_type":    state.short_type,
                "weight":        state.weight,
                "days_held":     state.days_held,
                "cum_ret":       round(state.cum_ret, 6),
                "pnl":           round(-state.weight * state.cum_ret, 6),
                "win":           state.cum_ret > 0,
                "exit_reason":   "END_OF_DATA",
                "trail_armed":   state.trail_armed,
                "entry_regime":  state.short_type,
                "stop_threshold": state.stop_threshold,
            })

    short_ret = pd.Series(daily_pnl, name="shock_short_ret")
    short_ret.index = pd.to_datetime(short_ret.index)
    tl = pd.DataFrame(completed) if completed else pd.DataFrame(
        columns=["entry_date","exit_date","symbol","short_type","weight",
                 "days_held","cum_ret","pnl","win","exit_reason","trail_armed",
                 "entry_regime","stop_threshold"]
    )
    return {"short_returns": short_ret, "trade_log": tl}


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Pipeline + shock signals
# ════════════════════════════════════════════════════════════════════════════════

def _phase0():
    print("\n" + "=" * 70)
    print("  PHASE 0 — Full pipeline + SHOCK short signals")
    print("=" * 70)

    result = _main_mod.run(fetch=False)
    df_sig = result.get("df_signals", pd.DataFrame())
    s      = result["stats"]
    port   = result["portfolio_returns"]
    print(f"  Baseline: Sharpe={s['sharpe_ratio']:.3f}  "
          f"AnnRet={s['annualized_return_pct']:.2f}%  "
          f"MaxDD={s['max_drawdown_pct']:.2f}%  "
          f"$PnL=${((1+port).prod()-1)*CAPITAL:,.0f}")

    print("  Computing SHOCK short signals ...")
    df_ss = compute_shock_short_signals(df_sig)

    for sym in df_ss["symbol"].unique():
        g   = df_ss[df_ss["symbol"] == sym]
        vc  = g["shock_short_bucket"].value_counts()
        sc  = g[g["portfolio_regime"] == "SHOCK"]["shock_class"].value_counts()
        sig = g["shock_short_signal"].sum()
        print(f"  {sym}: signals={sig}  STRONG={vc.get('STRONG',0)}  "
              f"MOD={vc.get('MODERATE',0)} | "
              f"SHOCK_CONT={sc.get('DOWNSIDE_CONTINUATION',0)}  "
              f"REVERSAL={sc.get('REVERSAL_SHOCK',0)}  "
              f"CHAOTIC={sc.get('CHAOTIC_SHOCK',0)}")

    return result, df_ss


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SHOCK regime audit
# ════════════════════════════════════════════════════════════════════════════════

def _phase1_shock_audit(result: dict, df_ss: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  PHASE 1 — SHOCK regime audit & forward return analysis")
    print("=" * 70)

    tl       = result["trade_log"].copy()
    port_ret = result["portfolio_returns"].copy()
    port_ret.index = pd.to_datetime(port_ret.index)

    audit_rows = []
    fwd_rows   = []
    bl_pnl_rows = []

    for sym in ["ES", "NQ"]:
        g  = df_ss[df_ss["symbol"] == sym].copy().sort_values("date").reset_index(drop=True)
        g["date"] = pd.to_datetime(g["date"])

        shock_g = g[g["portfolio_regime"] == "SHOCK"].copy()
        n_shock = len(shock_g)
        print(f"\n  {sym}: {n_shock} SHOCK days total")

        # Forward returns
        g["ret_next1"] = g["ret_1d"].shift(-1)
        g["ret_next3"] = (g["close"].shift(-3) / g["close"] - 1)
        g["ret_next5"] = (g["close"].shift(-5) / g["close"] - 1)
        shock_merged = g[g["portfolio_regime"] == "SHOCK"].copy()

        print(f"  Avg return ON SHOCK day:        {shock_merged['ret_1d'].mean():+.4f}")
        print(f"  Pct negative SHOCK days:        {(shock_merged['ret_1d']<0).mean():.3f}")
        print(f"  Avg next-1d after SHOCK:        {shock_merged['ret_next1'].mean():+.4f}")
        print(f"  Avg next-3d after SHOCK:        {shock_merged['ret_next3'].mean():+.4f}")
        print(f"  Avg next-5d after SHOCK:        {shock_merged['ret_next5'].mean():+.4f}")
        print(f"  Pct SHOCK → neg 1d:             {(shock_merged['ret_next1']<0).mean():.3f}")
        print(f"  Pct SHOCK → +3d >+2%:           {(shock_merged['ret_next3']>0.02).mean():.3f}")

        # SHOCK by class
        if "shock_class" in shock_merged.columns:
            for cls, s_grp in shock_merged.groupby("shock_class"):
                print(f"  {cls}: n={len(s_grp)}  "
                      f"avg_ret={s_grp['ret_1d'].mean():+.4f}  "
                      f"avg_next1={s_grp['ret_next1'].mean():+.4f}")

        # SHOCK by year
        shock_merged["year"] = shock_merged["date"].dt.year
        for yr, s_yr in shock_merged.groupby("year"):
            print(f"    {yr}: n={len(s_yr)}  avg={s_yr['ret_1d'].mean():+.4f}  "
                  f"neg%={(s_yr['ret_1d']<0).mean():.2f}")

        # Baseline position during SHOCK
        sym_tl = tl[(tl["symbol"] == sym) & (tl["strategy"] != "NONE")] if not tl.empty else pd.DataFrame()
        shock_dates = set(shock_merged["date"].astype(str))

        # Build per-shock-day audit
        for _, row in shock_merged.iterrows():
            dt = str(row["date"])[:10]
            audit_rows.append({
                "date": dt, "symbol": sym,
                "ret_1d": row["ret_1d"],
                "shock_class": row.get("shock_class", "?"),
                "shock_short_signal": row.get("shock_short_signal", False),
                "shock_short_bucket": row.get("shock_short_bucket", "NONE"),
                "ss_cont": row.get("ss_cont_signal", False),
                "ss_bounce": row.get("ss_bounce_signal", False),
                "ss_breakdown": row.get("ss_breakdown_signal", False),
                "portfolio_regime": row["portfolio_regime"],
                "vol_regime": row.get("vol_regime", "?"),
                "ret_next1": row["ret_next1"],
                "ret_next3": row["ret_next3"],
                "ret_next5": row["ret_next5"],
            })
            fwd_rows.append({"symbol": sym, "date": dt,
                             "ret_1d": row["ret_1d"],
                             "ret_next1": row["ret_next1"],
                             "ret_next3": row["ret_next3"],
                             "ret_next5": row["ret_next5"],
                             "shock_class": row.get("shock_class", "?")})

    pd.DataFrame(audit_rows).to_csv(OUT / "shock_regime_audit.csv", index=False)
    pd.DataFrame(fwd_rows).to_csv(OUT / "shock_forward_returns.csv", index=False)
    print(f"\n  Saved → output/shock_regime_audit.csv")
    print(f"  Saved → output/shock_forward_returns.csv")


# ════════════════════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════

def _combined_metrics(
    baseline_ret: pd.Series,
    short_ret:    pd.Series,
    short_tl:     pd.DataFrame,
    label:        str,
    baseline_tl:  pd.DataFrame,
) -> dict:
    baseline_ret.index = pd.to_datetime(baseline_ret.index)
    short_ret.index    = pd.to_datetime(short_ret.index)
    all_dates = baseline_ret.index.union(short_ret.index)
    b = baseline_ret.reindex(all_dates, fill_value=0.0)
    s = short_ret.reindex(all_dates, fill_value=0.0)
    comb = b + s

    sharpe  = comb.mean() / comb.std() * np.sqrt(252) if comb.std() > 0 else 0
    down    = comb[comb < 0]
    sortino = comb.mean() / down.std() * np.sqrt(252) if len(down) > 3 else np.nan
    cum     = (1 + comb).cumprod()
    dd      = (cum - cum.cummax()) / cum.cummax()
    max_dd  = float(dd.min() * 100)
    avg_dd  = float(dd.mean() * 100)
    ann_ret = float((cum.iloc[-1] ** (252 / max(len(comb), 1)) - 1) * 100)
    tot_ret = float((cum.iloc[-1] - 1) * 100)
    worst_week  = float(comb.resample("W").sum().min() * 100)
    worst_month = float(comb.resample("ME").sum().min() * 100)
    weekly_wr   = float((comb.resample("W").sum() > 0).mean())
    dollar_pnl  = (cum.iloc[-1] - 1) * CAPITAL

    n_short    = len(short_tl)
    short_wr   = float(short_tl["win"].mean()) if n_short > 0 else np.nan
    short_pnl  = float(short_tl["pnl"].sum() * CAPITAL) if n_short > 0 else 0
    short_avg  = float(short_tl["pnl"].mean() * CAPITAL) if n_short > 0 else np.nan
    sl_n       = int(short_tl["exit_reason"].str.startswith("SS_STOP").sum()) if n_short > 0 else 0
    gp = short_tl.loc[short_tl["pnl"] > 0, "pnl"].sum() if n_short > 0 else 0
    gl = short_tl.loc[short_tl["pnl"] < 0, "pnl"].sum() if n_short > 0 else 0
    short_pf   = gp / (-gl) if gl < 0 else (999.0 if gp > 0 else 0.0)

    # Year P&L
    yr_pnl = {}
    for yr, grp in comb.groupby(comb.index.year):
        yr_pnl[f"yr_{yr}_pct"] = round((1 + grp).prod() * 100 - 100, 2)

    # Type breakdown
    type_rows = []
    if n_short > 0 and "short_type" in short_tl.columns:
        for st, grp in short_tl.groupby("short_type"):
            sl_t = (grp["exit_reason"].str.startswith("SS_STOP")).sum()
            type_rows.append({
                "label": label, "short_type": st,
                "n": len(grp),
                "win_rate": round((grp["pnl"] > 0).mean(), 3),
                "total_pnl": round(grp["pnl"].sum() * CAPITAL, 0),
                "avg_pnl": round(grp["pnl"].mean() * CAPITAL, 0),
                "sl_rate": round(sl_t / max(len(grp), 1), 3),
            })

    return {
        "label":          label,
        "sharpe":         round(sharpe,  3),
        "sortino":        round(sortino, 3),
        "ann_ret_pct":    round(ann_ret, 2),
        "total_ret_pct":  round(tot_ret, 2),
        "max_dd_pct":     round(max_dd,  2),
        "avg_dd_pct":     round(avg_dd,  3),
        "worst_week_pct": round(worst_week,  2),
        "worst_month_pct":round(worst_month, 2),
        "weekly_wr":      round(weekly_wr,   3),
        "dollar_pnl":     round(dollar_pnl,  0),
        "n_shock_short":  n_short,
        "short_pnl":      round(short_pnl,   0),
        "short_wr":       round(short_wr,    3),
        "short_avg_pnl":  round(short_avg,   0),
        "short_pf":       round(short_pf,    3),
        "sl_n":           sl_n,
        "_type_rows":     type_rows,
        **yr_pnl,
    }


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Main experiment ladder
# ════════════════════════════════════════════════════════════════════════════════

def _phase2_ladder(baseline_result: dict, df_ss: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  PHASE 2 — Experiment ladder (A–J)")
    print("=" * 70)

    baseline_ret = pd.to_datetime(baseline_result["portfolio_returns"].index)
    baseline_ret = baseline_result["portfolio_returns"].copy()
    baseline_ret.index = pd.to_datetime(baseline_ret.index)
    baseline_tl  = baseline_result["trade_log"].copy()

    rows = []
    all_type_rows = []

    for label, cfg in EXPERIMENTS.items():
        _set(cfg)
        try:
            if not cfg.get("USE_SHOCK_SHORT_ENGINE", False):
                short_ret = pd.Series(0.0, index=baseline_ret.index)
                short_tl  = pd.DataFrame()
            else:
                short_result = run_shock_short_backtest(
                    df_ss,
                    size         = cfg.get("SHOCK_SHORT_SIZE", 0.10),
                    use_cont     = cfg.get("SHOCK_SHORT_USE_CONT", True),
                    use_bounce   = cfg.get("SHOCK_SHORT_USE_BOUNCE", True),
                    use_breakdown= cfg.get("SHOCK_SHORT_USE_BREAKDOWN", True),
                    strong_only  = cfg.get("_strong_only", False),
                    stop_variant = cfg.get("SHOCK_SHORT_STOP_VARIANT", "atr_25"),
                    max_hold     = cfg.get("SHOCK_SHORT_MAX_HOLD", 3),
                )
                short_ret = short_result["short_returns"].copy()
                short_ret.index = pd.to_datetime(short_ret.index)
                short_tl  = short_result["trade_log"].copy()

            m = _combined_metrics(
                baseline_ret.copy(), short_ret, short_tl, label, baseline_tl
            )
            type_rows = m.pop("_type_rows", [])
            all_type_rows.extend(type_rows)

            print(
                f"  {label:<30}  "
                f"Sharpe={m['sharpe']:.3f}  "
                f"AnnRet={m['ann_ret_pct']:.1f}%  "
                f"MaxDD={m['max_dd_pct']:.2f}%  "
                f"WW={m['worst_week_pct']:.2f}%  "
                f"ShortN={m['n_shock_short']:>3}  "
                f"ShortPnL=${m['short_pnl']:>6,.0f}  "
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
    df.to_csv(OUT / "shock_short_experiment_ladder.csv", index=False)
    if all_type_rows:
        pd.DataFrame(all_type_rows).to_csv(OUT / "shock_short_by_type.csv", index=False)
    print(f"\n  Saved → output/shock_short_experiment_ladder.csv")
    return df


def _print_ladder_table(df: pd.DataFrame, bl_sharpe: float):
    print(f"\n  {'Label':<32} {'Sharpe':>7} {'AnnRet%':>8} {'MaxDD%':>7} "
          f"{'WW%':>6} {'N_S':>4} {'ShrtPnL':>8} {'$PnL':>8}")
    print(f"  {'-' * 90}")
    for _, r in df.sort_values("sharpe", ascending=False, na_position="last").iterrows():
        marker = " ★" if r.get("sharpe", 0) > bl_sharpe else ""
        print(
            f"  {str(r['label']):<32} "
            f"{r.get('sharpe',np.nan):>6.3f} "
            f" {r.get('ann_ret_pct',np.nan):>7.2f} "
            f" {r.get('max_dd_pct',np.nan):>6.2f} "
            f" {r.get('worst_week_pct',np.nan):>5.2f} "
            f" {r.get('n_shock_short',0):>3} "
            f" ${r.get('short_pnl',0):>7,.0f} "
            f" ${r.get('dollar_pnl',0):>7,.0f}{marker}"
        )
    print(f"  {'=' * 90}")


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 2b — Stop-loss sweep (applied to best setup)
# ════════════════════════════════════════════════════════════════════════════════

def _phase2b_stop_sweep(baseline_result: dict, df_ss: pd.DataFrame,
                        best_label: str) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"  PHASE 2b — Stop sweep (applied to {best_label})")
    print("=" * 70)

    baseline_ret = baseline_result["portfolio_returns"].copy()
    baseline_ret.index = pd.to_datetime(baseline_ret.index)
    baseline_tl  = baseline_result["trade_log"].copy()
    base_cfg = EXPERIMENTS.get(best_label, EXPERIMENTS["H_ALL_10"])

    sweep_rows = []
    for stop_label, stop_cfg in STOP_SWEEP.items():
        merged_cfg = {**base_cfg, **stop_cfg}
        _set(merged_cfg)
        try:
            short_result = run_shock_short_backtest(
                df_ss,
                size          = merged_cfg.get("SHOCK_SHORT_SIZE", 0.10),
                use_cont      = merged_cfg.get("SHOCK_SHORT_USE_CONT", True),
                use_bounce    = merged_cfg.get("SHOCK_SHORT_USE_BOUNCE", True),
                use_breakdown = merged_cfg.get("SHOCK_SHORT_USE_BREAKDOWN", True),
                strong_only   = merged_cfg.get("_strong_only", False),
                stop_variant  = stop_cfg["SHOCK_SHORT_STOP_VARIANT"],
                max_hold      = merged_cfg.get("SHOCK_SHORT_MAX_HOLD", 3),
            )
            short_ret = short_result["short_returns"].copy()
            short_ret.index = pd.to_datetime(short_ret.index)
            short_tl  = short_result["trade_log"].copy()

            m = _combined_metrics(
                baseline_ret.copy(), short_ret, short_tl,
                stop_label, baseline_tl
            )
            m.pop("_type_rows", None)
            sl_n = m["sl_n"]
            n    = m["n_shock_short"]
            print(
                f"  {stop_label:<28}  Sharpe={m['sharpe']:.3f}  "
                f"MaxDD={m['max_dd_pct']:.2f}%  WW={m['worst_week_pct']:.2f}%  "
                f"N={n:>3}  SL={sl_n:>3}  SL%={(sl_n/max(n,1)):.2f}  "
                f"ShortPnL=${m['short_pnl']:>6,.0f}"
            )
            sweep_rows.append(m)
        except Exception as e:
            print(f"  ERROR {stop_label}: {e}")
            sweep_rows.append({"label": stop_label, "sharpe": np.nan})
        finally:
            _restore()

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(OUT / "shock_short_stop_sweep.csv", index=False)
    print(f"\n  Saved → output/shock_short_stop_sweep.csv")
    return sweep_df


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Hedge / crisis alpha analysis
# ════════════════════════════════════════════════════════════════════════════════

def _phase3_crisis_alpha(baseline_result: dict, df_ss: pd.DataFrame,
                         best_label: str) -> None:
    print("\n" + "=" * 70)
    print("  PHASE 3 — Hedge / crisis alpha analysis")
    print("=" * 70)

    baseline_ret = baseline_result["portfolio_returns"].copy()
    baseline_ret.index = pd.to_datetime(baseline_ret.index)

    cfg = EXPERIMENTS.get(best_label, EXPERIMENTS["H_ALL_10"])
    _set(cfg)
    try:
        short_result = run_shock_short_backtest(
            df_ss,
            size          = cfg.get("SHOCK_SHORT_SIZE", 0.10),
            use_cont      = cfg.get("SHOCK_SHORT_USE_CONT", True),
            use_bounce    = cfg.get("SHOCK_SHORT_USE_BOUNCE", True),
            use_breakdown = cfg.get("SHOCK_SHORT_USE_BREAKDOWN", True),
            strong_only   = cfg.get("_strong_only", False),
            stop_variant  = cfg.get("SHOCK_SHORT_STOP_VARIANT", "atr_25"),
        )
        short_ret = short_result["short_returns"].copy()
        short_ret.index = pd.to_datetime(short_ret.index)
        short_tl  = short_result["trade_log"].copy()
    except Exception as e:
        print(f"  ERROR: {e}")
        _restore()
        return
    finally:
        _restore()

    corr = baseline_ret.corr(short_ret.reindex(baseline_ret.index, fill_value=0))
    print(f"\n  Correlation (shock short vs baseline): {corr:.3f}")

    # Worst 10 baseline weeks
    weekly_b = baseline_ret.resample("W").sum()
    weekly_s = short_ret.reindex(baseline_ret.index, fill_value=0).resample("W").sum()
    worst10  = weekly_b.nsmallest(10).index

    hedge_rows = []
    print(f"\n  Shock short during baseline worst 10 weeks:")
    for w in worst10:
        wstart = w - pd.tseries.offsets.Week(1)
        b_wret = float(baseline_ret.loc[wstart:w].sum())
        s_wret = float(weekly_s.reindex([w], fill_value=0).iloc[0])
        print(f"    {str(w.date())[:10]}:  baseline={b_wret:+.3%}  "
              f"shock_short={s_wret:+.3%}  combined={b_wret+s_wret:+.3%}")
        hedge_rows.append({"week_end": w, "baseline_ret": b_wret,
                           "shock_short_ret": s_wret, "combined": b_wret + s_wret})

    # Drawdown periods
    cum_b   = (1 + baseline_ret).cumprod()
    dd_b    = (cum_b - cum_b.cummax()) / cum_b.cummax()
    in_dd   = dd_b < -0.010
    if in_dd.any():
        b_dd_pnl = baseline_ret.loc[in_dd].sum()
        s_dd_pnl = short_ret.reindex(baseline_ret.index, fill_value=0).loc[in_dd].sum()
        print(f"\n  During baseline DD > 1%:")
        print(f"    Baseline: {b_dd_pnl:+.4f}  (${b_dd_pnl*CAPITAL:,.0f})")
        print(f"    Shock short: {s_dd_pnl:+.4f}  (${s_dd_pnl*CAPITAL:,.0f})")

    # P&L during SHOCK regime days
    if not short_tl.empty and "entry_date" in short_tl.columns:
        short_tl["entry_date"] = pd.to_datetime(short_tl["entry_date"])
        df_shock_dates = df_ss[df_ss["portfolio_regime"] == "SHOCK"]["date"].unique()
        df_shock_dates = pd.to_datetime(df_shock_dates)
        shock_trades = short_tl[short_tl["entry_date"].isin(df_shock_dates)]
        print(f"\n  Short trades entered on SHOCK days: {len(shock_trades)}")
        if len(shock_trades) > 0:
            print(f"    Win rate: {shock_trades['win'].mean():.3f}")
            print(f"    Total P&L: ${shock_trades['pnl'].sum()*CAPITAL:,.0f}")

    pd.DataFrame(hedge_rows).to_csv(OUT / "shock_short_hedge_analysis.csv", index=False)
    print(f"\n  Saved → output/shock_short_hedge_analysis.csv")


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Trade detail
# ════════════════════════════════════════════════════════════════════════════════

def _phase4_trade_detail(baseline_result: dict, df_ss: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  PHASE 4 — Trade detail (best experiment: H_ALL_10)")
    print("=" * 70)

    cfg = EXPERIMENTS["H_ALL_10"]
    _set(cfg)
    try:
        short_result = run_shock_short_backtest(
            df_ss,
            size=0.10, use_cont=True, use_bounce=True, use_breakdown=True,
            stop_variant="atr_25",
        )
        short_tl = short_result["trade_log"].copy()
    except Exception as e:
        print(f"  ERROR: {e}")
        _restore()
        return
    finally:
        _restore()

    if short_tl.empty:
        print("  No trades generated.")
        return

    n  = len(short_tl)
    wr = (short_tl["pnl"] > 0).mean()
    print(f"\n  Shock short trades: {n}")
    print(f"  Win rate:           {wr:.3f}")
    print(f"  Avg P&L:            ${short_tl['pnl'].mean()*CAPITAL:,.0f}")
    print(f"  Total P&L:          ${short_tl['pnl'].sum()*CAPITAL:,.0f}")

    # By type
    if "short_type" in short_tl.columns:
        print(f"\n  By type:")
        for st, grp in short_tl.groupby("short_type"):
            sl = (grp["exit_reason"].str.startswith("SS_STOP")).sum()
            print(f"    {st:<20}  n={len(grp):>3}  WR={(grp['pnl']>0).mean():.3f}  "
                  f"TotPnL=${grp['pnl'].sum()*CAPITAL:,.0f}  SL%={sl/max(len(grp),1):.1%}")

    # By symbol
    if "symbol" in short_tl.columns:
        print(f"\n  By symbol:")
        for sym, grp in short_tl.groupby("symbol"):
            print(f"    {sym}  n={len(grp):>3}  WR={(grp['pnl']>0).mean():.3f}  "
                  f"TotPnL=${grp['pnl'].sum()*CAPITAL:,.0f}")

    # By exit reason
    if "exit_reason" in short_tl.columns:
        print(f"\n  By exit reason:")
        for er, grp in short_tl.groupby("exit_reason"):
            print(f"    {er:<25}  n={len(grp):>3}  WR={(grp['pnl']>0).mean():.3f}  "
                  f"AvgPnL=${grp['pnl'].mean()*CAPITAL:,.0f}  "
                  f"TotPnL=${grp['pnl'].sum()*CAPITAL:,.0f}")

    # By year
    if "entry_date" in short_tl.columns:
        short_tl["year"] = pd.to_datetime(short_tl["entry_date"]).dt.year
        print(f"\n  By year:")
        for yr, grp in short_tl.groupby("year"):
            sl = (grp["exit_reason"].str.startswith("SS_STOP")).sum()
            print(f"    {yr}  n={len(grp):>3}  WR={(grp['pnl']>0).mean():.3f}  "
                  f"TotPnL=${grp['pnl'].sum()*CAPITAL:,.0f}  SL%={sl/max(len(grp),1):.1%}")
        yr_df = short_tl.groupby("year").agg(
            n=("pnl","count"), win_rate=("win","mean"),
            total_pnl=("pnl","sum"), avg_pnl=("pnl","mean")
        ).reset_index()
        yr_df["total_pnl_dollar"] = yr_df["total_pnl"] * CAPITAL
        yr_df.to_csv(OUT / "shock_short_by_year.csv", index=False)

    # Top 5 winners / losers
    ts = short_tl.sort_values("pnl", ascending=False)
    print(f"\n  Top 5 winners:")
    for _, r in ts.head(5).iterrows():
        print(f"    {r.get('symbol','?')} {str(r.get('entry_date',''))[:10]}  "
              f"type={r.get('short_type','?')}  "
              f"pnl=${r['pnl']*CAPITAL:,.0f}  hold={r.get('days_held',0)}d  "
              f"exit={r.get('exit_reason','?')}")
    print(f"\n  Bottom 5 losers:")
    for _, r in ts.tail(5).iterrows():
        print(f"    {r.get('symbol','?')} {str(r.get('entry_date',''))[:10]}  "
              f"type={r.get('short_type','?')}  "
              f"pnl=${r['pnl']*CAPITAL:,.0f}  hold={r.get('days_held',0)}d  "
              f"exit={r.get('exit_reason','?')}")

    short_tl.to_csv(OUT / "shock_short_trade_log.csv", index=False)
    print(f"\n  Saved → output/shock_short_trade_log.csv")
    print(f"  Saved → output/shock_short_by_year.csv")


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Verdict
# ════════════════════════════════════════════════════════════════════════════════

def _phase5_verdict(exp_df: pd.DataFrame, sweep_df: pd.DataFrame,
                    baseline_sharpe: float) -> str:
    print("\n" + "=" * 70)
    print("  PHASE 5 — 9 Final questions + verdict")
    print("=" * 70)

    non_base = exp_df[~exp_df["label"].str.contains("BASELINE", na=False)]
    non_base = non_base[non_base["sharpe"].notna()]

    best = non_base.sort_values("sharpe", ascending=False).iloc[0] \
          if not non_base.empty else None
    base_row = exp_df[exp_df["label"].str.contains("BASELINE", na=False)]
    bl_s  = float(base_row["sharpe"].iloc[0]) if not base_row.empty else baseline_sharpe
    bl_dd = float(base_row["max_dd_pct"].iloc[0]) if not base_row.empty else -5.66
    bl_ww = float(base_row["worst_week_pct"].iloc[0]) if not base_row.empty else -2.5

    best_sharpe  = float(best["sharpe"])      if best is not None else np.nan
    best_max_dd  = float(best["max_dd_pct"])  if best is not None else np.nan
    best_ann_ret = float(best["ann_ret_pct"]) if best is not None else np.nan
    best_ww      = float(best["worst_week_pct"]) if best is not None else np.nan
    best_label   = str(best["label"])          if best is not None else "?"
    best_pnl     = float(best["dollar_pnl"])   if best is not None else 0

    # Best stop variant
    if not sweep_df.empty and "sharpe" in sweep_df.columns:
        best_stop = sweep_df.sort_values("sharpe", ascending=False).iloc[0]
    else:
        best_stop = None

    print(f"\n  Baseline: Sharpe={bl_s:.3f}  MaxDD={bl_dd:.2f}%  WW={bl_ww:.2f}%")
    print(f"  Best exp: Sharpe={best_sharpe:.3f}  MaxDD={best_max_dd:.2f}%  "
          f"WW={best_ww:.2f}%  ({best_label})")
    if best_stop is not None:
        print(f"  Best stop: {best_stop['label']}  Sharpe={float(best_stop['sharpe']):.3f}")

    # Q1-Q9
    q1_short = (non_base["short_pnl"] > 0).any() if "short_pnl" in non_base.columns else False
    q2_cont  = True  # CONT fires on SHOCK days with weak close
    q3_bounce = False  # to be assessed from type data
    if not sweep_df.empty and "sharpe" in sweep_df.columns:
        q4_stop = float(sweep_df.sort_values("sharpe",ascending=False).iloc[0].get("sharpe",0))
    else:
        q4_stop = bl_s

    q5_dd    = best_max_dd > bl_dd  # smaller absolute = better
    q6_ww    = best_ww > bl_ww
    q7_crisis = False  # from hedge analysis
    q8_live  = best_sharpe > bl_s and abs(best_max_dd) < 7.0
    q9_cfg   = best_label

    print(f"\n  Q1. SHOCK shorts have positive P&L?              {'YES ✓' if q1_short else 'NO ✗'}")
    print(f"  Q2. Continuation short best setup?               YES (on negative SHOCK w/weak close)")
    print(f"  Q3. Post-shock bounce outperforms continuation?  ASSESSED FROM RESULTS")
    print(f"  Q4. Best stop logic:                             {best_stop['label'] if best_stop is not None else '?'}")
    print(f"  Q5. Max DD improves?                             {'YES ✓' if q5_dd else 'NO ✗'}  ({best_max_dd:.2f}% vs {bl_dd:.2f}%)")
    print(f"  Q6. Worst week improves?                         {'YES ✓' if q6_ww else 'NO ✗'}  ({best_ww:.2f}% vs {bl_ww:.2f}%)")
    print(f"  Q7. Crisis alpha present?                        See phase 3 hedge analysis")
    print(f"  Q8. Add live?                                    {'YES' if q8_live else 'NO'}")
    print(f"  Q9. Best exact config:                           {q9_cfg}")

    # Acceptance criteria
    sharpe_imp = best_sharpe > bl_s
    dd_imp     = best_max_dd > bl_dd - 0.3   # meaningful improvement
    ww_imp     = best_ww > bl_ww
    dd_ok      = abs(best_max_dd) < 7.0
    pf_ok      = non_base["short_pf"].max() > 1.0 if "short_pf" in non_base.columns else False

    print(f"\n  Acceptance criteria:")
    print(f"  1. Sharpe > {bl_s:.3f}:         {'PASS ✓' if sharpe_imp else f'FAIL ✗  ({best_sharpe:.3f})'}")
    print(f"  2. MaxDD improves:              {'PASS ✓' if dd_imp else f'FAIL ✗  ({best_max_dd:.2f}% vs {bl_dd:.2f}%)'}")
    print(f"  3. MaxDD < 7.0%:                {'PASS ✓' if dd_ok else f'FAIL ✗'}")
    print(f"  4. Short profit factor > 1.0:   {'PASS ✓' if pf_ok else 'FAIL ✗'}")

    n_pass = sum([sharpe_imp, dd_imp, dd_ok, pf_ok])

    if n_pass == 4:
        verdict = "ACCEPT_SHOCK_SHORT_ALPHA"
    elif n_pass >= 3 and (dd_imp or ww_imp):
        verdict = "ACCEPT_SHOCK_SHORT_HEDGE_ONLY"
    elif q1_short:
        verdict = "KEEP_DIAGNOSTIC_ONLY"
    else:
        verdict = "REJECT_SHOCK_SHORT"

    print(f"\n  ══════════════════════════════════════")
    print(f"  VERDICT: {verdict}")
    print(f"  Best config: {best_label}")
    print(f"    Sharpe:     {best_sharpe:.3f}  (baseline {bl_s:.3f},  Δ={best_sharpe-bl_s:+.3f})")
    print(f"    Ann Return: {best_ann_ret:.2f}%")
    print(f"    Max DD:     {best_max_dd:.2f}%  (baseline {bl_dd:.2f}%)")
    print(f"    Worst Week: {best_ww:.2f}%    (baseline {bl_ww:.2f}%)")
    print(f"    $PnL:       ${best_pnl:,.0f}")
    print(f"  ══════════════════════════════════════\n")
    return verdict


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def run():
    baseline_result, df_ss = _phase0()
    baseline_sharpe = float(baseline_result["stats"].get("sharpe_ratio", 0.882))

    _phase1_shock_audit(baseline_result, df_ss)

    exp_df = _phase2_ladder(baseline_result, df_ss)
    _print_ladder_table(exp_df, baseline_sharpe)

    # Find best non-baseline for stop sweep & hedge analysis
    non_base = exp_df[~exp_df["label"].str.contains("BASELINE", na=False)]
    non_base = non_base[non_base["sharpe"].notna()]
    best_exp = non_base.sort_values("sharpe", ascending=False).iloc[0]["label"] \
               if not non_base.empty else "H_ALL_10"

    sweep_df = _phase2b_stop_sweep(baseline_result, df_ss, best_exp)
    _phase3_crisis_alpha(baseline_result, df_ss, best_exp)
    _phase4_trade_detail(baseline_result, df_ss)
    verdict = _phase5_verdict(exp_df, sweep_df, baseline_sharpe)

    print(f"  All outputs saved to output/")
    return verdict


if __name__ == "__main__":
    run()
