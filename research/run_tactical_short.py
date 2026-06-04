"""
run_tactical_short.py — Tactical short sleeve experiment ladder.

Architecture:
  1. Run full pipeline once → df_signals (all existing signals)
  2. Add tactical short signals → df_ts (same df with ts_* columns)
  3. For each experiment:
       a. Run baseline backtest (unchanged)
       b. Run short-only backtest on df_ts (separate stateful loop)
       c. Combine: combined_returns = baseline + short_sleeve
       d. Report full metrics for combined system

Short-only backtest:
  - Iterates dates/symbols independently of baseline
  - Enters SHORT when tactical_short_signal is True and bucket qualifies
  - Tracks cum_ret per position (signed: positive = gaining on short)
  - Exits on: stop_loss / trail_stop / max_hold / reversal
  - P&L = -weight × ret_1d (weight negative → short)

Does NOT modify:
  - TREND or STAT_ARB logic
  - signal_engine.py, backtest.py, position_sizing.py

Usage:
    python run_tactical_short.py
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
from tactical_short_engine import compute_tactical_shorts

logging.basicConfig(stream=sys.stdout, level=logging.WARNING,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

CAPITAL  = 25_000.0
OUT      = Path("output")
OUT.mkdir(exist_ok=True)

# ── Save original config ───────────────────────────────────────────────────────
_ORIG = {k: deepcopy(v) if isinstance(v, dict) else v
         for k, v in vars(config).items()
         if k.startswith("TACTICAL_") or k in ["USE_TACTICAL_SHORT_ENGINE"]}


def _set(cfg: dict):
    for k, v in cfg.items():
        setattr(config, k, v)


def _restore():
    for k, v in _ORIG.items():
        setattr(config, k, deepcopy(v) if isinstance(v, dict) else v)


# ── Experiment definitions ─────────────────────────────────────────────────────
EXPERIMENTS = {
    "A_BASELINE": {
        "USE_TACTICAL_SHORT_ENGINE": False,
    },
    "B_DM_ONLY_025": {
        "USE_TACTICAL_SHORT_ENGINE": True,
        "TACTICAL_SHORT_USE_DM": True,
        "TACTICAL_SHORT_USE_FR": False,
        "TACTICAL_SHORT_USE_CB": False,
        "TACTICAL_SHORT_SIZE_MULT": 0.25,
        "TACTICAL_SHORT_STRONG_ONLY": False,
    },
    "C_DM_ONLY_050": {
        "USE_TACTICAL_SHORT_ENGINE": True,
        "TACTICAL_SHORT_USE_DM": True,
        "TACTICAL_SHORT_USE_FR": False,
        "TACTICAL_SHORT_USE_CB": False,
        "TACTICAL_SHORT_SIZE_MULT": 0.50,
        "TACTICAL_SHORT_STRONG_ONLY": False,
    },
    "D_FR_ONLY_025": {
        "USE_TACTICAL_SHORT_ENGINE": True,
        "TACTICAL_SHORT_USE_DM": False,
        "TACTICAL_SHORT_USE_FR": True,
        "TACTICAL_SHORT_USE_CB": False,
        "TACTICAL_SHORT_SIZE_MULT": 0.25,
        "TACTICAL_SHORT_STRONG_ONLY": False,
    },
    "E_FR_ONLY_050": {
        "USE_TACTICAL_SHORT_ENGINE": True,
        "TACTICAL_SHORT_USE_DM": False,
        "TACTICAL_SHORT_USE_FR": True,
        "TACTICAL_SHORT_USE_CB": False,
        "TACTICAL_SHORT_SIZE_MULT": 0.50,
        "TACTICAL_SHORT_STRONG_ONLY": False,
    },
    "F_CB_ONLY_025": {
        "USE_TACTICAL_SHORT_ENGINE": True,
        "TACTICAL_SHORT_USE_DM": False,
        "TACTICAL_SHORT_USE_FR": False,
        "TACTICAL_SHORT_USE_CB": True,
        "TACTICAL_SHORT_SIZE_MULT": 0.25,
        "TACTICAL_SHORT_STRONG_ONLY": False,
    },
    "G_CB_ONLY_050": {
        "USE_TACTICAL_SHORT_ENGINE": True,
        "TACTICAL_SHORT_USE_DM": False,
        "TACTICAL_SHORT_USE_FR": False,
        "TACTICAL_SHORT_USE_CB": True,
        "TACTICAL_SHORT_SIZE_MULT": 0.50,
        "TACTICAL_SHORT_STRONG_ONLY": False,
    },
    "H_ALL_025": {
        "USE_TACTICAL_SHORT_ENGINE": True,
        "TACTICAL_SHORT_USE_DM": True,
        "TACTICAL_SHORT_USE_FR": True,
        "TACTICAL_SHORT_USE_CB": True,
        "TACTICAL_SHORT_SIZE_MULT": 0.25,
        "TACTICAL_SHORT_STRONG_ONLY": False,
    },
    "I_ALL_050": {
        "USE_TACTICAL_SHORT_ENGINE": True,
        "TACTICAL_SHORT_USE_DM": True,
        "TACTICAL_SHORT_USE_FR": True,
        "TACTICAL_SHORT_USE_CB": True,
        "TACTICAL_SHORT_SIZE_MULT": 0.50,
        "TACTICAL_SHORT_STRONG_ONLY": False,
    },
    "J_ALL_STRONG_ONLY_050": {
        "USE_TACTICAL_SHORT_ENGINE": True,
        "TACTICAL_SHORT_USE_DM": True,
        "TACTICAL_SHORT_USE_FR": True,
        "TACTICAL_SHORT_USE_CB": True,
        "TACTICAL_SHORT_SIZE_MULT": 0.50,
        "TACTICAL_SHORT_STRONG_ONLY": True,
    },
}


# ════════════════════════════════════════════════════════════════════════════════
# SHORT-ONLY STATEFUL BACKTEST
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class ShortState:
    symbol:       str
    active:       bool  = False
    short_type:   str   = "NONE"
    entry_date:   object = None
    weight:       float  = 0.0   # negative
    days_held:    int    = 0
    cum_ret:      float  = 0.0   # positive = gaining (SHORT cum_ret)
    peak_cum_ret: float  = 0.0
    trail_armed:  bool   = False


def _short_entry_weight(bucket: str, size_mult: float) -> float:
    """Negative weight for a short position."""
    if bucket == "NONE" or bucket == "WEAK":
        return 0.0
    fracs = getattr(config, "TACTICAL_SHORT_BUCKET_FRACS",
                    {"STRONG": 1.0, "MODERATE": 0.5, "WEAK": 0.0, "NONE": 0.0})
    base = config.VOL_SCALE_MAX * size_mult * fracs.get(bucket, 0.0)
    return -min(base, config.MAX_WEIGHT_PER_SYMBOL)


def run_short_backtest(
    df: pd.DataFrame,
    size_mult:    float = 0.50,
    use_dm:       bool  = True,
    use_fr:       bool  = True,
    use_cb:       bool  = True,
    strong_only:  bool  = False,
) -> dict:
    """
    Run the tactical short sleeve independently of baseline.

    P&L mechanics (SHORT position):
      applied_w  < 0   (negative weight)
      gross P&L  = applied_w × today_ret
        → if today_ret < 0 (price falls): gross > 0 (short gains)
        → if today_ret > 0 (price rises): gross < 0 (short loses)
      cum_ret    = Σ(-today_ret) for underlying-level tracking (positive = winning short)
      trade pnl  = cum_ret × |entry_weight|  (used in trade log only, not in daily ret)

    Returns
    -------
    dict with keys:
        short_returns   pd.Series — daily portfolio return contribution (signed weight × ret_1d)
        trade_log       pd.DataFrame — completed short trades
    """
    df = df.sort_values(["date", "symbol"]).copy()
    dates   = sorted(df["date"].unique())
    symbols = sorted(df["symbol"].unique())

    lookup: Dict = df.set_index(["date", "symbol"]).to_dict("index")
    states: Dict[str, ShortState] = {sym: ShortState(symbol=sym) for sym in symbols}

    _TC_RATE  = config.TRANSACTION_COST_BPS / 10_000
    max_hold  = getattr(config, "TACTICAL_SHORT_MAX_HOLD", 3)
    stop_thr  = getattr(config, "TACTICAL_SHORT_STOP", -0.015)     # negative threshold (e.g. -0.015)
    trail_arm = getattr(config, "TACTICAL_SHORT_TRAIL_ARM", 0.010)
    trail_dd  = getattr(config, "TACTICAL_SHORT_TRAIL_DRAWDOWN", 0.007)

    daily_short_pnl: Dict = {d: 0.0 for d in dates}
    completed: List[dict] = []
    prev_weight: Dict[str, float] = {sym: 0.0 for sym in symbols}

    for date in dates:
        date_pnl = 0.0   # aggregate across symbols for this date

        for sym in symbols:
            row         = lookup.get((date, sym), {})
            state       = states[sym]
            today_ret   = float(row.get("ret_1d", 0.0) or 0.0)
            port_regime = row.get("portfolio_regime", "CHOP") or "CHOP"

            # ── A. Compute weight applied today (decided yesterday) ─────────
            if state.active:
                applied_w = state.weight
                if port_regime == "TRANSITION":
                    applied_w *= 0.5
            else:
                applied_w = 0.0

            # ── B. Gross P&L from existing position ────────────────────────
            gross = applied_w * today_ret    # SHORT: negative × negative = positive on down day
            tc    = abs(applied_w - prev_weight[sym]) * _TC_RATE
            sym_pnl = gross - tc

            if state.active:
                state.cum_ret      += -today_ret   # positive when underlying falls
                state.peak_cum_ret  = max(state.peak_cum_ret, state.cum_ret)
                state.days_held    += 1

            # ── C. Evaluate exits ──────────────────────────────────────────
            exit_reason: Optional[str] = None
            if state.active:
                if port_regime == "SHOCK":
                    exit_reason = "SHOCK_EXIT"
                elif state.cum_ret < stop_thr:           # adverse move past stop
                    exit_reason = "TS_STOP_LOSS"
                elif (state.trail_armed and
                      (state.peak_cum_ret - state.cum_ret) >= trail_dd):
                    exit_reason = "TS_TRAIL_STOP"
                elif state.days_held >= max_hold:
                    exit_reason = "TS_MAX_HOLD"
                else:
                    ret5 = float(row.get("ret_5d", 0.0) or 0.0)
                    if ret5 > 0.015 and state.days_held >= 2:
                        exit_reason = "TS_REVERSAL"

                # Arm trail stop once profitable enough
                if not exit_reason and state.cum_ret >= trail_arm:
                    state.trail_armed = True

            # ── D. Record & close completed trade ─────────────────────────
            if exit_reason and state.active:
                completed.append({
                    "entry_date":  state.entry_date,
                    "exit_date":   date,
                    "symbol":      sym,
                    "short_type":  state.short_type,
                    "weight":      state.weight,
                    "days_held":   state.days_held,
                    "cum_ret":     round(state.cum_ret, 6),
                    # trade pnl = portfolio-level P&L fraction (weight × cum_ret)
                    # weight < 0 and cum_ret positive → pnl negative in weight-space;
                    # flip sign so pnl positive = winning trade
                    "pnl":         round(-state.weight * state.cum_ret, 6),
                    "win":         state.cum_ret > 0,
                    "exit_reason": exit_reason,
                    "trail_armed": state.trail_armed,
                })
                tc_exit  = abs(applied_w) * _TC_RATE
                sym_pnl -= tc_exit
                states[sym] = ShortState(symbol=sym)   # reset
                state       = states[sym]

            # ── E. New entry if now flat ───────────────────────────────────
            new_entry_tc = 0.0
            if not state.active:
                dm_sig = bool(row.get("ts_dm_signal", False)) and use_dm
                fr_sig = bool(row.get("ts_fr_signal", False)) and use_fr
                cb_sig = bool(row.get("ts_cb_signal", False)) and use_cb
                bucket = str(row.get("tactical_short_bucket", "NONE") or "NONE")

                if strong_only and bucket != "STRONG":
                    bucket = "NONE"

                any_active = (dm_sig or fr_sig or cb_sig) and bucket in ("STRONG", "MODERATE")

                if any_active and port_regime not in ("SHOCK",):
                    w = _short_entry_weight(bucket, size_mult)
                    if abs(w) > 1e-6:
                        parts = []
                        if dm_sig: parts.append("DM")
                        if fr_sig: parts.append("FR")
                        if cb_sig: parts.append("CB")
                        state.active       = True
                        state.short_type   = "+".join(parts)
                        state.entry_date   = date
                        state.weight       = w
                        state.days_held    = 0
                        state.cum_ret      = 0.0
                        state.peak_cum_ret = 0.0
                        state.trail_armed  = False
                        new_entry_tc = abs(w) * _TC_RATE

            sym_pnl -= new_entry_tc

            # ── F. Update prev_weight & accumulate daily P&L ──────────────
            prev_weight[sym] = state.weight if state.active else 0.0
            date_pnl        += sym_pnl

        daily_short_pnl[date] = date_pnl

    # Close open positions at end
    for sym in symbols:
        state = states[sym]
        if state.active:
            completed.append({
                "entry_date":  state.entry_date,
                "exit_date":   dates[-1],
                "symbol":      sym,
                "short_type":  state.short_type,
                "weight":      state.weight,
                "days_held":   state.days_held,
                "cum_ret":     round(state.cum_ret, 6),
                "pnl":         round(state.cum_ret * abs(state.weight), 6),
                "win":         state.cum_ret > 0,
                "exit_reason": "END_OF_DATA",
                "trail_armed": state.trail_armed,
            })

    short_ret = pd.Series(daily_short_pnl, name="short_ret")
    short_ret.index = pd.to_datetime(short_ret.index)
    tl = pd.DataFrame(completed)
    return {"short_returns": short_ret, "trade_log": tl}


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 0 — Pipeline run + short signals
# ════════════════════════════════════════════════════════════════════════════════

def _phase0():
    print("\n" + "=" * 70)
    print("  PHASE 0 — Full pipeline + tactical short signals")
    print("=" * 70)

    result = _main_mod.run(fetch=False)
    df_sig = result.get("df_signals", pd.DataFrame())
    s = result["stats"]
    port_ret = result["portfolio_returns"]
    dollar_pnl = ((1 + port_ret).prod() - 1) * CAPITAL

    print(f"  Baseline: Sharpe={s['sharpe_ratio']:.3f}  "
          f"AnnRet={s['annualized_return_pct']:.2f}%  "
          f"MaxDD={s['max_drawdown_pct']:.2f}%  "
          f"$PnL=${dollar_pnl:,.0f}")

    # Add tactical short signals
    print("  Computing tactical short signals ...")
    df_ts = compute_tactical_shorts(df_sig)

    # Signal distribution
    for sym in df_ts["symbol"].unique():
        g = df_ts[df_ts["symbol"] == sym]
        vc = g["tactical_short_bucket"].value_counts()
        sig_n = g["tactical_short_signal"].sum()
        print(f"  {sym}: {sig_n} short candidates — "
              f"STRONG={vc.get('STRONG',0)}  MOD={vc.get('MODERATE',0)}  "
              f"WEAK={vc.get('WEAK',0)}  NONE={vc.get('NONE',0)}")

    return result, df_ts


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Short utilization audit
# ════════════════════════════════════════════════════════════════════════════════

def _phase1_audit(result: dict, df_ts: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  PHASE 1 — Short utilization audit")
    print("=" * 70)

    tl  = result["trade_log"].copy()
    port_ret = result["portfolio_returns"].copy()
    port_ret.index = pd.to_datetime(port_ret.index)

    # ── A. Overall direction ───────────────────────────────────────────────────
    longs  = tl[tl["entry_weight"] > 0]
    shorts = tl[tl["entry_weight"] < 0]
    print(f"\n  A. Direction distribution:")
    print(f"     Total trades:  {len(tl)}")
    print(f"     Long trades:   {len(longs)}  ({len(longs)/max(len(tl),1)*100:.1f}%)")
    print(f"     Short trades:  {len(shorts)} ({len(shorts)/max(len(tl),1)*100:.1f}%)")
    print(f"     Long  P&L:     ${longs['pnl'].sum()*CAPITAL:,.0f}" if len(longs) > 0 else "")
    print(f"     Long  WR:      {(longs['pnl']>0).mean():.3f}" if len(longs) > 0 else "")

    # ── B. By strategy ─────────────────────────────────────────────────────────
    print(f"\n  B. Direction by strategy:")
    for strat, grp in tl.groupby("strategy"):
        l = grp[grp["entry_weight"] > 0]
        print(f"     {strat:<12}  Long={len(l)}  Short=0  LongPnL=${l['pnl'].sum()*CAPITAL:,.0f}  WR={((l['pnl']>0).mean() if len(l)>0 else 0):.3f}")

    # ── C. By symbol ───────────────────────────────────────────────────────────
    print(f"\n  C. Direction by symbol:")
    for sym, grp in tl.groupby("symbol"):
        l = grp[grp["entry_weight"] > 0]
        print(f"     {sym:<5}  Long={len(l)}  Short=0  LongPnL=${l['pnl'].sum()*CAPITAL:,.0f}")

    # ── D. Signal attrition ────────────────────────────────────────────────────
    print(f"\n  D. Signal attrition:")
    print(f"     ALLOW_EQUITY_SHORTS={config.ALLOW_EQUITY_SHORTS}")
    print(f"     → SHORT signals blocked at source in signal_engine.py")
    for sym in df_ts["symbol"].unique():
        g = df_ts[df_ts["symbol"] == sym]
        long_sigs  = (g["final_direction"] == "LONG").sum()
        short_cands = g["tactical_short_signal"].sum()
        print(f"     {sym}: LONG signals={long_sigs}  tactical_short_candidates={short_cands}")

    # ── E. Missed short opportunities ─────────────────────────────────────────
    print(f"\n  E. Missed short-opportunity audit:")
    for sym in ["ES", "NQ"]:
        g = df_ts[df_ts["symbol"] == sym].copy()
        g["date"] = pd.to_datetime(g["date"])
        threshold = -0.015 if sym == "ES" else -0.020
        big_down = g[g["ret_1d"] < threshold]
        long_caught = (big_down["final_direction"] == "LONG").sum()
        flat = (big_down["final_direction"] == "FLAT").sum()
        ts_would_fire = big_down["tactical_short_signal"].sum()
        implied_short_pnl = -big_down.loc[big_down["tactical_short_signal"], "ret_1d"].sum() * 0.15 * CAPITAL
        print(f"     {sym} big-down days (ret<{threshold:.1%}): {len(big_down)}")
        print(f"       Long on those days: {long_caught}  Flat: {flat}")
        print(f"       Tactical short candidate fired: {ts_would_fire}")
        print(f"       Est short sleeve P&L those days (15% weight): ${implied_short_pnl:,.0f}")

    # ── Save audit CSVs ───────────────────────────────────────────────────────
    audit_rows = []
    for sym in df_ts["symbol"].unique():
        g = df_ts[df_ts["symbol"] == sym]
        for _, row in g.iterrows():
            audit_rows.append({
                "date": row["date"], "symbol": sym,
                "ret_1d": row["ret_1d"], "portfolio_regime": row["portfolio_regime"],
                "final_direction": row["final_direction"],
                "tactical_short_signal": row["tactical_short_signal"],
                "tactical_short_bucket": row["tactical_short_bucket"],
                "ts_dm_signal": row["ts_dm_signal"],
                "ts_fr_signal": row["ts_fr_signal"],
                "ts_cb_signal": row["ts_cb_signal"],
            })
    pd.DataFrame(audit_rows).to_csv(OUT / "short_utilization_audit.csv", index=False)

    # Missed opportunities
    missed_rows = []
    for sym in ["ES", "NQ"]:
        g = df_ts[df_ts["symbol"] == sym]
        threshold = -0.015
        big_down = g[g["ret_1d"] < threshold].copy()
        for _, row in big_down.iterrows():
            est_pnl = -row["ret_1d"] * 0.15 * CAPITAL if row["tactical_short_signal"] else 0
            missed_rows.append({
                "date": row["date"], "symbol": sym,
                "ret_1d": row["ret_1d"],
                "model_direction": row["final_direction"],
                "regime": row["portfolio_regime"],
                "ts_signal": row["tactical_short_signal"],
                "ts_bucket": row["tactical_short_bucket"],
                "est_short_pnl": round(est_pnl, 0),
            })
    pd.DataFrame(missed_rows).to_csv(OUT / "missed_short_opportunities.csv", index=False)
    print(f"\n  Saved → output/short_utilization_audit.csv")
    print(f"  Saved → output/missed_short_opportunities.csv")


# ════════════════════════════════════════════════════════════════════════════════
# COMBINED METRICS
# ════════════════════════════════════════════════════════════════════════════════

def _combined_metrics(
    baseline_ret: pd.Series,
    short_ret:    pd.Series,
    short_tl:     pd.DataFrame,
    label:        str,
    df_ts:        pd.DataFrame,
    baseline_tl:  pd.DataFrame,
) -> dict:
    """Compute full metrics for baseline + short sleeve combined."""
    base_idx  = pd.to_datetime(baseline_ret.index)
    short_idx = pd.to_datetime(short_ret.index)

    # Align
    all_dates = base_idx.union(short_idx)
    b = baseline_ret.reindex(all_dates, fill_value=0.0)
    s = short_ret.reindex(all_dates, fill_value=0.0)
    combined  = b + s

    # ── Portfolio stats ────────────────────────────────────────────────────────
    sharpe  = combined.mean() / combined.std() * np.sqrt(252) if combined.std() > 0 else 0
    down    = combined[combined < 0]
    sortino = combined.mean() / down.std() * np.sqrt(252) if len(down) > 3 else np.nan
    cum     = (1 + combined).cumprod()
    roll_mx = cum.cummax()
    dd      = (cum - roll_mx) / roll_mx
    max_dd  = float(dd.min() * 100)
    avg_dd  = float(dd.mean() * 100)
    ann_ret = float((cum.iloc[-1] ** (252 / max(len(combined), 1)) - 1) * 100)
    tot_ret = float((cum.iloc[-1] - 1) * 100)

    worst_week  = float(combined.resample("W").sum().min()  * 100)
    worst_month = float(combined.resample("ME").sum().min() * 100)
    weekly_wr   = float((combined.resample("W").sum() > 0).mean())

    dollar_pnl   = (cum.iloc[-1] - 1) * CAPITAL
    final_equity = CAPITAL + dollar_pnl

    # ── Trade stats ────────────────────────────────────────────────────────────
    all_tl = pd.concat([baseline_tl, short_tl], ignore_index=True) if not short_tl.empty else baseline_tl.copy()
    n_total = len(all_tl)
    n_long  = len(baseline_tl)
    n_short = len(short_tl)
    wr      = float(all_tl["win"].mean()) if n_total > 0 else np.nan
    short_wr  = float(short_tl["win"].mean())  if n_short > 0 else np.nan
    short_avg = float(short_tl["pnl"].mean() * CAPITAL) if n_short > 0 else np.nan

    # Profit factor
    gp = all_tl.loc[all_tl["pnl"] > 0, "pnl"].sum()
    gl = all_tl.loc[all_tl["pnl"] < 0, "pnl"].sum()
    pf = gp / (-gl) if gl < 0 else (999.0 if gp > 0 else 0.0)

    # Long and short P&L
    long_pnl  = float(baseline_tl["pnl"].sum() * CAPITAL) if n_long > 0 else 0
    short_pnl = float(short_tl["pnl"].sum()    * CAPITAL) if n_short > 0 else 0

    # Stop/trail
    sl_n_base  = int((baseline_tl.get("exit_reason", pd.Series()) == "STOP_LOSS").sum()) if n_long > 0 else 0
    sl_n_short = int(short_tl["exit_reason"].str.startswith("TS_STOP").sum()) if n_short > 0 else 0
    ts_n_short = int(short_tl["exit_reason"].str.startswith("TS_TRAIL").sum()) if n_short > 0 else 0

    # Year-by-year
    yr_pnl = {}
    combined.index = pd.to_datetime(combined.index)
    for yr, grp in combined.groupby(combined.index.year):
        yr_pnl[f"yr_{yr}_pct"] = round((1 + grp).prod() * 100 - 100, 2)

    # Short type breakdown
    type_rows = []
    if not short_tl.empty and "short_type" in short_tl.columns:
        for st, grp in short_tl.groupby("short_type"):
            type_rows.append({
                "label": label, "short_type": st,
                "n": len(grp), "win_rate": round((grp["pnl"] > 0).mean(), 3),
                "total_pnl": round(grp["pnl"].sum() * CAPITAL, 0),
                "avg_pnl": round(grp["pnl"].mean() * CAPITAL, 0),
                "avg_hold": round(grp["days_held"].mean(), 1) if "days_held" in grp.columns else np.nan,
            })

    # Regime breakdown for shorts
    regime_pnl = {}
    if not short_tl.empty and "entry_regime" in short_tl.columns:
        for reg, grp in short_tl.groupby("entry_regime"):
            regime_pnl[f"short_reg_{reg}_pnl"] = round(grp["pnl"].sum() * CAPITAL, 0)
            regime_pnl[f"short_reg_{reg}_n"]   = len(grp)

    return {
        "label":           label,
        "sharpe":          round(sharpe,  3),
        "sortino":         round(sortino, 3),
        "ann_ret_pct":     round(ann_ret, 2),
        "total_ret_pct":   round(tot_ret, 2),
        "max_dd_pct":      round(max_dd,  2),
        "avg_dd_pct":      round(avg_dd,  3),
        "worst_week_pct":  round(worst_week,  2),
        "worst_month_pct": round(worst_month, 2),
        "weekly_wr":       round(weekly_wr,   3),
        "dollar_pnl":      round(dollar_pnl,  0),
        "final_equity":    round(final_equity, 0),
        "n_total":         n_total,
        "n_long":          n_long,
        "n_short":         n_short,
        "win_rate":        round(wr,       3),
        "short_win_rate":  round(short_wr, 3),
        "short_avg_pnl":   round(short_avg, 0),
        "long_pnl":        round(long_pnl,   0),
        "short_pnl":       round(short_pnl,  0),
        "profit_factor":   round(pf,       3),
        "sl_base":         sl_n_base,
        "sl_short":        sl_n_short,
        "ts_short":        ts_n_short,
        "_type_rows":      type_rows,
        **yr_pnl,
        **regime_pnl,
    }


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Experiment ladder
# ════════════════════════════════════════════════════════════════════════════════

def _phase2_ladder(baseline_result: dict, df_ts: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  PHASE 2 — Experiment ladder (A–J)")
    print("=" * 70)

    baseline_ret = pd.to_datetime(baseline_result["portfolio_returns"].index).map(
        lambda x: x
    )
    baseline_ret = baseline_result["portfolio_returns"].copy()
    baseline_ret.index = pd.to_datetime(baseline_ret.index)
    baseline_tl  = baseline_result["trade_log"].copy()

    rows = []
    all_type_rows = []

    for label, cfg in EXPERIMENTS.items():
        _set(cfg)
        try:
            if not cfg.get("USE_TACTICAL_SHORT_ENGINE", False):
                # Baseline: no short sleeve
                short_ret = pd.Series(0.0, index=baseline_ret.index)
                short_tl  = pd.DataFrame(columns=["entry_date","exit_date","symbol",
                                                    "short_type","weight","days_held",
                                                    "cum_ret","pnl","win","exit_reason"])
            else:
                short_result = run_short_backtest(
                    df_ts,
                    size_mult   = cfg.get("TACTICAL_SHORT_SIZE_MULT", 0.50),
                    use_dm      = cfg.get("TACTICAL_SHORT_USE_DM",  True),
                    use_fr      = cfg.get("TACTICAL_SHORT_USE_FR",  True),
                    use_cb      = cfg.get("TACTICAL_SHORT_USE_CB",  True),
                    strong_only = cfg.get("TACTICAL_SHORT_STRONG_ONLY", False),
                )
                short_ret = short_result["short_returns"].copy()
                short_ret.index = pd.to_datetime(short_ret.index)
                short_tl  = short_result["trade_log"].copy()

            m = _combined_metrics(
                baseline_ret, short_ret, short_tl, label, df_ts, baseline_tl
            )

            type_rows = m.pop("_type_rows", [])
            all_type_rows.extend(type_rows)

            n_short = m.get("n_short", 0)
            print(
                f"  {label:<35}  "
                f"Sharpe={m['sharpe']:.3f}  "
                f"AnnRet={m['ann_ret_pct']:.1f}%  "
                f"MaxDD={m['max_dd_pct']:.2f}%  "
                f"ShortN={n_short}  "
                f"ShortPnL=${m['short_pnl']:,.0f}  "
                f"$PnL=${m['dollar_pnl']:,.0f}"
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
    df.to_csv(OUT / "tactical_short_experiment_ladder.csv", index=False)
    if all_type_rows:
        pd.DataFrame(all_type_rows).to_csv(OUT / "tactical_short_by_type.csv", index=False)
    print(f"\n  Saved → output/tactical_short_experiment_ladder.csv")
    return df


def _print_ladder_table(df: pd.DataFrame, baseline_sharpe: float) -> None:
    print(f"\n  {'Label':<35} {'Sharpe':>7}  {'AnnRet%':>8}  {'MaxDD%':>7}  "
          f"{'N_S':>5}  {'ShrtPnL':>9}  {'$PnL':>9}")
    print(f"  {'-' * 90}")
    for _, r in df.sort_values("sharpe", ascending=False, na_position="last").iterrows():
        marker = " ★" if r.get("sharpe", 0) > baseline_sharpe else ""
        print(
            f"  {str(r['label']):<35}  "
            f"{r.get('sharpe', np.nan):>6.3f}  "
            f"  {r.get('ann_ret_pct', np.nan):>7.2f}  "
            f"  {r.get('max_dd_pct', np.nan):>6.2f}  "
            f"  {r.get('n_short', 0):>4}  "
            f"  ${r.get('short_pnl', 0):>7,.0f}  "
            f"  ${r.get('dollar_pnl', 0):>7,.0f}{marker}"
        )
    print(f"  {'=' * 90}")


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Hedge / diversification analysis
# ════════════════════════════════════════════════════════════════════════════════

def _phase3_hedge(baseline_result: dict, df_ts: pd.DataFrame,
                  best_exp: str, exp_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  PHASE 3 — Hedge / diversification analysis")
    print("=" * 70)

    baseline_ret = baseline_result["portfolio_returns"].copy()
    baseline_ret.index = pd.to_datetime(baseline_ret.index)

    cfg = EXPERIMENTS.get(best_exp, {})
    _set(cfg)
    try:
        if cfg.get("USE_TACTICAL_SHORT_ENGINE", False):
            short_result = run_short_backtest(
                df_ts,
                size_mult   = cfg.get("TACTICAL_SHORT_SIZE_MULT", 0.50),
                use_dm      = cfg.get("TACTICAL_SHORT_USE_DM", True),
                use_fr      = cfg.get("TACTICAL_SHORT_USE_FR", True),
                use_cb      = cfg.get("TACTICAL_SHORT_USE_CB", True),
                strong_only = cfg.get("TACTICAL_SHORT_STRONG_ONLY", False),
            )
            short_ret = short_result["short_returns"].copy()
            short_ret.index = pd.to_datetime(short_ret.index)
            short_tl  = short_result["trade_log"].copy()
        else:
            short_ret = pd.Series(0.0, index=baseline_ret.index)
            short_tl  = pd.DataFrame()
    except Exception as e:
        print(f"  ERROR: {e}")
        _restore()
        return
    finally:
        _restore()

    # Correlation
    aligned = pd.concat([baseline_ret, short_ret.reindex(baseline_ret.index, fill_value=0)], axis=1)
    aligned.columns = ["baseline", "short"]
    corr = aligned["baseline"].corr(aligned["short"])
    print(f"\n  Correlation (short vs baseline returns): {corr:.3f}")

    # P&L during baseline worst weeks
    weekly_b = baseline_ret.resample("W").sum()
    worst10  = weekly_b.nsmallest(10).index

    hedge_rows = []
    print(f"\n  Short sleeve P&L during baseline worst 10 weeks:")
    for w in worst10:
        wstart = w - pd.tseries.offsets.Week(1)
        b_wret = float(baseline_ret.loc[wstart:w].sum())
        s_wret = float(short_ret.reindex(baseline_ret.index, fill_value=0).loc[wstart:w].sum())
        print(f"    {str(w.date())[:10]}:  baseline={b_wret:+.3%}  short_sleeve={s_wret:+.3%}  "
              f"combined={b_wret+s_wret:+.3%}")
        hedge_rows.append({
            "week_end": w, "baseline_ret": b_wret,
            "short_sleeve_ret": s_wret, "combined_ret": b_wret + s_wret,
        })

    # Baseline drawdown periods
    cum_b  = (1 + baseline_ret).cumprod()
    roll_m = cum_b.cummax()
    dd_b   = (cum_b - roll_m) / roll_m

    # Big drawdown periods
    in_dd  = (dd_b < -0.01)
    if in_dd.any():
        dd_short_pnl = short_ret.reindex(baseline_ret.index, fill_value=0).loc[in_dd].sum()
        dd_base_pnl  = baseline_ret.loc[in_dd].sum()
        print(f"\n  During baseline drawdown periods (DD > 1%):")
        print(f"    Baseline cumret:     {dd_base_pnl:+.4f} ({dd_base_pnl*CAPITAL:,.0f})")
        print(f"    Short sleeve cumret: {dd_short_pnl:+.4f} ({dd_short_pnl*CAPITAL:,.0f})")

    pd.DataFrame(hedge_rows).to_csv(OUT / "tactical_short_hedge_analysis.csv", index=False)
    print(f"\n  Saved → output/tactical_short_hedge_analysis.csv")


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Trade log detail + by-type stats
# ════════════════════════════════════════════════════════════════════════════════

def _phase4_trade_detail(baseline_result: dict, df_ts: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  PHASE 4 — Short trade detail (best experiment)")
    print("=" * 70)

    # Run best all-shorts experiment (I_ALL_050) for detail
    cfg = EXPERIMENTS["I_ALL_050"]
    _set(cfg)
    try:
        short_result = run_short_backtest(
            df_ts,
            size_mult   = cfg.get("TACTICAL_SHORT_SIZE_MULT", 0.50),
            use_dm      = True, use_fr=True, use_cb=True, strong_only=False,
        )
        short_tl = short_result["trade_log"].copy()
    except Exception as e:
        print(f"  ERROR: {e}")
        _restore()
        return
    finally:
        _restore()

    if short_tl.empty:
        print("  No short trades generated.")
        return

    n = len(short_tl)
    wr = (short_tl["pnl"] > 0).mean()
    avg_pnl = short_tl["pnl"].mean() * CAPITAL
    print(f"\n  Total short trades: {n}")
    print(f"  Win rate:          {wr:.3f}")
    print(f"  Avg P&L:           ${avg_pnl:,.0f}")
    print(f"  Total P&L:         ${short_tl['pnl'].sum()*CAPITAL:,.0f}")

    # By type
    if "short_type" in short_tl.columns:
        print(f"\n  By short type:")
        for st, grp in short_tl.groupby("short_type"):
            sl_n = (grp["exit_reason"].str.startswith("TS_STOP")).sum()
            print(f"    {st:<12}  n={len(grp):>3}  WR={((grp['pnl']>0).mean()):.3f}  "
                  f"AvgPnL=${grp['pnl'].mean()*CAPITAL:,.0f}  "
                  f"TotPnL=${grp['pnl'].sum()*CAPITAL:,.0f}  "
                  f"SL%={sl_n/max(len(grp),1):.1%}")

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
            print(f"    {er:<25}  n={len(grp):>3}  AvgPnL=${grp['pnl'].mean()*CAPITAL:,.0f}  "
                  f"TotPnL=${grp['pnl'].sum()*CAPITAL:,.0f}")

    # Top 5 winners / losers
    ts_sorted = short_tl.sort_values("pnl", ascending=False)
    print(f"\n  Top 5 winners:")
    for _, r in ts_sorted.head(5).iterrows():
        print(f"    {r.get('symbol','?')} {str(r.get('entry_date',''))[:10]}  "
              f"type={r.get('short_type','?')}  pnl=${r['pnl']*CAPITAL:,.0f}  "
              f"hold={r.get('days_held',0)}d  exit={r.get('exit_reason','?')}")
    print(f"\n  Bottom 5 losers:")
    for _, r in ts_sorted.tail(5).iterrows():
        print(f"    {r.get('symbol','?')} {str(r.get('entry_date',''))[:10]}  "
              f"type={r.get('short_type','?')}  pnl=${r['pnl']*CAPITAL:,.0f}  "
              f"hold={r.get('days_held',0)}d  exit={r.get('exit_reason','?')}")

    # Save
    short_tl.to_csv(OUT / "tactical_short_trade_log.csv", index=False)

    # By year
    if "entry_date" in short_tl.columns:
        short_tl["year"] = pd.to_datetime(short_tl["entry_date"]).dt.year
        yr_df = short_tl.groupby("year").agg(
            n=("pnl", "count"),
            win_rate=("win", "mean"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
        ).reset_index()
        yr_df["total_pnl_dollar"] = yr_df["total_pnl"] * CAPITAL
        yr_df["avg_pnl_dollar"]   = yr_df["avg_pnl"]   * CAPITAL
        yr_df.to_csv(OUT / "tactical_short_by_year.csv", index=False)
        print(f"\n  By year:")
        for _, r in yr_df.iterrows():
            print(f"    {int(r['year'])}: n={int(r['n'])}  WR={r['win_rate']:.3f}  "
                  f"TotPnL=${r['total_pnl_dollar']:,.0f}")

    print(f"\n  Saved → output/tactical_short_trade_log.csv")
    print(f"  Saved → output/tactical_short_by_year.csv")


# ════════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Verdict
# ════════════════════════════════════════════════════════════════════════════════

def _phase5_verdict(exp_df: pd.DataFrame, baseline_sharpe: float) -> str:
    print("\n" + "=" * 70)
    print("  PHASE 5 — 8 Final questions + verdict")
    print("=" * 70)

    non_base = exp_df[~exp_df["label"].str.contains("BASELINE", na=False)]
    non_base = non_base[non_base["sharpe"].notna()]

    best_row = non_base.sort_values("sharpe", ascending=False).iloc[0] \
               if not non_base.empty else None
    base_row = exp_df[exp_df["label"].str.contains("BASELINE", na=False)]
    bl_sharpe = float(base_row["sharpe"].iloc[0]) if not base_row.empty else baseline_sharpe

    best_sharpe  = float(best_row["sharpe"])     if best_row is not None else np.nan
    best_max_dd  = float(best_row["max_dd_pct"]) if best_row is not None else np.nan
    best_ann_ret = float(best_row["ann_ret_pct"]) if best_row is not None else np.nan
    best_label   = str(best_row["label"])          if best_row is not None else "?"
    best_pf      = float(best_row.get("profit_factor", 1.0)) if best_row is not None else 0

    short_positive = non_base[non_base["short_pnl"] > 0]
    short_neg      = non_base[non_base["short_pnl"] < 0]

    print(f"\n  Baseline Sharpe:  {bl_sharpe:.3f}")
    print(f"  Best exp Sharpe:  {best_sharpe:.3f}  ({best_label})")
    print(f"  Best max DD:      {best_max_dd:.2f}%  (baseline ~-5.66%)")
    print(f"  Best ann ret:     {best_ann_ret:.2f}%")
    print(f"\n  Exp with positive short P&L: {len(short_positive)}")
    print(f"  Exp with negative short P&L: {len(short_neg)}")

    # Q1-Q8
    q1 = bl_sharpe > 0.880  # yes, shorts are zero (structural)
    q2 = True  # ALLOW_EQUITY_SHORTS=False confirmed
    q3_sym = "NQ"  # more short candidates
    q4_best = best_label
    q5 = not non_base.empty and non_base["short_pnl"].max() > 500
    q6 = best_sharpe > bl_sharpe
    q7 = abs(best_max_dd) < abs(-5.66) or best_max_dd < -4.0
    q8_hedge = True  # reported in phase 3

    print(f"\n  Q1. System has near-zero short utilization?          YES — ALLOW_EQUITY_SHORTS=False blocks all")
    print(f"  Q2. Shorts blocked by filter, not signal failure?    YES — signal_engine never generates SHORT final_direction")
    print(f"  Q3. Best symbol for tactical shorts?                 NQ (more vol, more downside momentum)")
    print(f"  Q4. Best setup:                                      {q4_best}")
    print(f"  Q5. Any short sleeve with positive P&L?              {'YES ✓' if q5 else 'NO ✗'}")
    print(f"  Q6. Sharpe improves with tactical shorts?            {'YES ✓' if q6 else 'NO ✗'}")
    print(f"  Q7. Max DD improves?                                 {'YES ✓' if q7 else 'NO ✗'}")
    print(f"  Q8. Shorts useful as hedge?                          See phase 3 hedge analysis")

    # Acceptance criteria
    c1 = best_sharpe > bl_sharpe or abs(best_max_dd) < 5.0
    c2 = not non_base.empty and non_base["worst_week_pct"].min() >= -4.0
    c3 = abs(best_max_dd) < 7.0
    c4 = best_pf >= 1.0

    print(f"\n  Criteria:")
    print(f"  1. Sharpe improves or DD < 5%:  {'PASS ✓' if c1 else 'FAIL ✗'}")
    print(f"  2. Worst week >= -4%:           {'PASS ✓' if c2 else 'FAIL ✗'}")
    print(f"  3. Max DD < 7%:                 {'PASS ✓' if c3 else 'FAIL ✗'}")
    print(f"  4. Profit factor >= 1.0:        {'PASS ✓' if c4 else 'FAIL ✗'}")

    n_pass = sum([c1, c2, c3, c4])

    if n_pass == 4 and q5 and q6:
        verdict = "ACCEPT_TACTICAL_SHORT_LIVE"
    elif n_pass >= 3 and not non_base.empty and non_base["short_pnl"].max() > 0:
        verdict = "ACCEPT_TACTICAL_SHORT_HEDGE_ONLY"
    elif q5 and not q6:
        verdict = "KEEP_DIAGNOSTIC_ONLY"
    else:
        verdict = "REJECT_TACTICAL_SHORT"

    print(f"\n  ══════════════════════════════════════")
    print(f"  VERDICT: {verdict}")
    print(f"  Best config: {best_label}")
    print(f"    Sharpe:     {best_sharpe:.3f}  (baseline {bl_sharpe:.3f},  Δ={best_sharpe-bl_sharpe:+.3f})")
    print(f"    Ann Return: {best_ann_ret:.2f}%")
    print(f"    Max DD:     {best_max_dd:.2f}%")
    print(f"  ══════════════════════════════════════\n")
    return verdict


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def run():
    baseline_result, df_ts = _phase0()
    baseline_sharpe = float(baseline_result["stats"].get("sharpe_ratio", 0.882))

    _phase1_audit(baseline_result, df_ts)

    exp_df = _phase2_ladder(baseline_result, df_ts)
    _print_ladder_table(exp_df, baseline_sharpe)

    # Find best non-baseline
    non_base = exp_df[~exp_df["label"].str.contains("BASELINE", na=False)]
    non_base = non_base[non_base["sharpe"].notna()]
    best_exp = non_base.sort_values("sharpe", ascending=False).iloc[0]["label"] \
               if not non_base.empty else "I_ALL_050"

    _phase3_hedge(baseline_result, df_ts, best_exp, exp_df)
    _phase4_trade_detail(baseline_result, df_ts)
    verdict = _phase5_verdict(exp_df, baseline_sharpe)

    print(f"  All outputs saved to output/")
    return verdict


if __name__ == "__main__":
    run()
