"""
backtest.py V3 — Stateful day-by-day backtest with winner extension,
trailing profit stops, stop losses, and add-to-winner. [CHANGED]

ARCHITECTURE CHANGE FROM V2:
  V2: vectorised — shift weight matrix by 1 day, multiply by returns.
  V3: iterative — PositionState per symbol tracks cum_ret, peak, trail flag.
      Sizing is done in position_sizing.py at entry time, not pre-computed.

EXIT PRIORITY ORDER (evaluated each day for every open position):
  1. SHOCK_EXIT          portfolio_regime == SHOCK → exit all
  2. STOP_LOSS           cum_ret < STOP_LOSS_THRESHOLD (-1.5%)
  3. SIGNAL_EXIT         signal says FLAT for fundamental reason (non-TREND)
  4. TRAIL_STOP          trailing profit fires
  5. FAST_EXIT           profit present + conviction faded
  6. MAX_HOLD_EXIT       non-TREND strategies only (timer-based)
  7. MAX_EXTENDED_HOLD   TREND hard cap at MAX_EXTENDED_HOLD_DAYS (7)

WINNER EXTENSION (TREND only):
  At day 3 (base MAX_HOLD_DAYS["TREND"]), check winner_conditions_hold.
  If True: keep holding (can_extend from signal_engine also checked).
  Extension continues until day 7 hard cap or conditions fail.

ADD-TO-WINNER (TREND only):
  After P&L update, if TREND position is profitable (cum_ret >= +0.5%)
  and max adds not reached: increase current_weight by 50% of entry weight.

TRANSACTION COSTS:
  Applied to abs weight change each day (same model as V2).

OUTPUT INTERFACE (identical to V2 for diagnostics compatibility):
  portfolio_returns   daily net return Series
  symbol_returns      ES and NQ net contribution DataFrame
  strategy_returns    daily return by strategy (4 columns)
  positions           weight-applied matrix (for attribution)
  trade_log           completed trades DataFrame
  pair_trades         SA pair trades only
  stats               performance summary dict
  daily_records       [NEW] per-day position detail for V3 diagnostics
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

import config
from position_sizing import compute_entry_weight
from winner_extension import (
    should_stop_loss,
    should_trail_exit,
    should_fast_exit,
    winner_conditions_hold,
    should_add_to_winner,
)

logger = logging.getLogger(__name__)

_STRATEGIES = ["TREND", "PULLBACK", "MEAN_REVERT", "STAT_ARB"]
_TC_RATE    = config.TRANSACTION_COST_BPS / 10_000


@dataclass
class PositionState:
    """All mutable state for a single symbol's current position."""
    symbol:          str
    direction:       str   = "FLAT"
    strategy:        str   = "NONE"
    entry_date:      Any   = None
    entry_weight:    float = 0.0
    entry_quality:   int   = 0
    high_conviction: bool  = False
    entry_regime:    str   = "NONE"
    entry_bias:      str   = "NONE"
    entry_reason:    str   = "NONE"
    entry_vol_scalar:float = np.nan
    current_weight:  float = 0.0
    max_weight:      float = 0.0
    days_held:       int   = 0
    cum_ret:         float = 0.0
    peak_cum_ret:    float = 0.0
    trail_active:    bool  = False
    extension_used:  bool  = False
    add_count:       int   = 0


def run_backtest(df: pd.DataFrame) -> dict:
    df      = df.sort_values(["date", "symbol"]).copy()
    dates   = sorted(df["date"].unique())
    symbols = sorted(df["symbol"].unique())

    # Fast O(1) row lookup
    lookup: Dict = df.set_index(["date", "symbol"]).to_dict("index")

    # Position state per symbol
    states: Dict[str, PositionState] = {sym: PositionState(symbol=sym) for sym in symbols}

    # Accumulators
    daily_pnl:      Dict = {date: {sym: 0.0 for sym in symbols} for date in dates}
    weight_applied: Dict = {date: {sym: 0.0 for sym in symbols} for date in dates}
    strategy_daily: Dict = {date: {sym: "NONE" for sym in symbols} for date in dates}
    completed_trades: List[dict] = []
    daily_records:    List[dict] = []
    prev_weight = {sym: 0.0 for sym in symbols}

    for date in dates:
        current_weights: Dict[str, float] = {}

        for sym in symbols:
            row   = lookup.get((date, sym), {})
            state = states[sym]

            # ── A. Record weight applied to today's return ─────────────────
            # (weight was decided at end of previous day; now it generates P&L)
            applied_w = state.current_weight if state.direction != "FLAT" else 0.0

            # TRANSITION regime: reduce existing positions to 50% (matches V2).
            # applied_w is scaled; state.current_weight keeps its full value so
            # that the weight restores automatically when regime returns to TREND.
            port_regime_a = row.get("portfolio_regime", "CHOP") or "CHOP"
            if port_regime_a == "TRANSITION" and state.direction != "FLAT":
                applied_w *= config.TRANSITION_SIZE_MULT

            weight_applied[date][sym] = applied_w
            strategy_daily[date][sym] = state.strategy if state.direction != "FLAT" else "NONE"

            # ── B. Apply today's return to open position ───────────────────
            today_ret = float(row.get("ret_1d", 0.0) or 0.0)
            if state.direction != "FLAT" and applied_w != 0.0:
                gross_pnl = applied_w * today_ret
                tc        = abs(applied_w - prev_weight.get(sym, 0.0)) * _TC_RATE
                daily_pnl[date][sym] = gross_pnl - tc

                # cum_ret tracks the underlying asset's cumulative return
                # (not portfolio-weighted) so exit thresholds remain intuitive.
                # LONG direction: +1% underlying = +1% cum_ret.
                signed_ret = today_ret if state.direction == "LONG" else -today_ret
                state.cum_ret      += signed_ret
                state.peak_cum_ret  = max(state.peak_cum_ret, state.cum_ret)
                state.days_held    += 1
            else:
                # Flat: only TC from any residual weight change
                tc = abs(applied_w - prev_weight.get(sym, 0.0)) * _TC_RATE
                daily_pnl[date][sym] = -tc

            # ── C. Evaluate exits ──────────────────────────────────────────
            port_regime = row.get("portfolio_regime", "CHOP") or "CHOP"
            exit_reason = None

            if state.direction != "FLAT":
                exit_reason = _evaluate_exits(state, row, port_regime)

            if exit_reason:
                _close_position(state, date, exit_reason, completed_trades)
                # Closing generates TC (going to zero)
                tc_exit = abs(applied_w) * _TC_RATE
                daily_pnl[date][sym] -= tc_exit

            # ── D. Add-to-winner (TREND positions still open) ─────────────
            if state.direction != "FLAT" and state.strategy == "TREND":
                bias    = row.get("weekly_bias",        "NEUTRAL") or "NEUTRAL"
                regime  = port_regime
                h4_sig  = row.get("h4_exec_signal",     "NONE") or "NONE"
                mom_str = float(row.get("momentum_strength", 0.0) or 0.0)
                if should_add_to_winner(state.strategy, state.cum_ret, state.add_count,
                                        bias, regime, h4_sig, mom_str):
                    add_w = state.entry_weight * config.ADD_WINNER_SIZE_FRAC
                    new_w = min(state.current_weight + add_w, config.MAX_WEIGHT_PER_SYMBOL)
                    tc_add = abs(new_w - state.current_weight) * _TC_RATE
                    state.current_weight = new_w
                    state.max_weight     = max(state.max_weight, new_w)
                    state.add_count     += 1
                    daily_pnl[date][sym] -= tc_add
                    logger.debug("ADD-TO-WINNER %s @ %s  new_w=%.4f", sym, date, new_w)

            # ── E. New entries ─────────────────────────────────────────────
            if state.direction == "FLAT":
                sig_dir    = row.get("final_direction", "FLAT") or "FLAT"
                sig_str    = row.get("strategy_used",   "NONE") or "NONE"
                sig_reason = row.get("signal_reason",   "")     or ""

                # Open when signal says non-FLAT direction.
                # HOLD signals are allowed for TREND — when backtest closes a
                # TREND position (trail stop / max-hold) but signal_engine still
                # shows valid TREND conditions, we should re-enter immediately.
                # For non-TREND strategies, HOLD signals come from an ongoing
                # position in signal_engine state — allow re-entry (the quality
                # gate blocks poor setups; re-entering a valid pullback/SA is fine).
                if sig_dir not in ("FLAT", "") and sig_str not in ("NONE", ""):
                    entry_w = compute_entry_weight(row, sig_str)
                    if abs(entry_w) > 1e-6:
                        _open_position(state, date, row, sig_dir, sig_str, entry_w)
                        tc_entry = abs(entry_w) * _TC_RATE
                        daily_pnl[date][sym] -= tc_entry

            # ── Record decided weight for this day ─────────────────────────
            current_weights[sym] = state.current_weight if state.direction != "FLAT" else 0.0

        # ── F. Gross exposure cap (both symbols combined) ──────────────────
        gross = sum(abs(w) for w in current_weights.values())
        if gross > config.MAX_COMBINED_EXPOSURE:
            scale = config.MAX_COMBINED_EXPOSURE / gross
            for sym in symbols:
                if states[sym].direction != "FLAT":
                    states[sym].current_weight *= scale
                    current_weights[sym]        = states[sym].current_weight

        # ── G. Record daily state for diagnostics ──────────────────────────
        for sym in symbols:
            state = states[sym]
            row   = lookup.get((date, sym), {})
            daily_records.append({
                "date":            date,
                "symbol":          sym,
                "weight":          current_weights.get(sym, 0.0),
                "weight_applied":  weight_applied[date][sym],
                "pnl":             daily_pnl[date][sym],
                "direction":       state.direction,
                "strategy":        state.strategy,
                "days_held":       state.days_held,
                "cum_ret":         state.cum_ret,
                "trail_active":    state.trail_active,
                "extension_used":  state.extension_used,
                "add_count":       state.add_count,
                "portfolio_regime": row.get("portfolio_regime", "CHOP"),
            })
            prev_weight[sym] = current_weights.get(sym, 0.0)

    # Close any positions still open at end of data
    for sym in symbols:
        state = states[sym]
        if state.direction != "FLAT":
            _close_position(state, dates[-1], "END_OF_DATA", completed_trades)

    return _build_output(
        dates, symbols, daily_pnl, weight_applied, strategy_daily,
        completed_trades, daily_records, df,
    )


# ── Exit evaluation ────────────────────────────────────────────────────────────

def _evaluate_exits(
    state: PositionState,
    row: dict,
    port_regime: str,
) -> Optional[str]:
    """
    Evaluate all exit conditions in priority order.
    Returns exit_reason string, or None to keep holding.
    Updates trail_active and extension_used as side effects.
    """
    # 1. SHOCK: exit everything immediately
    if port_regime == "SHOCK":
        return "SHOCK_EXIT"

    # 2. Stop loss: trade clearly wrong
    if should_stop_loss(state.cum_ret):
        return "STOP_LOSS"

    # 3. Signal fundamental exit (non-TREND): trust signal engine's MR/SA exits
    if state.strategy != "TREND":
        sig_dir    = row.get("final_direction", "FLAT") or "FLAT"
        sig_reason = row.get("signal_reason",   "")     or ""
        if sig_dir == "FLAT" and "MAX_HOLD_EXIT" not in sig_reason:
            return "SIGNAL_FUNDAMENTAL_EXIT"

    # 4. Trailing profit stop (update arm flag as side effect)
    fire_trail, new_trail = should_trail_exit(
        state.cum_ret, state.peak_cum_ret, state.trail_active
    )
    state.trail_active = new_trail
    if fire_trail:
        return "TRAIL_STOP"

    # 5. Fast exit: protect profit when conviction fades
    h4_sig = row.get("h4_exec_signal", "NONE") or "NONE"
    conf   = float(row.get("confidence_score", 1.0) or 1.0)
    if should_fast_exit(state.cum_ret, h4_sig, conf):
        return "FAST_EXIT"

    # 6. Max hold for non-TREND strategies (backtest enforces as safety net)
    if state.strategy != "TREND":
        max_h = config.MAX_HOLD_DAYS.get(state.strategy, 3)
        if state.days_held >= max_h:
            return f"MAX_HOLD_EXIT_{state.strategy}"
        return None  # non-TREND: no further checks needed

    # 7. TREND: winner extension check at base hold limit
    base_max = config.MAX_HOLD_DAYS.get("TREND", 3)
    if state.days_held >= base_max:
        bias    = row.get("weekly_bias",        "NEUTRAL") or "NEUTRAL"
        regime  = row.get("portfolio_regime",   "CHOP")    or "CHOP"
        h4_sig2 = row.get("h4_exec_signal",     "NONE")    or "NONE"
        mom_str = float(row.get("momentum_strength", 0.0) or 0.0)
        can_ext = bool(row.get("can_extend", False))

        if winner_conditions_hold(bias, regime, h4_sig2, mom_str) and can_ext:
            state.extension_used = True
            # Fall through — check hard cap below
        else:
            return "MAX_HOLD_EXIT_TREND"

    # 8. TREND hard cap (even with extension)
    if state.days_held >= config.MAX_EXTENDED_HOLD_DAYS:
        return "MAX_EXTENDED_HOLD_EXIT"

    return None


# ── Position open / close ──────────────────────────────────────────────────────

def _open_position(
    state: PositionState,
    date: Any,
    row: dict,
    direction: str,
    strategy: str,
    entry_weight: float,
) -> None:
    """Open a new position and reset all state."""
    state.direction       = direction
    state.strategy        = strategy
    state.entry_date      = date
    state.entry_weight    = abs(entry_weight)
    state.entry_quality   = int(row.get("entry_quality", 0) or 0)
    state.high_conviction = bool(row.get("high_conviction", False))
    state.entry_regime    = row.get("portfolio_regime", "UNKNOWN") or "UNKNOWN"
    state.entry_bias      = row.get("weekly_bias",      "UNKNOWN") or "UNKNOWN"
    state.entry_reason    = row.get("signal_reason",    "UNKNOWN") or "UNKNOWN"
    state.entry_vol_scalar= float(row.get("vol_scalar", np.nan) or np.nan)
    state.current_weight  = entry_weight
    state.max_weight      = abs(entry_weight)
    state.days_held       = 0   # incremented on first day's return
    state.cum_ret         = 0.0
    state.peak_cum_ret    = 0.0
    state.trail_active    = False
    state.extension_used  = False
    state.add_count       = 0


def _close_position(
    state: PositionState,
    exit_date: Any,
    exit_reason: str,
    trades: List[dict],
) -> None:
    """Record a completed trade and reset state to flat."""
    if state.entry_date is not None:
        trades.append({
            "symbol":           state.symbol,
            "strategy":         state.strategy,
            "direction":        state.direction,
            "entry_date":       state.entry_date,
            "exit_date":        exit_date,
            "holding_days":     state.days_held,
            "entry_weight":     state.entry_weight,
            "max_weight":       state.max_weight,
            "pnl":              state.cum_ret,
            "entry_regime":     state.entry_regime,
            "entry_bias":       state.entry_bias,
            "entry_reason":     state.entry_reason,
            "exit_reason":      exit_reason,
            "entry_quality":    state.entry_quality,
            "high_conviction":  state.high_conviction,
            "entry_vol_scalar": state.entry_vol_scalar,
            "trail_active":     state.trail_active,
            "extension_used":   state.extension_used,
            "add_count":        state.add_count,
        })

    # Reset to flat
    state.direction       = "FLAT"
    state.strategy        = "NONE"
    state.entry_date      = None
    state.entry_weight    = 0.0
    state.current_weight  = 0.0
    state.max_weight      = 0.0
    state.days_held       = 0
    state.cum_ret         = 0.0
    state.peak_cum_ret    = 0.0
    state.trail_active    = False
    state.extension_used  = False
    state.add_count       = 0


# ── Output construction ────────────────────────────────────────────────────────

def _build_output(
    dates, symbols, daily_pnl, weight_applied, strategy_daily,
    completed_trades, daily_records, df_signals,
) -> dict:
    idx = pd.DatetimeIndex(dates)

    # Portfolio and symbol returns
    port_ret_data = {d: sum(daily_pnl[d].values()) for d in dates}
    portfolio_returns = pd.Series(port_ret_data, index=idx, name="portfolio")

    sym_ret_data = {sym: {d: daily_pnl[d][sym] for d in dates} for sym in symbols}
    symbol_returns = pd.DataFrame(sym_ret_data, index=idx)

    # Strategy attribution
    strat_ret_data = {s: {d: 0.0 for d in dates} for s in _STRATEGIES}
    for d in dates:
        for sym in symbols:
            strat = strategy_daily[d][sym]
            pnl   = daily_pnl[d][sym]
            if strat in strat_ret_data:
                strat_ret_data[strat][d] += pnl
    strategy_returns = pd.DataFrame(strat_ret_data, index=idx)

    # Positions (weight applied to each day's return — for attribution analysis)
    pos_data = {sym: {d: weight_applied[d][sym] for d in dates} for sym in symbols}
    positions = pd.DataFrame(pos_data, index=idx)

    # Trade log
    if completed_trades:
        tl = pd.DataFrame(completed_trades)
        tl["win"] = tl["pnl"] > 0
    else:
        tl = pd.DataFrame()

    # Pair trades (SA NQ-only)
    pair_trades = tl[tl["strategy"] == "STAT_ARB"].copy() if not tl.empty else pd.DataFrame()

    # Stats
    stats = _compute_stats(portfolio_returns, symbol_returns, positions, tl, strategy_returns)

    logger.info(
        "Backtest V3 complete | Sharpe: %.3f  MaxDD: %.1f%%  Trades: %d  Win: %.1f%%",
        stats["sharpe_ratio"],
        stats["max_drawdown_pct"],
        stats.get("total_trades", 0),
        100 * stats.get("overall_win_rate", 0),
    )

    return {
        "portfolio_returns": portfolio_returns,
        "symbol_returns":    symbol_returns,
        "strategy_returns":  strategy_returns,
        "positions":         positions,
        "trade_log":         tl,
        "pair_trades":       pair_trades,
        "stats":             stats,
        "daily_records":     pd.DataFrame(daily_records),
    }


# ── Statistics ────────────────────────────────────────────────────────────────

def _compute_stats(returns, symbol_returns, positions, trade_log, strategy_returns) -> dict:
    ann = config.TRADING_DAYS_PER_YEAR

    cumret  = (1 + returns).cumprod()
    total_r = cumret.iloc[-1] - 1
    ann_r   = (1 + total_r) ** (ann / max(len(returns), 1)) - 1
    ann_vol = returns.std() * np.sqrt(ann)
    sharpe  = ann_r / ann_vol if ann_vol > 1e-9 else 0.0

    dd    = (cumret - cumret.cummax()) / cumret.cummax()
    maxdd = dd.min()

    nonzero = returns[returns != 0]
    hit     = (nonzero > 0).mean() if len(nonzero) else 0.0
    gp      = nonzero[nonzero > 0].sum()
    gl      = nonzero[nonzero < 0].sum()
    pf      = abs(gp / gl) if gl < 0 else np.nan

    gross_exp = positions.abs().sum(axis=1)

    recent_10 = returns.iloc[-10:] if len(returns) >= 10 else returns
    recent_30 = returns.iloc[-30:] if len(returns) >= 30 else returns
    recent_10_r = float((1 + recent_10).cumprod().iloc[-1] - 1)
    recent_30_r = float((1 + recent_30).cumprod().iloc[-1] - 1)

    roll_vol       = returns.rolling(30).std() * np.sqrt(ann)
    roll_ret       = returns.rolling(30).mean() * ann
    rolling_sharpe = (roll_vol > 1e-9) * (roll_ret / roll_vol.replace(0, np.nan))
    avg_rs         = float(rolling_sharpe.dropna().mean())

    strat_totals = {s: float(strategy_returns[s].sum()) for s in strategy_returns.columns}
    trade_stats  = _trade_stats(trade_log)

    return {
        "total_return_pct":         round(total_r * 100,     2),
        "annualized_return_pct":    round(ann_r   * 100,     2),
        "annualized_vol_pct":       round(ann_vol * 100,     2),
        "sharpe_ratio":             round(sharpe,            3),
        "max_drawdown_pct":         round(maxdd * 100,       2),
        "hit_rate":                 round(hit,               4),
        "profit_factor":            round(pf, 3) if not np.isnan(pf) else 0.0,
        "avg_gross_exposure":       round(gross_exp.mean(),  4),
        "pct_time_in_market":       round((gross_exp > 1e-6).mean(), 4),
        "recent_10d_return_pct":    round(recent_10_r * 100, 2),
        "recent_30d_return_pct":    round(recent_30_r * 100, 2),
        "avg_rolling_30d_sharpe":   round(avg_rs,            3),
        "symbol_contribution":      symbol_returns.sum().to_dict(),
        "strategy_totals":          strat_totals,
        "n_days":                   len(returns),
        **trade_stats,
    }


def _trade_stats(tl: pd.DataFrame) -> dict:
    empty = {
        "total_trades": 0, "overall_win_rate": 0.0, "profit_factor": 0.0,
        "avg_holding_days": 0.0, "median_holding_days": 0.0,
        "trades_by_strategy": {}, "win_rate_by_strategy": {},
        "trades_long": 0, "trades_short": 0,
        "long_win_rate": 0.0, "short_win_rate": 0.0,
        "high_conv_win_rate": 0.0, "high_conv_count": 0,
        "quality_4plus_win_rate": 0.0,
        # V3 additions
        "stop_loss_count": 0, "trail_stop_count": 0, "fast_exit_count": 0,
        "extension_count": 0, "add_to_winner_count": 0,
        "extended_win_rate": 0.0,
    }
    if tl is None or tl.empty:
        return empty

    wins = tl["win"]
    gp   = tl.loc[wins,  "pnl"].sum()
    gl   = tl.loc[~wins, "pnl"].sum()
    pf   = abs(gp / gl) if gl < 0 else np.nan

    by_s   = tl.groupby("strategy")["win"].agg(["count", "mean"]).to_dict("index")
    long_  = tl[tl["direction"] == "LONG"]
    short_ = tl[tl["direction"] == "SHORT"]

    hc_col = "high_conviction" if "high_conviction" in tl.columns else None
    hc_wr  = 0.0
    hc_cnt = 0
    if hc_col:
        hc    = tl[tl[hc_col] == True]
        hc_cnt = len(hc)
        hc_wr  = round(hc["win"].mean(), 4) if hc_cnt > 0 else 0.0

    q4_wr = 0.0
    if "entry_quality" in tl.columns:
        q4    = tl[tl["entry_quality"] >= 4]
        q4_wr = round(q4["win"].mean(), 4) if len(q4) > 0 else 0.0

    # V3 exit reason stats
    sl_cnt   = int((tl["exit_reason"] == "STOP_LOSS").sum())          if "exit_reason" in tl.columns else 0
    ts_cnt   = int((tl["exit_reason"] == "TRAIL_STOP").sum())         if "exit_reason" in tl.columns else 0
    fe_cnt   = int((tl["exit_reason"] == "FAST_EXIT").sum())          if "exit_reason" in tl.columns else 0
    ext_cnt  = int(tl["extension_used"].sum())                        if "extension_used" in tl.columns else 0
    add_cnt  = int((tl["add_count"] > 0).sum())                      if "add_count" in tl.columns else 0
    ext_wr   = 0.0
    if "extension_used" in tl.columns:
        ext_trades = tl[tl["extension_used"] == True]
        ext_wr = round(ext_trades["win"].mean(), 4) if len(ext_trades) > 0 else 0.0

    return {
        "total_trades":           int(len(tl)),
        "overall_win_rate":       round(wins.mean(), 4),
        "profit_factor":          round(pf, 3) if not np.isnan(pf) else 0.0,
        "avg_holding_days":       round(tl["holding_days"].mean(),   1),
        "median_holding_days":    round(tl["holding_days"].median(), 1),
        "trades_by_strategy":     {s: int(v["count"])        for s, v in by_s.items()},
        "win_rate_by_strategy":   {s: round(v["mean"], 4)   for s, v in by_s.items()},
        "trades_long":            int(len(long_)),
        "trades_short":           int(len(short_)),
        "long_win_rate":          round(long_["win"].mean(),  4) if len(long_)  else 0.0,
        "short_win_rate":         round(short_["win"].mean(), 4) if len(short_) else 0.0,
        "high_conv_win_rate":     hc_wr,
        "high_conv_count":        hc_cnt,
        "quality_4plus_win_rate": q4_wr,
        # V3 additions
        "stop_loss_count":        sl_cnt,
        "trail_stop_count":       ts_cnt,
        "fast_exit_count":        fe_cnt,
        "extension_count":        ext_cnt,
        "add_to_winner_count":    add_cnt,
        "extended_win_rate":      ext_wr,
    }
