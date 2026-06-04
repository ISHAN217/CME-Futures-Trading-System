"""
signal_engine.py — Stateful, regime-gated competition signal engine. [CHANGED]

Key changes from base system:

1. TRANSITION regime [NEW]
   When portfolio_regime == TRANSITION:
     - Existing positions: apply TRANSITION_SIZE_MULT (0.50) to weights
     - New entries: blocked. Only exits are allowed.
     - No 2-day buffer logic needed — TRANSITION IS the buffer.

2. TREND pullback entries [NEW]
   When regime == TREND, bias == LONG, mr_signal == PULLBACK:
     - Enter LONG with strategy="PULLBACK" (separate from TREND)
     - Max hold: 2 days (vs 3 for TREND)
     - Smaller size (handled in portfolio.py)
     - This captures fast reversion alpha within the trend

3. Entry quality gating [NEW]
   compute_entry_quality() from intraday_4h_engine is called here
   (needs weekly_bias context that 4H engine doesn't have).
   h4_quality < H4_QUALITY_LOW (2) → block entry
   h4_quality >= H4_QUALITY_HIGH (4) → allow high-conviction upscaling in portfolio

4. High-conviction upscaling signal [NEW]
   When momentum_strength > HIGH_CONV_MOM_THRESH (0.65)
   AND h4_quality >= HIGH_CONV_QUALITY_MIN (4):
   → sets high_conviction flag for portfolio.py to apply +20% size

5. Competition max holds [CHANGED]
   TREND: 3 days (was 5), PULLBACK: 2, MR: 2, SA: 3

6. SA runs in TREND and CHOP [CHANGED]
   In base system, SA only activated in CHOP. Here it can activate
   in TREND too, because the SA signal is a spread signal independent
   of regime. SA entries are blocked if TREND position already held.

State tracked per symbol:
    direction     current active position
    strategy      TREND / PULLBACK / MEAN_REVERT / STAT_ARB / NONE
    days_held     consecutive days in current position
    prev_regime   previous regime (for TRANSITION detection)
    in_transition whether currently in TRANSITION regime

Output per (date, symbol):
    final_direction   LONG / FLAT
    strategy_used     TREND / PULLBACK / MEAN_REVERT / STAT_ARB / NONE
    days_held         consecutive days in current position
    signal_reason     human-readable trade rationale
    confidence_score  float in [0, 1]
    h4_quality        entry quality score 1–5 (set at entry)
    high_conviction   bool — portfolio should apply +20% size
    in_transition     bool — portfolio should apply 0.5× size
"""

import logging
from copy import deepcopy
from typing import Dict, Any

import numpy as np
import pandas as pd

import config
from intraday_4h_engine import compute_entry_quality

logger = logging.getLogger(__name__)

_SYMBOLS = ["ES", "NQ"]

_INIT_STATE: Dict[str, Any] = {
    "direction":     "FLAT",
    "strategy":      "NONE",
    "days_held":     0,
    "prev_regime":   "NONE",
    "entry_quality": 0,
}


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stateful forward-pass over (date, symbol) pairs.
    Adds: final_direction, strategy_used, days_held, signal_reason,
          confidence_score, h4_quality, high_conviction, in_transition.
    """
    df = df.sort_values(["date", "symbol"]).copy()

    dates   = sorted(df["date"].unique())
    sym_set = sorted(df["symbol"].unique())

    states = {sym: deepcopy(_INIT_STATE) for sym in sym_set}

    lookup = df.set_index(["date", "symbol"]).to_dict("index")

    results = []

    for date in dates:
        es_row = lookup.get((date, "ES"), {})
        nq_row = lookup.get((date, "NQ"), {})

        port_regime  = es_row.get("portfolio_regime",               "CHOP")
        port_senti   = es_row.get("portfolio_sentiment_flag",       "NORMAL")
        port_senti_m = es_row.get("portfolio_sentiment_multiplier", 1.0)

        sa_signal = es_row.get("sa_signal",       "FLAT")
        sa_es_dir = es_row.get("sa_es_direction", "FLAT")
        sa_nq_dir = es_row.get("sa_nq_direction", "FLAT")

        # SA: can activate in both TREND and CHOP (not TRANSITION or SHOCK)
        sa_active = _evaluate_sa(
            port_regime, sa_signal, sa_es_dir, sa_nq_dir,
            states["ES"], states["NQ"],
        )

        for sym in sym_set:
            row   = lookup.get((date, sym), {})
            state = states[sym]

            # ── SHOCK: exit all immediately ──────────────────────────────
            if port_regime == "SHOCK":
                _force_flat(state)
                results.append(_make_result(
                    date, sym, state,
                    reason="SHOCK_EXIT",
                    confidence=0.0,
                    quality=0,
                    high_conv=False,
                    in_transition=False,
                    can_extend=False,
                ))
                continue

            # ── TRANSITION: hold at half size, no new entries ─────────────
            if port_regime == "TRANSITION":
                in_trans = True
                if state["direction"] != "FLAT":
                    state["days_held"] += 1
                    max_h = config.MAX_HOLD_DAYS.get(state["strategy"], 3)
                    if state["days_held"] >= max_h:
                        _force_flat(state)
                        reason = "MAX_HOLD_TRANSITION_EXIT"
                    else:
                        reason = "TRANSITION_HOLD"
                else:
                    reason = "TRANSITION_FLAT"

                results.append(_make_result(
                    date, sym, state,
                    reason=reason,
                    confidence=0.4,
                    quality=state.get("entry_quality", 0),
                    high_conv=False,
                    in_transition=True,
                    can_extend=False,
                ))
                state["prev_regime"] = port_regime
                continue

            # ── Normal regimes: TREND, CHOP ───────────────────────────────
            in_trans = False
            state["prev_regime"] = port_regime

            if port_regime == "TREND":
                # Check for PULLBACK sub-mode first (fast alpha)
                mr_mode  = row.get("mr_signal_mode", "NONE")
                mr_sig   = row.get("mr_signal",      "FLAT")

                if mr_mode == "TREND_PULLBACK" and mr_sig == "PULLBACK":
                    direction, reason, confidence, quality = _pullback_decision(row, state, sym)

                elif sa_active:
                    sa_dir = sa_es_dir if sym == "ES" else sa_nq_dir
                    if sa_dir == "FLAT":
                        # SA doesn't apply to this symbol (ES hedge disabled).
                        # Fall through to normal TREND decision — don't close ES positions.
                        direction, reason, confidence, quality = _trend_decision(row, state, sym)
                    else:
                        direction, reason, confidence, quality = _sa_decision(row, state, sym, sa_dir)
                else:
                    direction, reason, confidence, quality = _trend_decision(row, state, sym)

            elif port_regime == "CHOP":
                # SA takes priority, MR is fallback
                if sa_active:
                    sa_dir = sa_es_dir if sym == "ES" else sa_nq_dir
                    if sa_dir == "FLAT":
                        direction, reason, confidence, quality = "FLAT", "SA_ES_HEDGE_DISABLED", 0.0, 0
                    else:
                        direction, reason, confidence, quality = _sa_decision(row, state, sym, sa_dir)
                else:
                    direction, reason, confidence, quality = _mr_decision(row, state, sym)
            else:
                direction, reason, confidence, quality = "FLAT", "UNKNOWN_REGIME", 0.0, 0

            # ── Entry quality gate (blocks low-quality new entries) ────────
            if direction != "FLAT" and direction != state["direction"]:
                # This is a new entry attempt — gate by quality
                if quality < config.H4_QUALITY_LOW:
                    direction = "FLAT"
                    reason    = f"BLOCKED_LOW_QUALITY_{quality}"
                    confidence = 0.0
                    quality    = 0

            # ── Sentiment gate ─────────────────────────────────────────────
            direction, reason, confidence = _apply_sentiment(
                direction, reason, confidence,
                port_senti, port_senti_m, state,
            )

            # ── Max hold enforcement ───────────────────────────────────────
            direction, reason = _check_max_hold(direction, reason, state)

            # ── CWT gating — applied after signal assignment, before recording ──
            if config.USE_CWT_CHOP_FILTER:
                cwt_label    = str(row.get("cwt_chop_label", "CWT_UNCLEAR") or "CWT_UNCLEAR")
                cur_strategy = _infer_strategy(reason)

                if cur_strategy == "MEAN_REVERT" and config.CWT_APPLY_TO_MEAN_REVERT:
                    if cwt_label != "CWT_RANGE_OSCILLATORY":
                        direction = "FLAT"
                        reason    = f"BLOCKED_CWT_MR_{cwt_label}"
                        confidence = 0.0
                        cur_strategy = "NONE"

                elif cur_strategy == "PULLBACK" and config.CWT_APPLY_TO_PULLBACK:
                    if cwt_label in ("CWT_CHAOTIC_NOISE", "CWT_COMPRESSION"):
                        direction = "FLAT"
                        reason    = f"BLOCKED_CWT_PB_{cwt_label}"
                        confidence = 0.0
                        cur_strategy = "NONE"
                    elif cwt_label not in ("CWT_RANGE_OSCILLATORY", "CWT_TREND_COHERENT"):
                        reason = (reason or "") + f"|CWT_{cwt_label}"

                elif cur_strategy == "STAT_ARB" and config.CWT_APPLY_TO_STAT_ARB:
                    # Tag only — no blocking in v1
                    if cwt_label not in ("CWT_UNCLEAR",):
                        reason = (reason or "") + f"|CWT_{cwt_label}"

            # ── High-conviction flag ───────────────────────────────────────
            mom_str = float(row.get("momentum_strength", 0.0) or 0.0)
            high_conv = (
                direction != "FLAT"
                and mom_str >= config.HIGH_CONV_MOM_THRESH
                and quality >= config.HIGH_CONV_QUALITY_MIN
            )

            # ── Winner extension eligibility [V3] ─────────────────────────
            # True when TREND conditions remain valid for backtest to extend.
            can_extend = _compute_can_extend(row, state, direction)

            # ── Update state ───────────────────────────────────────────────
            _update_state(state, direction, reason, quality)

            results.append(_make_result(
                date, sym, state,
                reason=reason,
                confidence=confidence,
                quality=state["entry_quality"],
                high_conv=high_conv,
                in_transition=False,
                can_extend=can_extend,
            ))

    result_df = pd.DataFrame(results)
    df = df.merge(result_df, on=["date", "symbol"], how="left")
    _log_signal_distribution(df)
    return df


# ── Strategy decision functions ───────────────────────────────────────────────

def _trend_decision(row, state, sym):
    """TREND strategy: LONG only, requires weekly_bias == LONG."""
    bias     = row.get("weekly_bias",    "NEUTRAL")
    exec_sig = row.get("h4_exec_signal", "NONE")
    r2       = float(row.get("r2_20d", 0.0) or 0.0)

    # Still in a TREND position → hold or exit
    if state["strategy"] == "TREND" and state["direction"] != "FLAT":
        if exec_sig == "WAIT":
            return state["direction"], "TREND_HOLD_WAIT", r2, state["entry_quality"]
        return state["direction"], "TREND_HOLD", r2, state["entry_quality"]

    if bias != "LONG":
        return "FLAT", "TREND_FLAT_NO_BIAS", 0.0, 0

    # 4H signal must be actionable (not WAIT, not NONE for quality scoring)
    if exec_sig == "WAIT":
        return "FLAT", "TREND_BLOCKED_H4_WAIT", 0.0, 0

    # Compute entry quality (needs both weekly_bias and h4 features)
    quality = compute_entry_quality(row)

    reason = f"TREND_LONG_{exec_sig}" if exec_sig != "NONE" else "TREND_LONG_DAILY"
    return "LONG", reason, r2, quality


def _pullback_decision(row, state, sym):
    """
    TREND-regime pullback mode.
    A 4H dip within a confirmed uptrend — expect 1-2 day reversion.
    """
    if getattr(config, "DISABLE_PULLBACK", False):
        return "FLAT", "PULLBACK_DISABLED", 0.0, 0

    bias     = row.get("weekly_bias",    "NEUTRAL")
    mr_z     = float(row.get("mr_zscore", 0.0) or 0.0)
    mr_str   = float(row.get("mr_strength", 0.0) or 0.0)

    # Still in an existing PULLBACK position
    if state["strategy"] == "PULLBACK" and state["direction"] != "FLAT":
        # Exit when z reverts above exit threshold (0.30)
        if mr_z > -config.MR_EXIT_THRESHOLD:
            return "FLAT", "PULLBACK_EXIT_REVERTED", 0.0, 0
        return state["direction"], "PULLBACK_HOLD", mr_str, state["entry_quality"]

    # Block if already in a different active position
    if state["direction"] != "FLAT":
        return state["direction"], f"{state['strategy']}_HOLD", 0.5, state["entry_quality"]

    # Entry
    quality = compute_entry_quality(row)
    return "LONG", "TREND_PULLBACK_ENTRY", min(mr_str, 1.0), quality


def _mr_decision(row, state, sym):
    """CHOP-regime mean reversion."""
    if getattr(config, "DISABLE_MEAN_REVERT", False):
        return "FLAT", "MR_DISABLED", 0.0, 0

    mr_signal = row.get("mr_signal",   "FLAT")
    mr_z      = float(row.get("mr_zscore",   0.0) or 0.0)
    mr_str    = float(row.get("mr_strength", 0.0) or 0.0)
    bias      = row.get("weekly_bias", "NEUTRAL")

    # Exit existing MR position
    if state["strategy"] == "MEAN_REVERT" and state["direction"] != "FLAT":
        if abs(mr_z) < config.MR_EXIT_THRESHOLD:
            return "FLAT", "MR_EXIT_REVERTED", 0.0, 0
        return state["direction"], "MR_HOLD", min(mr_str, 1.0), state["entry_quality"]

    if mr_signal == "FLAT" or mr_signal == "PULLBACK":
        # PULLBACK is handled in _pullback_decision; ignore here
        return "FLAT", "MR_FLAT", 0.0, 0

    # NEUTRAL weekly bias → disable MR (proven: 30% win rate in directionless markets)
    if bias == "NEUTRAL":
        return "FLAT", "MR_BLOCKED_NEUTRAL_WEEKLY", 0.0, 0

    # MR SHORT in LONG bias → fighting the trend (38% win rate, -1% PnL in base)
    if mr_signal == "SHORT" and bias == "LONG":
        return "FLAT", "MR_SHORT_BLOCKED_LONG_WEEKLY", 0.0, 0

    quality = compute_entry_quality(row)
    reason  = "MR_LONG" if mr_signal == "LONG" else "MR_SHORT"
    return mr_signal, reason, min(mr_str, 1.0), quality


def _sa_decision(row, state, sym, sa_dir):
    """Stat arb strategy."""
    spread_z = float(row.get("spread_zscore", 0.0) or 0.0)
    sa_str   = float(row.get("sa_strength",   0.0) or 0.0)

    # Exit existing SA position
    if state["strategy"] == "STAT_ARB" and state["direction"] != "FLAT":
        if abs(spread_z) < config.SA_EXIT_THRESHOLD:
            return "FLAT", "SA_EXIT_REVERTED", 0.0, 0
        return state["direction"], "SA_HOLD", min(sa_str, 1.0), state["entry_quality"]

    # New entry
    reason = f"SA_ENTRY_{sa_dir}_{sym}"
    # Quality scoring not applied to SA entries (spread-driven, not bias-driven)
    return sa_dir, reason, min(sa_str, 1.0), 3  # default quality 3 for SA


def _evaluate_sa(port_regime, sa_signal, sa_es_dir, sa_nq_dir, state_es, state_nq):
    """SA can activate in TREND or CHOP (not TRANSITION or SHOCK)."""
    if port_regime in ("TRANSITION", "SHOCK"):
        return False
    if sa_signal != "ENTRY":
        return False
    if sa_es_dir == "FLAT" and sa_nq_dir == "FLAT":
        return False
    # Don't interrupt an existing TREND/PULLBACK NQ position
    if state_nq["strategy"] in ("TREND", "PULLBACK") and state_nq["direction"] != "FLAT":
        return False
    return True


# ── Filters ───────────────────────────────────────────────────────────────────

def _apply_sentiment(direction, reason, confidence, flag, multiplier, state):
    if flag == "NORMAL":
        return direction, reason, confidence

    # HIGH_RISK: block new entries; allow existing to hold
    if flag == "HIGH_RISK" and direction != state["direction"]:
        return "FLAT", f"BLOCKED_{flag}", 0.0

    # EVENT_ACTIVE: allow entries, note risk in reason (size handled in portfolio)
    if flag == "EVENT_ACTIVE":
        return direction, reason + "_EVENT", confidence * 0.8

    return direction, reason, confidence


def _check_max_hold(direction, reason, state):
    if state["strategy"] == "NONE" or state["direction"] == "FLAT":
        return direction, reason

    # [V3] TREND max-hold is now managed by backtest.py (winner extension logic).
    # Signal engine never force-exits TREND on a timer — it only exits TREND
    # when bias/regime fundamentals change. Backtest applies the hard cap.
    if state["strategy"] == "TREND":
        return direction, reason

    max_hold = config.MAX_HOLD_DAYS.get(state["strategy"], 3)
    if state["days_held"] >= max_hold and state["direction"] != "FLAT":
        return "FLAT", f"MAX_HOLD_EXIT_{state['strategy']}"

    return direction, reason


# ── State helpers ──────────────────────────────────────────────────────────────

def _update_state(state, direction, reason, quality=0):
    prev_dir = state["direction"]

    if direction == "FLAT":
        state["direction"]     = "FLAT"
        state["strategy"]      = "NONE"
        state["days_held"]     = 0
        state["entry_quality"] = 0
    elif direction == prev_dir:
        state["days_held"] += 1
    else:
        state["direction"]     = direction
        state["days_held"]     = 1
        state["strategy"]      = _infer_strategy(reason)
        state["entry_quality"] = quality


def _infer_strategy(reason: str) -> str:
    r = reason.upper()
    if "PULLBACK" in r and "TREND" in r:
        return "PULLBACK"
    if "TREND" in r:
        return "TREND"
    if "MR_" in r or "MEAN" in r:
        return "MEAN_REVERT"
    if "SA_" in r or "STAT_ARB" in r:
        return "STAT_ARB"
    return "NONE"


def _force_flat(state):
    state["direction"]     = "FLAT"
    state["strategy"]      = "NONE"
    state["days_held"]     = 0
    state["entry_quality"] = 0


def _compute_can_extend(row: dict, state: dict, direction: str) -> bool:
    """
    [V3] Check whether a TREND position qualifies for winner extension.
    True when holding TREND-LONG and all extension conditions remain valid.
    Called before state is updated (state still reflects prior direction).
    """
    # Only relevant if we're currently in a TREND position
    if state["strategy"] != "TREND" or state["direction"] == "FLAT":
        return False
    if direction == "FLAT":
        # Signal says exit — not eligible for extension
        return False
    bias    = row.get("weekly_bias",        "NEUTRAL")
    regime  = row.get("portfolio_regime",   "CHOP")
    h4_sig  = row.get("h4_exec_signal",     "NONE")
    mom_str = float(row.get("momentum_strength", 0.0) or 0.0)
    return (
        bias   == "LONG"
        and regime == "TREND"
        and h4_sig not in ("WAIT", "NONE")
        and mom_str > 0.0
    )


def _make_result(date, sym, state, reason, confidence, quality, high_conv, in_transition, can_extend=False) -> dict:
    return {
        "date":             date,
        "symbol":           sym,
        "final_direction":  state["direction"],
        "strategy_used":    state["strategy"],
        "days_held":        state["days_held"],
        "signal_reason":    reason,
        "confidence_score": round(float(confidence), 4),
        "entry_quality":    quality,
        "high_conviction":  high_conv,
        "in_transition":    in_transition,
        "can_extend":       can_extend,
    }


def _log_signal_distribution(df: pd.DataFrame) -> None:
    for sym in df["symbol"].unique():
        g  = df[df["symbol"] == sym]
        vc = g["final_direction"].value_counts()
        sv = g["strategy_used"].value_counts()
        hc = g["high_conviction"].sum() if "high_conviction" in g.columns else 0
        ce = g["can_extend"].sum()      if "can_extend"      in g.columns else 0
        logger.info(
            "Signals %s — LONG: %d  FLAT: %d  "
            "| TREND: %d  PULLBACK: %d  MR: %d  SA: %d  "
            "high_conv: %d  can_extend: %d",
            sym,
            vc.get("LONG", 0), vc.get("FLAT", 0),
            sv.get("TREND",       0),
            sv.get("PULLBACK",    0),
            sv.get("MEAN_REVERT", 0),
            sv.get("STAT_ARB",    0),
            int(hc), int(ce),
        )
