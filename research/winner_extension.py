"""
winner_extension.py — V3 winner extension, trailing stops, and exit logic. [NEW]

Pure stateless functions. All take primitive scalars (no dataclass imports).
Called from backtest.py on each open position each day.

Exit functions (return True = exit now):
  should_stop_loss(cum_ret)
  should_trail_exit(cum_ret, peak_cum_ret, trail_active) → (exit, new_trail_active)
  should_fast_exit(cum_ret, h4_signal, confidence)

Extension / add functions (return True = do the action):
  winner_conditions_hold(weekly_bias, portfolio_regime, h4_exec_signal, momentum_strength)
  should_add_to_winner(strategy, cum_ret, add_count, weekly_bias, portfolio_regime, h4_exec_signal, momentum_strength)

Design rules:
  - No side effects. All state updates happen in backtest.py.
  - Each function tests exactly one decision.
  - Parameters are explicit primitives, not dicts, to keep tests simple.
"""

from typing import Tuple

import config


# ── Exit: stop loss ───────────────────────────────────────────────────────────

def should_stop_loss(cum_ret: float) -> bool:
    """
    Exit immediately: trade has exceeded the max loss threshold.

    WHY: At -1.5% cumulative return on a position, the trade thesis is
    clearly wrong. Staying in costs more than getting out. This is a
    hard floor — no extension or trail logic overrides a stop loss.
    """
    return cum_ret < config.STOP_LOSS_THRESHOLD


# ── Exit: trailing profit stop ────────────────────────────────────────────────

def should_trail_exit(
    cum_ret: float,
    peak_cum_ret: float,
    trail_active: bool,
) -> Tuple[bool, bool]:
    """
    Trailing profit stop — arms once trade is sufficiently profitable.

    Returns (should_exit, new_trail_active).

    WHY: Without a trailing stop, a +1.5% winner can turn into a -0.5%
    loser by holding too long. The trailing stop arms at +1.0% and fires
    when we give back 0.7% from the peak. This captures at least +0.3%
    on any trade that reaches +1.0%.
    """
    new_trail = trail_active

    # Arm the trailing stop once we reach the profit threshold
    if not trail_active and cum_ret >= config.TRAIL_ARM_THRESHOLD:
        new_trail = True

    # Fire if we've drawn down from peak beyond the stop drawdown
    if new_trail and (peak_cum_ret - cum_ret) >= config.TRAIL_STOP_DRAWDOWN:
        return True, new_trail

    return False, new_trail


# ── Exit: fast exit on conviction fade ────────────────────────────────────────

def should_fast_exit(cum_ret: float, h4_signal: str, confidence: float) -> bool:
    """
    Protect profits when conviction fades, but only if the trade is profitable.

    WHY: When a trade has reached +0.8% AND the 4H signal flips to WAIT
    (momentum gone) or confidence drops below 0.30, the edge of holding
    has declined. Taking profits here beats holding and giving them back.

    Only triggers above FAST_EXIT_PROFIT_FLOOR — below that, the trade
    has not yet earned "profits to protect" and should be allowed to develop.
    """
    if cum_ret < config.FAST_EXIT_PROFIT_FLOOR:
        return False

    if h4_signal == "WAIT":
        return True

    if confidence < config.FAST_EXIT_CONFIDENCE_DROP:
        return True

    return False


# ── Extension condition check ─────────────────────────────────────────────────

def winner_conditions_hold(
    weekly_bias: str,
    portfolio_regime: str,
    h4_exec_signal: str,
    momentum_strength: float,
) -> bool:
    """
    Check whether a TREND position deserves extension past BASE_MAX_HOLD_DAYS.

    Returns True when ALL of:
      - weekly_bias == LONG          (directional regime still valid)
      - portfolio_regime == TREND    (structural trend intact)
      - h4_exec_signal not WAIT/NONE (4H execution conditions OK)
      - momentum_strength > 0        (some multi-TF momentum remains)

    WHY: Extension is only justified when the original trade thesis
    remains fully intact. If any of these fail, the 3-day timer is
    the right exit.
    """
    if weekly_bias != "LONG":
        return False
    if portfolio_regime != "TREND":
        return False
    # WAIT always blocks — momentum is stalling.
    # NONE is allowed in relaxed mode (4H quiet = steady trend, not reversal).
    if h4_exec_signal == "WAIT":
        return False
    if h4_exec_signal == "NONE" and not getattr(config, "EXTEND_ALLOW_H4_NONE", False):
        return False
    if momentum_strength <= 0.0:
        return False
    return True


# ── Add-to-winner check ───────────────────────────────────────────────────────

def should_add_to_winner(
    strategy: str,
    cum_ret: float,
    add_count: int,
    weekly_bias: str,
    portfolio_regime: str,
    h4_exec_signal: str,
    momentum_strength: float,
) -> bool:
    """
    Decide whether to add to an existing profitable TREND position.

    Requirements (all must hold):
      - Strategy is TREND (add-to-winner is TREND-only)
      - cum_ret >= ADD_WINNER_MIN_CUM_RET (+0.5%)
      - add_count < ADD_WINNER_MAX_ADDS (max 1 add per trade)
      - All winner conditions still hold (see winner_conditions_hold)

    WHY: Adding to a winner amplifies the edge on the best trades.
    The +0.5% threshold ensures the trade is genuinely profitable
    before adding. The 1-add limit prevents runaway pyramiding.
    """
    if strategy != "TREND":
        return False
    if cum_ret < config.ADD_WINNER_MIN_CUM_RET:
        return False
    if add_count >= config.ADD_WINNER_MAX_ADDS:
        return False
    return winner_conditions_hold(
        weekly_bias, portfolio_regime, h4_exec_signal, momentum_strength,
    )
