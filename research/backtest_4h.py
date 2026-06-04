"""
backtest_4h.py — 4H-resolution backtest engine.

Design:
  - State machine ticks on every 4H bar close
  - Entry signal at bar close → order placed → fills at NEXT bar open
  - Stop is checked intrabar (high/low of next bars)
  - Trail stop activates after reaching +1x ATR profit
  - Hold timer: MAX_BARS_HELD (default 18 = ~3 calendar days for 6-bar instruments)
  - Aggregates to daily portfolio returns for Sharpe/MaxDD computation

Position sizing:
  - vol_scalar = TARGET_DAILY_VOL / daily_atr_pct  (capped at VOL_SCALE_MAX)
  - trade weight = vol_scalar * WEIGHT_PER_SYMBOL
  - Portfolio return per bar = sum(direction * bar_return * weight)

Multiple symbols:
  - Each symbol runs its own state machine (no cross-instrument blocking)
  - Portfolio return is sum of all symbol returns for that bar's date

Output:
  - "portfolio_returns":  {date_str: return}  (daily)
  - "trade_log":  list of dicts per closed trade
  - "stats":  Sharpe, annual return, MaxDD, etc.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any


# ── Config defaults (can override at call time) ───────────────────────────────

_CFG = dict(
    # Position sizing
    TARGET_DAILY_VOL   = 0.01,      # 1% daily vol target
    VOL_SCALE_MAX      = 0.30,      # cap on vol_scalar × weight
    WEIGHT_PER_SYMBOL  = 0.25,      # base weight (25% NAV per symbol slot)

    # Stop & trail
    STOP_ATR_MULT      = 2.0,       # initial stop: entry ± STOP_ATR_MULT × 4H_ATR
    TRAIL_ATR_MULT     = 2.5,       # trail stop: peak ∓ TRAIL_ATR_MULT × 4H_ATR
    TRAIL_ACTIVATE_ATR = 1.0,       # trail activates once profit > this × 4H_ATR

    # Hold timer
    MAX_BARS_HELD      = 18,        # 18 × 4H = 3 calendar days

    # Regime filter for entries
    # LONG only in BULL_TREND / BULL_RANGE / NEUTRAL
    # SHORT only in BEAR_TREND
    ALLOW_LONG_REGIMES  = {"BULL_TREND", "BULL_RANGE", "NEUTRAL", "BEAR_RANGE"},
    ALLOW_SHORT_REGIMES = {"BEAR_TREND"},

    # Require min ADX for entry (0 = disabled)
    MIN_ADX_LONG        = 15.0,
    MIN_ADX_SHORT       = 15.0,

    # Daily trend score gate (requires daily_trend_score column from features_4h)
    MIN_TREND_SCORE_LONG  = -0.20,   # allow longs unless strongly bearish
    MIN_TREND_SCORE_SHORT = -1.0,    # no gate for shorts (regime handles it)

    # Starting capital
    STARTING_NAV       = 25_000.0,

    # Slippage (as fraction of price, one-way)
    SLIPPAGE_PCT       = 0.0002,    # 0.02% per trade

    # Commission per unit (percentage of notional)
    COMMISSION_PCT     = 0.0001,    # 0.01% per trade
)


# ── Position state ────────────────────────────────────────────────────────────

class _Pos:
    """Track a single open position."""
    __slots__ = (
        "symbol", "direction", "entry_idx", "entry_price",
        "atr", "stop_price", "trail_active", "peak_price",
        "weight", "bars_held", "entry_date",
    )

    def __init__(self, symbol: str, direction: int, entry_idx: int,
                 entry_price: float, atr: float, weight: float, entry_date: str,
                 stop_mult: float):
        self.symbol       = symbol
        self.direction    = direction       # +1 long / -1 short
        self.entry_idx    = entry_idx
        self.entry_price  = entry_price
        self.atr          = atr
        self.stop_price   = entry_price - direction * stop_mult * atr
        self.trail_active = False
        self.peak_price   = entry_price
        self.weight       = weight
        self.bars_held    = 0
        self.entry_date   = entry_date

    def update_trail(self, bar_high: float, bar_low: float, trail_atr: float, activate_atr: float):
        """Update peak and potentially activate trailing stop."""
        if self.direction == 1:
            self.peak_price = max(self.peak_price, bar_high)
            profit_atr = (self.peak_price - self.entry_price) / max(self.atr, 1e-8)
        else:
            self.peak_price = min(self.peak_price, bar_low)
            profit_atr = (self.entry_price - self.peak_price) / max(self.atr, 1e-8)

        if profit_atr >= activate_atr:
            self.trail_active = True

    def trail_stop_price(self, trail_atr: float) -> float:
        return self.peak_price - self.direction * trail_atr * self.atr

    def check_exit(
        self,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        trail_atr: float,
        activate_atr: float,
        max_bars: int,
    ) -> tuple[str | None, float | None]:
        """
        Return (reason, exit_price) or (None, None) if still holding.
        Called at NEXT bar after entry (to simulate fill-at-open semantics
        for stop and immediate check for time exit).
        """
        self.bars_held += 1

        # Update trailing
        self.update_trail(bar_high, bar_low, trail_atr, activate_atr)

        # ── Stop loss (intrabar) ──────────────────────────────────────────
        if self.direction == 1:
            if bar_low <= self.stop_price:
                # Fill at stop or at open if gapped below
                fill = min(bar_open, self.stop_price)
                return "STOP", fill
        else:
            if bar_high >= self.stop_price:
                fill = max(bar_open, self.stop_price)
                return "STOP", fill

        # ── Trailing stop ─────────────────────────────────────────────────
        if self.trail_active:
            ts = self.trail_stop_price(trail_atr)
            if self.direction == 1 and bar_low <= ts:
                fill = min(bar_open, ts)
                return "TRAIL", fill
            if self.direction == -1 and bar_high >= ts:
                fill = max(bar_open, ts)
                return "TRAIL", fill

        # ── Hold timer ────────────────────────────────────────────────────
        if self.bars_held >= max_bars:
            return "TIME", bar_close

        return None, None


# ── Engine ────────────────────────────────────────────────────────────────────

def run_backtest_4h(df: pd.DataFrame, **kwargs) -> dict[str, Any]:
    """
    Run 4H backtest.

    Parameters
    ----------
    df : DataFrame output of compute_4h_features(), must include:
        datetime, date, symbol, open, high, low, close,
        hb_signal_long, hb_signal_short, hb_atr, hb_atr_pct,
        daily_regime (optional), daily_adx (optional), daily_trend_score (optional)

    kwargs : override _CFG keys

    Returns
    -------
    dict with keys: portfolio_returns, trade_log, stats
    """
    cfg = {**_CFG, **kwargs}

    # Normalise and sort
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["date"]     = pd.to_datetime(df["date"])
    df = df.sort_values(["datetime", "symbol"]).reset_index(drop=True)

    # Fill optional columns — handles both missing columns AND NaN values
    if "daily_regime"       not in df.columns: df["daily_regime"]       = "NEUTRAL"
    if "daily_adx"          not in df.columns: df["daily_adx"]          = 25.0
    if "daily_trend_score"  not in df.columns: df["daily_trend_score"]  = 0.0
    if "daily_atr_pct"      not in df.columns: df["daily_atr_pct"]      = df["hb_atr_pct"]
    # Fallback: NaN atr (no daily context) → use 4H ATR as proxy
    df["daily_atr_pct"] = df["daily_atr_pct"].fillna(df["hb_atr_pct"]).fillna(0.01)
    df["daily_adx"]     = df["daily_adx"].fillna(25.0)
    df["daily_trend_score"] = df["daily_trend_score"].fillna(0.0)

    symbols = df["symbol"].unique().tolist()
    positions: dict[str, _Pos | None] = {s: None for s in symbols}

    # Bar-level return accumulator: date_str → list of returns
    bar_returns: dict[str, list[float]] = {}
    trade_log: list[dict] = []

    # Precompute vol_scalar per bar (used at entry)
    df["_vol_scalar"] = (
        cfg["TARGET_DAILY_VOL"] / df["daily_atr_pct"].clip(lower=0.002)
    ).clip(upper=cfg["VOL_SCALE_MAX"] / cfg["WEIGHT_PER_SYMBOL"])

    bars = df.to_dict("records")
    n = len(bars)

    # Build index: for each (symbol, bar_idx) → next bar's row
    # We need next-bar-open for entry fills and exit checks
    sym_idx: dict[str, list[int]] = {s: [] for s in symbols}
    for i, row in enumerate(bars):
        sym_idx[row["symbol"]].append(i)

    # Map global bar index → position in sym_idx for each symbol
    sym_pos_of: dict[int, int] = {}
    for s, idxs in sym_idx.items():
        for pos_in_list, global_i in enumerate(idxs):
            sym_pos_of[global_i] = pos_in_list

    def _next_bar(sym: str, cur_global_i: int) -> dict | None:
        """Return the next bar dict for the same symbol, or None if last."""
        idxs = sym_idx[sym]
        p = sym_pos_of.get(cur_global_i)
        if p is None or p + 1 >= len(idxs):
            return None
        return bars[idxs[p + 1]]

    def _record_bar_return(date_str: str, ret: float):
        bar_returns.setdefault(date_str, [])
        bar_returns[date_str].append(ret)

    # ── State machine ─────────────────────────────────────────────────────
    for i, bar in enumerate(bars):
        sym  = bar["symbol"]
        pos  = positions[sym]
        date = bar["date"].strftime("%Y-%m-%d") if hasattr(bar["date"], "strftime") else str(bar["date"])[:10]

        # ── Manage open position ──────────────────────────────────────────
        if pos is not None:
            reason, exit_px = pos.check_exit(
                bar_open  = bar["open"],
                bar_high  = bar["high"],
                bar_low   = bar["low"],
                bar_close = bar["close"],
                trail_atr  = cfg["TRAIL_ATR_MULT"],
                activate_atr = cfg["TRAIL_ACTIVATE_ATR"],
                max_bars   = cfg["MAX_BARS_HELD"],
            )
            if reason is not None:
                # Close position
                raw_ret = pos.direction * (exit_px - pos.entry_price) / pos.entry_price
                cost    = cfg["SLIPPAGE_PCT"] + cfg["COMMISSION_PCT"]
                net_ret = raw_ret - cost
                wt_ret  = net_ret * pos.weight

                _record_bar_return(date, wt_ret)
                trade_log.append({
                    "symbol":      sym,
                    "direction":   "LONG" if pos.direction == 1 else "SHORT",
                    "entry_date":  pos.entry_date,
                    "exit_date":   date,
                    "entry_price": pos.entry_price,
                    "exit_price":  exit_px,
                    "bars_held":   pos.bars_held,
                    "exit_reason": reason,
                    "raw_return":  raw_ret,
                    "pnl":         net_ret,
                    "strategy":    f"4H_{'LONG' if pos.direction==1 else 'SHORT'}",
                })
                positions[sym] = None
                pos = None

        # ── Check for new entry signal ────────────────────────────────────
        if pos is None:
            sig_long  = bar.get("hb_signal_long",  0)
            sig_short = bar.get("hb_signal_short", 0)

            # Prefer long signal over short if both fire
            if sig_long and not sig_short:
                candidate_dir = 1
            elif sig_short and not sig_long:
                candidate_dir = -1
            else:
                candidate_dir = 0

            if candidate_dir != 0:
                # Safely handle NaN regime (when no daily context available)
                _regime_raw = bar.get("daily_regime", None)
                regime = "NEUTRAL" if (_regime_raw is None or (isinstance(_regime_raw, float) and np.isnan(_regime_raw))) else str(_regime_raw)
                _adx_raw = bar.get("daily_adx", None)
                adx = 25.0 if (_adx_raw is None or (isinstance(_adx_raw, float) and np.isnan(_adx_raw))) else float(_adx_raw)
                _ts_raw = bar.get("daily_trend_score", None)
                ts  = 0.0  if (_ts_raw  is None or (isinstance(_ts_raw,  float) and np.isnan(_ts_raw)))  else float(_ts_raw)

                # Regime filter
                if candidate_dir == 1:
                    allowed = (regime in cfg["ALLOW_LONG_REGIMES"] and
                               adx >= cfg["MIN_ADX_LONG"] and
                               ts  >= cfg["MIN_TREND_SCORE_LONG"])
                else:
                    allowed = (regime in cfg["ALLOW_SHORT_REGIMES"] and
                               adx >= cfg["MIN_ADX_SHORT"] and
                               ts  <= cfg["MIN_TREND_SCORE_SHORT"])

                if allowed:
                    nxt = _next_bar(sym, i)
                    if nxt is not None:
                        entry_px  = nxt["open"] * (1 + candidate_dir * cfg["SLIPPAGE_PCT"])
                        atr       = bar["hb_atr"]
                        vol_sc    = bar["_vol_scalar"]
                        weight    = min(vol_sc * cfg["WEIGHT_PER_SYMBOL"], cfg["VOL_SCALE_MAX"])
                        positions[sym] = _Pos(
                            symbol       = sym,
                            direction    = candidate_dir,
                            entry_idx    = i,
                            entry_price  = entry_px,
                            atr          = atr,
                            weight       = weight,
                            entry_date   = date,
                            stop_mult    = cfg["STOP_ATR_MULT"],
                        )

    # ── Aggregate to daily returns ─────────────────────────────────────────
    daily_rets: dict[str, float] = {}
    for d, rets in bar_returns.items():
        daily_rets[d] = float(np.sum(rets))

    # All unique trading dates in the data (for full-period Sharpe/CAGR)
    all_dates_unique = sorted({
        b["date"].strftime("%Y-%m-%d") if hasattr(b["date"], "strftime") else str(b["date"])[:10]
        for b in bars
    })

    # ── Compute stats ─────────────────────────────────────────────────────
    stats = _compute_stats(daily_rets, trade_log, cfg["STARTING_NAV"],
                           all_dates=all_dates_unique)

    return {
        "portfolio_returns": daily_rets,
        "trade_log":         trade_log,
        "stats":             stats,
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

def _compute_stats(daily_rets: dict, trade_log: list, starting_nav: float,
                   all_dates: list | None = None) -> dict:
    """
    Compute portfolio stats.

    all_dates: sorted list of ALL trading bar dates (used to fill zeros for
    inactive days so Sharpe and CAGR are computed over the full period).
    If None, we use only active-trade days (less accurate but still usable).
    """
    pr_active = pd.Series(daily_rets)
    pr_active.index = pd.to_datetime(pr_active.index)
    pr_active = pr_active.sort_index()

    if len(pr_active) == 0:
        return {"sharpe_ratio": 0, "annualized_return_pct": 0,
                "max_drawdown_pct": 0, "total_return_pct": 0,
                "total_trades": 0, "overall_win_rate": 0}

    # Build full daily series: zero on inactive days (portfolio holds cash)
    if all_dates is not None and len(all_dates) > 1:
        full_idx = pd.to_datetime(sorted(set(all_dates)))
        pr = pr_active.reindex(full_idx, fill_value=0.0)
    else:
        pr = pr_active

    # Compound total return (active days only — inactive days contribute 0)
    total_ret = (1 + pr_active).prod() - 1

    # CAGR: use actual calendar span
    start_dt   = pr.index[0]
    end_dt     = pr.index[-1]
    span_years = max((end_dt - start_dt).days / 365.25, 1.0 / 252)
    cagr = (1 + total_ret) ** (1.0 / span_years) - 1

    # Sharpe: annualised over active-trade days (filter zero days to avoid dilution)
    # We report "active-days Sharpe" — measures signal quality when deployed
    pr_nonzero = pr[pr != 0.0]
    if len(pr_nonzero) >= 5:
        mean_r = pr_nonzero.mean()
        std_r  = pr_nonzero.std()
        # Annualise assuming the strategy runs ~252 days/year when active
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 1e-8 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown: equity curve with zeros (portfolio stays flat when not trading)
    cum  = (1 + pr).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    max_dd = dd.min()

    # Trade stats
    n_trades = len(trade_log)
    if n_trades > 0:
        tl = pd.DataFrame(trade_log)
        win_rate = (tl["pnl"] > 0).mean()
        avg_pnl  = tl["pnl"].mean()
    else:
        win_rate = 0.0
        avg_pnl  = 0.0

    return {
        "sharpe_ratio":         round(sharpe, 4),
        "annualized_return_pct":round(cagr * 100, 3),
        "max_drawdown_pct":     round(max_dd * 100, 3),
        "total_return_pct":     round(total_ret * 100, 3),
        "total_trades":         n_trades,
        "overall_win_rate":     round(win_rate, 4),
        "avg_trade_pnl_pct":    round(avg_pnl * 100, 4),
    }
