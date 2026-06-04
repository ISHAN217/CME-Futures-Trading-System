"""
run_4h_execution.py — Daily pipeline signals, 4H-resolution execution.

Core idea (as designed):
  - Signal generated at daily close (full ES/NQ pipeline, proven Sharpe 1.4)
  - Instead of entering at next-day daily open, enter at first liquid 4H bar (12:00 UTC)
  - Position checked at EVERY 4H bar: stop-loss fires intraday, not just at EOD
  - Hold timer in 4H bars (default 48 bars = 8 calendar days)

Why this beats pure 4H signals:
  - Keeps the proven multi-signal daily engine (momentum + MR + stat-arb + regime)
  - Gets 4H execution for better stop response (exits same day vs next day)
  - Same signal frequency; just tighter intraday risk control

Data requirements:
  - Daily pipeline:  2019-01-02 → present  (yfinance, already working)
  - 4H bars for ES/NQ: /tmp/es_nq_4h.parquet  (Databento 2018-2026, just built)
"""

from __future__ import annotations
import contextlib
import logging
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config
from run_frequency import (
    build_base_df, build_fixed_stack,
    inject_h4_brk_chop, inject_h4_mom_chop_nq,
    override_config,
    _BASE_HOLD_DAYS, _BASE_MAX_WT_PER_SYM,
)
from backtest import run_backtest   # daily baseline for comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# ES/NQ 4H liquid entry bar: 12:00 UTC (contains RTH open in both EDT and EST)
# Fallback: 16:00 UTC (also very liquid, RTH afternoon)
ENTRY_HOURS   = [12, 16, 8]          # preference order for entry bar
STOP_ATR_MULT = 1.5                  # stop: entry × (1 - STOP_ATR_MULT × daily_atr_pct)
TRAIL_ATR_MULT = 2.5                 # trail from peak, activates at TRAIL_ACTIVATE_ATR
TRAIL_ACTIVATE = 1.0                 # trail activates after 1× ATR profit
MAX_HOLD_BARS  = 48                  # 48 × 4H = 8 calendar days
SLIPPAGE_PCT   = 0.0002              # 0.02% per side
COMMISSION_PCT = 0.0001

# ── Position dataclass ────────────────────────────────────────────────────────

class _Pos:
    __slots__ = ('symbol','direction','entry_price','stop_price','weight',
                 'atr','bars_held','peak','trail_active','entry_date','extend_count')

    def __init__(self, symbol, direction, entry_price, stop_price,
                 weight, atr, entry_date):
        self.symbol       = symbol
        self.direction    = direction    # +1 long / -1 short
        self.entry_price  = entry_price
        self.stop_price   = stop_price
        self.weight       = weight
        self.atr          = atr
        self.bars_held    = 0
        self.peak         = entry_price
        self.trail_active = False
        self.entry_date   = entry_date
        self.extend_count = 0

    def check_bar(self, bar_open, bar_high, bar_low, bar_close,
                  max_bars, trail_mult, trail_activate):
        """Return (exit_reason, exit_price) or (None, None)."""
        self.bars_held += 1

        # Update peak and trail activation
        if self.direction == 1:
            self.peak = max(self.peak, bar_high)
            profit_atr = (self.peak - self.entry_price) / max(self.atr, 1e-8)
        else:
            self.peak = min(self.peak, bar_low)
            profit_atr = (self.entry_price - self.peak) / max(self.atr, 1e-8)
        if profit_atr >= trail_activate:
            self.trail_active = True

        # Stop loss (intrabar)
        if self.direction == 1 and bar_low <= self.stop_price:
            return "STOP", min(bar_open, self.stop_price)
        if self.direction == -1 and bar_high >= self.stop_price:
            return "STOP", max(bar_open, self.stop_price)

        # Trailing stop
        if self.trail_active:
            ts = (self.peak - self.direction * trail_mult * self.atr)
            if self.direction == 1 and bar_low <= ts:
                return "TRAIL", min(bar_open, ts)
            if self.direction == -1 and bar_high >= ts:
                return "TRAIL", max(bar_open, ts)

        # Hold timer
        if self.bars_held >= max_bars:
            return "TIME", bar_close

        return None, None


# ── Core execution engine ─────────────────────────────────────────────────────

def run_4h_execution(
    df_signals: pd.DataFrame,
    df_4h:      pd.DataFrame,
    max_hold_bars: int   = MAX_HOLD_BARS,
    stop_atr_mult: float = STOP_ATR_MULT,
    trail_atr_mult: float= TRAIL_ATR_MULT,
    trail_activate: float= TRAIL_ACTIVATE,
    weight_scale: float  = 1.0,
    vol_scale_max: float = 0.28,
    max_weight_per_sym: float = 0.50,
    allow_extend: bool   = False,    # reset hold timer on same-direction re-signal
    max_extends: int     = 3,        # cap on how many times timer can reset
) -> dict:
    """
    Execute daily pipeline signals at 4H bar resolution.

    Parameters
    ----------
    df_signals : daily pipeline DataFrame (output of build_fixed_stack + injects)
        Must have: date, symbol, final_direction, vol_scalar, atr_pct, mtf_entry_size
    df_4h : 4H OHLCV bars (from /tmp/es_nq_4h.parquet)
        Must have: datetime, date, root, open, high, low, close, volume
    """
    # ── Prep signal lookup ────────────────────────────────────────────────
    sig = df_signals.copy()
    # Use date STRINGS as keys to avoid UTC-aware vs naive mismatches
    sig['_dstr'] = pd.to_datetime(sig['date']).dt.strftime('%Y-%m-%d')

    def _safe_float(val, default):
        try:
            v = float(val)
            return default if (v != v) else v   # NaN check: NaN != NaN
        except (TypeError, ValueError):
            return default

    # Build lookup: (date_str, symbol) → signal info for that day
    sig_info = {}
    for _, row in sig.iterrows():
        if row['final_direction'] == 'FLAT':
            continue
        sig_info[(row['_dstr'], row['symbol'])] = {
            'direction':  1 if row['final_direction'] == 'LONG' else -1,
            'vol_scalar': _safe_float(row.get('vol_scalar'), 0.25),
            'atr_pct':    _safe_float(row.get('atr_pct'),    0.010),
            'mtf_size':   _safe_float(row.get('mtf_entry_size'), 0.20),
        }

    # Build per-symbol lookup for all signal directions (LONG / SHORT only)
    # Used to detect REVERSAL (not just FLAT) → exit current position
    all_dir = {}   # (date_str, symbol) → direction (+1/-1) — only non-FLAT
    for _, row in sig.iterrows():
        if row['final_direction'] == 'FLAT':
            continue
        all_dir[(row['_dstr'], row['symbol'])] = (
            1 if row['final_direction'] == 'LONG' else -1
        )

    # ── Prep 4H bars ─────────────────────────────────────────────────────
    h4 = df_4h.copy()
    h4['datetime'] = pd.to_datetime(h4['datetime'], utc=True)
    h4['date']     = h4['datetime'].dt.normalize()
    h4['symbol']   = h4['root']
    h4['hour']     = h4['datetime'].dt.hour
    h4 = h4.sort_values(['symbol', 'datetime']).reset_index(drop=True)

    # Map: (date, symbol, hour) → bar index for fast lookup
    bar_lookup = {}
    for i, row in h4.iterrows():
        bar_lookup[(row['date'], row['symbol'], row['hour'])] = i

    # ── State machine ─────────────────────────────────────────────────────
    positions: dict[str, _Pos | None] = {}
    bar_returns: dict[str, list] = {}
    trade_log: list[dict]        = []

    def _get_weight(info, ws):
        raw = info['vol_scalar'] * info['mtf_size'] * ws
        return min(raw, max_weight_per_sym, vol_scale_max)

    def _record(date_str, ret):
        bar_returns.setdefault(date_str, []).append(ret)

    bars = h4.to_dict('records')
    n    = len(bars)

    # Build per-symbol ordered bar list for "next bar" lookup
    sym_bars: dict[str, list[int]] = {}
    for i, b in enumerate(bars):
        sym_bars.setdefault(b['symbol'], []).append(i)
    sym_pos_of = {}
    for s, idxs in sym_bars.items():
        for p, gi in enumerate(idxs):
            sym_pos_of[gi] = p

    def _next_bar_idx(sym, cur_gi):
        idxs = sym_bars.get(sym, [])
        p = sym_pos_of.get(cur_gi)
        if p is None or p + 1 >= len(idxs): return None
        return idxs[p + 1]

    # Pending entries: signal fires on day D close → enter at first liquid bar of D+1
    pending: dict[str, dict] = {}   # symbol → entry info to execute at next liquid bar

    for gi, bar in enumerate(bars):
        sym   = bar['symbol']
        bdate = bar['date']
        bhour = bar['hour']
        bdate_str = bdate.strftime('%Y-%m-%d')

        pos = positions.get(sym)

        # ── Execute pending entry at first liquid bar of the entry day ────
        if sym in pending and pos is None:
            pend = pending[sym]
            if bdate_str == pend['entry_day'] and bhour in ENTRY_HOURS:
                # Enter: open of this bar
                info = pend['info']
                entry_px = bar['open'] * (1 + pend['direction'] * SLIPPAGE_PCT)
                atr_price = entry_px * info['atr_pct']
                weight    = _get_weight(info, weight_scale)
                stop_px   = entry_px - pend['direction'] * stop_atr_mult * atr_price
                pos = _Pos(
                    symbol      = sym,
                    direction   = pend['direction'],
                    entry_price = entry_px,
                    stop_price  = stop_px,
                    weight      = weight,
                    atr         = atr_price,
                    entry_date  = bdate_str,
                )
                positions[sym] = pos
                del pending[sym]

        # ── Manage existing position ──────────────────────────────────────
        if pos is not None:
            reason, exit_px = pos.check_bar(
                bar['open'], bar['high'], bar['low'], bar['close'],
                max_hold_bars, trail_atr_mult, trail_activate,
            )
            # Exit on signal REVERSAL only (LONG → SHORT or SHORT → LONG)
            # NOT on FLAT — that just means "no new entry", not "close now"
            if reason is None and bhour in ENTRY_HOURS:
                new_dir = all_dir.get((bdate_str, sym))
                if new_dir is not None and new_dir != pos.direction:
                    reason, exit_px = 'SIGNAL_REV', bar['open']

            if reason is not None:
                raw_ret  = pos.direction * (exit_px - pos.entry_price) / pos.entry_price
                net_ret  = raw_ret - SLIPPAGE_PCT - COMMISSION_PCT
                wt_ret   = net_ret * pos.weight
                _record(bdate_str, wt_ret)
                trade_log.append({
                    'symbol':      sym,
                    'direction':   'LONG' if pos.direction == 1 else 'SHORT',
                    'entry_date':  pos.entry_date,
                    'exit_date':   bdate_str,
                    'entry_price': pos.entry_price,
                    'exit_price':  exit_px,
                    'bars_held':   pos.bars_held,
                    'exit_reason': reason,
                    'pnl':         net_ret,
                    'weight':      pos.weight,
                    'strategy':    '4H_EXEC',
                })
                positions[sym] = None
                pos = None

        # ── Queue new entry signals ───────────────────────────────────────
        # Signal fires at EOD (daily close). We queue for D+1 first liquid bar.
        # In our 4H bar loop, the 20:00 UTC bar is the "daily close" bar.
        if bhour == 20:
            key = (bdate_str, sym)
            if key in sig_info:
                info = sig_info[key]
                new_dir = info['direction']

                # ── EXTENSION: same-direction signal while position is open ──
                if (allow_extend and pos is not None
                        and pos.direction == new_dir
                        and pos.extend_count < max_extends):
                    pos.bars_held   = 0          # reset hold timer
                    pos.extend_count += 1
                    # Re-anchor stop to current bar close
                    cur_close = bar['close']
                    atr_price = cur_close * info['atr_pct']
                    pos.stop_price = cur_close - new_dir * stop_atr_mult * atr_price
                    pos.atr        = max(pos.atr, atr_price)  # keep larger ATR

                # ── NEW ENTRY: queue for D+1 if no position and no pending ──
                elif pos is None and sym not in pending:
                    next_day_ts = pd.to_datetime(bdate_str) + pd.Timedelta(days=1)
                    while next_day_ts.dayofweek >= 5:
                        next_day_ts += pd.Timedelta(days=1)
                    next_day_str = next_day_ts.strftime('%Y-%m-%d')
                    pending[sym] = {
                        'entry_day': next_day_str,
                        'direction': new_dir,
                        'info':      info,
                    }

    # ── Portfolio daily returns ───────────────────────────────────────────
    daily_rets = {d: float(np.sum(r)) for d, r in bar_returns.items()}
    all_dates  = sorted({b['date'].strftime('%Y-%m-%d') for b in bars})

    stats = _compute_stats(daily_rets, trade_log, all_dates)
    return {'portfolio_returns': daily_rets, 'trade_log': trade_log, 'stats': stats}


def _compute_stats(daily_rets, trade_log, all_dates):
    pr_active = pd.Series(daily_rets)
    pr_active.index = pd.to_datetime(pr_active.index)
    pr_active = pr_active.sort_index()

    if len(pr_active) == 0:
        return {k: 0 for k in ['sharpe_ratio','annualized_return_pct',
                                'max_drawdown_pct','total_return_pct',
                                'total_trades','overall_win_rate']}

    full_idx = pd.to_datetime(sorted(set(all_dates)))
    pr = pr_active.reindex(full_idx, fill_value=0.0)

    total_ret  = (1 + pr_active).prod() - 1
    span_years = max((pr.index[-1] - pr.index[0]).days / 365.25, 1/252)
    cagr = (1 + total_ret) ** (1.0 / span_years) - 1

    pr_nz  = pr[pr != 0.0]
    sharpe = (pr_nz.mean() / pr_nz.std() * np.sqrt(252)) if len(pr_nz) > 5 and pr_nz.std() > 1e-8 else 0.0

    cum    = (1 + pr).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    n_tr  = len(trade_log)
    wr    = (pd.DataFrame(trade_log)['pnl'] > 0).mean() if n_tr > 0 else 0.0
    return {
        'sharpe_ratio':          round(sharpe, 4),
        'annualized_return_pct': round(cagr * 100, 3),
        'max_drawdown_pct':      round(max_dd * 100, 3),
        'total_return_pct':      round(total_ret * 100, 3),
        'total_trades':          n_tr,
        'overall_win_rate':      round(float(wr), 4),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_result(label, result, n_years):
    s   = result['stats']
    tl  = pd.DataFrame(result['trade_log']) if result['trade_log'] else pd.DataFrame()
    pr  = pd.Series(result['portfolio_returns'])
    pr.index = pd.to_datetime(pr.index)
    ann = pr.groupby(pr.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    tr_pw = s['total_trades'] / (n_years * 52)
    final = 25000 * (1 + s['total_return_pct'] / 100)

    logger.info(
        "%-32s  Sh=%.3f  Ann=%+.2f%%  DD=%.2f%%  Trades=%d(%.2f/wk)  WR=%.1f%%  $25k->$%s",
        label,
        s['sharpe_ratio'], s['annualized_return_pct'], s['max_drawdown_pct'],
        s['total_trades'], tr_pw, s['overall_win_rate'] * 100,
        f"{final:,.0f}",
    )

    if not tl.empty:
        for ex in ['STOP','TRAIL','TIME','SIGNAL_FLAT']:
            sub = tl[tl['exit_reason'] == ex]
            if len(sub):
                logger.info("  %-14s n=%3d  WR=%5.1f%%  avg=%+.3f%%",
                            ex, len(sub), (sub['pnl'] > 0).mean() * 100,
                            sub['pnl'].mean() * 100)
        logger.info("  Avg hold: %.1f bars (%.1f days)", tl['bars_held'].mean(),
                    tl['bars_held'].mean() / 6)

    return s, ann


# ── Multi-slot engine: 3 concurrent independent strategies ───────────────────
#   Slot 0  ES_TREND  — all ES non-FLAT signals
#   Slot 1  NQ_TREND  — NQ non-FLAT signals where strategy != STAT_ARB
#   Slot 2  NQ_SA     — NQ stat-arb signals (sa_signal='ENTRY'), independent slot

SLOT_DEFS = [
    ('ES', 'TREND'),   # ES main slot
    ('NQ', 'TREND'),   # NQ trend/reset slot
    ('NQ', 'SA'),      # NQ stat-arb slot  ← the missing 182 trades
]


def run_4h_multislot(
    df_signals:     pd.DataFrame,
    df_4h:          pd.DataFrame,
    max_hold_bars:  int   = 24,
    stop_atr_mult:  float = 1.0,
    trail_atr_mult: float = 2.0,
    trail_activate: float = 1.0,
    weight_scale:   float = 1.0,
    vol_scale_max:  float = 0.84,
    max_weight_per_sym: float = 1.5,
    allow_extend:   bool  = True,
    max_extends:    int   = 3,
    # Quality filters — None = no filter, 'STRONG' = STRONG or EXCEPTIONAL only
    trend_edge_min: str | None = None,   # trend_continuation_edge_bucket floor
    sa_quality_min: str | None = None,   # statarb_quality_bucket_hump floor
) -> dict:
    """
    3-slot concurrent 4H execution.
    Each slot manages its own position, stop, and hold timer independently.
    NQ can hold a TREND position AND an SA position simultaneously.
    """

    def _safe_float(val, default=0.0):
        try:
            v = float(val)
            return default if (v != v) else v
        except (TypeError, ValueError):
            return default

    _QUALITY_RANK = {'MODERATE': 0, 'GOOD': 1, 'STRONG': 2, 'EXCEPTIONAL': 3}
    _trend_min    = _QUALITY_RANK.get(trend_edge_min, -1) if trend_edge_min else -1
    _sa_min       = _QUALITY_RANK.get(sa_quality_min, -1) if sa_quality_min else -1

    # ── Build signal lookups ──────────────────────────────────────────────
    sig = df_signals.copy()
    sig['_dstr'] = pd.to_datetime(sig['date']).dt.strftime('%Y-%m-%d')

    # Slot TREND: (date_str, symbol) → info   [non-FLAT, non-pure-SA rows]
    trend_info = {}
    all_trend_dir = {}   # for SIGNAL_REV detection

    # Slot SA: (date_str, 'NQ') → info   [sa_signal=='ENTRY' NQ rows]
    sa_info = {}

    for _, row in sig.iterrows():
        sym   = str(row['symbol'])
        dstr  = row['_dstr']
        fdir  = str(row.get('final_direction', 'FLAT'))
        strat = str(row.get('strategy_used', 'NONE'))

        # ── TREND slot signal ─────────────────────────────────────────────
        if fdir != 'FLAT':
            # For NQ: exclude rows that are pure STAT_ARB (no trend component)
            is_pure_sa = (sym == 'NQ') and (strat == 'STAT_ARB')
            if not is_pure_sa:
                # Quality gate: trend_continuation_edge_bucket
                edge_bkt = str(row.get('trend_continuation_edge_bucket', 'MODERATE'))
                edge_rank = _QUALITY_RANK.get(edge_bkt, 0)
                if edge_rank >= _trend_min:
                    direction = 1 if fdir == 'LONG' else -1
                    info = {
                        'direction':  direction,
                        'vol_scalar': _safe_float(row.get('vol_scalar'), 0.25),
                        'atr_pct':    _safe_float(row.get('atr_pct'),    0.010),
                        'mtf_size':   _safe_float(row.get('mtf_entry_size'), 0.20),
                    }
                    trend_info[(dstr, sym)]    = info
                    all_trend_dir[(dstr, sym)] = direction

        # ── SA slot signal (NQ only) ─────────────────────────────────────
        if sym == 'NQ':
            sa_sig = str(row.get('sa_signal', 'FLAT'))
            sa_nq  = str(row.get('sa_nq_direction', 'FLAT'))
            if sa_sig == 'ENTRY' and sa_nq == 'LONG':
                # Quality gate: statarb_quality_bucket_hump
                sa_bkt  = str(row.get('statarb_quality_bucket_hump', 'GOOD'))
                sa_rank = _QUALITY_RANK.get(sa_bkt, 1)
                if sa_rank >= _sa_min:
                    sa_info[(dstr, 'NQ')] = {
                        'direction':  1,
                        'vol_scalar': _safe_float(row.get('vol_scalar'), 0.25),
                        'atr_pct':    _safe_float(row.get('atr_pct'),    0.010),
                        'mtf_size':   _safe_float(row.get('mtf_entry_size'), 0.15),
                    }

    logger.info("Multi-slot signals — TREND_ES=%d  TREND_NQ=%d  SA_NQ=%d",
                sum(1 for (_, s) in trend_info if s == 'ES'),
                sum(1 for (_, s) in trend_info if s == 'NQ'),
                len(sa_info))

    # ── Prep 4H bars ─────────────────────────────────────────────────────
    h4 = df_4h.copy()
    h4['datetime'] = pd.to_datetime(h4['datetime'], utc=True)
    h4['date']     = h4['datetime'].dt.normalize()
    h4['symbol']   = h4['root']
    h4['hour']     = h4['datetime'].dt.hour
    h4 = h4.sort_values(['symbol', 'datetime']).reset_index(drop=True)
    bars = h4.to_dict('records')

    # ── State: one _Pos per slot, one pending entry per slot ─────────────
    positions: dict[tuple, _Pos | None] = {k: None for k in SLOT_DEFS}
    pending:   dict[tuple, dict]        = {}
    bar_returns: dict[str, list]        = {}
    trade_log:   list[dict]             = []

    def _weight(info):
        raw = info['vol_scalar'] * info['mtf_size'] * weight_scale
        return min(raw, max_weight_per_sym, vol_scale_max)

    def _record(dstr, ret):
        bar_returns.setdefault(dstr, []).append(ret)

    for bar in bars:
        sym       = bar['symbol']
        bdate_str = bar['date'].strftime('%Y-%m-%d')
        bhour     = bar['hour']

        # Which slots live on this symbol?
        my_slots = [k for k in SLOT_DEFS if k[0] == sym]

        for slot_key in my_slots:
            _, slot_type = slot_key
            pos = positions[slot_key]

            # ── Execute pending entry at first liquid bar of entry_day ────
            if slot_key in pending and pos is None:
                pend = pending[slot_key]
                if bdate_str == pend['entry_day'] and bhour in ENTRY_HOURS:
                    info     = pend['info']
                    entry_px = bar['open'] * (1 + pend['direction'] * SLIPPAGE_PCT)
                    atr_px   = entry_px * info['atr_pct']
                    stop_px  = entry_px - pend['direction'] * stop_atr_mult * atr_px
                    pos = _Pos(sym, pend['direction'], entry_px, stop_px,
                               _weight(info), atr_px, bdate_str)
                    positions[slot_key] = pos
                    del pending[slot_key]

            # ── Manage existing position ──────────────────────────────────
            if pos is not None:
                reason, exit_px = pos.check_bar(
                    bar['open'], bar['high'], bar['low'], bar['close'],
                    max_hold_bars, trail_atr_mult, trail_activate,
                )
                # SIGNAL_REV only for TREND slot (SA is always LONG, no reversal)
                if reason is None and bhour in ENTRY_HOURS and slot_type == 'TREND':
                    new_dir = all_trend_dir.get((bdate_str, sym))
                    if new_dir is not None and new_dir != pos.direction:
                        reason, exit_px = 'SIGNAL_REV', bar['open']

                if reason is not None:
                    raw_ret = pos.direction * (exit_px - pos.entry_price) / pos.entry_price
                    net_ret = raw_ret - SLIPPAGE_PCT - COMMISSION_PCT
                    _record(bdate_str, net_ret * pos.weight)
                    trade_log.append({
                        'symbol':      sym,
                        'slot':        slot_type,
                        'strategy':    f'{sym}_{slot_type}',
                        'direction':   'LONG' if pos.direction == 1 else 'SHORT',
                        'entry_date':  pos.entry_date,
                        'exit_date':   bdate_str,
                        'bars_held':   pos.bars_held,
                        'exit_reason': reason,
                        'pnl':         net_ret,
                        'weight':      pos.weight,
                    })
                    positions[slot_key] = None
                    pos = None

            # ── Queue new signals at the 20:00 "daily close" bar ─────────
            if bhour == 20:
                # Select lookup based on slot type
                if slot_type == 'TREND':
                    sdict = trend_info
                    skey  = (bdate_str, sym)
                else:   # SA
                    sdict = sa_info
                    skey  = (bdate_str, 'NQ')

                if skey in sdict:
                    info    = sdict[skey]
                    new_dir = info['direction']

                    # Extension: same-direction re-signal while position open
                    if (allow_extend and pos is not None
                            and pos.direction == new_dir
                            and pos.extend_count < max_extends):
                        pos.bars_held    = 0
                        pos.extend_count += 1
                        cur_close = bar['close']
                        atr_px    = cur_close * info['atr_pct']
                        pos.stop_price = cur_close - new_dir * stop_atr_mult * atr_px
                        pos.atr = max(pos.atr, atr_px)

                    # New entry: slot free, not already pending
                    elif pos is None and slot_key not in pending:
                        nxt = pd.to_datetime(bdate_str) + pd.Timedelta(days=1)
                        while nxt.dayofweek >= 5:
                            nxt += pd.Timedelta(days=1)
                        pending[slot_key] = {
                            'entry_day': nxt.strftime('%Y-%m-%d'),
                            'direction': new_dir,
                            'info':      info,
                        }

    # ── Aggregate & stats ─────────────────────────────────────────────────
    daily_rets = {d: float(np.sum(v)) for d, v in bar_returns.items()}
    all_dates  = sorted({b['date'].strftime('%Y-%m-%d') for b in bars})
    stats = _compute_stats(daily_rets, trade_log, all_dates)
    return {'portfolio_returns': daily_rets, 'trade_log': trade_log, 'stats': stats}


def _print_multislot(label, result, n_years):
    """Extended reporting that breaks down by slot."""
    s   = result['stats']
    tl  = pd.DataFrame(result['trade_log'])
    pr  = pd.Series(result['portfolio_returns'])
    pr.index = pd.to_datetime(pr.index)
    ann = pr.groupby(pr.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    final = 25000 * (1 + s['total_return_pct'] / 100)

    logger.info(
        "%-34s  Sh=%.3f  Ann=%+.2f%%  DD=%.2f%%  Trades=%d(%.2f/wk)  WR=%.1f%%  $25k->$%s",
        label, s['sharpe_ratio'], s['annualized_return_pct'], s['max_drawdown_pct'],
        s['total_trades'], s['total_trades'] / (n_years * 52),
        s['overall_win_rate'] * 100, f"{final:,.0f}",
    )

    if not tl.empty:
        # Per-slot breakdown
        for slot in tl['strategy'].unique():
            sub = tl[tl['strategy'] == slot]
            wr  = (sub['pnl'] > 0).mean()
            avg = sub['pnl'].mean() * 100
            logger.info("  %-12s  n=%3d  WR=%5.1f%%  avg=%+.3f%%  hold=%.1fb",
                        slot, len(sub), wr * 100, avg, sub['bars_held'].mean())

        # Exit type breakdown
        for ex in ['STOP', 'TRAIL', 'TIME', 'SIGNAL_REV']:
            sub = tl[tl['exit_reason'] == ex]
            if len(sub):
                logger.info("  %-12s  n=%3d  WR=%5.1f%%  avg=%+.3f%%",
                            ex, len(sub), (sub['pnl'] > 0).mean() * 100,
                            sub['pnl'].mean() * 100)

    return s, ann


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("4H EXECUTION EXPERIMENT — daily signals, 4H exits")
    logger.info("=" * 60)

    # ── Build daily pipeline signals ──────────────────────────────────────
    logger.info("Building daily pipeline signals...")
    df0     = build_base_df()
    df_sig  = build_fixed_stack(df0)
    df_sig  = inject_h4_brk_chop(df_sig)
    df_sig  = inject_h4_mom_chop_nq(df_sig)

    sig_period = (df_sig['date'].min(), df_sig['date'].max())
    n_years = (pd.to_datetime(sig_period[1]) - pd.to_datetime(sig_period[0])).days / 365.25
    logger.info("Signal period: %s → %s  (%.1f years)", *sig_period, n_years)

    # ── Load Databento 4H bars ────────────────────────────────────────────
    logger.info("Loading ES/NQ 4H bars...")
    df_4h = pd.read_parquet('/tmp/es_nq_4h.parquet')
    df_4h['datetime'] = pd.to_datetime(df_4h['datetime'], utc=True)
    # Trim to signal period (±1 day buffer)
    start_4h = pd.to_datetime(sig_period[0], utc=True) - pd.Timedelta(days=2)
    end_4h   = pd.to_datetime(sig_period[1], utc=True) + pd.Timedelta(days=2)
    df_4h = df_4h[(df_4h['datetime'] >= start_4h) & (df_4h['datetime'] <= end_4h)].copy()
    logger.info("4H bars: %d  symbols=%s", len(df_4h), df_4h['root'].unique().tolist())

    results = []

    # ── BASELINE: daily system (standard config) ──────────────────────────
    logger.info("\n--- BASELINE: Daily system (1x scale) ---")
    with override_config(
        MAX_HOLD_DAYS=_BASE_HOLD_DAYS, MAX_EXTENDED_HOLD_DAYS=8,
        EXTEND_ALLOW_H4_NONE=True, VOL_SCALE_MAX=0.28,
        MAX_WEIGHT_PER_SYMBOL=_BASE_MAX_WT_PER_SYM,
        STAT_ARB_NQ_WEIGHT=0.20, MAX_COMBINED_EXPOSURE=0.50,
    ):
        r_daily = run_backtest(df_sig)

    s_d = r_daily['stats']
    pr_d = pd.Series(r_daily['portfolio_returns'])
    pr_d.index = pd.to_datetime(pr_d.index)
    ann_d = pr_d.groupby(pr_d.index.year).apply(lambda x: (1+x).prod()-1) * 100
    tr_pw = s_d['total_trades'] / (n_years * 52)
    final_d = 25000 * (1 + s_d['total_return_pct'] / 100)
    logger.info(
        "%-32s  Sh=%.3f  Ann=%+.2f%%  DD=%.2f%%  Trades=%d(%.2f/wk)  WR=%.1f%%  $25k->$%s",
        "DAILY_BASELINE",
        s_d['sharpe_ratio'], s_d['annualized_return_pct'], s_d['max_drawdown_pct'],
        s_d['total_trades'], tr_pw, s_d['overall_win_rate'] * 100, f"{final_d:,.0f}",
    )
    results.append(('DAILY_BASELINE', s_d, ann_d))

    # ── EXP 1: 4H execution, default params ──────────────────────────────
    logger.info("\n--- EXP 1: 4H execution, default (48-bar hold, 1.5x stop) ---")
    r1 = run_4h_execution(df_sig, df_4h, weight_scale=1.0)
    s1, ann1 = _print_result("4H_EXEC_BASE", r1, n_years)
    results.append(('4H_EXEC_BASE', s1, ann1))

    # ── EXP 2: Tighter stop (1.0x ATR) — best Sharpe found ───────────────
    logger.info("\n--- EXP 2: Tight stop (1.0x ATR), 48-bar hold ---")
    r2 = run_4h_execution(df_sig, df_4h, stop_atr_mult=1.0, trail_atr_mult=2.0,
                          weight_scale=1.0)
    s2, ann2 = _print_result("4H_TIGHT_STOP_1x", r2, n_years)
    results.append(('4H_TIGHT_STOP_1x', s2, ann2))

    # ── EXP 3: Tight stop, shorter hold (24 bars = 4 days) ───────────────
    logger.info("\n--- EXP 3: Tight stop, 24-bar hold (4 days) ---")
    r3 = run_4h_execution(df_sig, df_4h, stop_atr_mult=1.0, trail_atr_mult=2.0,
                          max_hold_bars=24, weight_scale=1.0)
    s3, ann3 = _print_result("4H_TIGHT_24BAR_1x", r3, n_years)
    results.append(('4H_TIGHT_24BAR_1x', s3, ann3))

    # ── EXP 4: Tight stop, 24-bar hold, 2x scale ─────────────────────────
    logger.info("\n--- EXP 4: Tight stop, 24-bar, 2x scale ---")
    r4 = run_4h_execution(df_sig, df_4h, stop_atr_mult=1.0, trail_atr_mult=2.0,
                          max_hold_bars=24, weight_scale=2.0,
                          vol_scale_max=0.56, max_weight_per_sym=1.0)
    s4, ann4 = _print_result("4H_TIGHT_24BAR_2x", r4, n_years)
    results.append(('4H_TIGHT_24BAR_2x', s4, ann4))

    # ── EXP 5: Tight stop, 24-bar hold, 3x scale ─────────────────────────
    logger.info("\n--- EXP 5: Tight stop, 24-bar, 3x scale ---")
    r5 = run_4h_execution(df_sig, df_4h, stop_atr_mult=1.0, trail_atr_mult=2.0,
                          max_hold_bars=24, weight_scale=3.0,
                          vol_scale_max=0.84, max_weight_per_sym=1.5)
    s5, ann5 = _print_result("4H_TIGHT_24BAR_3x", r5, n_years)
    results.append(('4H_TIGHT_24BAR_3x', s5, ann5))

    # ── EXP 6: Tight stop, 24-bar hold, 5x scale ─────────────────────────
    logger.info("\n--- EXP 6: Tight stop, 24-bar, 5x scale (daily return target) ---")
    r6 = run_4h_execution(df_sig, df_4h, stop_atr_mult=1.0, trail_atr_mult=2.0,
                          max_hold_bars=24, weight_scale=5.0,
                          vol_scale_max=1.40, max_weight_per_sym=2.5)
    s6, ann6 = _print_result("4H_TIGHT_24BAR_5x", r6, n_years)
    results.append(('4H_TIGHT_24BAR_5x', s6, ann6))

    # ── EXP 7: No fixed stop — trail + signal_rev + time only ────────────
    # Hypothesis: the 70 stopped trades (100% loss) recover if held to TIME.
    # If so, removing the fixed stop should boost WR and total return.
    logger.info("\n--- EXP 7: No fixed stop (999x ATR), trail+time only, 24-bar, 2x ---")
    r7 = run_4h_execution(df_sig, df_4h, stop_atr_mult=999.0, trail_atr_mult=2.0,
                          trail_activate=1.5, max_hold_bars=24, weight_scale=2.0,
                          vol_scale_max=0.56, max_weight_per_sym=1.0)
    s7, ann7 = _print_result("4H_NOSTOP_24BAR_2x", r7, n_years)
    results.append(('4H_NOSTOP_24BAR_2x', s7, ann7))

    # ── EXP 8: Tight stop + hold extension (reset timer on re-signal) ────
    # Hypothesis: extending the hold on same-direction re-signal adds more
    # profitable time exits, increasing total return without hurting Sharpe.
    logger.info("\n--- EXP 8: Tight stop + extend (3x resets), 24-bar, 2x scale ---")
    r8 = run_4h_execution(df_sig, df_4h, stop_atr_mult=1.0, trail_atr_mult=2.0,
                          max_hold_bars=24, weight_scale=2.0,
                          vol_scale_max=0.56, max_weight_per_sym=1.0,
                          allow_extend=True, max_extends=3)
    s8, ann8 = _print_result("4H_EXT_24BAR_2x", r8, n_years)
    results.append(('4H_EXT_24BAR_2x', s8, ann8))

    # ── EXP 9: No stop + extension, 3x scale — target Ann ~4-5% ─────────
    logger.info("\n--- EXP 9: No stop + extend, 24-bar, 3x scale ---")
    r9 = run_4h_execution(df_sig, df_4h, stop_atr_mult=999.0, trail_atr_mult=2.0,
                          trail_activate=1.5, max_hold_bars=24, weight_scale=3.0,
                          vol_scale_max=0.84, max_weight_per_sym=1.5,
                          allow_extend=True, max_extends=3)
    s9, ann9 = _print_result("4H_NOSTOP_EXT_3x", r9, n_years)
    results.append(('4H_NOSTOP_EXT_3x', s9, ann9))

    # ── EXP 10: EXT (tight stop + extend) at 3x scale ────────────────────
    # EXP 8 was the best (Sh=2.563). Scale it to target Ann ~3.2%.
    logger.info("\n--- EXP 10: Tight stop + extend, 24-bar, 3x scale ---")
    r10 = run_4h_execution(df_sig, df_4h, stop_atr_mult=1.0, trail_atr_mult=2.0,
                           max_hold_bars=24, weight_scale=3.0,
                           vol_scale_max=0.84, max_weight_per_sym=1.5,
                           allow_extend=True, max_extends=3)
    s10, ann10 = _print_result("4H_EXT_24BAR_3x", r10, n_years)
    results.append(('4H_EXT_24BAR_3x', s10, ann10))

    # ── EXP 11: EXT at 4x scale — target Ann ~4.3% ───────────────────────
    logger.info("\n--- EXP 11: Tight stop + extend, 24-bar, 4x scale ---")
    r11 = run_4h_execution(df_sig, df_4h, stop_atr_mult=1.0, trail_atr_mult=2.0,
                           max_hold_bars=24, weight_scale=4.0,
                           vol_scale_max=1.12, max_weight_per_sym=2.0,
                           allow_extend=True, max_extends=3)
    s11, ann11 = _print_result("4H_EXT_24BAR_4x", r11, n_years)
    results.append(('4H_EXT_24BAR_4x', s11, ann11))

    # ── EXP 12: EXT with tighter extend stop (0.75x ATR re-anchor) ───────
    # What if we re-anchor the stop very tightly on extension to lock in gains?
    logger.info("\n--- EXP 12: Tight stop + tight extend (stop=0.75x), 24-bar, 2x ---")
    r12 = run_4h_execution(df_sig, df_4h, stop_atr_mult=0.75, trail_atr_mult=1.5,
                           max_hold_bars=24, weight_scale=2.0,
                           vol_scale_max=0.56, max_weight_per_sym=1.0,
                           allow_extend=True, max_extends=3)
    s12, ann12 = _print_result("4H_EXT_TIGHT_2x", r12, n_years)
    results.append(('4H_EXT_TIGHT_2x', s12, ann12))

    # ── EXP 13: EXT at 5x scale — push past daily return level ──────────
    logger.info("\n--- EXP 13: Tight stop + extend, 24-bar, 5x scale ---")
    r13 = run_4h_execution(df_sig, df_4h, stop_atr_mult=1.0, trail_atr_mult=2.0,
                           max_hold_bars=24, weight_scale=5.0,
                           vol_scale_max=1.40, max_weight_per_sym=2.5,
                           allow_extend=True, max_extends=3)
    s13, ann13 = _print_result("4H_EXT_24BAR_5x", r13, n_years)
    results.append(('4H_EXT_24BAR_5x', s13, ann13))

    # ── EXP 14: EXT, early trail (activate at 0.5x ATR) ─────────────────
    # Trail activates sooner → locks in gains faster → improve WR on trail exits
    logger.info("\n--- EXP 14: EXT + early trail (activate=0.5x), 3x scale ---")
    r14 = run_4h_execution(df_sig, df_4h, stop_atr_mult=1.0, trail_atr_mult=2.0,
                           trail_activate=0.5, max_hold_bars=24, weight_scale=3.0,
                           vol_scale_max=0.84, max_weight_per_sym=1.5,
                           allow_extend=True, max_extends=3)
    s14, ann14 = _print_result("4H_EXT_ETRL_3x", r14, n_years)
    results.append(('4H_EXT_ETRL_3x', s14, ann14))

    # ══════════════════════════════════════════════════════════════════════
    # MULTI-SLOT: 3 concurrent independent strategies
    # ══════════════════════════════════════════════════════════════════════

    logger.info("\n" + "=" * 60)
    logger.info("MULTI-SLOT EXPERIMENTS — ES_TREND + NQ_TREND + NQ_SA")
    logger.info("=" * 60)

    # ── MS1: 1x scale baseline ────────────────────────────────────────────
    logger.info("\n--- MS1: Multi-slot, 1x scale ---")
    ms1 = run_4h_multislot(df_sig, df_4h, weight_scale=1.0,
                            vol_scale_max=0.28, max_weight_per_sym=0.5)
    sms1, ams1 = _print_multislot("4H_MS_1x", ms1, n_years)
    results.append(('4H_MS_1x', sms1, ams1))

    # ── MS2: 3x scale (match EXT 3x for apples-to-apples) ────────────────
    logger.info("\n--- MS2: Multi-slot, 3x scale ---")
    ms2 = run_4h_multislot(df_sig, df_4h, weight_scale=3.0,
                            vol_scale_max=0.84, max_weight_per_sym=1.5)
    sms2, ams2 = _print_multislot("4H_MS_3x", ms2, n_years)
    results.append(('4H_MS_3x', sms2, ams2))

    # ── MS3: Tuned — tighter SA stop (SA slots noisier), wider extend ─────
    logger.info("\n--- MS3: Multi-slot, 3x, SA stop=1.5x (SA slots noisier) ---")
    ms3 = run_4h_multislot(df_sig, df_4h, weight_scale=3.0,
                            stop_atr_mult=1.5, trail_atr_mult=2.5,
                            vol_scale_max=0.84, max_weight_per_sym=1.5)
    sms3, ams3 = _print_multislot("4H_MS_3x_W15STOP", ms3, n_years)
    results.append(('4H_MS_3x_W15STOP', sms3, ams3))

    # ── MS4: Quality filter — SA=STRONG, trend edge=STRONG ───────────────
    # Only enter high-conviction signals. Reduces choppy-year entries.
    logger.info("\n--- MS4: Multi-slot, 3x, SA_STRONG + TREND_EDGE_STRONG filter ---")
    ms4 = run_4h_multislot(df_sig, df_4h, weight_scale=3.0,
                            vol_scale_max=0.84, max_weight_per_sym=1.5,
                            sa_quality_min='STRONG', trend_edge_min='STRONG')
    sms4, ams4 = _print_multislot("4H_MS_3x_QFILTER", ms4, n_years)
    results.append(('4H_MS_3x_QFILTER', sms4, ams4))

    # ── MS5: SA quality filter only (keep all trend signals) ─────────────
    logger.info("\n--- MS5: Multi-slot, 3x, SA=STRONG filter only ---")
    ms5 = run_4h_multislot(df_sig, df_4h, weight_scale=3.0,
                            vol_scale_max=0.84, max_weight_per_sym=1.5,
                            sa_quality_min='STRONG')
    sms5, ams5 = _print_multislot("4H_MS_3x_SA_STRONG", ms5, n_years)
    results.append(('4H_MS_3x_SA_STRONG', sms5, ams5))

    # ── MS6: BEST COMBO — wider stop + SA quality filter ─────────────────
    # Combines: 1.5x stop (reduces choppy stop-outs) + SA_STRONG (removes low-quality SA)
    logger.info("\n--- MS6: Multi-slot, 3x, SA_STRONG + 1.5x stop (best combo) ---")
    ms6 = run_4h_multislot(df_sig, df_4h, weight_scale=3.0,
                            stop_atr_mult=1.5, trail_atr_mult=2.5,
                            vol_scale_max=0.84, max_weight_per_sym=1.5,
                            sa_quality_min='STRONG')
    sms6, ams6 = _print_multislot("4H_MS_3x_BEST", ms6, n_years)
    results.append(('4H_MS_3x_BEST', sms6, ams6))

    # ── Summary table ────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("4H EXECUTION vs DAILY BASELINE — RESULTS")
    print("=" * 90)
    hdr = f"{'label':<28} {'sharpe':>7} {'d_sh':>7} {'annual':>8} {'d_ann':>7} {'max_dd':>8} {'trades':>7} {'wr':>6}  {'$25k->':>12}"
    print(hdr)
    print("-" * 90)
    base_sh  = results[0][1]['sharpe_ratio']
    base_ann = results[0][1]['annualized_return_pct']
    for lbl, s, ann in results:
        d_sh  = s['sharpe_ratio']         - base_sh
        d_ann = s['annualized_return_pct'] - base_ann
        final = 25000 * (1 + s['total_return_pct'] / 100)
        print(f"{lbl:<28} {s['sharpe_ratio']:>7.3f} {d_sh:>+7.3f} "
              f"{s['annualized_return_pct']:>+7.2f}% {d_ann:>+6.2f}% "
              f"{s['max_drawdown_pct']:>7.2f}% {s['total_trades']:>7d} "
              f"{s['overall_win_rate']*100:>5.1f}%  ${final:>10,.0f}")

    print()
    print("=" * 90)
    print("ANNUAL RETURNS BY YEAR")
    print("=" * 90)
    years = sorted(results[0][2].index)
    print(f"{'label':<28} " + "  ".join(f"yr_{y}" for y in years))
    for lbl, s, ann in results:
        row = f"{lbl:<28} " + "  ".join(f"{ann.get(y,0):+6.1f}%" for y in years)
        print(row)


if __name__ == '__main__':
    main()
