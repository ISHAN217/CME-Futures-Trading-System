"""
run_trade_timing_analysis.py — Trade timing quality analysis.
=============================================================

For every completed TREND LONG trade, reconstruct the full price swing and answer:
  1. Where did we enter? (% from swing low)
  2. How long did we hold? (days)
  3. How much of the available move did we capture? (capture_rate)
  4. How much did we miss BEFORE entry? (pre_entry_miss_pct)
  5. How much did we miss AFTER exit? (post_exit_miss_pct)
  6. Are we buying early / middle / late?
  7. Are we exiting before or after the peak?

Methodology:
  - "Full swing" for each trade = swing_low (20d before entry) to swing_high (20d after exit)
  - entry_close  ≈ closing price on entry_date
  - exit_close   ≈ closing price on exit_date
  - pre_entry_miss  = entry_close - swing_low  (price move we missed before getting in)
  - captured        = exit_close - entry_close  (price move we captured)
  - post_exit_miss  = swing_high - exit_close  (price move we missed after exiting)
  - total_move      = swing_high - swing_low
  - entry_pct       = pre_entry_miss / total_move   → 0%=bottom, 100%=top
  - capture_rate    = captured / total_move
  - entry_timing    = EARLY (<30%), MIDDLE (30-70%), LATE (>70%)
  - exit_timing     = EARLY (left >50% on table), ON_TIME, LATE (price fell after exit)
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import config
from data_loader        import load_daily, load_4h
from features           import compute_features
from momentum_engine    import compute_momentum
from weekly_bias        import compute_weekly_bias
from regime_engine      import compute_regime
from volatility_engine  import compute_volatility_features
from intraday_4h_engine import compute_4h_features, aggregate_4h_to_daily, merge_4h_into_daily
from mean_reversion     import compute_mean_reversion
from stat_arb           import compute_stat_arb
from sentiment_layer    import compute_sentiment
from signal_engine      import compute_signals
from signal_quality_engine          import compute_signal_quality
from trend_continuation_edge_engine import compute_trend_continuation_edge
from backtest           import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT_DIR = "/Users/ishanbhardwaj/cme_competition_system/output/timing"
os.makedirs(OUT_DIR, exist_ok=True)

SWING_LOOKBACK  = 20   # trading days before entry to find swing low
SWING_LOOKAHEAD = 20   # trading days after exit to find swing high


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_df_and_trades():
    df        = load_daily()
    df_4h_raw = load_4h()
    df = compute_features(df)
    df = compute_momentum(df)
    df = compute_weekly_bias(df)
    df = compute_regime(df)
    df = compute_volatility_features(df)

    if config.USE_CWT_CHOP_FILTER:
        from cwt_chop_filter import compute_cwt_chop_features
        df = compute_cwt_chop_features(df)
    else:
        for col, val in [("cwt_chop_label","CWT_UNCLEAR"),("cwt_trade_allowed",1),
                         ("cwt_size_multiplier",1.0),("cwt_confidence",0.0),
                         ("cwt_total_energy",0.0),("cwt_entropy",0.5),("cwt_energy_z",0.0),
                         ("cwt_compression_flag",0),("cwt_expansion_flag",0),
                         ("cwt_high_energy_ratio",1/3),("cwt_mid_energy_ratio",1/3),
                         ("cwt_slow_energy_ratio",1/3)]:
            if col not in df.columns: df[col] = val

    if df_4h_raw is not None:
        df_4h_feat  = compute_4h_features(df_4h_raw)
        df_4h_daily = aggregate_4h_to_daily(df_4h_feat)
    else:
        df_4h_daily = None
    df = merge_4h_into_daily(df, df_4h_daily)
    df = compute_mean_reversion(df)
    df = compute_stat_arb(df)
    df = compute_sentiment(df)
    df = compute_signals(df)
    df = compute_signal_quality(df)
    df = compute_trend_continuation_edge(df)

    df["date"] = pd.to_datetime(df["date"])
    result = run_backtest(df)
    tl = result.get("trade_log")
    return df, tl


# ─────────────────────────────────────────────────────────────────────────────
# PRICE SERIES LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def build_price_index(df: pd.DataFrame) -> dict:
    """Build {symbol: pd.Series(close, index=date)} for fast lookup."""
    idx = {}
    for sym, g in df.groupby("symbol"):
        g2 = g.set_index("date").sort_index()
        idx[sym] = g2["close"]
    return idx


def get_close(price_idx: dict, sym: str, date) -> float:
    s = price_idx.get(sym)
    if s is None: return np.nan
    date = pd.Timestamp(date)
    if date in s.index: return float(s[date])
    # nearest available date
    avail = s.index[s.index <= date]
    return float(s[avail[-1]]) if len(avail) > 0 else np.nan


def swing_low_before(price_idx: dict, sym: str, date, lookback: int) -> float:
    s = price_idx.get(sym)
    if s is None: return np.nan
    date = pd.Timestamp(date)
    avail = s.index[s.index <= date]
    window = avail[-lookback:] if len(avail) >= lookback else avail
    return float(s[window].min()) if len(window) > 0 else np.nan


def swing_high_after(price_idx: dict, sym: str, date, lookahead: int) -> float:
    s = price_idx.get(sym)
    if s is None: return np.nan
    date = pd.Timestamp(date)
    avail = s.index[s.index >= date]
    window = avail[:lookahead] if len(avail) >= lookahead else avail
    return float(s[window].max()) if len(window) > 0 else np.nan


def peak_during_hold(price_idx: dict, sym: str, entry_date, exit_date) -> float:
    s = price_idx.get(sym)
    if s is None: return np.nan
    ed = pd.Timestamp(entry_date); xd = pd.Timestamp(exit_date)
    window = s[(s.index >= ed) & (s.index <= xd)]
    return float(window.max()) if len(window) > 0 else np.nan


def trough_during_hold(price_idx: dict, sym: str, entry_date, exit_date) -> float:
    s = price_idx.get(sym)
    if s is None: return np.nan
    ed = pd.Timestamp(entry_date); xd = pd.Timestamp(exit_date)
    window = s[(s.index >= ed) & (s.index <= xd)]
    return float(window.min()) if len(window) > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# TIMING ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def enrich_timing(tl: pd.DataFrame, price_idx: dict) -> pd.DataFrame:
    tl = tl.copy()
    tl["entry_date"] = pd.to_datetime(tl["entry_date"])
    tl["exit_date"]  = pd.to_datetime(tl["exit_date"])

    rows = []
    for _, row in tl.iterrows():
        sym    = row["symbol"]
        ed     = row["entry_date"]
        xd     = row["exit_date"]
        dirn   = row.get("direction", "LONG")
        strat  = row.get("strategy",  "TREND")
        pnl    = float(row.get("pnl",  0.0))     # cum_ret = underlying return
        hold   = int(row.get("holding_days", 1))
        e_reg  = row.get("entry_regime",  "")
        e_bias = row.get("entry_bias",    "")
        e_qual = row.get("entry_quality", 3)
        exit_r = row.get("exit_reason",   "")
        h_conv = bool(row.get("high_conviction", False))
        add_c  = int(row.get("add_count", 0))

        ec = get_close(price_idx, sym, ed)
        xc = get_close(price_idx, sym, xd)

        if dirn == "LONG":
            sw_low  = swing_low_before(price_idx, sym, ed,  SWING_LOOKBACK)
            sw_high = swing_high_after(price_idx, sym, xd, SWING_LOOKAHEAD)
            peak_h  = peak_during_hold(price_idx, sym, ed, xd)
            trough_h= trough_during_hold(price_idx, sym, ed, xd)

            pre_entry_miss  = ec - sw_low    if (not np.isnan(ec) and not np.isnan(sw_low))  else np.nan
            captured_raw    = xc - ec        if (not np.isnan(ec) and not np.isnan(xc))      else np.nan
            post_exit_miss  = sw_high - xc   if (not np.isnan(xc) and not np.isnan(sw_high)) else np.nan
            total_move      = sw_high - sw_low if (not np.isnan(sw_low) and not np.isnan(sw_high)) else np.nan
            peak_miss       = peak_h - xc    if (not np.isnan(peak_h) and not np.isnan(xc))  else np.nan
            intra_dd        = ec - trough_h  if (not np.isnan(ec) and not np.isnan(trough_h)) else np.nan

            entry_pct       = pre_entry_miss / total_move      if total_move and total_move > 0 else np.nan
            capture_rate    = captured_raw   / total_move      if total_move and total_move > 0 else np.nan
            post_exit_pct   = post_exit_miss / total_move      if total_move and total_move > 0 else np.nan
            capture_of_avail= captured_raw   / (total_move - pre_entry_miss) \
                              if (total_move and pre_entry_miss and (total_move - pre_entry_miss) > 0) else np.nan
            pre_pct         = pre_entry_miss / ec              if ec and ec > 0 else np.nan
            capt_pct        = captured_raw   / ec              if ec and ec > 0 else np.nan
            post_pct        = post_exit_miss / xc              if xc and xc > 0 else np.nan
            peak_miss_pct   = peak_miss      / ec              if ec and ec > 0 else np.nan
            dd_pct          = intra_dd       / ec              if ec and ec > 0 else np.nan

        else:   # SHORT
            sw_high = swing_high_after(price_idx, sym, ed, SWING_LOOKBACK)  # look back for high
            sw_low  = swing_low_before(price_idx, sym, xd, SWING_LOOKAHEAD) # look ahead for low
            # Invert everything for short
            pre_entry_miss = sw_high - ec    if (not np.isnan(sw_high) and not np.isnan(ec))  else np.nan
            captured_raw   = ec - xc         if (not np.isnan(ec) and not np.isnan(xc))       else np.nan
            post_exit_miss = xc - sw_low     if (not np.isnan(xc) and not np.isnan(sw_low))   else np.nan
            total_move     = sw_high - sw_low if (not np.isnan(sw_high) and not np.isnan(sw_low)) else np.nan
            peak_h  = trough_during_hold(price_idx, sym, ed, xd)
            trough_h= peak_during_hold(price_idx, sym, ed, xd)
            peak_miss      = xc - peak_h     if (not np.isnan(peak_h) and not np.isnan(xc))  else np.nan
            intra_dd       = trough_h - ec   if (not np.isnan(trough_h) and not np.isnan(ec)) else np.nan

            entry_pct       = pre_entry_miss / total_move      if total_move and total_move > 0 else np.nan
            capture_rate    = captured_raw   / total_move      if total_move and total_move > 0 else np.nan
            post_exit_pct   = post_exit_miss / total_move      if total_move and total_move > 0 else np.nan
            capture_of_avail= captured_raw   / (total_move - pre_entry_miss) \
                              if (total_move and pre_entry_miss and (total_move - pre_entry_miss) > 0) else np.nan
            pre_pct         = pre_entry_miss / ec              if ec and ec > 0 else np.nan
            capt_pct        = captured_raw   / ec              if ec and ec > 0 else np.nan
            post_pct        = post_exit_miss / xc              if xc and xc > 0 else np.nan
            peak_miss_pct   = peak_miss      / ec              if ec and ec > 0 else np.nan
            dd_pct          = intra_dd       / ec              if ec and ec > 0 else np.nan

        # Entry timing label
        if entry_pct is not None and not np.isnan(entry_pct):
            if entry_pct < 0.30:  entry_timing = "EARLY"
            elif entry_pct < 0.60: entry_timing = "MIDDLE"
            else:                  entry_timing = "LATE"
        else:
            entry_timing = "UNKNOWN"

        # Exit timing label
        if post_exit_pct is not None and not np.isnan(post_exit_pct) and \
           capture_rate  is not None and not np.isnan(capture_rate):
            if post_exit_pct > 0.40:   exit_timing = "TOO_EARLY"
            elif post_exit_pct < 0.05: exit_timing = "NEAR_PEAK"
            else:                       exit_timing = "OK"
        else:
            exit_timing = "UNKNOWN"

        rows.append({
            "symbol":           sym,
            "strategy":         strat,
            "direction":        dirn,
            "entry_date":       ed,
            "exit_date":        xd,
            "holding_days":     hold,
            "entry_regime":     e_reg,
            "entry_bias":       e_bias,
            "entry_quality":    e_qual,
            "high_conviction":  h_conv,
            "add_count":        add_c,
            "exit_reason":      exit_r,
            "pnl_cum_ret":      round(pnl, 5),
            "entry_close":      round(ec, 2)  if not np.isnan(ec)  else np.nan,
            "exit_close":       round(xc, 2)  if not np.isnan(xc)  else np.nan,
            "swing_low":        round(sw_low,  2) if not np.isnan(sw_low)  else np.nan,
            "swing_high":       round(sw_high, 2) if not np.isnan(sw_high) else np.nan,
            "peak_during_hold": round(peak_h,  2) if not np.isnan(peak_h)  else np.nan,
            # Absolute price moves
            "pre_entry_miss_pts":  round(pre_entry_miss,2) if not np.isnan(pre_entry_miss)  else np.nan,
            "captured_pts":        round(captured_raw,  2) if not np.isnan(captured_raw)    else np.nan,
            "post_exit_miss_pts":  round(post_exit_miss,2) if not np.isnan(post_exit_miss)  else np.nan,
            "total_move_pts":      round(total_move,    2) if not np.isnan(total_move)      else np.nan,
            "peak_miss_during_hold_pts": round(peak_miss,2) if not np.isnan(peak_miss)      else np.nan,
            "intra_trade_dd_pts":  round(intra_dd,     2) if not np.isnan(intra_dd)         else np.nan,
            # As % of entry price
            "pre_entry_miss_pct":  round(pre_pct * 100, 3)   if pre_pct and not np.isnan(pre_pct)     else np.nan,
            "captured_pct":        round(capt_pct * 100, 3)  if capt_pct and not np.isnan(capt_pct)   else np.nan,
            "post_exit_miss_pct":  round(post_pct * 100, 3)  if post_pct and not np.isnan(post_pct)   else np.nan,
            "peak_miss_pct":       round(peak_miss_pct*100,3) if peak_miss_pct and not np.isnan(peak_miss_pct) else np.nan,
            "intra_dd_pct":        round(dd_pct * 100, 3)    if dd_pct and not np.isnan(dd_pct)       else np.nan,
            # Fractions of total swing
            "entry_pct_of_swing":  round(entry_pct * 100, 1)    if entry_pct and not np.isnan(entry_pct)    else np.nan,
            "capture_rate_pct":    round(capture_rate * 100, 1)  if capture_rate and not np.isnan(capture_rate) else np.nan,
            "post_exit_pct_of_swing": round(post_exit_pct*100,1) if post_exit_pct and not np.isnan(post_exit_pct) else np.nan,
            "capture_of_available_pct": round(capture_of_avail*100,1) if capture_of_avail and not np.isnan(capture_of_avail) else np.nan,
            "entry_timing":     entry_timing,
            "exit_timing":      exit_timing,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS & PRINTING
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(enriched: pd.DataFrame) -> None:
    # Filter to LONG trades only (most informative; SHORT trades very rare)
    longs = enriched[enriched["direction"] == "LONG"].copy()
    n_tot = len(enriched); n_long = len(longs)

    print(f"\n{'═'*80}")
    print(f"  TRADE TIMING ANALYSIS  —  {n_tot} trades total  ({n_long} LONG)")
    print(f"{'═'*80}")

    def _pct(s, col):
        v = s[col].dropna()
        if len(v) == 0: return "N/A"
        return f"{v.mean():+.2f}%  (med {v.median():+.2f}%,  p25={v.quantile(.25):+.2f}%,  p75={v.quantile(.75):+.2f}%)"

    def _pct_n(s, col):
        v = s[col].dropna()
        if len(v) == 0: return "N/A"
        return f"{v.mean():.1f}%  (med {v.median():.1f}%)"

    print(f"\n  ── Q1: WHERE DID WE ENTER? (% from 20d swing low to entry, relative to full swing)")
    print(f"     Mean entry_pct_of_swing:  {_pct_n(longs, 'entry_pct_of_swing')}")
    ep = longs["entry_pct_of_swing"].dropna()
    if len(ep):
        print(f"     Interpretation: We enter when {ep.mean():.0f}% of the 20-day swing is already complete.")
        print(f"     → {(ep < 30).mean()*100:.0f}% of entries are EARLY  (bottom third of move)")
        print(f"     → {((ep >= 30) & (ep < 60)).mean()*100:.0f}% of entries are MIDDLE (middle third)")
        print(f"     → {(ep >= 60).mean()*100:.0f}% of entries are LATE   (top third of move)")

    print(f"\n  ── Q2: HOW LONG DID WE HOLD?")
    hd = longs["holding_days"].dropna()
    if len(hd):
        print(f"     Mean: {hd.mean():.1f} days    Median: {hd.median():.1f} days")
        print(f"     ≤1d: {(hd<=1).mean()*100:.0f}%   ≤3d: {(hd<=3).mean()*100:.0f}%   ≤5d: {(hd<=5).mean()*100:.0f}%   >5d: {(hd>5).mean()*100:.0f}%")
        for er, g in longs.groupby("exit_reason"):
            gd = g["holding_days"].dropna()
            print(f"     [{er:30s}]  n={len(g):3d}  avg={gd.mean():.1f}d")

    print(f"\n  ── Q3: HOW MUCH OF THE MOVE DID WE CAPTURE?")
    cr = longs["capture_rate_pct"].dropna()
    av = longs["capture_of_available_pct"].dropna()
    if len(cr):
        print(f"     Capture rate (% of full 40d swing):  {_pct_n(longs, 'capture_rate_pct')}")
        print(f"     Capture of available (% of move from our entry onward): {_pct_n(longs, 'capture_of_available_pct')}")
        print(f"     Captured as % of entry price:  {_pct(longs, 'captured_pct')}")

    print(f"\n  ── Q4: HOW MUCH DID WE MISS BEFORE ENTRY?")
    if len(longs):
        print(f"     Pre-entry move (% of entry price):  {_pct(longs, 'pre_entry_miss_pct')}")
        print(f"     Pre-entry as % of total swing:      {_pct_n(longs, 'entry_pct_of_swing')}")

    print(f"\n  ── Q5: HOW MUCH DID WE MISS AFTER EXIT?")
    if len(longs):
        print(f"     Post-exit move (% of exit price):   {_pct(longs, 'post_exit_miss_pct')}")
        print(f"     Post-exit as % of total swing:      {_pct_n(longs, 'post_exit_pct_of_swing')}")
        print(f"     Peak-miss during hold (left inside trade): {_pct(longs, 'peak_miss_pct')}")

    print(f"\n  ── Q6: EARLY / MIDDLE / LATE?")
    et = longs["entry_timing"].value_counts()
    for label in ["EARLY","MIDDLE","LATE","UNKNOWN"]:
        n = et.get(label, 0)
        print(f"     {label:<10}  {n:4d}  ({n/max(len(longs),1)*100:.0f}%)")

    print(f"\n  ── EXIT TIMING")
    xt = longs["exit_timing"].value_counts()
    for label in ["TOO_EARLY","OK","NEAR_PEAK","UNKNOWN"]:
        n = xt.get(label, 0)
        print(f"     {label:<15}  {n:4d}  ({n/max(len(longs),1)*100:.0f}%)")

    print(f"\n  ── BREAKDOWN BY ENTRY TIMING")
    print(f"  {'Entry Timing':<12}  {'N':>4}  {'Avg Cap%':>9}  {'Avg Post%':>10}  {'WinRate':>8}  {'AvgPnL':>8}")
    for timing in ["EARLY","MIDDLE","LATE"]:
        g = longs[longs["entry_timing"] == timing]
        if len(g) == 0: continue
        cap  = g["capture_rate_pct"].mean()
        post = g["post_exit_pct_of_swing"].mean()
        wr   = (g["pnl_cum_ret"] > 0).mean()
        apnl = g["pnl_cum_ret"].mean() * 100
        print(f"  {timing:<12}  {len(g):>4}  {cap:>8.1f}%  {post:>9.1f}%  {wr:>7.0%}  {apnl:>+7.3f}%")

    print(f"\n  ── BREAKDOWN BY EXIT REASON vs CAPTURE RATE")
    print(f"  {'Exit Reason':<30}  {'N':>4}  {'Avg Hold':>8}  {'Avg Cap%':>9}  {'Post Exit%':>11}  {'WinRate':>8}")
    for er, g in longs.groupby("exit_reason"):
        cap  = g["capture_rate_pct"].mean()
        post = g["post_exit_pct_of_swing"].mean()
        wr   = (g["pnl_cum_ret"] > 0).mean()
        hold = g["holding_days"].mean()
        print(f"  {er:<30}  {len(g):>4}  {hold:>7.1f}d  {cap:>8.1f}%  {post:>10.1f}%  {wr:>7.0%}")

    print(f"\n  ── BREAKDOWN BY SYMBOL")
    for sym, g in longs.groupby("symbol"):
        ep_m = g["entry_pct_of_swing"].mean()
        cr_m = g["capture_rate_pct"].mean()
        po_m = g["post_exit_pct_of_swing"].mean()
        wr   = (g["pnl_cum_ret"] > 0).mean()
        print(f"  {sym}:  n={len(g):3d}  entry@{ep_m:.0f}% of swing  capture={cr_m:.0f}%  post_miss={po_m:.0f}%  WR={wr:.0%}")

    print(f"\n  ── BREAKDOWN BY YEAR")
    longs2 = longs.copy()
    longs2["year"] = pd.to_datetime(longs2["entry_date"]).dt.year
    for yr, g in sorted(longs2.groupby("year")):
        ep_m = g["entry_pct_of_swing"].mean()
        cr_m = g["capture_rate_pct"].mean()
        po_m = g["post_exit_pct_of_swing"].mean()
        wr   = (g["pnl_cum_ret"] > 0).mean()
        print(f"  {yr}:  n={len(g):3d}  entry@{ep_m:.0f}%  capture={cr_m:.0f}%  post_miss={po_m:.0f}%  WR={wr:.0%}")

    print(f"\n  ── HIGH CONVICTION vs NORMAL")
    for hc, g in longs.groupby("high_conviction"):
        label = "HIGH_CONV" if hc else "NORMAL"
        ep_m = g["entry_pct_of_swing"].mean()
        cr_m = g["capture_rate_pct"].mean()
        wr   = (g["pnl_cum_ret"] > 0).mean()
        print(f"  {label:<12}  n={len(g):3d}  entry@{ep_m:.0f}%  capture={cr_m:.0f}%  WR={wr:.0%}")

    print(f"\n  ── WITH ADD-TO-WINNER vs WITHOUT")
    for add, g in longs.groupby(longs["add_count"] > 0):
        label = "ADDED" if add else "NO_ADD"
        ep_m = g["entry_pct_of_swing"].mean()
        cr_m = g["capture_rate_pct"].mean()
        hold = g["holding_days"].mean()
        wr   = (g["pnl_cum_ret"] > 0).mean()
        print(f"  {label:<12}  n={len(g):3d}  entry@{ep_m:.0f}%  capture={cr_m:.0f}%  hold={hold:.1f}d  WR={wr:.0%}")

    # Implied diagnosis
    print(f"\n{'═'*80}")
    print(f"  TIMING DIAGNOSIS")
    print(f"{'═'*80}")
    ep_all = longs["entry_pct_of_swing"].dropna()
    cr_all = longs["capture_rate_pct"].dropna()
    po_all = longs["post_exit_pct_of_swing"].dropna()

    if len(ep_all) and len(cr_all) and len(po_all):
        ep_m = ep_all.mean(); cr_m = cr_all.mean(); po_m = po_all.mean()

        if ep_m > 50:
            print(f"  ENTRY: LATE  — entering after {ep_m:.0f}% of the swing has already moved.")
            print(f"    → We are momentum chasers at daily resolution.")
            print(f"    → The trend signal fires AFTER price has established structure.")
        elif ep_m < 25:
            print(f"  ENTRY: EARLY — entering when only {ep_m:.0f}% of the swing is done.")
            print(f"    → We are anticipating the trend early.")
        else:
            print(f"  ENTRY: MIDDLE — entering at {ep_m:.0f}% of the swing (reasonable).")

        if po_m > 30:
            print(f"\n  EXIT: TOO EARLY — leaving {po_m:.0f}% of the total move on the table after exit.")
            print(f"    → Stops and trail stops are firing before the trend completes.")
        elif po_m < 10:
            print(f"\n  EXIT: NEAR PEAK — exiting near the top (only {po_m:.0f}% left after).")
        else:
            print(f"\n  EXIT: OK — {po_m:.0f}% remaining after exit is acceptable.")

        print(f"\n  CAPTURE: capturing {cr_m:.0f}% of the 40-day swing around each trade.")
        print(f"    pre={ep_m:.0f}%  captured={cr_m:.0f}%  post={po_m:.0f}%  (sum≈{ep_m+cr_m+po_m:.0f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  TRADE TIMING ANALYSIS")
    print("=" * 80)

    df, tl = build_df_and_trades()
    if tl is None or tl.empty:
        print("  ERROR: No trades in trade log.")
        return

    logger.info("Trade log: %d trades", len(tl))
    price_idx = build_price_index(df)

    logger.info("Enriching timing for %d trades ...", len(tl))
    enriched = enrich_timing(tl, price_idx)
    enriched.to_csv(os.path.join(OUT_DIR, "trade_timing_enriched.csv"), index=False)
    logger.info("Saved trade_timing_enriched.csv")

    print_summary(enriched)
    print(f"\n  Full data saved to: {OUT_DIR}/trade_timing_enriched.csv")


if __name__ == "__main__":
    main()
