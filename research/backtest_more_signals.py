#!/usr/bin/env python3
"""
backtest_more_signals.py — Five More Signal Types

SIGNAL A: Turn-of-Month LONG (calendar effect, ~60/yr)
  Last 3 + first 2 trading days of each month → institutional inflow bias.
  Enter LONG at 10:00 if ES is above its 9:30 open. Exit EOD.
  Academic: Ariel (1987), Lakonishok & Smidt (1988).

SIGNAL B: Day Streak + ORB Alignment (~40/yr)
  When prior 3 calendar days all closed in same direction AND today's ORB
  confirms that direction → momentum continuation.
  Structural rationale: 3-day streak = trend, not noise.

SIGNAL C: Overnight Gap Alignment with ORB (~55/yr)
  Gap direction must MATCH the ORB breakout direction.
  Both ES gap AND OR break pointing same way = dual confirmation.
  Tests if adding gap filter to primary improves WR.

SIGNAL D: Session Wick Fill (~45/yr)
  By 12:00 PM, if session low is >0.5% below 9:30 open AND price has
  returned above the 9:30 open → sellers exhausted, LONG entry.
  Opposite: session high >0.5% above open + price below open → SHORT.
  Fires at midday on failed directional days.

SIGNAL E: 15-Minute OR Breakout (~80/yr)
  Use 9:30–9:45 as the OR (narrower = earlier + more signals).
  Both ES + NQ must confirm same direction by 10:00.
  Hypothesis: tighter, faster OR → more directional days captured.

Anti-overfitting:
  - All parameters pre-stated before running
  - IS = 2018-2021, OOS = 2022-2026
  - Single threshold per signal (no grid search)
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25
RTH_S = time(9, 30); RTH_E = time(16, 0)
OR_S  = time(9, 30); OR_E  = time(10, 0); ORB_C = time(10, 30)
EOD   = time(15, 30)
MIN_ES = 8; MAX_ES = 60; MIN_NQ = 30; MAX_NQ = 175
ACCT   = 25_000; RISK = 0.02
IS_END = date(2021, 12, 31); OS_START = date(2022, 1, 1)


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def sh(dr):
    return dr.mean() / dr.std(ddof=1) * np.sqrt(252) if len(dr) > 1 and dr.std() > 0 else 0.0


def prow(lbl, tdf, r_col="r", pad=46):
    if not len(tdf):
        print(f"  — {lbl:<{pad}}  n=0"); return
    wr  = (tdf[r_col] > 0).mean() * 100; ar = tdf[r_col].mean()
    dr  = tdf.groupby("date")[r_col].sum(); s = sh(dr)
    nyr = tdf["date"].nunique() / 8.1
    wins = tdf[tdf[r_col] > 0]; losses = tdf[tdf[r_col] <= 0]
    wl = (abs(wins[r_col].mean() / losses[r_col].mean())
          if len(losses) and losses[r_col].mean() != 0 else 99.0)
    flag = "✅" if (ar > 0 and s > 2.5) else ("🟡" if (ar > 0 and s > 1.5) else "❌")
    print(f"  {flag} {lbl:<{pad}}  {nyr:.0f}/yr  WR={wr:5.1f}%  "
          f"Avg={ar:+.4f}  W/L={wl:.2f}x  Sh={s:.2f}")


def yr_row(tdf, yr, r_col="r"):
    yt = tdf[tdf["year"] == yr]
    if not len(yt): return
    tag  = "IS " if yr <= 2021 else "OOS"
    wr   = (yt[r_col] > 0).mean() * 100; ar = yt[r_col].mean()
    dr   = yt.groupby("date")[r_col].sum(); s = sh(dr)
    flag = "✅" if (ar > 0 and s > 1.5) else ("🟡" if ar > 0 else "❌")
    print(f"  {flag} {yr}: n={len(yt):3d}  WR={wr:5.1f}%  AvgR={ar:+.4f}  Sh={s:.2f} {tag}")


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL A — TURN OF MONTH
# ─────────────────────────────────────────────────────────────────────────────
def get_tom_dates(dates):
    """Last 3 + first 2 trading days of each calendar month."""
    tom = set()
    month_days = {}
    for d in dates:
        k = (d.year, d.month)
        if k not in month_days: month_days[k] = []
        month_days[k].append(d)
    for k, ds in month_days.items():
        ds_sorted = sorted(ds)
        for d in ds_sorted[:2]:   tom.add(d)   # first 2
        for d in ds_sorted[-3:]:  tom.add(d)   # last 3
    return tom


def run_tom(df, dates, by_d, min_move_pct=0.0):
    """
    Turn-of-Month LONG.
    On TOM days: if ES_close_10:00 > ES_open_9:30 (+min_move_pct%),
    enter LONG at 10:00 bar close. Exit at 15:30 EOD.
    Stop: 10:00 bar close − OR_range (same risk as primary system OR).
    """
    tom = get_tom_dates(dates)
    trades = []
    for i, d in enumerate(dates[1:], 1):
        if d not in tom: continue
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 30: continue

        # OR quality
        or_b = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] < OR_E)]
        if not len(or_b): continue
        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min(); es_R = es_H - es_L
        if not (MIN_ES <= es_R <= MAX_ES): continue

        open_930   = or_b.iloc[0]["ES_open"]
        close_1000 = or_b.iloc[-1]["ES_close"]
        move_pct   = (close_1000 - open_930) / open_930 * 100

        if move_pct < min_move_pct: continue   # must be positive (TOM bias is LONG)

        ep   = close_1000 + SLIP
        sp   = ep - es_R - SLIP           # stop = 1 OR range below entry
        risk = abs(ep - sp)
        if risk < 1: continue

        # Exit at EOD
        eod_bar = rth[rth["time_et"] >= EOD]
        if not len(eod_bar): continue

        # Check if stop hit between 10:00 and EOD
        fwd = rth[rth["time_et"] >= OR_E].reset_index(drop=True)
        rv = 0.0; why = "EOD"
        for _, bar in fwd.iterrows():
            if bar["time_et"] >= EOD:
                rv = (bar["ES_close"] - ep - SLIP*2) / risk; why = "EOD"; break
            if bar["ES_low"] <= sp: rv = -1.0; why = "STOP"; break

        ts = pd.Timestamp(d); iso = ts.isocalendar()
        trades.append(dict(
            date=d, year=d.year, direction="LONG", r=rv,
            usd=round(rv*(ACCT*RISK), 2), win=rv > 0,
            exit=why, move_pct=round(move_pct, 3),
            iso_wk=(int(iso.year), int(iso.week)),
        ))
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL B — DAY STREAK + ORB
# ─────────────────────────────────────────────────────────────────────────────
def run_day_streak(df, dates, by_d, streak_days=3, max_break=0.20):
    """
    3-day closing streak + ORB confirmation.
    Requires last N closes all up (for LONG) or all down (for SHORT)
    AND ES + NQ both break the OR in that same direction.
    Standard primary-like entry: breakout bar close, stop=OR range, target=3R.
    """
    # Pre-compute daily closes
    daily_close = {}
    for d in dates:
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= RTH_E)]
        if not len(rth): continue
        daily_close[d] = rth["ES_close"].iloc[-1]

    trades = []
    for i, d in enumerate(dates[streak_days:], streak_days):
        # Check streak
        prev_dates = dates[i-streak_days:i]
        closes = [daily_close.get(pd) for pd in prev_dates]
        if any(c is None for c in closes): continue

        streak_up   = all(closes[j] > closes[j-1] for j in range(1, len(closes)))
        streak_down = all(closes[j] < closes[j-1] for j in range(1, len(closes)))
        if not (streak_up or streak_down): continue

        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 20: continue

        or_b = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] < OR_E)]
        if not len(or_b): continue
        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min(); es_R = es_H - es_L
        nq_or = day[(day["time_et"] >= OR_S) & (day["time_et"] < OR_E)]
        nq_H = nq_or["NQ_high"].max(); nq_L = nq_or["NQ_low"].min(); nq_R = nq_H - nq_L
        if not (MIN_ES <= es_R <= MAX_ES and MIN_NQ <= nq_R <= MAX_NQ): continue

        post    = rth[(rth["time_et"] >= OR_E) & (rth["time_et"] < ORB_C)]
        nq_post = day[(day["time_et"] >= OR_E) & (day["time_et"] < ORB_C)]
        es_b = nq_b = None; entry_row = None; bd_es = bd_nq = 0
        for _, r in post.iterrows():
            if r["time_et"] < time(10, 2): continue
            if es_b is None:
                if r["ES_high"] > es_H: es_b = "L"; bd_es=(r["ES_high"]-es_H)/es_R
                elif r["ES_low"] < es_L: es_b = "S"; bd_es=(es_L-r["ES_low"])/es_R
            nqr = nq_post[nq_post["time_et"] == r["time_et"]]
            if nq_b is None and len(nqr):
                nr = nqr.iloc[0]
                if nr["NQ_high"] > nq_H: nq_b="L"; bd_nq=(nr["NQ_high"]-nq_H)/nq_R
                elif nr["NQ_low"] < nq_L: nq_b="S"; bd_nq=(nq_L-nr["NQ_low"])/nq_R
            if es_b and nq_b: entry_row = r; break

        if not (es_b and nq_b and es_b == nq_b): continue
        if bd_es > max_break or bd_nq > max_break: continue

        di = "LONG" if es_b == "L" else "SHORT"
        if streak_up   and di != "LONG":  continue   # streak up → only LONG
        if streak_down and di != "SHORT": continue   # streak down → only SHORT

        ep = entry_row["ES_close"] + SLIP if di=="LONG" else entry_row["ES_close"] - SLIP
        sp = es_L - SLIP if di=="LONG" else es_H + SLIP
        tp = ep + 3*es_R if di=="LONG" else ep - 3*es_R
        risk = abs(ep - sp)

        fwd = rth[rth["time_et"] > entry_row["time_et"]].reset_index(drop=True)
        rv = 0.0; why = "NONE"
        for _, bar in fwd.iterrows():
            if bar["time_et"] >= EOD:
                pnl = (bar["ES_close"]-ep-SLIP*2) if di=="LONG" else (ep-bar["ES_close"]-SLIP*2)
                rv = pnl/risk; why="EOD"; break
            if di=="LONG":
                if bar["ES_low"]  <= sp: rv=-1.0; why="STOP"; break
                if bar["ES_high"] >= tp: rv=3.0-SLIP/risk; why="TGT"; break
            else:
                if bar["ES_high"] >= sp: rv=-1.0; why="STOP"; break
                if bar["ES_low"]  <= tp: rv=3.0-SLIP/risk; why="TGT"; break

        ts = pd.Timestamp(d); iso = ts.isocalendar()
        trades.append(dict(
            date=d, year=d.year, direction=di, r=rv,
            usd=round(rv*(ACCT*RISK),2), win=rv>0, exit=why,
            iso_wk=(int(iso.year), int(iso.week)),
        ))
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL C — GAP + ORB ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
def run_gap_orb(df, dates, by_d, min_gap_pct=0.10, max_break=0.20):
    """
    Gap direction MUST match ORB breakout direction.
    Gap = (today 9:30 open) / (yesterday 4PM close) - 1.
    Both ES gap AND OR break same direction → dual confirmation.
    Uses primary-like entry (market fill at breakout bar close).
    Both ES + NQ must confirm.
    """
    trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 20: continue

        # Previous day close
        prev = by_d.get(dates[i-1])
        if prev is None: continue
        pr = prev[(prev["time_et"] >= RTH_S) & (prev["time_et"] <= RTH_E)]
        if not len(pr): continue
        prev_close = pr["ES_close"].iloc[-1]

        # Gap
        or_b = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] < OR_E)]
        if not len(or_b): continue
        rth_open = or_b.iloc[0]["ES_open"]
        gap_pct  = (rth_open - prev_close) / prev_close * 100

        if abs(gap_pct) < min_gap_pct: continue  # no meaningful gap

        gap_dir = "L" if gap_pct > 0 else "S"

        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min(); es_R = es_H - es_L
        nq_or = day[(day["time_et"] >= OR_S) & (day["time_et"] < OR_E)]
        nq_H = nq_or["NQ_high"].max(); nq_L = nq_or["NQ_low"].min(); nq_R = nq_H - nq_L
        if not (MIN_ES <= es_R <= MAX_ES and MIN_NQ <= nq_R <= MAX_NQ): continue

        post    = rth[(rth["time_et"] >= OR_E) & (rth["time_et"] < ORB_C)]
        nq_post = day[(day["time_et"] >= OR_E) & (day["time_et"] < ORB_C)]
        es_b = nq_b = None; entry_row = None; bd_es = bd_nq = 0
        for _, r in post.iterrows():
            if r["time_et"] < time(10, 2): continue
            if es_b is None:
                if r["ES_high"] > es_H: es_b="L"; bd_es=(r["ES_high"]-es_H)/es_R
                elif r["ES_low"] < es_L: es_b="S"; bd_es=(es_L-r["ES_low"])/es_R
            nqr = nq_post[nq_post["time_et"] == r["time_et"]]
            if nq_b is None and len(nqr):
                nr = nqr.iloc[0]
                if nr["NQ_high"] > nq_H: nq_b="L"; bd_nq=(nr["NQ_high"]-nq_H)/nq_R
                elif nr["NQ_low"] < nq_L: nq_b="S"; bd_nq=(nq_L-nr["NQ_low"])/nq_R
            if es_b and nq_b: entry_row = r; break

        if not (es_b and nq_b and es_b == nq_b): continue
        if bd_es > max_break or bd_nq > max_break: continue
        if es_b != gap_dir: continue   # ← KEY FILTER: gap must match breakout

        di = "LONG" if es_b == "L" else "SHORT"
        ep = entry_row["ES_close"] + SLIP if di=="LONG" else entry_row["ES_close"] - SLIP
        sp = es_L - SLIP if di=="LONG" else es_H + SLIP
        tp = ep + 3*es_R if di=="LONG" else ep - 3*es_R
        risk = abs(ep - sp)

        fwd = rth[rth["time_et"] > entry_row["time_et"]].reset_index(drop=True)
        rv = 0.0; why = "NONE"
        for _, bar in fwd.iterrows():
            if bar["time_et"] >= EOD:
                pnl = (bar["ES_close"]-ep-SLIP*2) if di=="LONG" else (ep-bar["ES_close"]-SLIP*2)
                rv = pnl/risk; why="EOD"; break
            if di=="LONG":
                if bar["ES_low"]  <= sp: rv=-1.0; why="STOP"; break
                if bar["ES_high"] >= tp: rv=3.0-SLIP/risk; why="TGT"; break
            else:
                if bar["ES_high"] >= sp: rv=-1.0; why="STOP"; break
                if bar["ES_low"]  <= tp: rv=3.0-SLIP/risk; why="TGT"; break

        ts = pd.Timestamp(d); iso = ts.isocalendar()
        trades.append(dict(
            date=d, year=d.year, direction=di, r=rv,
            usd=round(rv*(ACCT*RISK),2), win=rv>0, exit=why,
            gap_pct=round(gap_pct,3),
            iso_wk=(int(iso.year), int(iso.week)),
        ))
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL D — SESSION WICK FILL
# ─────────────────────────────────────────────────────────────────────────────
def run_wick_fill(df, dates, by_d, wick_pct=0.50, min_return=0.0):
    """
    Midday session wick fill.
    At 12:00 PM:
      LONG : session LOW > 0.5% below 9:30 open AND current > 9:30 open
             → selling failed, buyers absorbing → LONG for afternoon continuation
      SHORT: session HIGH > 0.5% above 9:30 open AND current < 9:30 open
             → buying failed, sellers absorbing → SHORT

    Entry  : 12:01 bar close ± slip
    Stop   : session extreme (low for LONG, high for SHORT) - 1pt
    Target : 2R
    """
    trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 30: continue

        # OR quality
        or_b = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] < OR_E)]
        if not len(or_b): continue
        es_R = or_b["ES_high"].max() - or_b["ES_low"].min()
        if not (MIN_ES <= es_R <= MAX_ES): continue

        open_930 = or_b.iloc[0]["ES_open"]

        # Stats at 12:00
        sess_noon = rth[rth["time_et"] <= time(12, 0)]
        if not len(sess_noon): continue
        sess_low   = sess_noon["ES_low"].min()
        sess_high  = sess_noon["ES_high"].max()
        close_1200 = sess_noon.iloc[-1]["ES_close"]

        down_wick_pct = (open_930 - sess_low) / open_930 * 100
        up_wick_pct   = (sess_high - open_930) / open_930 * 100

        # Entry at 12:01
        entry_bar = rth[rth["time_et"] == time(12, 1)]
        if not len(entry_bar): continue
        ep_raw = entry_bar.iloc[0]["ES_close"]

        di = None
        if down_wick_pct >= wick_pct and close_1200 > open_930:
            di = "LONG"   # sellers failed: price went below open but came back
        elif up_wick_pct >= wick_pct and close_1200 < open_930:
            di = "SHORT"  # buyers failed: price went above open but came back

        if di is None: continue

        ep = ep_raw + SLIP if di=="LONG" else ep_raw - SLIP
        sp = sess_low  - 1.0 - SLIP if di=="LONG" else sess_high + 1.0 + SLIP
        risk = abs(ep - sp)
        if risk < 1 or risk > 30: continue
        tp = ep + 2*risk if di=="LONG" else ep - 2*risk

        fwd = rth[rth["time_et"] > time(12, 1)].reset_index(drop=True)
        rv = 0.0; why = "NONE"
        for _, bar in fwd.iterrows():
            if bar["time_et"] >= EOD:
                pnl = (bar["ES_close"]-ep-SLIP*2) if di=="LONG" else (ep-bar["ES_close"]-SLIP*2)
                rv = pnl/risk; why="EOD"; break
            if di=="LONG":
                if bar["ES_low"]  <= sp: rv=-1.0; why="STOP"; break
                if bar["ES_high"] >= tp: rv=2.0-SLIP/risk; why="TGT"; break
            else:
                if bar["ES_high"] >= sp: rv=-1.0; why="STOP"; break
                if bar["ES_low"]  <= tp: rv=2.0-SLIP/risk; why="TGT"; break

        ts = pd.Timestamp(d); iso = ts.isocalendar()
        trades.append(dict(
            date=d, year=d.year, direction=di, r=rv,
            usd=round(rv*(ACCT*RISK),2), win=rv>0, exit=why,
            wick=round(max(down_wick_pct, up_wick_pct), 3),
            iso_wk=(int(iso.year), int(iso.week)),
        ))
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL E — 15-MINUTE OR BREAKOUT
# ─────────────────────────────────────────────────────────────────────────────
def run_15min_orb(df, dates, by_d, max_break=0.20):
    """
    15-minute OR breakout.
    OR = 9:30–9:45 (first 15 bars). Breakout watch = 9:46–10:15.
    Both ES + NQ must confirm same direction.
    Standard entry: breakout bar close, stop=OR range, target=3R.
    """
    OR15_E = time(9, 45); ORB15_C = time(10, 15)
    trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 20: continue

        or15 = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] <= OR15_E)]
        if not len(or15): continue
        es_H = or15["ES_high"].max(); es_L = or15["ES_low"].min(); es_R = es_H - es_L
        nq_or15 = day[(day["time_et"] >= OR_S) & (day["time_et"] <= OR15_E)]
        nq_H = nq_or15["NQ_high"].max(); nq_L = nq_or15["NQ_low"].min(); nq_R = nq_H - nq_L

        if not (MIN_ES <= es_R <= MAX_ES and MIN_NQ <= nq_R <= MAX_NQ): continue

        post    = rth[(rth["time_et"] > OR15_E) & (rth["time_et"] <= ORB15_C)]
        nq_post = day[(day["time_et"] > OR15_E) & (day["time_et"] <= ORB15_C)]
        es_b = nq_b = None; entry_row = None; bd_es = bd_nq = 0
        for _, r in post.iterrows():
            if es_b is None:
                if r["ES_high"] > es_H: es_b="L"; bd_es=(r["ES_high"]-es_H)/es_R
                elif r["ES_low"] < es_L: es_b="S"; bd_es=(es_L-r["ES_low"])/es_R
            nqr = nq_post[nq_post["time_et"] == r["time_et"]]
            if nq_b is None and len(nqr):
                nr = nqr.iloc[0]
                if nr["NQ_high"] > nq_H: nq_b="L"; bd_nq=(nr["NQ_high"]-nq_H)/nq_R
                elif nr["NQ_low"] < nq_L: nq_b="S"; bd_nq=(nq_L-nr["NQ_low"])/nq_R
            if es_b and nq_b: entry_row = r; break

        if not (es_b and nq_b and es_b == nq_b): continue
        if bd_es > max_break or bd_nq > max_break: continue

        di = "LONG" if es_b == "L" else "SHORT"
        ep = entry_row["ES_close"] + SLIP if di=="LONG" else entry_row["ES_close"] - SLIP
        sp = es_L - SLIP if di=="LONG" else es_H + SLIP
        tp = ep + 3*es_R if di=="LONG" else ep - 3*es_R
        risk = abs(ep - sp)

        fwd = rth[rth["time_et"] > entry_row["time_et"]].reset_index(drop=True)
        rv = 0.0; why = "NONE"
        for _, bar in fwd.iterrows():
            if bar["time_et"] >= EOD:
                pnl = (bar["ES_close"]-ep-SLIP*2) if di=="LONG" else (ep-bar["ES_close"]-SLIP*2)
                rv = pnl/risk; why="EOD"; break
            if di=="LONG":
                if bar["ES_low"]  <= sp: rv=-1.0; why="STOP"; break
                if bar["ES_high"] >= tp: rv=3.0-SLIP/risk; why="TGT"; break
            else:
                if bar["ES_high"] >= sp: rv=-1.0; why="STOP"; break
                if bar["ES_low"]  <= tp: rv=3.0-SLIP/risk; why="TGT"; break

        ts = pd.Timestamp(d); iso = ts.isocalendar()
        trades.append(dict(
            date=d, year=d.year, direction=di, r=rv,
            usd=round(rv*(ACCT*RISK),2), win=rv>0, exit=why,
            es_R=round(es_R,1),
            iso_wk=(int(iso.year), int(iso.week)),
        ))
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading …")
    df    = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    W = 72
    results = {}

    # ── RUN ALL ──────────────────────────────────────────────────────────────
    for name, fn, kwargs in [
        ("A: Turn of Month",    run_tom,         {}),
        ("B: Day Streak + ORB", run_day_streak,  {}),
        ("C: Gap + ORB Align",  run_gap_orb,     {}),
        ("D: Session Wick Fill",run_wick_fill,   {}),
        ("E: 15-min OR Break",  run_15min_orb,   {}),
    ]:
        print(f"  Running {name} …", end=" ", flush=True)
        t = fn(df, dates, by_d, **kwargs)
        results[name] = t
        print(f"{len(t)} trades ({len(t)/8.1:.0f}/yr)")

    print()

    # ── INDIVIDUAL RESULTS ────────────────────────────────────────────────────
    for name, t in results.items():
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        print("=" * W)
        print(f"  {name}")
        print("=" * W)
        print(f"  Total: {len(t)} trades ({len(t)/8.1:.0f}/yr)\n")
        prow("Full period  2018-2026", t)
        prow("IS  2018-2021",          IS_t)
        prow("OOS 2022-2026",          OOS_t)

        if len(t):
            # Direction
            for di in ["LONG", "SHORT"]:
                sub = t[t["direction"] == di] if "direction" in t.columns else pd.DataFrame()
                if len(sub): prow(f"  {di}", sub)

            print()
            for yr in sorted(t["year"].unique()): yr_row(t, yr)
        print()

    # ── RANKING ───────────────────────────────────────────────────────────────
    print("=" * W)
    print("  RANKING — ALL 5 NEW SIGNALS")
    print("=" * W)
    print(f"\n  {'Signal':<30} {'n/yr':>5}  {'WR':>7}  {'OOS Sh':>8}  {'IS Sh':>7}  verdict")
    print("  " + "─" * 70)

    rows = []
    for name, t in results.items():
        if not len(t): continue
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        nyr  = len(t) / 8.1
        wr   = (t["r"] > 0).mean() * 100
        ar   = t["r"].mean()
        s_is = sh(IS_t.groupby("date")["r"].sum()) if len(IS_t) else 0
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        if ar > 0 and s_oo > 2.5:   v = "✅ ADD"
        elif ar > 0 and s_oo > 1.5: v = "🟡 MARGINAL"
        else:                        v = "❌ SKIP"
        rows.append((name, nyr, wr, s_is, s_oo, v, ar))

    rows.sort(key=lambda x: -x[4])
    for name, nyr, wr, s_is, s_oo, v, ar in rows:
        print(f"  {v[:2]} {name:<30} {nyr:>5.0f}/yr  {wr:>6.1f}%  "
              f"OOS={s_oo:>5.2f}  IS={s_is:>5.2f}  {v}")

    # ── COMPARE WITH KNOWN WINNERS ────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  CONTEXT vs PREVIOUSLY VALIDATED SIGNALS")
    print(f"  {'─' * 70}")
    ref = [
        ("Double Test at Level", 65, 38.7, 3.42, 4.61, "✅ In system"),
        ("Primary ORB",          18, 60.6, 4.94, 3.48, "✅ In system"),
        ("MTF Scalp",           127, 44.7, 2.68, 2.03, "✅ In system"),
    ]
    print(f"\n  {'Signal':<30} {'n/yr':>5}  {'WR':>7}  {'IS Sh':>7}  {'OOS Sh':>8}  status")
    print("  " + "─" * 70)
    for name, nyr, wr, s_is, s_oo, status in ref:
        print(f"  {name:<30} {nyr:>5}/yr  {wr:>6.1f}%  IS={s_is:>5.2f}  OOS={s_oo:>5.2f}  {status}")
    print()
    for name, nyr, wr, s_is, s_oo, v, ar in rows:
        vs = "✅ ADD" if v.startswith("✅") else ("🟡 MARGINAL" if v.startswith("🟡") else "❌ SKIP")
        print(f"  {name:<30} {nyr:>5.0f}/yr  {wr:>6.1f}%  IS={s_is:>5.2f}  OOS={s_oo:>5.2f}  {vs}")
    print()


if __name__ == "__main__":
    main()
