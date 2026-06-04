#!/usr/bin/env python3
"""
backtest_cl_improved.py — CL/MCL Improved System

Compares 7 approaches against the original 20-bar rolling breakout:
  A. Original   : 20-bar rolling breakout, fixed 10t stop / 20t target
  B. OR-only    : Fixed 9:30-10:00 ET opening range breakout
  C. OR + PDH/L : OR breakout confirmed by prior-day CL high/low
  D. OR + PWH/PWL: OR breakout confirmed by prior-week CL high/low
  E. Full struct : OR + PDH/L + PWH/PWL combined (mirrors ES/NQ system)
  F. ES/NQ align : Full struct + ES/NQ direction filter
  G. ATR stops  : Full struct + ATR-based adaptive stops/targets

CL data  : CL_1min_continuous.parquet (Jun 2021 – Apr 2026)
ES/NQ    : ES_NQ_1min_aligned.parquet (for alignment filter)
"""

import numpy as np
import pandas as pd
from datetime import time, date, timedelta
from collections import deque

CL_DATA   = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/CL_1min_continuous.parquet"
ESNG_DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"

# ── Session windows (ET) ──────────────────────────────────────────────────────
RTH_START   = time(9, 30)    # CL active session start (aligns with ES/NQ)
RTH_END     = time(14, 30)   # CL NYMEX floor session end
OR_START    = time(9, 30)    # Opening range start
OR_END      = time(10, 0)    # Opening range end
ENTRY_END   = time(11, 0)    # No entries after this
FLAT_TIME   = time(14, 0)    # Hard flatten for CL (before floor close)

# ── Original system parameters ────────────────────────────────────────────────
ORIG_WINDOW = 20             # rolling bars for breakout
ORIG_VOL_WINDOW = 60         # rolling bars for volume pct
ORIG_VOL_PCT = 70            # percentile threshold
ORIG_ENTRY_START = time(9, 5)   # 08:05 CT = 09:05 ET
ORIG_ENTRY_END   = time(11, 44) # 10:44 CT = 11:44 ET
ORIG_FLAT        = time(12, 0)  # 11:00 CT = 12:00 ET
ORIG_STOP_T  = 10            # ticks
ORIG_TGT_T   = 20            # ticks
ORIG_TIME_STOP_MIN = 35      # minutes

# ── Improved system parameters ────────────────────────────────────────────────
TICK = 0.01                  # CL tick size in $/bbl
PMULT = 3.0                  # 3R target (same as ES/NQ system)
STOP_R = 1.0                 # 1R stop (OR range)
SLIP  = 0.02                 # 2 ticks slippage per side
ATR_PERIOD = 14              # ATR lookback
ATR_STOP_MULT = 0.8          # stop = 0.8 × ATR (tight)
ATR_TGT_MULT  = 2.0          # target = 2.0 × ATR


def load_data():
    print("Loading CL data …")
    cl = pd.read_parquet(CL_DATA)
    cl["ts_et"]   = pd.to_datetime(cl["ts_et"])
    if cl["ts_et"].dt.tz is None:
        cl["ts_et"] = cl["ts_et"].dt.tz_localize("America/New_York")
    else:
        cl["ts_et"] = cl["ts_et"].dt.tz_convert("America/New_York")
    cl["date_et"] = cl["ts_et"].dt.date
    cl["time_et"] = cl["ts_et"].dt.time
    cl = cl.sort_values("ts_et").reset_index(drop=True)
    print(f"  CL: {len(cl):,} bars  {cl['date_et'].min()} → {cl['date_et'].max()}")

    print("Loading ES/NQ data …")
    en = pd.read_parquet(ESNG_DATA)
    en["ts_et"]   = pd.to_datetime(en["timestamp"]).dt.tz_convert("America/New_York")
    en["date_et"] = en["ts_et"].dt.date
    en["time_et"] = en["ts_et"].dt.time
    en = en.sort_values("ts_et").reset_index(drop=True)
    print(f"  ES/NQ: {len(en):,} bars  {en['date_et'].min()} → {en['date_et'].max()}")

    return cl, en


def compute_weekly_hl_cl(by_d, dates):
    """Prior week CL RTH H/L."""
    weekly = {}
    for d in dates:
        d_ts = pd.Timestamp(d)
        iso  = d_ts.isocalendar()
        key  = (int(iso.year), int(iso.week))
        day  = by_d.get(d)
        if day is None: continue
        rth  = day[(day["time_et"] >= RTH_START) & (day["time_et"] <= RTH_END)]
        if len(rth) == 0: continue
        h = rth["CL_high"].max(); l = rth["CL_low"].min()
        if key not in weekly:
            weekly[key] = [h, l]
        else:
            weekly[key][0] = max(weekly[key][0], h)
            weekly[key][1] = min(weekly[key][1], l)
    return weekly


def get_pwhl(d, weekly):
    d_ts = pd.Timestamp(d)
    iso  = d_ts.isocalendar()
    pw   = int(iso.week) - 1
    yr   = int(iso.year)
    if pw == 0:
        yr -= 1
        pw = int(pd.Timestamp(f"{yr}-12-28").isocalendar().week)
    key = (yr, pw)
    if key in weekly:
        return weekly[key][0], weekly[key][1]
    return None, None


def sim_trade_cl(day, entry_t, direction, entry_px, stop_px, tgt_px, flat_time,
                 time_stop_mins=None):
    """Simulate a CL trade. Returns (r_val, exit_reason)."""
    rng = abs(entry_px - stop_px)
    after = day[day["time_et"] > entry_t]
    entry_dt = None
    for _, bar in after.iterrows():
        t = bar["time_et"]
        if t >= flat_time:
            sign = 1 if direction == "LONG" else -1
            net  = sign * (bar["CL_close"] - entry_px) - SLIP * 2
            return (net / rng) if rng > 0 else 0.0, "FLAT"
        if time_stop_mins is not None and entry_dt is None:
            entry_dt = bar.name
        if direction == "LONG":
            if bar["CL_low"]  <= stop_px: return (-1.0 - SLIP*2/rng), "STOP"
            if bar["CL_high"] >= tgt_px:  return (PMULT  - SLIP/rng),  "TGT"
        else:
            if bar["CL_high"] >= stop_px: return (-1.0 - SLIP*2/rng), "STOP"
            if bar["CL_low"]  <= tgt_px:  return (PMULT  - SLIP/rng),  "TGT"
    return 0.0, "NONE"


# ─── SYSTEM A: Original 20-bar rolling breakout ───────────────────────────────
def run_original(cl, dates, by_d):
    trades = []
    for d in dates:
        day = by_d.get(d)
        if day is None: continue
        sess = day[(day["time_et"] >= time(6,0)) & (day["time_et"] <= ORIG_FLAT)]
        if len(sess) < ORIG_WINDOW + 5: continue

        highs  = sess["CL_high"].values
        lows   = sess["CL_low"].values
        vols   = sess["CL_volume"].values
        times  = sess["time_et"].values

        long_fired = short_fired = False

        for i in range(ORIG_WINDOW, len(sess)):
            t = times[i]
            if t < ORIG_ENTRY_START: continue
            if t >= ORIG_ENTRY_END: break

            roll_h = highs[i-ORIG_WINDOW:i].max()
            roll_l = lows[i-ORIG_WINDOW:i].min()
            cur_h  = highs[i]; cur_l = lows[i]

            vol_start = max(0, i - ORIG_VOL_WINDOW)
            vol_hist  = vols[vol_start:i]
            vol_thresh = np.percentile(vol_hist, ORIG_VOL_PCT) if len(vol_hist) >= 10 else 0
            vol_ok    = vols[i] >= vol_thresh

            entry_px = sess["CL_close"].values[i]
            after    = day[day["time_et"] > t]

            if not long_fired and cur_h > roll_h and vol_ok:
                long_fired = True
                tgt  = entry_px + ORIG_TGT_T * TICK
                stop = entry_px - ORIG_STOP_T * TICK
                rv, why = sim_trade_cl(day, t, "LONG", entry_px, stop, tgt,
                                       ORIG_FLAT, ORIG_TIME_STOP_MIN)
                r_norm = rv * (ORIG_TGT_T * TICK) / abs(entry_px - stop) if abs(entry_px-stop) > 0 else rv
                trades.append(dict(date=d, year=d.year, dir="LONG",
                                   r=r_norm, win=r_norm>0, exit=why, sys="A_ORIG"))

            if not short_fired and cur_l < roll_l and vol_ok:
                short_fired = True
                tgt  = entry_px - ORIG_TGT_T * TICK
                stop = entry_px + ORIG_STOP_T * TICK
                rv, why = sim_trade_cl(day, t, "SHORT", entry_px, stop, tgt,
                                       ORIG_FLAT, ORIG_TIME_STOP_MIN)
                r_norm = rv * (ORIG_TGT_T * TICK) / abs(entry_px - stop) if abs(entry_px-stop) > 0 else rv
                trades.append(dict(date=d, year=d.year, dir="SHORT",
                                   r=r_norm, win=r_norm>0, exit=why, sys="A_ORIG"))

    return pd.DataFrame(trades)


# ─── SYSTEMS B-G: Structural approach ────────────────────────────────────────
def run_structural(cl, en, dates, by_d, weekly,
                   use_pdhl=False, use_pwhl=False,
                   use_esng_filter=False, use_atr=False,
                   sys_label="B"):

    # Pre-compute daily ES/NQ OR direction for alignment filter
    esng_dir = {}
    if use_esng_filter and en is not None:
        en_by_d = {d: g for d, g in en.groupby("date_et")}
        for d in dates:
            eday = en_by_d.get(d)
            if eday is None: continue
            or_b = eday[(eday["time_et"] >= OR_START) & (eday["time_et"] < OR_END)]
            if len(or_b) == 0: continue
            es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min()
            nq_H = or_b["NQ_high"].max(); nq_L = or_b["NQ_low"].min()
            post = eday[(eday["time_et"] >= OR_END) & (eday["time_et"] < time(10,30))]
            es_b = nq_b = None
            for _, r in post.iterrows():
                if es_b is None:
                    if r["ES_high"] > es_H: es_b = "L"
                    elif r["ES_low"] < es_L: es_b = "S"
                if nq_b is None:
                    if r["NQ_high"] > nq_H: nq_b = "L"
                    elif r["NQ_low"] < nq_L: nq_b = "S"
                if es_b and nq_b: break
            if es_b and nq_b and es_b == nq_b:
                esng_dir[d] = "LONG" if es_b == "L" else "SHORT"

    trades = []

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue

        # Opening range
        or_b = day[(day["time_et"] >= OR_START) & (day["time_et"] < OR_END)]
        if len(or_b) < 5: continue
        or_H = or_b["CL_high"].max()
        or_L = or_b["CL_low"].min()
        or_R = or_H - or_L
        if or_R < 0.10: continue   # min CL OR range 10 ticks

        # ATR
        if use_atr:
            rth_hist = day[(day["time_et"] >= time(9,0)) & (day["time_et"] < OR_END)]
            if len(rth_hist) < ATR_PERIOD: continue
            h = rth_hist["CL_high"].values; l = rth_hist["CL_low"].values
            tr = np.maximum(h[1:]-l[1:], np.abs(h[1:]-l[:-1]), np.abs(l[1:]-h[:-1]))
            atr = tr[-ATR_PERIOD:].mean() if len(tr) >= ATR_PERIOD else or_R
            stop_dist = atr * ATR_STOP_MULT
            tgt_mult_r = ATR_TGT_MULT * atr / or_R   # expressed as R-multiples
        else:
            stop_dist  = or_R          # 1R stop = 1 OR range
            tgt_mult_r = PMULT         # 3R target

        # Prior day H/L
        if use_pdhl:
            prev = by_d.get(dates[i-1])
            if prev is None: continue
            prev_rth = prev[(prev["time_et"] >= RTH_START) & (prev["time_et"] <= RTH_END)]
            if len(prev_rth) == 0: continue
            PDH = prev_rth["CL_high"].max()
            PDL = prev_rth["CL_low"].min()
        else:
            PDH = PDL = None

        # Prior week H/L
        if use_pwhl:
            PWH, PWL = get_pwhl(d, weekly)
        else:
            PWH = PWL = None

        # Watch window: 10:00–11:00 ET
        watch = day[(day["time_et"] >= OR_END) & (day["time_et"] < ENTRY_END)]

        long_fired = short_fired = False

        for _, bar in watch.iterrows():
            t = bar["time_et"]

            # OR breakout
            orb_long  = bar["CL_high"] > or_H
            orb_short = bar["CL_low"]  < or_L

            # PDH/L confirmation
            if PDH:
                pdhl_long  = bar["CL_high"] > PDH
                pdhl_short = bar["CL_low"]  < PDL
            else:
                pdhl_long = pdhl_short = True   # not required

            # PWH/PWL confirmation
            if PWH:
                pwhl_long  = bar["CL_high"] > PWH
                pwhl_short = bar["CL_low"]  < PWL
            else:
                pwhl_long = pwhl_short = True   # not required

            # ES/NQ alignment
            esng_ok_long  = True
            esng_ok_short = True
            if use_esng_filter:
                esng = esng_dir.get(d)
                if esng == "LONG":  esng_ok_short = False
                elif esng == "SHORT": esng_ok_long  = False

            # LONG signal
            if (not long_fired and orb_long and pdhl_long and
                    pwhl_long and esng_ok_long):
                long_fired = True
                if PDH: ep = PDH
                elif PWH: ep = PWH
                else: ep = or_H
                sp = ep - stop_dist
                tp = ep + stop_dist * tgt_mult_r
                rv, why = sim_trade_cl(day, t, "LONG", ep, sp, tp, FLAT_TIME)
                trades.append(dict(date=d, year=d.year, dir="LONG",
                                   r=rv, win=rv>0, exit=why, sys=sys_label))

            # SHORT signal
            if (not short_fired and orb_short and pdhl_short and
                    pwhl_short and esng_ok_short):
                short_fired = True
                if PDL: ep = PDL
                elif PWL: ep = PWL
                else: ep = or_L
                sp = ep + stop_dist
                tp = ep - stop_dist * tgt_mult_r
                rv, why = sim_trade_cl(day, t, "SHORT", ep, sp, tp, FLAT_TIME)
                trades.append(dict(date=d, year=d.year, dir="SHORT",
                                   r=rv, win=rv>0, exit=why, sys=sys_label))

    return pd.DataFrame(trades)


def stats(tdf, label):
    if len(tdf) == 0:
        return dict(label=label, n=0, wr=0, avg_r=0, ev=0, sharpe=0, pf=0)
    wr    = tdf["win"].mean()
    avg_r = tdf["r"].mean()
    ev    = wr * PMULT - (1 - wr)
    wins  = tdf[tdf["win"]]["r"].sum()
    loss  = abs(tdf[~tdf["win"]]["r"].sum())
    pf    = wins / loss if loss > 0 else 999
    dr    = tdf.groupby("date")["r"].sum()
    sh    = dr.mean() / dr.std(ddof=1) * np.sqrt(252) if dr.std() > 0 else 0
    return dict(label=label, n=len(tdf), n_days=tdf["date"].nunique(),
                wr=wr, avg_r=avg_r, ev=ev, sharpe=round(sh, 3),
                pf=round(pf, 2), tdf=tdf)


def print_stats(s, pad=38):
    print(f"  {s['label']:<{pad}}  n={s['n']:5d}  WR={s['wr']*100:5.1f}%  "
          f"AvgR={s['avg_r']:+.3f}  EV={s['ev']:+.3f}  "
          f"Sharpe={s['sharpe']:5.3f}  PF={s['pf']:.2f}x")


def main():
    cl, en = load_data()
    dates = sorted(cl["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in cl.groupby("date_et")}

    print("Pre-computing weekly H/L …")
    weekly = compute_weekly_hl_cl(by_d, dates)
    print(f"  {len(weekly)} weeks\n")

    W = 76
    def sec(t): print("\n" + "="*W + f"\n  {t}\n" + "="*W)

    # Run all systems
    print("Running all systems …")
    print("  A: Original 20-bar rolling …")
    A = stats(run_original(cl, dates, by_d),          "A. Original (20-bar rolling)")
    print("  B: OR only …")
    B = stats(run_structural(cl, en, dates, by_d, weekly,
                             sys_label="B"),            "B. OR breakout only")
    print("  C: OR + PDH/L …")
    C = stats(run_structural(cl, en, dates, by_d, weekly,
                             use_pdhl=True, sys_label="C"), "C. OR + PDH/L")
    print("  D: OR + PWH/PWL …")
    D = stats(run_structural(cl, en, dates, by_d, weekly,
                             use_pwhl=True, sys_label="D"), "D. OR + PWH/PWL")
    print("  E: Full structural (OR+PDH/L+PWH/PWL) …")
    E = stats(run_structural(cl, en, dates, by_d, weekly,
                             use_pdhl=True, use_pwhl=True,
                             sys_label="E"),           "E. Full struct (OR+PDH/L+PWH/PWL)")
    print("  F: Full struct + ES/NQ filter …")
    F = stats(run_structural(cl, en, dates, by_d, weekly,
                             use_pdhl=True, use_pwhl=True,
                             use_esng_filter=True,
                             sys_label="F"),           "F. Full struct + ES/NQ align")
    print("  G: Full struct + ATR stops …")
    G = stats(run_structural(cl, en, dates, by_d, weekly,
                             use_pdhl=True, use_pwhl=True,
                             use_atr=True, sys_label="G"), "G. Full struct + ATR stops")

    all_sys = [A, B, C, D, E, F, G]

    # ── Main comparison ───────────────────────────────────────────────────────
    sec("1. SYSTEM COMPARISON  (Jun 2021 – Apr 2026)")
    print(f"  {'System':<38}  {'Trades':>6}  {'Days':>5}  {'WR':>6}  "
          f"{'AvgR':>7}  {'EV':>7}  {'Sharpe':>7}  {'PF':>6}")
    print(f"  {'─'*76}")
    for s in all_sys:
        print_stats(s)

    # ── Year-by-year ──────────────────────────────────────────────────────────
    sec("2. YEAR-BY-YEAR  (A=Original vs E=Full Structural)")
    print(f"  {'Year':<6}  {'A: WR':>7}  {'A: PnL/trade':>13}  │  "
          f"{'E: WR':>7}  {'E: PnL/trade':>13}  {'Change':>8}")
    print(f"  {'─'*64}")
    for yr in sorted(A["tdf"]["year"].unique()):
        a_sub = A["tdf"][A["tdf"]["year"]==yr]
        e_sub = E["tdf"][E["tdf"]["year"]==yr]
        a_wr = a_sub["win"].mean()*100 if len(a_sub) else 0
        e_wr = e_sub["win"].mean()*100 if len(e_sub) else 0
        a_ar = a_sub["r"].mean() if len(a_sub) else 0
        e_ar = e_sub["r"].mean() if len(e_sub) else 0
        flag = " ✅" if e_wr > a_wr else " ⚠️"
        print(f"  {yr:<6}  {a_wr:>6.1f}%  {a_ar:>+12.3f}R  │  "
              f"{e_wr:>6.1f}%  {e_ar:>+12.3f}R  {e_wr-a_wr:>+6.1f}pp{flag}")

    # ── 2025 deterioration fix ────────────────────────────────────────────────
    sec("3. 2025 DETERIORATION — Did structural approach fix it?")
    for s in all_sys:
        sub = s["tdf"][s["tdf"]["year"]==2025] if len(s["tdf"]) else pd.DataFrame()
        wr  = sub["win"].mean()*100 if len(sub) else 0
        ar  = sub["r"].mean() if len(sub) else 0
        n   = len(sub)
        fix = " ✅ FIXED" if wr >= 55 else (" 🟡 IMPROVED" if wr > 47.9 else " ❌ STILL WEAK")
        print(f"  {s['label']:<38}  n={n:4d}  WR={wr:5.1f}%  AvgR={ar:+.3f}R{fix}")

    # ── Exit reason ───────────────────────────────────────────────────────────
    sec("4. EXIT REASON BREAKDOWN  (E = Full Structural)")
    for reason in ["TGT","STOP","FLAT","NONE"]:
        sub = E["tdf"][E["tdf"]["exit"]==reason]
        if len(sub)==0: continue
        wr  = sub["win"].mean()*100
        ar  = sub["r"].mean()
        pct = len(sub)/len(E["tdf"])*100
        print(f"  {reason:<6}  {len(sub):>5} trades ({pct:5.1f}%)  "
              f"WR={wr:5.1f}%  AvgR={ar:+.3f}R")

    # ── Streak and drawdown ───────────────────────────────────────────────────
    sec("5. RISK PROFILE  (A=Original vs E=Full Structural)")
    for s in [A, E]:
        dr   = s["tdf"].groupby("date")["r"].sum()
        dd   = (dr.cumsum() - dr.cumsum().cummax()).min()
        streak = max_loss = cur = 0
        for w in s["tdf"]["win"]:
            if not w: cur += 1; max_loss = max(max_loss, cur)
            else: cur = 0
        print(f"  {s['label']:<38}  MaxDD={dd:.2f}R  "
              f"MaxConsecLoss={max_loss}  Sharpe={s['sharpe']:.3f}")

    # ── Signal frequency ──────────────────────────────────────────────────────
    sec("6. SIGNAL FREQUENCY PER WEEK")
    total_weeks = len(dates) / 5
    for s in all_sys:
        n_days = s["tdf"]["date"].nunique() if len(s["tdf"]) else 0
        yrs    = (dates[-1] - dates[0]).days / 365.25
        pw     = n_days / yrs / 52
        print(f"  {s['label']:<38}  {n_days:4d} signal days  "
              f"~{n_days/yrs:.0f}/yr  ~{pw:.2f}/wk")

    # ── Recommendation ────────────────────────────────────────────────────────
    best = max([B,C,D,E,F,G], key=lambda s: s["sharpe"])
    sec("7. RECOMMENDATION")
    print(f"  Best system     : {best['label']}")
    print(f"  Sharpe          : {best['sharpe']:.3f}  (vs original: {A['sharpe']:.3f})")
    wr_imp = (best['wr'] - A['wr']) * 100
    print(f"  WR improvement  : {wr_imp:+.1f}pp  ({A['wr']*100:.1f}% → {best['wr']*100:.1f}%)")
    print(f"  2025 WR         : "
          f"{best['tdf'][best['tdf']['year']==2025]['win'].mean()*100:.1f}%"
          f"  (vs original: 47.9%)")
    print()
    print("  Dollar P&L at practical sizes (using best system AvgR):")
    for contracts, pt in [(1,1),(3,3),(5,5),(10,10)]:
        avg_r_usd = best["avg_r"] * 10 * contracts   # 10 ticks = 1R at MCL
        annual    = avg_r_usd * (best["n"] / 4.8)
        weekly    = annual / 52
        print(f"    {contracts:>2} MCL  AvgR/trade=${avg_r_usd:+.2f}  "
              f"~${annual:+.0f}/yr  ~${weekly:+.0f}/wk")
    print()


if __name__ == "__main__":
    main()
