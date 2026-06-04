#!/usr/bin/env python3
"""
backtest_cl_regime.py — Regime filters for CL OR+PDH/L system

Root cause of 2024-2025 decay: OR CoV (coefficient of variation) rose from
0.45 to 0.92, meaning opening ranges became wildly erratic → breakouts unreliable.

Tests 5 regime filters + combinations:
  F1. OR CoV filter          : skip if rolling 20-day OR CoV > threshold
  F2. OR/ATR ratio filter    : skip if OR < X% of 14-day ATR (OR too small)
  F3. OR absolute floor      : skip if OR < rolling 30th percentile (weak day)
  F4. Adaptive win-rate      : reduce size when recent 15-trade WR < threshold
  F5. Combined best          : F1 + F2 combined

Base system: OR + PDH/L (Tier 1 + Tier 3 combined — best trade-off from frequency analysis)
"""

import numpy as np
import pandas as pd
from datetime import time
from collections import deque

CL_DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/CL_1min_continuous.parquet"

RTH_START = time(9,30); RTH_END = time(14,30)
OR_START  = time(9,30); OR_END  = time(10,0)
ENTRY_END = time(11,0); FLAT    = time(14,0)
SLIP = 0.02; PMULT = 3.0
ATR_PERIOD = 14; COV_PERIOD = 20; FLOOR_PERIOD = 30


def load():
    cl = pd.read_parquet(CL_DATA)
    cl["ts_et"] = pd.to_datetime(cl["ts_et"])
    if cl["ts_et"].dt.tz is None:
        cl["ts_et"] = cl["ts_et"].dt.tz_localize("America/New_York")
    else:
        cl["ts_et"] = cl["ts_et"].dt.tz_convert("America/New_York")
    cl["date_et"] = cl["ts_et"].dt.date
    cl["time_et"] = cl["ts_et"].dt.time
    return cl.sort_values("ts_et").reset_index(drop=True)


def weekly_hl(by_d, dates):
    w = {}
    for d in dates:
        ts = pd.Timestamp(d); iso = ts.isocalendar()
        key = (int(iso.year), int(iso.week))
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"]>=RTH_START)&(day["time_et"]<=RTH_END)]
        if not len(rth): continue
        h = rth["CL_high"].max(); l = rth["CL_low"].min()
        if key not in w: w[key] = [h, l]
        else: w[key][0]=max(w[key][0],h); w[key][1]=min(w[key][1],l)
    return w


def sim(day, entry_t, direction, ep, sp, tp):
    rng = abs(ep - sp)
    for _, bar in day[day["time_et"] > entry_t].iterrows():
        t = bar["time_et"]
        if t >= FLAT:
            s = 1 if direction == "LONG" else -1
            return round((s*(bar["CL_close"]-ep) - SLIP*2)/rng, 3), "FLAT"
        if direction == "LONG":
            if bar["CL_low"]  <= sp: return round(-1.0-SLIP*2/rng, 3), "STOP"
            if bar["CL_high"] >= tp: return round(PMULT-SLIP/rng,   3), "TGT"
        else:
            if bar["CL_high"] >= sp: return round(-1.0-SLIP*2/rng, 3), "STOP"
            if bar["CL_low"]  <= tp: return round(PMULT-SLIP/rng,   3), "TGT"
    return 0.0, "NONE"


def build_daily_stats(cl, dates, by_d):
    """Pre-compute daily OR range, ATR, CoV for every trading day."""
    stats = {}
    or_hist  = deque(maxlen=COV_PERIOD)
    atr_hist = deque(maxlen=ATR_PERIOD)
    or_floor_hist = deque(maxlen=FLOOR_PERIOD)

    for d in dates:
        day = by_d.get(d)
        if day is None: continue

        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        rth  = day[(day["time_et"]>=RTH_START)&(day["time_et"]<=RTH_END)]
        if len(or_b) < 5 or len(rth) < 5: continue

        or_H = or_b["CL_high"].max(); or_L = or_b["CL_low"].min()
        or_R = or_H - or_L
        rth_R = rth["CL_high"].max() - rth["CL_low"].min()

        # Rolling CoV of OR ranges
        cov = (np.std(list(or_hist))/np.mean(list(or_hist))
               if len(or_hist) >= 10 and np.mean(list(or_hist)) > 0
               else None)

        # Rolling ATR (14-day avg of daily RTH range)
        atr = np.mean(list(atr_hist)) if len(atr_hist) >= 5 else None

        # OR as % of ATR
        or_atr_pct = (or_R / atr) if atr and atr > 0 else None

        # OR floor (30th percentile of recent ORs)
        or_floor_30 = (np.percentile(list(or_floor_hist), 30)
                       if len(or_floor_hist) >= 10 else None)

        stats[d] = dict(
            or_R=or_R, rth_R=rth_R,
            cov=cov, atr=atr,
            or_atr_pct=or_atr_pct,
            or_floor_30=or_floor_30,
        )

        # Update histories (use PREVIOUS days' data for next day's filter — no lookahead)
        or_hist.append(or_R)
        atr_hist.append(rth_R)
        or_floor_hist.append(or_R)

    return stats


def run_with_filter(cl, dates, by_d, wkly, daily_stats, filter_fn, sys_label):
    """
    filter_fn(day_stats) -> "FULL" | "HALF" | "SKIP"
    day_stats = dict from build_daily_stats for this date.
    """
    trades = []
    recent_wins = deque(maxlen=15)   # for adaptive win-rate filter

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue

        ds = daily_stats.get(d)
        if ds is None: continue

        or_R = ds["or_R"]
        if or_R < 0.10: continue

        # Apply regime filter
        size_mode = filter_fn(ds, list(recent_wins))
        if size_mode == "SKIP": continue

        # Prior day H/L
        prev = by_d.get(dates[i-1])
        if prev is None: continue
        prev_rth = prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if not len(prev_rth): continue
        PDH = prev_rth["CL_high"].max(); PDL = prev_rth["CL_low"].min()

        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        or_H = or_b["CL_high"].max(); or_L = or_b["CL_low"].min()

        watch = day[(day["time_et"]>=OR_END)&(day["time_et"]<ENTRY_END)]
        long_fired = short_fired = False
        pdh_broken = pdl_broken = False

        for _, bar in watch.iterrows():
            if bar["CL_high"] > PDH: pdh_broken = True
            if bar["CL_low"]  < PDL: pdl_broken = True

            orb_long  = bar["CL_high"] > or_H
            orb_short = bar["CL_low"]  < or_L

            if not long_fired and orb_long:
                long_fired = True
                ep = PDH if pdh_broken else or_H
                sp = ep - or_R; tp = ep + or_R * PMULT
                rv, why = sim(day, bar["time_et"], "LONG", ep, sp, tp)
                trades.append(dict(date=d, year=d.year, r=rv, win=rv>0,
                                   exit=why, sys=sys_label, size=size_mode))
                recent_wins.append(int(rv > 0))

            if not short_fired and orb_short:
                short_fired = True
                ep = PDL if pdl_broken else or_L
                sp = ep + or_R; tp = ep - or_R * PMULT
                rv, why = sim(day, bar["time_et"], "SHORT", ep, sp, tp)
                trades.append(dict(date=d, year=d.year, r=rv, win=rv>0,
                                   exit=why, sys=sys_label, size=size_mode))
                recent_wins.append(int(rv > 0))

    return pd.DataFrame(trades)


def prow(label, tdf, pad=42):
    if not len(tdf):
        print(f"  {label:<{pad}}  n=   0"); return
    wr=tdf["win"].mean(); ar=tdf["r"].mean()
    ev=wr*PMULT-(1-wr)
    yr=4.83; n_yr=len(tdf)/yr
    dr=tdf.groupby("date")["r"].sum()
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    flag="✅" if wr>0.57 else ("🟡" if wr>0.50 else "❌")
    print(f"  {flag} {label:<{pad}}  n={len(tdf):4d} ({n_yr:.0f}/yr)  "
          f"WR={wr*100:5.1f}%  EV={ev:+.3f}  Sh={sh:.2f}")


def pyr(label, tdf):
    print(f"\n  {label}")
    print(f"  {'Year':<6}  {'n':>4}  {'WR':>6}  {'EV':>7}  {'Status':>10}")
    print(f"  {'─'*42}")
    for yr, sub in tdf.groupby("year"):
        wr=sub["win"].mean(); ev=wr*PMULT-(1-wr)
        flag = "✅ good" if ev>1.0 else ("🟡 ok" if ev>0.5 else "❌ bad")
        print(f"  {yr:<6}  {len(sub):>4}  {wr*100:>5.1f}%  {ev:>+6.3f}R  {flag}")


def main():
    print("Loading …")
    cl = load()
    dates = sorted(cl["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in cl.groupby("date_et")}
    wkly  = weekly_hl(by_d, dates)

    print("Computing daily regime stats …")
    ds = build_daily_stats(cl, dates, by_d)
    print(f"  {len(ds)} trading days profiled\n")

    W = 78
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── Baseline (no filter) ──────────────────────────────────────────────────
    baseline = run_with_filter(cl, dates, by_d, wkly, ds,
                               lambda s, w: "FULL", "BASE")

    # ── F1: OR CoV filter — various thresholds ────────────────────────────────
    sec("F1. OR COEFFICIENT OF VARIATION FILTER")
    print("  Logic: skip day if rolling 20-day OR CoV > threshold")
    print("  CoV = std/mean of recent OR ranges. High CoV = erratic ORs = bad regime")
    print()
    print(f"  {'Threshold':<20}  {'n':>4}  {'n/yr':>5}  {'WR':>6}  {'EV':>7}  "
          f"{'Sharpe':>7}  {'Days skipped':>13}")
    print(f"  {'─'*70}")
    f1_results = {}
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, None]:
        label = f"CoV < {thresh}" if thresh else "No filter"
        tdf = run_with_filter(cl, dates, by_d, wkly, ds,
                              lambda s, w, t=thresh: (
                                  "SKIP" if (t and s["cov"] and s["cov"] > t) else "FULL"),
                              f"F1_{thresh}")
        if not len(tdf): continue
        wr=tdf["win"].mean(); ev=wr*PMULT-(1-wr)
        dr=tdf.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        skipped = len(baseline)-len(tdf) if thresh else 0
        pct_skip = skipped/len(baseline)*100 if len(baseline) else 0
        flag = "✅" if wr>0.57 else ("🟡" if wr>0.50 else "")
        print(f"  {flag} {label:<20}  n={len(tdf):4d}  {len(tdf)/4.83:>5.0f}/yr  "
              f"WR={wr*100:5.1f}%  EV={ev:+.4f}  Sh={sh:.3f}  "
              f"-{pct_skip:.0f}% trades")
        f1_results[thresh] = dict(wr=wr, sh=sh, n=len(tdf), ev=ev)

    # ── F2: OR/ATR ratio filter ───────────────────────────────────────────────
    sec("F2. OR/ATR RATIO FILTER")
    print("  Logic: skip if OR range < X% of 14-day avg ATR (OR too small = no energy)")
    print()
    print(f"  {'Threshold':<22}  {'n':>4}  {'WR':>6}  {'EV':>7}  {'Sharpe':>7}")
    print(f"  {'─'*56}")
    f2_results = {}
    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        label = f"OR/ATR >= {thresh:.0%}"
        tdf = run_with_filter(cl, dates, by_d, wkly, ds,
                              lambda s, w, t=thresh: (
                                  "SKIP" if (s["or_atr_pct"] and s["or_atr_pct"] < t)
                                  else "FULL"),
                              f"F2_{thresh}")
        if not len(tdf): continue
        wr=tdf["win"].mean(); ev=wr*PMULT-(1-wr)
        dr=tdf.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        flag = "✅" if wr>0.57 else ("🟡" if wr>0.50 else "")
        print(f"  {flag} {label:<22}  n={len(tdf):4d}  WR={wr*100:5.1f}%  "
              f"EV={ev:+.4f}  Sh={sh:.3f}")
        f2_results[thresh] = dict(wr=wr, sh=sh, n=len(tdf), ev=ev)

    # ── F3: OR absolute floor filter ─────────────────────────────────────────
    sec("F3. OR ABSOLUTE FLOOR FILTER")
    print("  Logic: skip if OR range < 30th pctile of last 30 ORs (below-average day)")
    print()
    print(f"  {'Threshold':<28}  {'n':>4}  {'WR':>6}  {'EV':>7}  {'Sharpe':>7}")
    print(f"  {'─'*60}")
    for pct_mult in [0.7, 0.8, 0.9, 1.0, 1.1]:
        label = f"OR > {pct_mult:.1f}× floor_30pct"
        tdf = run_with_filter(cl, dates, by_d, wkly, ds,
                              lambda s, w, m=pct_mult: (
                                  "SKIP" if (s["or_floor_30"] and
                                             s["or_R"] < s["or_floor_30"] * m)
                                  else "FULL"),
                              f"F3_{pct_mult}")
        if not len(tdf): continue
        wr=tdf["win"].mean(); ev=wr*PMULT-(1-wr)
        dr=tdf.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        flag = "✅" if wr>0.57 else ("🟡" if wr>0.50 else "")
        print(f"  {flag} {label:<28}  n={len(tdf):4d}  WR={wr*100:5.1f}%  "
              f"EV={ev:+.4f}  Sh={sh:.3f}")

    # ── F4: Adaptive WR filter ────────────────────────────────────────────────
    sec("F4. ADAPTIVE WIN-RATE FILTER")
    print("  Logic: if last 15 signals < X% WR, go to HALF size (don't skip entirely)")
    print()
    print(f"  {'Threshold':<20}  {'n':>4}  {'WR':>6}  {'EV':>7}  {'Sharpe':>7}")
    print(f"  {'─'*56}")
    for wr_thresh in [0.35, 0.40, 0.45, 0.50]:
        label = f"Reduce if WR<{wr_thresh:.0%}"
        tdf = run_with_filter(cl, dates, by_d, wkly, ds,
                              lambda s, w, t=wr_thresh: (
                                  "HALF" if (len(w) >= 10 and
                                             sum(w)/len(w) < t)
                                  else "FULL"),
                              f"F4_{wr_thresh}")
        if not len(tdf): continue
        wr=tdf["win"].mean(); ev=wr*PMULT-(1-wr)
        dr=tdf.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        flag = "✅" if wr>0.57 else ("🟡" if wr>0.50 else "")
        print(f"  {flag} {label:<20}  n={len(tdf):4d}  WR={wr*100:5.1f}%  "
              f"EV={ev:+.4f}  Sh={sh:.3f}")

    # ── F5: Best combination ──────────────────────────────────────────────────
    sec("F5. COMBINED FILTER  —  Best F1 + Best F2")
    best_f1 = max(f1_results, key=lambda k: f1_results[k]["sh"] if k else -999)
    best_f2 = max(f2_results, key=lambda k: f2_results[k]["sh"])
    print(f"  Best F1 threshold: CoV < {best_f1}  (Sharpe={f1_results[best_f1]['sh']:.3f})")
    print(f"  Best F2 threshold: OR/ATR >= {best_f2:.0%}  (Sharpe={f2_results[best_f2]['sh']:.3f})")
    print()
    combos = [
        (f"F1(CoV<{best_f1}) + F2(OR/ATR>={best_f2:.0%})",
         lambda s, w, t1=best_f1, t2=best_f2: (
             "SKIP" if ((t1 and s["cov"] and s["cov"] > t1) or
                        (s["or_atr_pct"] and s["or_atr_pct"] < t2))
             else "FULL")),
        (f"F1(CoV<0.60) + F2(OR/ATR>=20%)",
         lambda s, w: (
             "SKIP" if ((s["cov"] and s["cov"] > 0.60) or
                        (s["or_atr_pct"] and s["or_atr_pct"] < 0.20))
             else "FULL")),
        (f"F1(CoV<0.65) + F2(OR/ATR>=15%)",
         lambda s, w: (
             "SKIP" if ((s["cov"] and s["cov"] > 0.65) or
                        (s["or_atr_pct"] and s["or_atr_pct"] < 0.15))
             else "FULL")),
    ]
    best_combo_tdf = None; best_combo_sh = -999; best_combo_label = ""
    for label, fn in combos:
        tdf = run_with_filter(cl, dates, by_d, wkly, ds, fn, "F5")
        if not len(tdf): continue
        wr=tdf["win"].mean(); ev=wr*PMULT-(1-wr)
        dr=tdf.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        flag="✅" if wr>0.57 else ("🟡" if wr>0.50 else "❌")
        print(f"  {flag} {label:<45}  n={len(tdf):4d} ({len(tdf)/4.83:.0f}/yr)  "
              f"WR={wr*100:.1f}%  EV={ev:+.3f}  Sh={sh:.3f}")
        if sh > best_combo_sh:
            best_combo_sh = sh; best_combo_tdf = tdf; best_combo_label = label

    # ── Year-by-year for best ─────────────────────────────────────────────────
    sec("YEAR-BY-YEAR COMPARISON  —  Baseline vs Best Combined Filter")
    print(f"  {'Year':<6}  {'Base n':>6}  {'Base WR':>8}  {'Base EV':>8}  │  "
          f"{'Filt n':>6}  {'Filt WR':>8}  {'Filt EV':>8}  {'Change':>8}")
    print(f"  {'─'*68}")
    for yr in sorted(baseline["year"].unique()):
        bs = baseline[baseline["year"]==yr]
        ft = best_combo_tdf[best_combo_tdf["year"]==yr] if best_combo_tdf is not None else pd.DataFrame()
        b_wr=(bs["win"].mean() if len(bs) else 0)
        f_wr=(ft["win"].mean() if len(ft) else 0)
        b_ev=b_wr*PMULT-(1-b_wr) if len(bs) else 0
        f_ev=f_wr*PMULT-(1-f_wr) if len(ft) else 0
        flag = " ✅" if f_ev > b_ev + 0.1 else (" ⚠️" if f_ev < 0.5 else "")
        print(f"  {yr:<6}  {len(bs):>6}  {b_wr*100:>7.1f}%  {b_ev:>+7.3f}R  │  "
              f"{len(ft):>6}  {f_wr*100:>7.1f}%  {f_ev:>+7.3f}R  {(f_wr-b_wr)*100:>+6.1f}pp{flag}")

    # ── Final summary ─────────────────────────────────────────────────────────
    sec("FINAL SUMMARY")
    print(f"  Base (no filter):   ", end=""); prow("", baseline, pad=0)
    if best_combo_tdf is not None:
        print(f"  Best filter:        ", end="")
        prow(f"  {best_combo_label}", best_combo_tdf, pad=4)

    if best_combo_tdf is not None and len(best_combo_tdf):
        wr=best_combo_tdf["win"].mean()
        ev=wr*PMULT-(1-wr)
        dr=best_combo_tdf.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        n_yr = len(best_combo_tdf)/4.83
        print()
        print(f"  Optimal CL system (with regime filter):")
        print(f"    Signal  : OR + PDH/L (Tier 1 + Tier 3)")
        print(f"    Filter  : {best_combo_label}")
        print(f"    Trades  : {len(best_combo_tdf):,}  ({n_yr:.0f}/yr, "
              f"{n_yr/52:.2f}/wk)")
        print(f"    WR      : {wr*100:.1f}%")
        print(f"    EV      : {ev:+.3f}R")
        print(f"    Sharpe  : {sh:.3f}")
    print()


if __name__ == "__main__":
    main()
