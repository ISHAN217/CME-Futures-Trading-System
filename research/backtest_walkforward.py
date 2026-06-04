#!/usr/bin/env python3
"""
backtest_walkforward.py — Overfitting / walk-forward audit

Tests whether the system's parameters (optimised on 2018-2026) hold up
on unseen data by splitting into:
  IN-SAMPLE  : 2018-2021  (parameters were implicitly calibrated here)
  OUT-SAMPLE : 2022-2026  (genuinely unseen)

Also checks year-by-year consistency to spot any performance decay.
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = ("/Users/ishanbhardwaj/cme_competition_system/output/mtf/"
        "ES_NQ_1min_aligned.parquet")

OR_START   = time(9, 30);  OR_END     = time(10, 0)
ORB_CUTOFF = time(10, 30); EOD_CLOSE  = time(15, 30)
RTH_START  = time(9, 30);  RTH_END    = time(16, 0)
WATCH_END  = time(11, 0)

MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=150
SLIP=0.25; PMULT=3.0


def run_orb(df):
    """Run ORB-only on full dataset. Returns per-trade DataFrame."""
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    trades = []

    for d in dates:
        day = by_d.get(d)
        if day is None: continue
        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if len(or_b)==0: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not (MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        post = day[(day["time_et"]>=OR_END)&(day["time_et"]<ORB_CUTOFF)]
        es_b=nq_b=None; entry_row=None
        for _,r in post.iterrows():
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"
                elif r["ES_low"]<es_L: es_b="S"
            if nq_b is None:
                if r["NQ_high"]>nq_H: nq_b="L"
                elif r["NQ_low"]<nq_L: nq_b="S"
            if es_b and nq_b: entry_row=r; break
        if not (es_b and nq_b and es_b==nq_b): continue

        direction="LONG" if es_b=="L" else "SHORT"
        et = entry_row["time_et"]
        after = day[day["time_et"]>et]

        for sym,H,L,R,ch,cl,cc in [
            ("ES",es_H,es_L,es_R,"ES_high","ES_low","ES_close"),
            ("NQ",nq_H,nq_L,nq_R,"NQ_high","NQ_low","NQ_close"),
        ]:
            ep=H if direction=="LONG" else L
            sp=L if direction=="LONG" else H
            tp=round(ep+PMULT*R,2) if direction=="LONG" else round(ep-PMULT*R,2)
            for _,bar in after.iterrows():
                t=bar["time_et"]
                if t>=EOD_CLOSE:
                    sign=1 if direction=="LONG" else -1
                    net=sign*(bar[cc]-ep)-SLIP*2
                    r_val=net/R if R>0 else 0; why="EOD"; break
                if direction=="LONG":
                    if bar[cl]<=sp: r_val=-1.0-SLIP*2/R; why="STOP"; break
                    if bar[ch]>=tp: r_val=PMULT-SLIP/R;  why="TGT";  break
                else:
                    if bar[ch]>=sp: r_val=-1.0-SLIP*2/R; why="STOP"; break
                    if bar[cl]<=tp: r_val=PMULT-SLIP/R;  why="TGT";  break
            else:
                continue
            trades.append(dict(date=d, sym=sym, r=r_val, win=r_val>0, exit=why,
                               year=d.year))
    return pd.DataFrame(trades)


def run_pdhl(df):
    """Run PDH/L on ORB-miss days. Returns per-trade DataFrame."""
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    trades = []

    for i,d in enumerate(dates[1:],1):
        day=by_d.get(d)
        if day is None: continue
        or_b=day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if len(or_b)==0: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not (MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        # ORB fired?
        post=day[(day["time_et"]>=OR_END)&(day["time_et"]<ORB_CUTOFF)]
        es_b=nq_b=None
        for _,r in post.iterrows():
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"
                elif r["ES_low"]<es_L: es_b="S"
            if nq_b is None:
                if r["NQ_high"]>nq_H: nq_b="L"
                elif r["NQ_low"]<nq_L: nq_b="S"
            if es_b and nq_b: break
        if es_b and nq_b and es_b==nq_b: continue  # ORB fired — skip

        prev=by_d.get(dates[i-1])
        if prev is None: continue
        prth=prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if len(prth)==0: continue
        PDH_ES=prth["ES_high"].max(); PDL_ES=prth["ES_low"].min()
        PDH_NQ=prth["NQ_high"].max(); PDL_NQ=prth["NQ_low"].min()

        watch=day[(day["time_et"]>=OR_END)&(day["time_et"]<WATCH_END)]
        eu=ed=nu=nd=False; direction=None; entry_row=None
        for _,r in watch.iterrows():
            if not eu and r["ES_high"]>PDH_ES: eu=True
            if not ed and r["ES_low"]<PDL_ES:  ed=True
            if not nu and r["NQ_high"]>PDH_NQ: nu=True
            if not nd and r["NQ_low"]<PDL_NQ:  nd=True
            if eu and nu and direction is None: direction="LONG";  entry_row=r; break
            if ed and nd and direction is None: direction="SHORT"; entry_row=r; break
            if (eu and nd) or (ed and nu): break

        if direction is None or entry_row is None: continue
        et=entry_row["time_et"]
        after=day[day["time_et"]>et]

        for sym,PDH,PDL,R,ch,cl,cc in [
            ("ES",PDH_ES,PDL_ES,es_R,"ES_high","ES_low","ES_close"),
            ("NQ",PDH_NQ,PDL_NQ,nq_R,"NQ_high","NQ_low","NQ_close"),
        ]:
            ep=PDH if direction=="LONG" else PDL
            sp=ep-R if direction=="LONG" else ep+R
            tp=round(ep+PMULT*R,2) if direction=="LONG" else round(ep-PMULT*R,2)
            for _,bar in after.iterrows():
                t=bar["time_et"]
                if t>=EOD_CLOSE:
                    sign=1 if direction=="LONG" else -1
                    net=sign*(bar[cc]-ep)-SLIP*2
                    r_val=net/R if R>0 else 0; why="EOD"; break
                if direction=="LONG":
                    if bar[cl]<=sp: r_val=-1.0-SLIP*2/R; why="STOP"; break
                    if bar[ch]>=tp: r_val=PMULT-SLIP/R;  why="TGT";  break
                else:
                    if bar[ch]>=sp: r_val=-1.0-SLIP*2/R; why="STOP"; break
                    if bar[cl]<=tp: r_val=PMULT-SLIP/R;  why="TGT";  break
            else:
                continue
            trades.append(dict(date=d, sym=sym, r=r_val, win=r_val>0, exit=why,
                               year=d.year))
    return pd.DataFrame(trades)


def stats(tdf, label):
    if len(tdf)==0:
        return dict(label=label, n=0, wr=0, avg_r=0, ev=0, sharpe=0, pf=0)
    wr    = tdf["win"].mean()
    avg_r = tdf["r"].mean()
    ev    = wr*PMULT-(1-wr)*1.0
    wins  = tdf[tdf["win"]]["r"].sum()
    loss  = abs(tdf[~tdf["win"]]["r"].sum())
    pf    = wins/loss if loss>0 else 999
    dr    = tdf.groupby("date")["r"].sum()
    sh    = dr.mean()/dr.std(ddof=1)*np.sqrt(252) if dr.std()>0 else 0
    return dict(label=label, n=len(tdf)//2, wr=wr, avg_r=avg_r, ev=ev,
                sharpe=round(sh,3), pf=round(pf,2))


def print_stats(s):
    print(f"  {s['label']:<28}  n={s['n']:4d}  WR={s['wr']*100:5.1f}%  "
          f"AvgR={s['avg_r']:+.3f}R  EV={s['ev']:+.3f}R  "
          f"Sharpe={s['sharpe']:5.3f}  PF={s['pf']:.2f}x")


def main():
    print("Loading data …")
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    df = df.sort_values("ts_et").reset_index(drop=True)
    print(f"  {len(df):,} bars  {sorted(df['date_et'].unique())[0]} → "
          f"{sorted(df['date_et'].unique())[-1]}\n")

    IS_END = date(2021, 12, 31)   # in-sample: 2018-2021
    OS_START = date(2022, 1, 1)   # out-of-sample: 2022-2026

    df_is = df[df["date_et"] <= IS_END].copy()
    df_os = df[df["date_et"] >= OS_START].copy()

    print("Running ORB backtest …")
    orb_all = run_orb(df)
    orb_is  = run_orb(df_is)
    orb_os  = run_orb(df_os)

    print("Running PDH/L backtest …")
    pdhl_all = run_pdhl(df)
    pdhl_is  = run_pdhl(df_is)
    pdhl_os  = run_pdhl(df_os)

    # ── Walk-forward ──────────────────────────────────────────────────────
    print()
    print("="*72)
    print("WALK-FORWARD TEST  —  ORB System")
    print("  Parameters fixed: 3R target, NQ max 150pts, OR 9:30–10:00 ET")
    print("="*72)
    print_stats(stats(orb_all, "Full sample  2018-2026"))
    print_stats(stats(orb_is,  "In-sample    2018-2021"))
    print_stats(stats(orb_os,  "Out-of-sample 2022-2026"))

    print()
    print("="*72)
    print("WALK-FORWARD TEST  —  PDH/L System")
    print("="*72)
    print_stats(stats(pdhl_all, "Full sample  2018-2026"))
    print_stats(stats(pdhl_is,  "In-sample    2018-2021"))
    print_stats(stats(pdhl_os,  "Out-of-sample 2022-2026"))

    # ── Year-by-year decay check ──────────────────────────────────────────
    print()
    print("="*72)
    print("YEAR-BY-YEAR  —  ORB (decay = overfitting signal)")
    print("="*72)
    print(f"  {'Year':<6} {'Days':>5} {'WR':>6} {'AvgR':>7} {'EV':>7} {'Sharpe':>7}")
    print(f"  {'─'*44}")
    for yr, sub in orb_all.groupby("year"):
        s = stats(sub, str(yr))
        dr = sub.groupby("date")["r"].sum()
        sh = dr.mean()/dr.std(ddof=1)*np.sqrt(252) if dr.std()>0 else 0
        marker = "  ← IS" if yr <= 2021 else "  ← OOS"
        print(f"  {yr:<6} {s['n']:>5} {s['wr']*100:>5.1f}% {s['avg_r']:>+6.3f}R "
              f"{s['ev']:>+6.3f}R {sh:>7.3f}{marker}")

    print()
    print("="*72)
    print("YEAR-BY-YEAR  —  PDH/L")
    print("="*72)
    print(f"  {'Year':<6} {'Days':>5} {'WR':>6} {'AvgR':>7} {'EV':>7} {'Sharpe':>7}")
    print(f"  {'─'*44}")
    for yr, sub in pdhl_all.groupby("year"):
        s = stats(sub, str(yr))
        dr = sub.groupby("date")["r"].sum()
        sh = dr.mean()/dr.std(ddof=1)*np.sqrt(252) if dr.std()>0 else 0
        marker = "  ← IS" if yr <= 2021 else "  ← OOS"
        print(f"  {yr:<6} {s['n']:>5} {s['wr']*100:>5.1f}% {s['avg_r']:>+6.3f}R "
              f"{s['ev']:>+6.3f}R {sh:>7.3f}{marker}")

    # ── Overfitting risk summary ──────────────────────────────────────────
    orb_wr_is  = stats(orb_is,  "")["wr"]
    orb_wr_os  = stats(orb_os,  "")["wr"]
    pdhl_wr_is = stats(pdhl_is, "")["wr"]
    pdhl_wr_os = stats(pdhl_os, "")["wr"]
    orb_ev_is  = stats(orb_is,  "")["ev"]
    orb_ev_os  = stats(orb_os,  "")["ev"]

    print()
    print("="*72)
    print("OVERFITTING RISK ASSESSMENT")
    print("="*72)
    print(f"  ORB  WR:  IS={orb_wr_is*100:.1f}%  OOS={orb_wr_os*100:.1f}%  "
          f"decay={((orb_wr_os-orb_wr_is)*100):+.1f}pp")
    print(f"  ORB  EV:  IS={orb_ev_is:+.3f}R  OOS={orb_ev_os:+.3f}R  "
          f"decay={orb_ev_os-orb_ev_is:+.3f}R")
    print(f"  PDH/L WR: IS={pdhl_wr_is*100:.1f}%  OOS={pdhl_wr_os*100:.1f}%  "
          f"decay={((pdhl_wr_os-pdhl_wr_is)*100):+.1f}pp")
    print()
    wr_decay = abs(orb_wr_os - orb_wr_is)
    if wr_decay < 0.03:
        print("  ✅  LOW overfitting risk — OOS performance close to IS")
    elif wr_decay < 0.06:
        print("  ⚠️   MODERATE overfitting risk — some decay in OOS")
    else:
        print("  ❌  HIGH overfitting risk — significant OOS decay")
    print()
    print("  Parameters tested on same data (data-mined):")
    print("  — 3R profit target     (chose best of 1.5R/2R/2.5R/3R)")
    print("  — NQ max 150pts        (chose based on WR bucketing)")
    print("  — PDH/L cutoff 11:00   (chose based on signal clustering)")
    print()


if __name__ == "__main__":
    main()
