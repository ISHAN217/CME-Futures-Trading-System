#!/usr/bin/env python3
"""
backtest_spread_fade.py — Two genuine improvements to the ORB system

1. ES/NQ SPREAD FADE (mean-reversion stat arb)
   The ES/NQ normalised spread is mean-reverting.
   When it breaks above its OR range (ES dramatically outperforming NQ),
   SHORT the spread: sell ES, buy NQ.
   Fade fires when NEITHER the regular ORB breakout happened
   (no directional signal) — so this is additive, not overlapping.

2. CORRELATION-WEIGHTED ORB
   Intraday ES/NQ correlation during the OR predicts ORB WR:
     Low corr (<0.70)  → WR=55.6%  → full/large size
     High corr (>0.95) → WR=39.8%  → skip or small size
   Tests whether weighting ORB trades by OR-period correlation
   improves risk-adjusted returns.

Both use proper market entry (close of signal bar) — no limit-order bugs.
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"

OR_START=time(9,30); OR_END=time(10,0); ORB_CUT=time(10,30)
EOD=time(15,30); RTH_START=time(9,30)
SLIP=0.25; MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=175


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


# ─── PART 1: SPREAD FADE ──────────────────────────────────────────────────────

def run_spread_fade(df, dates, by_d,
                    stop_mult=1.0,    # stop at entry + stop_mult×sp_R
                    tgt_mult=1.0,     # target = revert by tgt_mult×sp_R
                    window_end=time(11,30),
                    skip_if_orb=True):
    """
    Normalised spread = ES_return/ES_OR_R - NQ_return/NQ_OR_R

    When spread > OR_high: ES is outperforming NQ beyond its normal OR range.
    FADE: SHORT ES, LONG NQ (expect reversion).

    When spread < OR_low: NQ outperforming ES.
    FADE: LONG ES, SHORT NQ.

    Entry: close of the bar where spread first exceeds OR boundary.
    Stop:  spread extends another stop_mult × sp_R beyond entry.
    Target: spread reverts tgt_mult × sp_R back toward OR boundary.

    P&L reported in spread-R units.
    In live trading: short ES contracts + long NQ contracts, sized to
    equal normalised dollar exposure.
    """
    trades = []

    for d in dates:
        day = by_d.get(d)
        if day is None: continue
        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if len(or_b)<10: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        es_open = or_b["ES_open"].iloc[0]; nq_open = or_b["NQ_open"].iloc[0]

        # OR spread range
        sp_vals = []
        for _,row in or_b.iterrows():
            sp_vals.append((row["ES_close"]-es_open)/es_R - (row["NQ_close"]-nq_open)/nq_R)
        sp_H=max(sp_vals); sp_L=min(sp_vals); sp_R=sp_H-sp_L
        if sp_R < 0.05: continue   # trivial spread range

        # If skip_if_orb: skip days where regular ORB fires (additive signal)
        if skip_if_orb:
            post_orb = day[(day["time_et"]>=OR_END)&(day["time_et"]<ORB_CUT)]
            es_b=nq_b=None
            for _,r in post_orb.iterrows():
                if es_b is None:
                    if r["ES_high"]>es_H: es_b="L"
                    elif r["ES_low"]<es_L: es_b="S"
                if nq_b is None:
                    if r["NQ_high"]>nq_H: nq_b="L"
                    elif r["NQ_low"]<nq_L: nq_b="S"
                if es_b and nq_b: break
            if es_b and nq_b and es_b==nq_b: continue   # ORB day — skip

        # Watch for spread to break OR range (fade direction)
        watch = day[(day["time_et"]>=OR_END)&(day["time_et"]<window_end)]
        signal_t=direction=None; entry_sp=None

        for _,bar in watch.iterrows():
            cur_sp = (bar["ES_close"]-es_open)/es_R - (bar["NQ_close"]-nq_open)/nq_R
            if cur_sp > sp_H and direction is None:
                direction="FADE_UP"   # ES over-extended, fade: SHORT ES, LONG NQ
                signal_t=bar["time_et"]; entry_sp=cur_sp; break
            if cur_sp < sp_L and direction is None:
                direction="FADE_DN"   # NQ over-extended, fade: LONG ES, SHORT NQ
                signal_t=bar["time_et"]; entry_sp=cur_sp; break

        if direction is None: continue

        # Stop and target in spread-R units
        if direction=="FADE_UP":
            stop_level = entry_sp + stop_mult*sp_R    # spread extends further → stop
            tgt_level  = entry_sp - tgt_mult*sp_R     # spread reverts → target
        else:
            stop_level = entry_sp - stop_mult*sp_R
            tgt_level  = entry_sp + tgt_mult*sp_R

        # Simulate spread over time
        after=day[day["time_et"]>signal_t]
        r_val=0.0; exit_r="NONE"
        for _,bar in after.iterrows():
            t=bar["time_et"]
            cur_sp=(bar["ES_close"]-es_open)/es_R-(bar["NQ_close"]-nq_open)/nq_R
            if t>=EOD:
                if direction=="FADE_UP":
                    r_val=round((entry_sp-cur_sp)/sp_R,3)
                else:
                    r_val=round((cur_sp-entry_sp)/sp_R,3)
                exit_r="EOD"; break
            if direction=="FADE_UP":
                if cur_sp>=stop_level: r_val=-stop_mult; exit_r="STOP"; break
                if cur_sp<=tgt_level:  r_val=+tgt_mult;  exit_r="TGT";  break
            else:
                if cur_sp<=stop_level: r_val=-stop_mult; exit_r="STOP"; break
                if cur_sp>=tgt_level:  r_val=+tgt_mult;  exit_r="TGT";  break

        trades.append(dict(
            date=d, year=d.year, direction=direction,
            r=r_val, win=r_val>0, exit=exit_r,
            entry_excess=round(abs(entry_sp-sp_H if direction=="FADE_UP" else entry_sp-sp_L),3),
            dow=pd.Timestamp(d).day_name(),
        ))
    return pd.DataFrame(trades)


# ─── PART 2: CORRELATION-WEIGHTED ORB ─────────────────────────────────────────

def run_corr_weighted_orb(df, dates, by_d, pmult=3.0):
    """
    ORB with proper market entry (close of breakout bar) +
    OR-period correlation as a size multiplier.

    Size multipliers:
      corr >= 0.95:  size = 0.0  (skip — WR=39.8%, negative edge)
      corr 0.85-0.95: size = 0.5  (half)
      corr 0.70-0.85: size = 1.0  (normal)
      corr < 0.70:   size = 1.5  (increase — WR=55.6%)

    Tracks: raw R (unscaled) and dollar-weighted R (scaled by size).
    """
    trades = []

    for d in dates:
        day = by_d.get(d)
        if day is None: continue
        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if len(or_b)<10: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        # OR correlation
        es_rets = or_b["ES_close"].pct_change().dropna()
        nq_rets = or_b["NQ_close"].pct_change().dropna()
        if len(es_rets)<5: continue
        corr = float(es_rets.corr(nq_rets))

        if corr >= 0.95:  size = 0.0
        elif corr >= 0.85: size = 0.5
        elif corr >= 0.70: size = 1.0
        else:              size = 1.5

        if size == 0.0: continue   # skip very-high-corr days

        # ORB detection
        post = day[(day["time_et"]>=OR_END)&(day["time_et"]<ORB_CUT)]
        es_b=nq_b=None; entry_row=None
        for _,r in post.iterrows():
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"
                elif r["ES_low"]<es_L: es_b="S"
            if nq_b is None:
                if r["NQ_high"]>nq_H: nq_b="L"
                elif r["NQ_low"]<nq_L: nq_b="S"
            if es_b and nq_b: entry_row=r; break
        if not(es_b and nq_b and es_b==nq_b): continue

        direction="LONG" if es_b=="L" else "SHORT"
        et=entry_row["time_et"]

        # Market entry: close of the breakout bar
        for sym,H,L,R,ch,cl,cc in [
            ("ES",es_H,es_L,es_R,"ES_high","ES_low","ES_close"),
            ("NQ",nq_H,nq_L,nq_R,"NQ_high","NQ_low","NQ_close"),
        ]:
            ep = entry_row[cc] + SLIP if direction=="LONG" else entry_row[cc]-SLIP
            sp = ep - R if direction=="LONG" else ep + R
            tp = ep + pmult*R if direction=="LONG" else ep - pmult*R

            after = day[day["time_et"]>et]
            rv=0.0; why="NONE"
            for _,bar in after.iterrows():
                t=bar["time_et"]
                if t>=EOD:
                    sign=1 if direction=="LONG" else -1
                    net=sign*(bar[cc]-ep)-SLIP*2
                    rv=round(net/R,3); why="EOD"; break
                if direction=="LONG":
                    if bar[cl]<=sp: rv=round(-1.0-SLIP*2/R,3); why="STOP"; break
                    if bar[ch]>=tp: rv=round(pmult-SLIP/R,3);  why="TGT";  break
                else:
                    if bar[ch]>=sp: rv=round(-1.0-SLIP*2/R,3); why="STOP"; break
                    if bar[cl]<=tp: rv=round(pmult-SLIP/R,3);  why="TGT";  break

            trades.append(dict(
                date=d, year=d.year, sym=sym, direction=direction,
                r=rv, r_weighted=rv*size, win=rv>0,
                exit=why, corr=round(corr,3), size=size,
                dow=pd.Timestamp(d).day_name(),
            ))
    return pd.DataFrame(trades)


def prow(label, sub, r_col="r", pad=36):
    if not len(sub): print(f"  {label:<{pad}}  n=0"); return
    wr=sub["win"].mean(); ar=sub[r_col].mean()
    n_sig = sub["date"].nunique() if "date" in sub.columns else len(sub)
    dr=sub.groupby("date")[r_col].sum() if "date" in sub.columns else pd.Series(sub[r_col].values)
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    n_yr=n_sig/8.1
    flag="✅" if wr>0.53 else ("🟡" if wr>0.46 else "❌")
    print(f"  {flag} {label:<{pad}}  n={n_sig:4d}  ({n_yr:.0f}/yr)  "
          f"WR={wr*100:5.1f}%  AvgR={ar:+.4f}  Sh={sh:.2f}")


def main():
    print("Loading …")
    df=load()
    dates=sorted(df["date_et"].unique())
    by_d={d:g.reset_index(drop=True) for d,g in df.groupby("date_et")}
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")
    IS_END=date(2021,12,31); OS_START=date(2022,1,1)

    W=72
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1: SPREAD FADE
    # ══════════════════════════════════════════════════════════════════════════
    sec("PART 1: ES/NQ SPREAD FADE  —  Mean-reversion stat arb")
    print("  Entry: FADE (reverse) when spread breaks beyond its OR range")
    print("  Skip: days when regular ORB fires (additive signal, no overlap)\n")

    # Test different stop/target ratios
    print("  Stop/Target sensitivity:")
    print(f"  {'Config':<28}  {'n':>4}  {'WR':>6}  {'AvgR':>7}  {'Sharpe':>7}")
    print(f"  {'─'*62}")
    best_sh=-999; best_cfg=None; best_tdf=None
    for stop_m, tgt_m in [(1.0,1.0),(1.0,1.5),(1.0,2.0),(0.5,1.0),(0.5,1.5),(2.0,1.0)]:
        tdf=run_spread_fade(df,dates,by_d,stop_mult=stop_m,tgt_mult=tgt_m)
        if not len(tdf): continue
        wr=tdf["win"].mean(); ar=tdf["r"].mean()
        dr=tdf.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        flag="✅" if wr>0.60 else ("🟡" if wr>0.50 else "❌")
        print(f"  {flag} stop={stop_m}R / target={tgt_m}R          "
              f"  n={len(tdf):4d}  WR={wr*100:5.1f}%  AvgR={ar:+.4f}  Sh={sh:.2f}")
        if sh>best_sh: best_sh=sh; best_cfg=(stop_m,tgt_m); best_tdf=tdf

    print(f"\n  Best config: stop={best_cfg[0]}R / target={best_cfg[1]}R")

    if best_tdf is not None and len(best_tdf):
        sec("SPREAD FADE — Deep dive on best config")
        print(f"  Year-by-year:")
        print(f"  {'Year':<6}  {'n':>4}  {'WR':>6}  {'AvgR':>7}  {'TGT%':>5} {'STP%':>5} {'EOD%':>5}  IS/OOS")
        print(f"  {'─'*58}")
        for yr,sub in best_tdf.groupby("year"):
            wr=sub["win"].mean(); ar=sub["r"].mean()
            t_p=(sub["exit"]=="TGT").mean()*100
            s_p=(sub["exit"]=="STOP").mean()*100
            e_p=(sub["exit"]=="EOD").mean()*100
            label="← IS" if yr<=2021 else "← OOS"
            flag="✅" if wr>0.55 else ("🟡" if wr>0.45 else "❌")
            print(f"  {flag} {yr:<6}  {len(sub):>4}  {wr*100:>5.1f}%  {ar:>+6.3f}R  "
                  f"{t_p:>4.0f}% {s_p:>4.0f}% {e_p:>4.0f}%  {label}")

        print()
        print("  Walk-forward:")
        for period,mask in [("IS  2018-2021",best_tdf["date"]<=IS_END),
                             ("OOS 2022-2026",best_tdf["date"]>=OS_START)]:
            prow(period, best_tdf[mask])

        print()
        print("  By direction:")
        prow("  FADE_UP (short ES, long NQ)", best_tdf[best_tdf["direction"]=="FADE_UP"])
        prow("  FADE_DN (long ES, short NQ)", best_tdf[best_tdf["direction"]=="FADE_DN"])

        print()
        print("  Entry excess beyond OR boundary (how far spread has extended):")
        best_tdf["excess_bucket"]=pd.cut(best_tdf["entry_excess"],
                                          bins=[0,0.1,0.2,0.5,99],
                                          labels=["<0.1R","0.1-0.2R","0.2-0.5R",">0.5R"])
        for bkt in ["<0.1R","0.1-0.2R","0.2-0.5R",">0.5R"]:
            prow(f"  Excess {bkt}", best_tdf[best_tdf["excess_bucket"]==bkt])

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2: CORRELATION-WEIGHTED ORB
    # ══════════════════════════════════════════════════════════════════════════
    sec("PART 2: CORRELATION-WEIGHTED ORB")
    print("  Skip corr≥0.95, half-size 0.85-0.95, full 0.70-0.85, 1.5× <0.70")
    print("  Uses proper market entry (close of breakout bar)\n")

    corr_tdf=run_corr_weighted_orb(df,dates,by_d,pmult=3.0)
    print(f"  Total legs: {len(corr_tdf):,}  Signal days: {corr_tdf['date'].nunique()}")
    print()

    # Unweighted vs weighted
    print("  Unweighted (ignore correlation, trade all equally):")
    prow("    All corr-selected days", corr_tdf, r_col="r")
    print("  Size-weighted (apply correlation multiplier):")
    prow("    All corr-selected days", corr_tdf, r_col="r_weighted")

    print()
    print("  By correlation bucket:")
    for corr_lo,corr_hi,size_label in [
        (0.70,0.85,"1.0× (full)"),(0.0,0.70,"1.5× (boost)")]:
        sub=corr_tdf[(corr_tdf["corr"]>=corr_lo)&(corr_tdf["corr"]<corr_hi)]
        prow(f"  corr {corr_lo:.2f}-{corr_hi:.2f}  size={size_label}", sub)

    print()
    print("  Year-by-year (weighted R):")
    print(f"  {'Year':<6}  {'n_days':>6}  {'WR':>6}  {'Unwtd AvgR':>11}  {'Wtd AvgR':>10}  IS/OOS")
    print(f"  {'─'*60}")
    for yr,sub in corr_tdf.groupby("year"):
        wr=sub["win"].mean()
        ar_raw=sub["r"].mean(); ar_wtd=sub["r_weighted"].mean()
        label="← IS" if yr<=2021 else "← OOS"
        flag="✅" if wr>0.52 else ("🟡" if wr>0.45 else "❌")
        print(f"  {flag} {yr:<6}  {sub['date'].nunique():>6}  {wr*100:>5.1f}%  "
              f"{ar_raw:>+10.3f}R  {ar_wtd:>+9.3f}R  {label}")

    print()
    print("  Walk-forward (weighted R):")
    for period,mask in [("IS  2018-2021",corr_tdf["date"]<=IS_END),
                         ("OOS 2022-2026",corr_tdf["date"]>=OS_START)]:
        prow(period, corr_tdf[mask], r_col="r_weighted")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    sec("FINAL COMPARISON  —  All approaches side-by-side")
    print(f"  {'System':<42}  {'n/yr':>6}  {'WR':>6}  {'AvgR':>7}  {'Sharpe':>7}")
    print(f"  {'─'*70}")
    print(f"  ORB verified (market entry, baseline)           22/yr  47.8%  +0.920R   3.08")
    print(f"  PDH/L Confirmed (market entry)                  36/yr  54.5%  +1.181R   3.14")

    for label,tdf,r_col in [
        ("Spread Fade (best config)",           best_tdf,   "r"         ),
        ("Corr-weighted ORB (weighted R)",      corr_tdf,   "r_weighted"),
    ]:
        if tdf is None or not len(tdf): continue
        wr=tdf["win"].mean(); ar=tdf[r_col].mean()
        dr=tdf.groupby("date")[r_col].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        n_yr=tdf["date"].nunique()/8.1
        flag="✅" if wr>0.53 else ("🟡" if wr>0.46 else "❌")
        print(f"  {flag} {label:<42}  {n_yr:>6.0f}/yr  {wr*100:>5.1f}%  {ar:>+6.3f}R  {sh:>7.2f}")
    print()


if __name__=="__main__":
    main()
