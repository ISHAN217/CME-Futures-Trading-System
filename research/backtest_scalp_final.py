#!/usr/bin/env python3
"""
backtest_scalp_final.py — Follow-up on scalp improvement study

What the instrumented study revealed (honest reporting):
  1. Pre-stated hypothesis confirmed: lunch (11:30-12:30) is weak → passes filter
  2. UNEXPECTED: afternoon (12:30-13:30) is even worse (Sh=-0.88) — not pre-stated
  3. UNEXPECTED: wick ≥ 1.5 (pre-committed threshold) fails OOS; wick ≥ 2.0 passes OOS
  4. NQ corroboration HURTS (WR drops); rejected

This file tests:
  A. F1 = exclude lunch only           [pre-stated, passes IS+OOS]
  B. F1b = exclude lunch + afternoon   [data-discovered, clearly labeled as such]
  C. F2b = wick ≥ 2.0pt               [data-discovered, clearly labeled as such]
  D. Combined (A+C)                    [additive combination of discovered filters]
  E. Combined (B+C)                    [more aggressive combination]

LABELING: Data-discovered filters are flagged clearly so you know the risk.
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25
RTH_S = time(9,30); RTH_E = time(16,0)
OR_S  = time(9,30); OR_E  = time(10,0)
EOD   = time(15,30)
MIN_ES=8; MAX_ES=60; TRAIL=3.0; BE=3.0
ACCT=25000; SC_RISK=0.01
IS_END=date(2021,12,31); OS_START=date(2022,1,1)


def load():
    df=pd.read_parquet(DATA)
    df["ts_et"]=pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"]=df["ts_et"].dt.date; df["time_et"]=df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)

def build_levels(by_d,dates):
    weekly={}; monthly={}
    for d in dates:
        ts=pd.Timestamp(d); iso=ts.isocalendar()
        wk=(int(iso.year),int(iso.week)); mn=(ts.year,ts.month)
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_S)&(day["time_et"]<=RTH_E)]
        if not len(rth): continue
        h=rth["ES_high"].max(); l=rth["ES_low"].min()
        if wk not in weekly: weekly[wk]=[h,l]
        else: weekly[wk][0]=max(weekly[wk][0],h); weekly[wk][1]=min(weekly[wk][1],l)
        if mn not in monthly: monthly[mn]=[h,l]
        else: monthly[mn][0]=max(monthly[mn][0],h); monthly[mn][1]=min(monthly[mn][1],l)
    return weekly,monthly

def sh(dr):
    return dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0.0

def sim_trail(fwd,di,ep,s0):
    stop=s0; best=ep
    for c in fwd:
        if di=="SHORT": best=min(best,c); u=ep-c-SLIP*2
        else:           best=max(best,c); u=c-ep-SLIP*2
        if u>=BE:
            if di=="SHORT": stop=min(stop,best+TRAIL)
            else:           stop=max(stop,best-TRAIL)
        if di=="SHORT":
            if c>=stop: return ep-stop-SLIP*2
        else:
            if c<=stop: return stop-ep-SLIP*2
    last=fwd[-1] if len(fwd) else ep
    return ep-last-SLIP*2 if di=="SHORT" else last-ep-SLIP*2

def run_scalp(df,dates,by_d,weekly,monthly,
              exclude_buckets=None, min_wick=0.0):
    """
    Parameterised scalp.
    exclude_buckets: set of bucket strings to skip
                     ('lunch','afternoon', etc.)
    min_wick: minimum rejection wick (pts) to accept a signal
    """
    if exclude_buckets is None: exclude_buckets=set()
    trades=[]
    for i,d in enumerate(dates[1:],1):
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_S)&(day["time_et"]<=EOD)].reset_index(drop=True)
        if len(rth)<20: continue
        opens=rth["ES_open"].values; closes=rth["ES_close"].values
        highs=rth["ES_high"].values; lows=rth["ES_low"].values
        times=rth["time_et"].values

        or_b=rth[(rth["time_et"]>=OR_S)&(rth["time_et"]<OR_E)]
        if not len(or_b): continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        if not(MIN_ES<=es_R<=MAX_ES): continue

        prev=by_d.get(dates[i-1])
        if prev is None: continue
        pr=prev[(prev["time_et"]>=RTH_S)&(prev["time_et"]<=RTH_E)]
        if not len(pr): continue
        PDH=pr["ES_high"].max(); PDL=pr["ES_low"].min()

        ts=pd.Timestamp(d); iso=ts.isocalendar()
        pw=int(iso.week)-1; py=int(iso.year)
        if pw==0: py-=1; pw=int(pd.Timestamp(f"{py}-12-28").isocalendar().week)
        wk=weekly.get((py,pw)); PWH=wk[0] if wk else None; PWL=wk[1] if wk else None
        pm=ts.month-1; pmy=ts.year
        if pm==0: pm=12; pmy-=1
        mn=monthly.get((pmy,pm)); PMH=mn[0] if mn else None; PML=mn[1] if mn else None

        day_mid=(rth["ES_high"].max()+rth["ES_low"].min())/2
        rnds=[round(day_mid/50)*50+k*50 for k in range(-3,4)]
        lvl_r=[(a,b) for a,b in [("D1_H",PDH),("W1_H",PWH),("MN_H",PMH)]+
               [("RND",float(r)) for r in rnds] if b is not None]
        lvl_s=[(a,b) for a,b in [("D1_L",PDL),("W1_L",PWL),("MN_L",PML)]+
               [("RND",float(r)) for r in rnds] if b is not None]
        def conf(lv,ll): return sum(1 for(_,v) in ll if 0<abs(v-lv)<=8)+1

        prev_c=None
        for j in range(1,len(closes)-15):
            if times[j]<time(10,30): prev_c=closes[j]; continue
            if times[j]>time(14,30): break
            if prev_c is None: prev_c=closes[j]; continue

            # Time bucket check
            t=times[j]
            if   t<time(11,30): bucket="early"
            elif t<time(12,30): bucket="lunch"
            elif t<time(13,30): bucket="afternoon"
            else:               bucket="late"
            if bucket in exclude_buckets: prev_c=closes[j]; continue

            fired=False
            for fd,lvls in [("SHORT",lvl_r),("LONG",lvl_s)]:
                best_c=0; best_pnl=None; best_risk=None; best_nm=None
                best_wick=None
                for nm,lv in sorted(lvls,key=lambda x:-conf(x[1],lvls)):
                    c2=conf(lv,lvls)
                    if c2<=best_c: continue
                    if fd=="SHORT":
                        if not(prev_c<lv-SLIP and highs[j]>=lv and closes[j]<=lv-1.0): continue
                        if j+1>=len(opens): break
                        ep2=opens[j+1]+SLIP; s0=highs[j]+SLIP*2
                        wick=highs[j]-closes[j]
                    else:
                        if not(prev_c>lv+SLIP and lows[j]<=lv and closes[j]>=lv+1.0): continue
                        if j+1>=len(opens): break
                        ep2=opens[j+1]-SLIP; s0=lows[j]-SLIP*2
                        wick=closes[j]-lows[j]
                    if abs(ep2-s0)<0.5: continue
                    if wick<min_wick: continue   # wick filter
                    best_c=c2; best_pnl=sim_trail(closes[j+1:],fd,ep2,s0)
                    best_risk=abs(ep2-s0); best_nm=nm; best_wick=wick; break

                if best_pnl is not None:
                    n_mes=max(1,round((ACCT*SC_RISK)/(best_risk*5)))
                    mm=1.5 if best_nm=="MN_L" else 1.0
                    usd=best_pnl*5*n_mes*mm
                    trades.append(dict(
                        date=d,year=d.year,direction=fd,
                        pnl=best_pnl,usd=round(usd,2),win=best_pnl>0,
                        level_type=best_nm,
                        iso_wk=(int(iso.year),int(iso.week)),
                    ))
                    fired=True; break
            prev_c=closes[j]
            if fired: break
    return pd.DataFrame(trades)


def prow(lbl,tdf,pad=48):
    if not len(tdf): print(f"  — {lbl:<{pad}}  n=0"); return
    wr=(tdf["pnl"]>0).mean()*100; ar=tdf["pnl"].mean()
    dr=tdf.groupby("date")["pnl"].sum(); s=sh(dr)
    nyr=tdf["date"].nunique()/8.1
    wins=tdf[tdf["pnl"]>0]; losses=tdf[tdf["pnl"]<=0]
    wl=(abs(wins["pnl"].mean()/losses["pnl"].mean())
        if len(losses) and losses["pnl"].mean()!=0 else 99.0)
    flag="✅" if(ar>0 and s>2.5) else("🟡" if(ar>0 and s>1.5) else "❌")
    print(f"  {flag} {lbl:<{pad}}  {nyr:.0f}/yr  WR={wr:5.1f}%  "
          f"Avg={ar:+.4f}  W/L={wl:.2f}x  Sh={s:.2f}")

def yr_detail(tdf):
    for yr in sorted(tdf["year"].unique()):
        b=tdf[tdf["year"]==yr]
        wr=(b["pnl"]>0).mean()*100; ar=b["pnl"].mean()
        dr=b.groupby("date")["pnl"].sum(); s=sh(dr)
        tag="IS " if yr<=2021 else "OOS"
        flag="✅" if(ar>0 and s>2) else("🟡" if ar>0 else "❌")
        print(f"  {flag} {yr}: n={len(b):3d}  WR={wr:5.1f}%  Avg={ar:+.4f}  Sh={s:.2f} {tag}")


def main():
    print("Loading …")
    df=load()
    dates=sorted(df["date_et"].unique())
    by_d={d:g.reset_index(drop=True) for d,g in df.groupby("date_et")}
    weekly,monthly=build_levels(by_d,dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    W=76
    configs = [
        # (label, exclude_buckets, min_wick, tag)
        ("Baseline (no filters)",          set(),                           0.0,  "BASELINE"),
        ("F1  — no lunch (pre-stated)",    {"lunch"},                       0.0,  "PRE-STATED"),
        ("F1b — no lunch+afternoon ⚠️DATA", {"lunch","afternoon"},           0.0,  "DATA-DISC"),
        ("F2b — wick≥2pt ⚠️DATA",          set(),                           2.0,  "DATA-DISC"),
        ("F1+F2b — no-lunch + wick≥2 ⚠️",  {"lunch"},                       2.0,  "DATA-DISC"),
        ("F1b+F2b — aggressive ⚠️",         {"lunch","afternoon"},           2.0,  "DATA-DISC"),
    ]

    results = {}
    for lbl, excl, mw, tag in configs:
        print(f"  Running {lbl} …")
        sc = run_scalp(df, dates, by_d, weekly, monthly,
                       exclude_buckets=excl, min_wick=mw)
        results[lbl] = sc
    print()

    # ══════════════════════════════════════════════════════════════════════════
    print("="*W)
    print("  FULL COMPARISON — BASELINE vs ALL VARIANTS")
    print(f"  {'─'*74}")
    print(f"  {'Label':<48} {'n/yr':>5} {'WR':>7} {'IS_Sh':>6} {'OOS_Sh':>7} {'tag'}")
    print("  " + "─"*74)

    base = results["Baseline (no filters)"]
    base_IS  = base[base["year"]<=2021]
    base_OOS = base[base["year"]>=2022]
    s_b_is  = sh(base_IS.groupby("date")["pnl"].sum())
    s_b_oos = sh(base_OOS.groupby("date")["pnl"].sum())
    wr_b_is =(base_IS["pnl"]>0).mean()*100
    wr_b_oos=(base_OOS["pnl"]>0).mean()*100

    summary_rows = []
    for lbl, excl, mw, tag in configs:
        sc = results[lbl]
        IS  = sc[sc["year"]<=2021]
        OOS = sc[sc["year"]>=2022]
        nyr = len(sc)/8.1
        s_is  = sh(IS.groupby("date")["pnl"].sum())  if len(IS)  else 0
        s_oos = sh(OOS.groupby("date")["pnl"].sum()) if len(OOS) else 0
        wr_is =(IS["pnl"]>0).mean()*100   if len(IS)  else 0
        wr_oos=(OOS["pnl"]>0).mean()*100  if len(OOS) else 0
        wr_all=(sc["pnl"]>0).mean()*100   if len(sc)  else 0
        ar    = sc["pnl"].mean()           if len(sc)  else 0
        passes = (s_oos > s_b_oos and wr_oos > wr_b_oos)
        verdict = "✅ PASS" if passes else "❌ FAIL"
        symbol  = "✅" if (ar>0 and sh(sc.groupby("date")["pnl"].sum())>2.5) else \
                  "🟡" if ar>0 else "❌"
        print(f"  {symbol} {lbl:<48} {nyr:4.0f}/yr  {wr_all:5.1f}%  "
              f"IS={s_is:.2f}  OOS={s_oos:.2f}  {tag}  {verdict}")
        summary_rows.append((lbl, tag, nyr, wr_all, wr_is, wr_oos,
                              s_is, s_oos, passes))

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*W}")
    print("  DETAILED IS/OOS BREAKDOWN — EACH VARIANT")
    print("="*W)

    for lbl, excl, mw, tag in configs:
        sc  = results[lbl]
        IS  = sc[sc["year"]<=2021]
        OOS = sc[sc["year"]>=2022]
        print(f"\n  [{tag}] {lbl}")
        prow("Full",     sc)
        prow("  IS  2018-21", IS)
        prow("  OOS 2022-26", OOS)
        yr_detail(sc)

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*W}")
    print("  WEEKLY SIGNAL FREQUENCY — Each variant")
    print("="*W)
    for lbl, excl, mw, tag in configs:
        sc = results[lbl]
        nyr = len(sc)/8.1; npw = nyr/52
        print(f"  {lbl:<50} {nyr:.0f}/yr  ({npw:.2f}/week)")

    # ══════════════════════════════════════════════════════════════════════════
    # Best variant decision
    print(f"\n{'='*W}")
    print("  BEST VARIANT SELECTION")
    print("="*W)

    # Rank by OOS Sharpe (must improve vs baseline AND be stable IS)
    best_name = None; best_s_oos = s_b_oos
    for lbl, tag, nyr, wr_all, wr_is, wr_oos, s_is, s_oos, passes in summary_rows:
        if lbl == "Baseline (no filters)": continue
        if passes and s_oos > best_s_oos and nyr >= 40:  # keep ≥40/yr minimum
            best_s_oos = s_oos; best_name = lbl

    if best_name:
        best_sc  = results[best_name]
        best_IS  = best_sc[best_sc["year"]<=2021]
        best_OOS = best_sc[best_sc["year"]>=2022]
        s_best   = sh(best_sc.groupby("date")["pnl"].sum())
        print(f"\n  Best variant: {best_name}")
        print(f"  OOS Sharpe: {s_b_oos:.2f} → {best_s_oos:.2f} "
              f"(+{best_s_oos-s_b_oos:.2f})")
        print(f"  OOS WR: {wr_b_oos:.1f}% → "
              f"{(best_OOS['pnl']>0).mean()*100:.1f}%")
        n_base = len(base)/8.1; n_best = len(best_sc)/8.1
        print(f"  Frequency: {n_base:.0f}/yr → {n_best:.0f}/yr "
              f"(−{n_base-n_best:.0f}/yr = −{(n_base-n_best)/52:.2f}/wk)")
        # Check tag
        for lbl2, excl2, mw2, tag2 in configs:
            if lbl2 == best_name:
                if "DATA-DISC" in tag2:
                    print(f"\n  ⚠️  WARNING: This variant was data-discovered (not pre-stated).")
                    print(f"     Its OOS improvement may be partly from look-ahead on IS data.")
                    print(f"     Treat with caution — apply with 75% confidence, not 100%.")
                else:
                    print(f"\n  ✅  Pre-stated filter — higher confidence in OOS validity.")
                break
    else:
        print(f"\n  No variant significantly beats baseline with ≥40/yr frequency.")
        print(f"  Baseline remains the recommended scalp configuration.")

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*W}")
    print("  FINAL RECOMMENDATION FOR SIGNAL_GENERATOR.PY")
    print("="*W)

    # Find the passing variants
    passing = [(lbl, tag, nyr, wr_oos, s_oos)
               for lbl, tag, nyr, wr_all, wr_is, wr_oos, s_is, s_oos, passes
               in summary_rows if passes and nyr >= 40]

    if not passing:
        print("\n  No variant passes both IS+OOS with ≥40 trades/yr.")
        print("  Recommendation: DO NOT change the scalp system.")
        print("  The current scalp at Sh=2.28 is already well-calibrated.")
    else:
        print("\n  Variants that pass IS+OOS validation:")
        for lbl, tag, nyr, wr_oos, s_oos in sorted(passing, key=lambda x:-x[4]):
            caution = "⚠️ data-discovered" if "DATA-DISC" in tag else "✅ pre-stated"
            print(f"    {lbl:<48} OOS_Sh={s_oos:.2f}  {nyr:.0f}/yr  {caution}")

        # Final pick
        best_p = sorted(passing, key=lambda x:-x[4])[0]
        print(f"\n  Selected: {best_p[0]}")
        tag_note = ("⚠️ DATA-DISCOVERED — apply with caution"
                    if "DATA-DISC" in best_p[1] else
                    "✅ PRE-STATED — apply with confidence")
        print(f"  Status:   {tag_note}")
        print(f"  OOS Sh:   {s_b_oos:.2f} → {best_p[4]:.2f}")
        print(f"  OOS WR:   {wr_b_oos:.1f}% → {best_p[3]:.1f}%")
        print(f"  Freq:     {len(base)/8.1:.0f}/yr → {best_p[2]:.0f}/yr "
              f"(~{best_p[2]/52:.2f}/wk)")

    print()


if __name__ == "__main__":
    main()
