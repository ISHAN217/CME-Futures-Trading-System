#!/usr/bin/env python3
"""
backtest_scalp_improved.py — Scalp WR improvement study

ANTI-OVERFITTING PROTOCOL:
  1. Hypotheses stated BEFORE looking at results (economic rationale first)
  2. IS = 2018-2021 for filter design; OOS = 2022-2026 for validation
  3. Each filter tested independently first (no stacking until individually validated)
  4. Parameters kept to absolute minimum — only round numbers, no grid search
  5. If IS improves but OOS doesn't confirm → reject the filter
  6. Report all trades lost by each filter (transparency on frequency impact)

BASELINE: Current scalp — WR=44.7%, Sh=2.28, ~127/yr

FILTER 1 — TIME OF DAY
  Economic rationale: ES/NQ has documented U-shaped volume curve. The 11:30–12:30
  "lunch" window has thinner participation — price touches levels and reverses for no
  structural reason. The 12:30–14:30 afternoon window is more committed.
  Pre-stated hypothesis: Excluding 11:30–12:30 will increase WR.

FILTER 2 — REJECTION WICK QUALITY
  Economic rationale: A large wick (bar_high far from bar_close for SHORT) indicates
  forceful institutional selling at the level. A tiny wick means price barely touched
  the level — no structural rejection. Larger minimum wick = higher conviction signal.
  Pre-stated hypothesis: Requiring wick ≥ 1.5pt will increase WR, fewer trades.

FILTER 3 — NQ CORROBORATION
  Economic rationale: The primary ORB system requires ES+NQ to agree. The scalp
  currently ignores NQ. If ES tests PDH but NQ is in the middle of its day range,
  there's no corroboration. When NQ is also in its upper 25% (for SHORT) or lower
  25% (for LONG), the move is more structural.
  Pre-stated hypothesis: Requiring NQ to corroborate will increase WR, fewer trades.
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25
RTH_S = time(9,30); RTH_E = time(16,0)
OR_S  = time(9,30); OR_E  = time(10,0)
EOD   = time(15,30)
MIN_ES=8; MAX_ES=60
TRAIL=3.0; BE=3.0
ACCT=25000; SC_RISK=0.01
IS_END   = date(2021,12,31)
OS_START = date(2022,1,1)


# ─────────────────────────────────────────────────────────
def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)

def build_levels(by_d, dates):
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

def sim_trail(fwd, di, ep, s0):
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


# ─────────────────────────────────────────────────────────
# INSTRUMENT SCALP — returns rich trade records with metadata
# ─────────────────────────────────────────────────────────
def run_scalp_instrumented(df, dates, by_d, weekly, monthly):
    """
    Identical logic to final_system run_scalp but stores extra metadata
    per trade for post-hoc analysis:
      signal_time  : time of signal bar
      wick_size    : abs(bar_high - bar_close) for SHORT,
                     abs(bar_close - bar_low)  for LONG
      nq_pos_pct   : NQ close at signal time as % of NQ day range so far
                     (0 = at day low, 100 = at day high)
      nq_in_top25  : bool — NQ close in top 25% of NQ day range (for SHORT signals)
      nq_in_bot25  : bool — NQ close in bot 25% of NQ day range (for LONG signals)
      time_bucket  : 'early'(10:30-11:30), 'lunch'(11:30-12:30),
                     'afternoon'(12:30-13:30), 'late'(13:30-14:30)
    """
    trades=[]
    for i,d in enumerate(dates[1:],1):
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_S)&(day["time_et"]<=EOD)].reset_index(drop=True)
        if len(rth)<20: continue
        opens=rth["ES_open"].values;  closes=rth["ES_close"].values
        highs=rth["ES_high"].values;  lows=rth["ES_low"].values
        times=rth["time_et"].values
        nq_closes=rth["NQ_close"].values
        nq_highs =rth["NQ_high"].values
        nq_lows  =rth["NQ_low"].values

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
                    best_c=c2
                    best_pnl=sim_trail(closes[j+1:],fd,ep2,s0)
                    best_risk=abs(ep2-s0); best_nm=nm; best_wick=wick; break

                if best_pnl is not None:
                    n_mes=max(1,round((ACCT*SC_RISK)/(best_risk*5)))
                    mm=1.5 if best_nm=="MN_L" else 1.0
                    usd=best_pnl*5*n_mes*mm

                    # NQ position at signal bar
                    nq_day_h=nq_highs[:j+1].max(); nq_day_l=nq_lows[:j+1].min()
                    nq_rng=nq_day_h-nq_day_l
                    nq_pos_pct = ((nq_closes[j]-nq_day_l)/nq_rng*100
                                  if nq_rng>1 else 50.0)
                    nq_in_top25 = nq_pos_pct >= 75
                    nq_in_bot25 = nq_pos_pct <= 25

                    t=times[j]
                    if   t<time(11,30): bucket="early"
                    elif t<time(12,30): bucket="lunch"
                    elif t<time(13,30): bucket="afternoon"
                    else:               bucket="late"

                    trades.append(dict(
                        date=d, year=d.year, direction=fd,
                        pnl=best_pnl, usd=round(usd,2), win=best_pnl>0,
                        level_type=best_nm,
                        signal_time=t, time_bucket=bucket,
                        wick_size=round(best_wick,2),
                        nq_pos_pct=round(nq_pos_pct,1),
                        nq_in_top25=nq_in_top25, nq_in_bot25=nq_in_bot25,
                        iso_wk=(int(iso.year),int(iso.week)),
                    ))
                    fired=True; break
            prev_c=closes[j]
            if fired: break
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────
def prow(lbl, tdf, pad=44):
    if not len(tdf):
        print(f"  — {lbl:<{pad}}  n=0"); return
    wr=(tdf["pnl"]>0).mean()*100
    ar=tdf["pnl"].mean()
    dr=tdf.groupby("date")["pnl"].sum(); s=sh(dr)
    nyr=tdf["date"].nunique()/8.1
    wins=tdf[tdf["pnl"]>0]; losses=tdf[tdf["pnl"]<=0]
    wl=(abs(wins["pnl"].mean()/losses["pnl"].mean())
        if len(losses) and losses["pnl"].mean()!=0 else 99.0)
    flag="✅" if(ar>0 and s>2.5) else("🟡" if(ar>0 and s>1.5) else "❌")
    print(f"  {flag} {lbl:<{pad}}  {nyr:.0f}/yr  WR={wr:5.1f}%  "
          f"Avg={ar:+.4f}  W/L={wl:.2f}x  Sh={s:.2f}")

def compare(lbl, full, filtered, note=""):
    n_full=len(full); n_filt=len(filtered)
    drop=n_full-n_filt
    if not n_full: return
    dr_full=full.groupby("date")["pnl"].sum()
    dr_filt=filtered.groupby("date")["pnl"].sum() if n_filt else pd.Series(dtype=float)
    s_full=sh(dr_full)
    s_filt=sh(dr_filt) if len(dr_filt) else 0.0
    wr_full=(full["pnl"]>0).mean()*100
    wr_filt=(filtered["pnl"]>0).mean()*100 if n_filt else 0.0
    delta_wr=wr_filt-wr_full; delta_sh=s_filt-s_full
    arrow_wr="↑" if delta_wr>0 else "↓"
    arrow_sh="↑" if delta_sh>0 else "↓"
    print(f"  {lbl:<46}  n: {n_full}→{n_filt} (−{drop})  "
          f"WR: {wr_full:.1f}%→{wr_filt:.1f}% ({arrow_wr}{abs(delta_wr):.1f}pp)  "
          f"Sh: {s_full:.2f}→{s_filt:.2f} ({arrow_sh}{abs(delta_sh):.2f}){note}")


def main():
    print("Loading …")
    df=load()
    dates=sorted(df["date_et"].unique())
    by_d={d:g.reset_index(drop=True) for d,g in df.groupby("date_et")}
    weekly,monthly=build_levels(by_d,dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    print("Running instrumented scalp …")
    sc=run_scalp_instrumented(df,dates,by_d,weekly,monthly)
    print(f"  → {len(sc)} trades ({len(sc)/8.1:.0f}/yr)\n")

    IS  = sc[sc["year"]<=2021]
    OOS = sc[sc["year"]>=2022]
    W=72

    # ══════════════════════════════════════════════════════════════════════════
    print("="*W)
    print("  BASELINE — CURRENT SCALP")
    print("="*W)
    prow("Full period  2018-2026", sc)
    prow("IS  2018-2021",          IS)
    prow("OOS 2022-2026",          OOS)

    # ── DIAGNOSTIC 1: Time-of-day breakdown ──────────────────────────────────
    print(f"\n{'─'*W}")
    print("  DIAGNOSTIC 1 — By time bucket")
    print(f"  {'─'*70}")
    print("  Pre-hypothesis: lunch (11:30-12:30) should be weakest bucket\n")
    for bk in ["early","lunch","afternoon","late"]:
        sub=sc[sc["time_bucket"]==bk]
        if not len(sub): continue
        lbl=f"  {bk:12} ({bk})"
        prow(f"{bk:<12} (10:30-11:30 ≡ early, etc.)", sub)
    print()
    # Bucket detail
    for bk,window in [("early","10:30–11:30"),("lunch","11:30–12:30"),
                      ("afternoon","12:30–13:30"),("late","13:30–14:30")]:
        sub=sc[sc["time_bucket"]==bk]
        if not len(sub): continue
        wr=(sub["pnl"]>0).mean()*100; ar=sub["pnl"].mean()
        nyr=len(sub)/8.1
        print(f"    {bk:<12} {window}  :  {nyr:.0f}/yr  WR={wr:5.1f}%  Avg={ar:+.4f}")

    # ── DIAGNOSTIC 2: Wick size breakdown ────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  DIAGNOSTIC 2 — By rejection wick size")
    print(f"  {'─'*70}")
    print("  Pre-hypothesis: larger wick = more forceful rejection = higher WR\n")
    bins=[(0,1,"<1pt"),(1,2,"1–2pt"),(2,3,"2–3pt"),(3,99,">3pt")]
    for lo,hi,lbl in bins:
        sub=sc[(sc["wick_size"]>=lo)&(sc["wick_size"]<hi)]
        if not len(sub): continue
        wr=(sub["pnl"]>0).mean()*100; ar=sub["pnl"].mean()
        nyr=len(sub)/8.1
        print(f"    wick {lbl:<8}  :  {nyr:.0f}/yr  WR={wr:5.1f}%  Avg={ar:+.4f}")

    # ── DIAGNOSTIC 3: NQ position breakdown ──────────────────────────────────
    print(f"\n{'─'*W}")
    print("  DIAGNOSTIC 3 — NQ corroboration")
    print(f"  {'─'*70}")
    print("  Pre-hypothesis: NQ in top/bot 25% should confirm ES signal → higher WR\n")

    sh_all=sc[sc["direction"]=="SHORT"]
    lg_all=sc[sc["direction"]=="LONG"]

    for label,sub,corr_col in [
        ("SHORT — NQ in top 25%",  sh_all[sh_all["nq_in_top25"]],  "nq_in_top25"),
        ("SHORT — NQ NOT top 25%", sh_all[~sh_all["nq_in_top25"]], "nq_in_top25"),
        ("LONG  — NQ in bot 25%",  lg_all[lg_all["nq_in_bot25"]],  "nq_in_bot25"),
        ("LONG  — NQ NOT bot 25%", lg_all[~lg_all["nq_in_bot25"]], "nq_in_bot25"),
    ]:
        if not len(sub): continue
        wr=(sub["pnl"]>0).mean()*100; ar=sub["pnl"].mean()
        nyr=len(sub)/8.1
        print(f"    {label:<35}  :  {nyr:.0f}/yr  WR={wr:5.1f}%  Avg={ar:+.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*W}")
    print("  FILTER 1 — TIME OF DAY  (exclude lunch 11:30–12:30)")
    print("="*W)
    print("  Rationale: thin liquidity → spurious level touches in lunch window\n")

    f1 = sc[sc["time_bucket"] != "lunch"]
    f1_IS  = f1[f1["year"]<=2021]
    f1_OOS = f1[f1["year"]>=2022]

    compare("Baseline vs Filter-1 (full period)", sc, f1)
    compare("Baseline vs Filter-1 (IS  2018-21)", IS,  f1_IS)
    compare("Baseline vs Filter-1 (OOS 2022-26)", OOS, f1_OOS)
    print()
    prow("Filter-1 full period", f1)
    prow("Filter-1 IS  2018-21", f1_IS)
    prow("Filter-1 OOS 2022-26", f1_OOS)

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*W}")
    print("  FILTER 2 — REJECTION WICK ≥ threshold")
    print("="*W)
    print("  Rationale: larger wick = genuine institutional rejection, not random noise\n")

    print("  Parameter sweep (pre-committed — pick the one with consistent IS+OOS):")
    for thr in [0.5, 1.0, 1.5, 2.0, 2.5]:
        fx = sc[sc["wick_size"] >= thr]
        fx_IS  = fx[fx["year"]<=2021]
        fx_OOS = fx[fx["year"]>=2022]
        s_full_is  = sh(IS.groupby("date")["pnl"].sum())
        s_filt_is  = sh(fx_IS.groupby("date")["pnl"].sum()) if len(fx_IS) else 0
        s_filt_oos = sh(fx_OOS.groupby("date")["pnl"].sum()) if len(fx_OOS) else 0
        wr_f=(fx["pnl"]>0).mean()*100 if len(fx) else 0
        wr_is=(fx_IS["pnl"]>0).mean()*100 if len(fx_IS) else 0
        wr_oos=(fx_OOS["pnl"]>0).mean()*100 if len(fx_OOS) else 0
        print(f"    wick ≥ {thr:.1f}pt  :  n={len(fx):4d}({len(fx)/8.1:.0f}/yr)  "
              f"WR={wr_f:5.1f}%  IS_Sh={s_filt_is:.2f}  OOS_Sh={s_filt_oos:.2f}  "
              f"IS_WR={wr_is:.1f}%  OOS_WR={wr_oos:.1f}%")

    print()
    best_wick = 1.5   # pre-commit to round number with clear rationale
    f2 = sc[sc["wick_size"] >= best_wick]
    f2_IS  = f2[f2["year"]<=2021]
    f2_OOS = f2[f2["year"]>=2022]
    print(f"  Selected: wick ≥ {best_wick}pt (pre-committed round number)\n")
    compare("Baseline vs Filter-2 (full period)", sc,  f2)
    compare("Baseline vs Filter-2 (IS  2018-21)", IS,  f2_IS)
    compare("Baseline vs Filter-2 (OOS 2022-26)", OOS, f2_OOS)
    print()
    prow("Filter-2 full period", f2)
    prow("Filter-2 IS  2018-21", f2_IS)
    prow("Filter-2 OOS 2022-26", f2_OOS)

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*W}")
    print("  FILTER 3 — NQ CORROBORATION")
    print("="*W)
    print("  Rationale: ES+NQ agreement required for primary ORB; same logic for scalp")
    print("  SHORT: only trade if NQ also in top 25% of day range so far")
    print("  LONG : only trade if NQ also in bot 25% of day range so far\n")

    f3_short = sc[(sc["direction"]=="SHORT") & sc["nq_in_top25"]]
    f3_long  = sc[(sc["direction"]=="LONG")  & sc["nq_in_bot25"]]
    f3 = pd.concat([f3_short, f3_long]).sort_values("date")
    f3_IS    = f3[f3["year"]<=2021]
    f3_OOS   = f3[f3["year"]>=2022]

    compare("Baseline vs Filter-3 (full period)", sc,  f3)
    compare("Baseline vs Filter-3 (IS  2018-21)", IS,  f3_IS)
    compare("Baseline vs Filter-3 (OOS 2022-26)", OOS, f3_OOS)
    print()
    prow("Filter-3 full period", f3)
    prow("Filter-3 IS  2018-21", f3_IS)
    prow("Filter-3 OOS 2022-26", f3_OOS)

    # ══════════════════════════════════════════════════════════════════════════
    # Combine only filters that worked in BOTH IS and OOS
    print(f"\n{'='*W}")
    print("  COMBINED — Only validated filters (must pass both IS + OOS)")
    print("="*W)
    print("  Validation criterion: OOS Sharpe must improve AND OOS WR must improve\n")

    # Evaluate each filter: IS Sharpe, OOS Sharpe
    def eval_filter(name, full_sub, is_sub, oos_sub, base_is, base_oos):
        s_full = sh(full_sub.groupby("date")["pnl"].sum()) if len(full_sub) else 0
        s_is   = sh(is_sub.groupby("date")["pnl"].sum())  if len(is_sub)   else 0
        s_oos  = sh(oos_sub.groupby("date")["pnl"].sum()) if len(oos_sub)  else 0
        s_b_is = sh(base_is.groupby("date")["pnl"].sum())
        s_b_oos= sh(base_oos.groupby("date")["pnl"].sum())
        wr_is  = (is_sub["pnl"]>0).mean()*100  if len(is_sub)  else 0
        wr_oos = (oos_sub["pnl"]>0).mean()*100 if len(oos_sub) else 0
        wr_b_is  = (base_is["pnl"]>0).mean()*100
        wr_b_oos = (base_oos["pnl"]>0).mean()*100
        valid = (s_oos > s_b_oos) and (wr_oos > wr_b_oos)
        flag  = "✅ PASSES" if valid else "❌ FAILS"
        print(f"  {flag}  {name}")
        print(f"         IS  WR: {wr_b_is:.1f}% → {wr_is:.1f}%  Sh: {s_b_is:.2f} → {s_is:.2f}")
        print(f"         OOS WR: {wr_b_oos:.1f}% → {wr_oos:.1f}%  Sh: {s_b_oos:.2f} → {s_oos:.2f}")
        return valid

    v1 = eval_filter("Filter-1 (no lunch)", f1, f1_IS, f1_OOS, IS, OOS)
    print()
    v2 = eval_filter(f"Filter-2 (wick≥{best_wick}pt)", f2, f2_IS, f2_OOS, IS, OOS)
    print()
    v3 = eval_filter("Filter-3 (NQ corroboration)", f3, f3_IS, f3_OOS, IS, OOS)

    # Build combined from validated filters only
    validated = []
    base_comb = sc.copy()
    if v1: validated.append("no-lunch"); base_comb = base_comb[base_comb["time_bucket"] != "lunch"]
    if v2: validated.append(f"wick≥{best_wick}"); base_comb = base_comb[base_comb["wick_size"] >= best_wick]
    if v3:
        validated.append("NQ-corr")
        short_v = base_comb[(base_comb["direction"]=="SHORT") & base_comb["nq_in_top25"]]
        long_v  = base_comb[(base_comb["direction"]=="LONG")  & base_comb["nq_in_bot25"]]
        base_comb = pd.concat([short_v, long_v]).sort_values("date")

    print(f"\n  Validated filters: {validated if validated else ['NONE']}")

    if validated:
        comb_IS  = base_comb[base_comb["year"]<=2021]
        comb_OOS = base_comb[base_comb["year"]>=2022]
        print()
        print("  COMBINED RESULT (validated filters only):")
        prow("Baseline (unfiltered)",   sc)
        prow("Combined filtered",        base_comb)
        prow("  IS  2018-2021",          comb_IS)
        prow("  OOS 2022-2026",          comb_OOS)
    else:
        print("\n  No filters passed both IS + OOS validation.")
        print("  VERDICT: Current scalp baseline cannot be improved without overfitting.")

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*W}")
    print("  YEAR-BY-YEAR DETAIL (Baseline vs Best Combined)")
    print("="*W)
    print(f"  {'Year':<6}  {'n':>4}  {'WR':>7}  {'AvgPnl':>8}  {'Sh':>6}  "
          f"│  {'n':>4}  {'WR':>7}  {'AvgPnl':>8}  {'Sh':>6}  tag")
    print(f"  {'':─<6}  {'─'*4}  {'─'*7}  {'─'*8}  {'─'*6}  "
          f"│  {'─'*4}  {'─'*7}  {'─'*8}  {'─'*6}")
    print(f"  {'':6}  {'BASELINE':^28}  │  {'COMBINED':^28}")

    for yr in sorted(sc["year"].unique()):
        b = sc[sc["year"]==yr]
        c = base_comb[base_comb["year"]==yr] if validated else b
        tag = "IS " if yr<=2021 else "OOS"
        wr_b=(b["pnl"]>0).mean()*100;  ar_b=b["pnl"].mean()
        wr_c=(c["pnl"]>0).mean()*100  if len(c) else 0.0
        ar_c=c["pnl"].mean()           if len(c) else 0.0
        dr_b=b.groupby("date")["pnl"].sum(); sh_b=sh(dr_b)
        dr_c=c.groupby("date")["pnl"].sum(); sh_c=sh(dr_c)
        print(f"  {yr}   {len(b):4d}  {wr_b:6.1f}%  {ar_b:+.4f}  {sh_b:6.2f}  "
              f"│  {len(c):4d}  {wr_c:6.1f}%  {ar_c:+.4f}  {sh_c:6.2f}  {tag}")

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*W}")
    print("  FINAL HONEST ASSESSMENT")
    print("="*W)
    s_base = sh(sc.groupby("date")["pnl"].sum())
    s_comb = sh(base_comb.groupby("date")["pnl"].sum()) if validated else s_base
    n_base = len(sc)/8.1
    n_comb = len(base_comb)/8.1 if validated else n_base
    wr_base=(sc["pnl"]>0).mean()*100
    wr_comb=(base_comb["pnl"]>0).mean()*100 if validated else wr_base

    print(f"\n  Baseline : {n_base:.0f}/yr  WR={wr_base:.1f}%  Sh={s_base:.2f}")
    print(f"  Combined : {n_comb:.0f}/yr  WR={wr_comb:.1f}%  Sh={s_comb:.2f}")
    print(f"  Filters  : {validated if validated else ['none validated']}")
    if validated:
        trade_loss = n_base - n_comb
        wr_gain    = wr_comb - wr_base
        sh_gain    = s_comb - s_base
        print(f"\n  Cost : −{trade_loss:.0f} trades/yr  ({trade_loss/52:.2f}/wk fewer)")
        print(f"  Gain : +{wr_gain:.1f}pp WR,  Sharpe {s_base:.2f} → {s_comb:.2f}")
        verdict = "ADD" if (sh_gain > 0.3 and wr_gain > 2) else "MARGINAL"
        print(f"\n  Verdict: {verdict}")
        if verdict == "ADD":
            print("  Recommendation: Apply validated filter(s) to signal_generator.py")
        else:
            print("  Recommendation: Gain is too small to justify added complexity/risk")
    else:
        print("\n  Verdict: NO IMPROVEMENT FOUND")
        print("  Current scalp is already near the quality ceiling for this pattern type.")
        print("  Adding filters would reduce frequency without confirmed WR improvement.")

    print()


if __name__ == "__main__":
    main()
