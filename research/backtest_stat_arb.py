#!/usr/bin/env python3
"""
backtest_stat_arb.py — Stat Arb + ORB concepts

Tests three distinct stat arb ideas:

  A. ES/NQ SPREAD OR BREAKOUT
     Compute the normalised ES/NQ spread (each instrument's return
     divided by its OR range) during 9:30-10:00. After 10:00, watch
     for the SPREAD to break out of its own OR range. Long ES/Short NQ
     when spread breaks up (ES outperforming), or reverse.
     Market-neutral — doesn't bet on market direction.

  B. ONE-SIDED DIVERGENCE CATCHUP
     Current system skips days where only ONE instrument breaks OR.
     Instead: when ES breaks OR high but NQ doesn't (or vice versa),
     LONG the LAGGARD expecting it to catch up. Bets on correlation
     mean-reverting (the pair re-aligning).

  C. CORR-ADJUSTED ORB SIZE
     Compute rolling 20-bar correlation of ES/NQ returns intraday.
     When correlation is LOWER than usual (instruments diverging),
     reduce ORB size (lower conviction). When correlation is high,
     increase size. Tests whether intraday correlation predicts ORB WR.

  D. POST-ORB SPREAD TRADE
     After ORB fires LONG on both legs, if after 30 minutes ES has
     moved >2R but NQ only <1R (spread extended), close ES and double
     NQ (expecting NQ catchup). Pure spread trade on existing position.
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"

OR_START  = time(9,30); OR_END    = time(10,0)
ORB_CUT   = time(10,30); EOD      = time(15,30)
RTH_START = time(9,30); RTH_END   = time(16,0)
SLIP=0.25; MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=175


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def sim_leg(day, entry_t, direction, ep, sp, tp, ch, cl, cc):
    rng = abs(ep - sp)
    for _, bar in day[day["time_et"] > entry_t].iterrows():
        t = bar["time_et"]
        if t >= EOD:
            sign = 1 if direction == "LONG" else -1
            net  = sign*(bar[cc]-ep) - SLIP*2
            return round(net/rng, 3), "EOD"
        if direction == "LONG":
            if bar[cl] <= sp: return round(-1.0-SLIP*2/rng, 3), "STOP"
            if bar[ch] >= tp: return round((tp-ep)/rng-SLIP/rng, 3), "TGT"
        else:
            if bar[ch] >= sp: return round(-1.0-SLIP*2/rng, 3), "STOP"
            if bar[cl] <= tp: return round((ep-tp)/rng-SLIP/rng, 3), "TGT"
    return 0.0, "NONE"


# ─── A. Spread OR Breakout ─────────────────────────────────────────────────────

def run_spread_orb(df, dates, by_d, pmult=2.0, window_end=time(11,0)):
    """
    Normalise each instrument's price by its OR range to get a dimensionless
    relative return. The SPREAD of those normalised returns has its own OR.
    Trade when the spread breaks out of its OR range after 10:00.

    Long ES + Short NQ when spread breaks up (ES outperforming).
    Long NQ + Short ES when spread breaks down.

    Size: 1 unit of each leg (equal normalised risk — OR range sets leg size).
    Stop: spread retraces half its OR range back through the OR boundary.
    Target: pmult × spread_OR_range from entry.
    """
    trades = []
    for d in dates:
        day = by_d.get(d)
        if day is None: continue
        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if len(or_b) < 10: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        es_open = or_b["ES_open"].iloc[0]; nq_open = or_b["NQ_open"].iloc[0]

        # Compute normalised spread during OR
        spread_vals = []
        for _, row in or_b.iterrows():
            es_ret = (row["ES_close"] - es_open) / es_R
            nq_ret = (row["NQ_close"] - nq_open) / nq_R
            spread_vals.append(es_ret - nq_ret)

        sp_H = max(spread_vals); sp_L = min(spread_vals)
        sp_R = sp_H - sp_L
        if sp_R < 0.05: continue   # trivially small spread range

        # Watch window
        watch = day[(day["time_et"]>=OR_END)&(day["time_et"]<window_end)]
        signal_t = direction = None; entry_spread = None

        for _, bar in watch.iterrows():
            es_ret = (bar["ES_close"] - es_open) / es_R
            nq_ret = (bar["NQ_close"] - nq_open) / nq_R
            cur_sp = es_ret - nq_ret

            if cur_sp > sp_H and direction is None:
                direction = "ES_LEADS"  # long ES, short NQ
                signal_t  = bar["time_et"]; entry_spread = cur_sp; break
            if cur_sp < sp_L and direction is None:
                direction = "NQ_LEADS"  # long NQ, short ES
                signal_t  = bar["time_et"]; entry_spread = cur_sp; break

        if direction is None: continue

        # Target: spread moves pmult × sp_R further in breakout direction
        if direction == "ES_LEADS":
            # Long ES @ market, Short NQ @ market
            tgt_spread = entry_spread + pmult * sp_R
            stop_spread = sp_H   # spread retreats to OR high boundary = stop
        else:
            tgt_spread = entry_spread - pmult * sp_R
            stop_spread = sp_L

        # Simulate: track spread over time
        after = day[day["time_et"] > signal_t]
        result = "NONE"; r_val = 0.0
        for _, bar in after.iterrows():
            t = bar["time_et"]
            if t >= EOD:
                es_ret = (bar["ES_close"]-es_open)/es_R
                nq_ret = (bar["NQ_close"]-nq_open)/nq_R
                cur_sp = es_ret - nq_ret
                move   = (cur_sp - entry_spread) if direction=="ES_LEADS" else (entry_spread - cur_sp)
                r_val  = round(move / sp_R, 3); result = "EOD"; break
            es_ret = (bar["ES_close"]-es_open)/es_R
            nq_ret = (bar["NQ_close"]-nq_open)/nq_R
            cur_sp = es_ret - nq_ret
            if direction == "ES_LEADS":
                if cur_sp <= stop_spread: r_val=-1.0; result="STOP"; break
                if cur_sp >= tgt_spread:  r_val=pmult; result="TGT";  break
            else:
                if cur_sp >= stop_spread: r_val=-1.0; result="STOP"; break
                if cur_sp <= tgt_spread:  r_val=pmult; result="TGT";  break

        trades.append(dict(date=d, year=d.year, direction=direction,
                           r=r_val, win=r_val>0, exit=result,
                           entry_min=signal_t.hour*60+signal_t.minute-600,
                           dow=pd.Timestamp(d).day_name()))
    return pd.DataFrame(trades)


# ─── B. One-sided divergence catchup ──────────────────────────────────────────

def run_catchup(df, dates, by_d, pmult=3.0):
    """
    When only ONE instrument breaks OR (other stays inside), LONG the laggard
    expecting it to catch up. Entry at market, stop = OR range, target = pmult×R.
    """
    trades = []
    for d in dates:
        day = by_d.get(d)
        if day is None: continue
        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if len(or_b)==0: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        post = day[(day["time_et"]>=OR_END)&(day["time_et"]<ORB_CUT)]
        es_b=nq_b=None; es_row=nq_row=None
        for _,r in post.iterrows():
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"; es_row=r
                elif r["ES_low"]<es_L: es_b="S"; es_row=r
            if nq_b is None:
                if r["NQ_high"]>nq_H: nq_b="L"; nq_row=r
                elif r["NQ_low"]<nq_L: nq_b="S"; nq_row=r

        # Both broke → normal ORB day, skip (not a divergence)
        if es_b and nq_b: continue
        # Neither broke → no signal
        if not es_b and not nq_b: continue

        # One-sided break — trade the LAGGARD
        if es_b and not nq_b:
            leader="ES"; laggard="NQ"
            direction = "LONG" if es_b=="L" else "SHORT"
            signal_row = es_row
            # Entry: close of the leader's breakout bar = when we know the divergence
            ep  = signal_row["NQ_close"] + SLIP if direction=="LONG" else signal_row["NQ_close"]-SLIP
            sp  = ep - nq_R if direction=="LONG" else ep + nq_R
            tp  = ep + pmult*nq_R if direction=="LONG" else ep - pmult*nq_R
            rv, why = sim_leg(day, signal_row["time_et"], direction,
                              ep, sp, tp, "NQ_high","NQ_low","NQ_close")
        else:
            leader="NQ"; laggard="ES"
            direction = "LONG" if nq_b=="L" else "SHORT"
            signal_row = nq_row
            ep  = signal_row["ES_close"] + SLIP if direction=="LONG" else signal_row["ES_close"]-SLIP
            sp  = ep - es_R if direction=="LONG" else ep + es_R
            tp  = ep + pmult*es_R if direction=="LONG" else ep - pmult*es_R
            rv, why = sim_leg(day, signal_row["time_et"], direction,
                              ep, sp, tp, "ES_high","ES_low","ES_close")

        trades.append(dict(date=d, year=d.year, leader=leader, laggard=laggard,
                           direction=direction, r=rv, win=rv>0, exit=why,
                           dow=pd.Timestamp(d).day_name()))
    return pd.DataFrame(trades)


# ─── C. Spread correlation filter for ORB ────────────────────────────────────

def run_corr_filter(df, dates, by_d, pmult=3.0, high_corr_thresh=0.85):
    """
    Compute the rolling intraday correlation of ES/NQ 1-min returns during the OR.
    High correlation days → both instruments agreeing → trade ORB normally.
    Low correlation days → divergence, uncertain → skip ORB.
    Returns two groups: high-corr ORB and low-corr ORB trades.
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
        corr = es_rets.corr(nq_rets)
        high_corr = (corr >= high_corr_thresh)

        # ORB signal
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

        direction = "LONG" if es_b=="L" else "SHORT"
        et = entry_row["time_et"]

        for sym,H,L,R,ch,cl,cc in [
            ("ES",es_H,es_L,es_R,"ES_high","ES_low","ES_close"),
            ("NQ",nq_H,nq_L,nq_R,"NQ_high","NQ_low","NQ_close"),
        ]:
            ep = H+SLIP if direction=="LONG" else L-SLIP
            sp = L-SLIP if direction=="LONG" else H+SLIP
            tp = ep+pmult*R if direction=="LONG" else ep-pmult*R
            rv,why=sim_leg(day,et,direction,ep,sp,tp,ch,cl,cc)
            trades.append(dict(date=d,year=d.year,sym=sym,direction=direction,
                               r=rv,win=rv>0,exit=why,
                               corr=round(corr,3),high_corr=high_corr,
                               dow=pd.Timestamp(d).day_name()))
    return pd.DataFrame(trades)


def prow(label, sub, pad=38):
    if not len(sub): print(f"  {label:<{pad}}  n=0"); return
    wr=sub["win"].mean(); ar=sub["r"].mean()
    dr=sub.groupby("date")["r"].sum()
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    n_yr=len(sub)/8.1/(2 if "sym" in sub.columns else 1)
    flag="✅" if wr>0.52 else ("🟡" if wr>0.46 else "❌")
    print(f"  {flag} {label:<{pad}}  n={len(sub):4d} ({n_yr:.0f}/yr)  "
          f"WR={wr*100:5.1f}%  AvgR={ar:+.3f}R  Sh={sh:.2f}")


def main():
    print("Loading …")
    df = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    W=72
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)
    IS_END=date(2021,12,31); OS_START=date(2022,1,1)

    # ── A. Spread ORB ──────────────────────────────────────────────────────────
    sec("A. ES/NQ SPREAD OR BREAKOUT  —  Market neutral pairs trade")
    print("  Normalises each instrument by its OR range, trades the SPREAD")
    print("  when it breaks out of its own opening range.\n")
    for tgt in [1.5, 2.0, 2.5, 3.0]:
        tdf = run_spread_orb(df, dates, by_d, pmult=tgt)
        prow(f"Spread ORB  {tgt}R target", tdf)

    spread_tdf = run_spread_orb(df, dates, by_d, pmult=2.0)
    print()
    print("  Year-by-year (2R target):")
    print(f"  {'Year':<6}  {'n':>4}  {'WR':>6}  {'AvgR':>7}  {'IS/OOS':>6}")
    print(f"  {'─'*38}")
    for yr, sub in spread_tdf.groupby("year"):
        wr=sub["win"].mean(); ar=sub["r"].mean()
        label="← IS" if yr<=2021 else "← OOS"
        print(f"  {yr:<6}  {len(sub):>4}  {wr*100:>5.1f}%  {ar:>+6.3f}R  {label}")

    print()
    for period, mask in [("IS  2018-2021",spread_tdf["date"]<=IS_END),
                         ("OOS 2022-2026",spread_tdf["date"]>=OS_START)]:
        prow(period, spread_tdf[mask])

    # ── B. One-sided catchup ───────────────────────────────────────────────────
    sec("B. ONE-SIDED DIVERGENCE CATCHUP  —  Long the laggard")
    print("  When ES breaks OR but NQ doesn't (or vice versa),")
    print("  buy the laggard expecting correlation to normalise.\n")
    catch_tdf = run_catchup(df, dates, by_d, pmult=3.0)
    prow("All catchup trades", catch_tdf)
    prow("  ES leads, long NQ", catch_tdf[catch_tdf["leader"]=="ES"])
    prow("  NQ leads, long ES", catch_tdf[catch_tdf["leader"]=="NQ"])
    print()
    print("  Year-by-year:")
    for yr, sub in catch_tdf.groupby("year"):
        wr=sub["win"].mean(); ar=sub["r"].mean()
        label="← IS" if yr<=2021 else "← OOS"
        flag="✅" if wr>0.52 else ("🟡" if wr>0.46 else "❌")
        print(f"  {flag} {yr}  n={len(sub):3d}  WR={wr*100:5.1f}%  AvgR={ar:+.3f}R  {label}")
    print()
    for period, mask in [("IS  2018-2021",catch_tdf["date"]<=IS_END),
                         ("OOS 2022-2026",catch_tdf["date"]>=OS_START)]:
        prow(period, catch_tdf[mask])

    # ── C. Correlation filter ──────────────────────────────────────────────────
    sec("C. INTRADAY CORRELATION FILTER  —  Only trade ORB on high-corr days")
    print("  Tests: does higher ES/NQ correlation during the OR predict better ORB WR?\n")
    corr_tdf = run_corr_filter(df, dates, by_d, pmult=3.0)
    prow("All ORB days",      corr_tdf)
    prow("High corr (≥0.85)", corr_tdf[corr_tdf["high_corr"]==True])
    prow("Low corr  (<0.85)", corr_tdf[corr_tdf["high_corr"]==False])
    print()
    print("  Correlation distribution:")
    for thresh, label in [(0.95,"0.95+"), (0.85,"0.85-0.95"),
                          (0.70,"0.70-0.85"), (0.0,"<0.70")]:
        if thresh==0.95:
            sub=corr_tdf[corr_tdf["corr"]>=0.95]
        elif thresh==0.85:
            sub=corr_tdf[(corr_tdf["corr"]>=0.85)&(corr_tdf["corr"]<0.95)]
        elif thresh==0.70:
            sub=corr_tdf[(corr_tdf["corr"]>=0.70)&(corr_tdf["corr"]<0.85)]
        else:
            sub=corr_tdf[corr_tdf["corr"]<0.70]
        prow(f"  Corr {label}", sub)

    # ── Final comparison ───────────────────────────────────────────────────────
    sec("SUMMARY  —  Stat arb vs baseline")
    print(f"  {'System':<42}  {'n/yr':>5}  {'WR':>6}  {'AvgR':>7}  {'Sh':>6}")
    print(f"  {'─'*66}")
    print(f"  ORB (verified baseline)                   22/yr  47.8%  +0.92R   3.08")
    print(f"  PDH/L Confirmed (market entry)            36/yr  54.5%  +1.18R   3.14")

    for label, tdf in [
        ("Spread ORB (2R)",     run_spread_orb(df,dates,by_d,2.0)),
        ("Catchup trade (3R)",  catch_tdf),
        ("ORB high-corr only",  corr_tdf[corr_tdf["high_corr"]==True]),
    ]:
        if not len(tdf): continue
        wr=tdf["win"].mean(); ar=tdf["r"].mean()
        dr=tdf.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        n_yr=len(tdf)/8.1/(2 if "sym" in tdf.columns else 1)
        flag="✅" if wr>0.52 else ("🟡" if wr>0.46 else "❌")
        print(f"  {flag} {label:<42}  {n_yr:>5.0f}/yr  {wr*100:>5.1f}%  {ar:>+6.3f}R  {sh:>6.2f}")
    print()


if __name__=="__main__":
    main()
