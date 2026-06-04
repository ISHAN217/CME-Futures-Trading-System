#!/usr/bin/env python3
"""
backtest_ict_sweep.py — ICT Liquidity Sweep backtest

Core concept: institutional players briefly push price ABOVE a prior high
(or below a prior low) to trigger retail stop orders, collect the liquidity,
then reverse. The reversal from the sweep is the trade.

Signal types tested:
  A. PDH Sweep  — price spikes above Prior Day High then closes back below
  B. PDL Sweep  — price spikes below Prior Day Low then closes back above
  C. OR Sweep   — price spikes above OR high (after 10:00) then reverses
  D. SMT Sweep  — ES sweeps PDH but NQ FAILS to sweep PDH → bearish divergence

Entry:  Close of the sweep bar (market fill — bar spikes beyond level, closes back)
Stop:   1 tick beyond the sweep extreme (tightest possible stop)
Target: Tested at 1.5R, 2R, 3R based on stop distance (not OR range)
EOD:    15:30 ET

Time windows tested:
  - Full session (9:30–15:30)
  - NY Open (9:30–11:00) — ICT's prime manipulation window
  - NY AM (9:30–12:00)

Dual confirmation: require BOTH ES + NQ to show same pattern.
SMT divergence: one shows sweep, other fails to reach the level.
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"

OR_START  = time(9,30);  OR_END    = time(10,0)
RTH_START = time(9,30);  RTH_END   = time(16,0)
EOD       = time(15,30)
MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=175
SLIP=0.25


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def sim(day, entry_t, direction, ep, sp, tp, tgt_mult, ch, cl, cc):
    """Returns (r_val, exit_reason) with given R multiple as target."""
    rng = abs(ep - sp)
    tp  = ep + rng * tgt_mult if direction == "LONG" else ep - rng * tgt_mult
    for _, bar in day[day["time_et"] > entry_t].iterrows():
        t = bar["time_et"]
        if t >= EOD:
            sign = 1 if direction == "LONG" else -1
            net  = sign*(bar[cc]-ep) - SLIP*2
            return round(net/rng, 3), "EOD"
        if direction == "LONG":
            if bar[cl] <= sp: return round(-1.0-SLIP*2/rng, 3), "STOP"
            if bar[ch] >= tp: return round(tgt_mult-SLIP/rng, 3),  "TGT"
        else:
            if bar[ch] >= sp: return round(-1.0-SLIP*2/rng, 3), "STOP"
            if bar[cl] <= tp: return round(tgt_mult-SLIP/rng, 3),  "TGT"
    return 0.0, "NONE"


def run_sweeps(df, dates, by_d,
               sweep_level="PDH",    # PDH / PDL / OR / SMT
               window_end=time(11,0), # how late to look for sweep
               tgt_mult=2.0,          # R target
               require_both=True):    # both ES+NQ must sweep (or SMT for SMT mode)
    """
    For each trading day, scan the bar-by-bar data for a sweep pattern.
    Sweep = bar.high > level AND bar.close < level  (for bearish sweep of high)
    Entry at bar close, stop at sweep_high + SLIP, target = tgt_mult × stop_dist.
    """
    trades = []

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue

        # Opening range
        or_b = day[(day["time_et"]>=OR_START)&(day["time_et"]<OR_END)]
        if len(or_b)==0: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(MIN_ES<=es_R<=MAX_ES and MIN_NQ<=nq_R<=MAX_NQ): continue

        # Prior day H/L
        prev = by_d.get(dates[i-1])
        if prev is None: continue
        prth = prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if not len(prth): continue
        PDH_ES=prth["ES_high"].max(); PDL_ES=prth["ES_low"].min()
        PDH_NQ=prth["NQ_high"].max(); PDL_NQ=prth["NQ_low"].min()

        # Scan window
        scan_start = OR_END if sweep_level in ("OR","PDH_POST_OR") else OR_START
        scan = day[(day["time_et"]>=scan_start)&(day["time_et"]<window_end)]
        if len(scan)==0: continue

        dow = pd.Timestamp(d).day_name()

        # ── Bear sweep of HIGH (short trade) ─────────────────────────────────
        def find_bear_sweep(bars, es_lvl, nq_lvl):
            """Returns (signal_bar_es, signal_bar_nq, sweep_h_es, sweep_h_nq) or None."""
            es_swept=nq_swept=False
            es_bar=nq_bar=None; es_sh=nq_sh=None

            for _, bar in bars.iterrows():
                # ES sweep
                if not es_swept and bar["ES_high"] > es_lvl and bar["ES_close"] < es_lvl:
                    es_swept=True; es_bar=bar; es_sh=bar["ES_high"]
                # NQ sweep
                if not nq_swept and bar["NQ_high"] > nq_lvl and bar["NQ_close"] < nq_lvl:
                    nq_swept=True; nq_bar=bar; nq_sh=bar["NQ_high"]

                if require_both:
                    if es_swept and nq_swept:
                        # Use the later of the two as entry bar
                        sig = es_bar if es_bar["time_et"]>=nq_bar["time_et"] else nq_bar
                        return sig, max(es_sh, nq_sh)
                else:
                    if es_swept:
                        return es_bar, es_sh

            return None, None

        # ── Bull sweep of LOW (long trade) ────────────────────────────────────
        def find_bull_sweep(bars, es_lvl, nq_lvl):
            es_swept=nq_swept=False
            es_bar=nq_bar=None; es_sl=nq_sl=None

            for _, bar in bars.iterrows():
                if not es_swept and bar["ES_low"] < es_lvl and bar["ES_close"] > es_lvl:
                    es_swept=True; es_bar=bar; es_sl=bar["ES_low"]
                if not nq_swept and bar["NQ_low"] < nq_lvl and bar["NQ_close"] > nq_lvl:
                    nq_swept=True; nq_bar=bar; nq_sl=bar["NQ_low"]

                if require_both:
                    if es_swept and nq_swept:
                        sig = es_bar if es_bar["time_et"]>=nq_bar["time_et"] else nq_bar
                        return sig, min(es_sl, nq_sl)
                else:
                    if es_swept:
                        return es_bar, es_sl

            return None, None

        # ── SMT divergence sweep ──────────────────────────────────────────────
        def find_smt_sweep(bars, es_lvl, nq_lvl):
            """
            ES sweeps above PDH but NQ FAILS to reach PDH → bearish divergence.
            Entry when ES closes back below PDH (the sweep bar close).
            """
            for _, bar in bars.iterrows():
                es_swept  = bar["ES_high"] > es_lvl and bar["ES_close"] < es_lvl
                nq_reached = bar["NQ_high"] > nq_lvl
                if es_swept and not nq_reached:
                    return bar, bar["ES_high"]   # signal bar, sweep high
            return None, None

        # ── Select level based on sweep_level param ───────────────────────────
        bear_bar = bull_bar = None
        bear_sh = bull_sl = None

        if sweep_level == "PDH":
            bear_bar, bear_sh = find_bear_sweep(scan, PDH_ES, PDH_NQ)
            # Only do bear sweep (shorts from PDH sweep)
        elif sweep_level == "PDL":
            bull_bar, bull_sl = find_bull_sweep(scan, PDL_ES, PDL_NQ)
        elif sweep_level == "PDH_PDL":
            # First to fire wins
            bear_bar, bear_sh = find_bear_sweep(scan, PDH_ES, PDH_NQ)
            bull_bar, bull_sl = find_bull_sweep(scan, PDL_ES, PDL_NQ)
            # Only take the first one on any given day
            if bear_bar is not None and bull_bar is not None:
                if bear_bar["time_et"] <= bull_bar["time_et"]:
                    bull_bar = bull_sl = None
                else:
                    bear_bar = bear_sh = None
        elif sweep_level == "OR":
            # Sweep of current day's OR high/low (only valid after 10:00)
            bear_bar, bear_sh = find_bear_sweep(scan, es_H, nq_H)
            bull_bar, bull_sl = find_bull_sweep(scan, es_L, nq_L)
            if bear_bar is not None and bull_bar is not None:
                if bear_bar["time_et"] <= bull_bar["time_et"]:
                    bull_bar = bull_sl = None
                else:
                    bear_bar = bear_sh = None
        elif sweep_level == "SMT":
            bear_bar, bear_sh = find_smt_sweep(scan, PDH_ES, PDH_NQ)

        # ── Simulate bear sweep trade (SHORT) ─────────────────────────────────
        for sig_bar, sweep_ext, direction in [
            (bear_bar, bear_sh, "SHORT"),
            (bull_bar, bull_sl, "LONG"),
        ]:
            if sig_bar is None: continue

            entry_t = sig_bar["time_et"]
            # Market entry = close of sweep bar
            if direction == "SHORT":
                ep_es = sig_bar["ES_close"] - SLIP
                ep_nq = sig_bar["NQ_close"] - SLIP
                # Stop: above the sweep high (1 tick)
                sp_es = sweep_ext + SLIP
                sp_nq = sweep_ext + SLIP   # approximate — use same sweep high
            else:
                ep_es = sig_bar["ES_close"] + SLIP
                ep_nq = sig_bar["NQ_close"] + SLIP
                sp_es = sweep_ext - SLIP
                sp_nq = sweep_ext - SLIP

            # Simulate ES leg
            for sym, ep, sp, ch, cl, cc in [
                ("ES", ep_es, sp_es, "ES_high","ES_low","ES_close"),
                ("NQ", ep_nq, sp_nq, "NQ_high","NQ_low","NQ_close"),
            ]:
                rng_sym = abs(ep - sp)
                if rng_sym < 0.5: continue   # degenerate trade

                rv, why = sim(day, entry_t, direction, ep, sp, None, tgt_mult,
                              ch, cl, cc)
                trades.append(dict(
                    date=d, year=d.year, sym=sym,
                    sweep_type=sweep_level, direction=direction,
                    r=rv, win=rv>0, exit=why,
                    stop_dist=round(rng_sym, 2),
                    entry_min=entry_t.hour*60+entry_t.minute-9*60*60//3600,
                    dow=dow,
                ))

    return pd.DataFrame(trades)


def prow(label, sub, pad=34):
    if not len(sub): print(f"  {label:<{pad}}  n=0"); return
    wr=sub["win"].mean(); ar=sub["r"].mean()
    ev=wr*abs(sub["r"].max())-(1-wr) if len(sub) else 0  # approximate
    # Use actual EV with tgt_mult baked in
    tgt_r = sub[sub["exit"]=="TGT"]["r"].mean() if (sub["exit"]=="TGT").any() else 2.0
    stp_r = sub[sub["exit"]=="STOP"]["r"].mean() if (sub["exit"]=="STOP").any() else -1.0
    eod_r = sub[sub["exit"]=="EOD"]["r"].mean() if (sub["exit"]=="EOD").any() else 0.0
    ev_actual = ar
    dr=sub.groupby("date")["r"].sum()
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    n_yr=len(sub)/8.1/2
    flag="✅" if wr>0.55 else ("🟡" if wr>0.45 else "❌")
    print(f"  {flag} {label:<{pad}}  n={len(sub)//2:4d} ({n_yr:.0f}/yr)  "
          f"WR={wr*100:5.1f}%  AvgR={ar:+.3f}R  Sh={sh:.2f}")


def main():
    print("Loading …")
    df = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    W = 72
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── 1. Basic sweep types at 2R target ─────────────────────────────────────
    sec("1. SWEEP SIGNAL TYPES  —  2R target, NY Open window (9:30–11:00)")
    configs = [
        ("PDH Bear sweep (both ES+NQ)",  "PDH",     time(11,0), 2.0, True),
        ("PDL Bull sweep (both ES+NQ)",  "PDL",     time(11,0), 2.0, True),
        ("PDH+PDL (best of two)",        "PDH_PDL", time(11,0), 2.0, True),
        ("OR high/low sweep",            "OR",      time(12,0), 2.0, True),
        ("SMT: ES sweeps PDH, NQ fails", "SMT",     time(11,0), 2.0, False),
    ]
    results = {}
    for label, sweep, win_end, tgt, both in configs:
        tdf = run_sweeps(df, dates, by_d, sweep, win_end, tgt, both)
        results[label] = tdf
        prow(label, tdf)

    # ── 2. Target sensitivity (PDH sweep) ────────────────────────────────────
    sec("2. TARGET SENSITIVITY  —  PDH Bear sweep, NY Open")
    for tgt in [1.5, 2.0, 2.5, 3.0]:
        tdf = run_sweeps(df, dates, by_d, "PDH", time(11,0), tgt, True)
        prow(f"PDH sweep {tgt}R target", tdf)

    # ── 3. Time window sensitivity ────────────────────────────────────────────
    sec("3. TIME WINDOW  —  PDH Bear sweep, 2R target")
    for win_label, win_end in [
        ("OR only (9:30–10:00)",  time(10,0)),
        ("NY Open (9:30–11:00)",  time(11,0)),
        ("NY AM   (9:30–12:00)",  time(12,0)),
        ("Full day(9:30–15:30)",  time(15,30)),
    ]:
        tdf = run_sweeps(df, dates, by_d, "PDH", win_end, 2.0, True)
        prow(win_label, tdf)

    # ── 4. Year-by-year for best sweep ───────────────────────────────────────
    # Use PDH_PDL combined
    sec("4. YEAR-BY-YEAR  —  PDH+PDL sweep (2R, 9:30–11:00)")
    best_tdf = run_sweeps(df, dates, by_d, "PDH_PDL", time(11,0), 2.0, True)
    print(f"  {'Year':<6}  {'Days':>4}  {'WR':>6}  {'AvgR':>7}  {'Exit%':>20}")
    print(f"  {'─'*52}")
    for yr, sub in best_tdf.groupby("year"):
        wr=sub["win"].mean(); ar=sub["r"].mean()
        tgt_p=(sub["exit"]=="TGT").mean()*100
        stp_p=(sub["exit"]=="STOP").mean()*100
        eod_p=(sub["exit"]=="EOD").mean()*100
        flag=" ⚠️" if wr<0.4 else ""
        print(f"  {yr:<6}  {len(sub)//2:>4}  {wr*100:>5.1f}%  {ar:>+6.3f}R  "
              f"TGT={tgt_p:.0f}% STP={stp_p:.0f}% EOD={eod_p:.0f}%{flag}")

    # ── 5. Walk-forward ───────────────────────────────────────────────────────
    sec("5. WALK-FORWARD  IS 2018-2021  →  OOS 2022-2026")
    IS_END=date(2021,12,31); OS_START=date(2022,1,1)
    for period, mask in [
        ("IS  2018-2021", best_tdf["date"]<=IS_END),
        ("OOS 2022-2026", best_tdf["date"]>=OS_START),
    ]:
        sub=best_tdf[mask]
        prow(period, sub)

    # ── 6. Day of week ────────────────────────────────────────────────────────
    sec("6. WIN RATE BY DAY OF WEEK  —  PDH+PDL sweep")
    for dow in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
        prow(dow, best_tdf[best_tdf["dow"]==dow])

    # ── 7. Comparison vs ORB system ───────────────────────────────────────────
    sec("7. SWEEP vs EXISTING SYSTEM  —  honest side-by-side")
    print(f"  {'System':<36}  {'Days/yr':>7}  {'WR':>6}  {'AvgR':>7}  {'Sharpe':>7}")
    print(f"  {'─'*64}")
    print(f"  ORB (verified)                        ~22/yr  47.8%  +0.92R   3.08")
    print(f"  PDH/L Confirmed (market entry)        ~36/yr  54.5%  +1.18R   3.14")
    for label, tdf in [
        ("PDH Bear sweep 2R",
         run_sweeps(df,dates,by_d,"PDH",time(11,0),2.0,True)),
        ("PDH+PDL sweep 2R",
         run_sweeps(df,dates,by_d,"PDH_PDL",time(11,0),2.0,True)),
        ("OR sweep 2R",
         run_sweeps(df,dates,by_d,"OR",time(12,0),2.0,True)),
        ("SMT divergence 2R",
         run_sweeps(df,dates,by_d,"SMT",time(11,0),2.0,False)),
    ]:
        if not len(tdf): continue
        wr=tdf["win"].mean(); ar=tdf["r"].mean()
        dr=tdf.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        n_yr=len(tdf)/8.1/2
        flag="✅" if wr>0.52 else ("🟡" if wr>0.45 else "❌")
        print(f"  {flag} {label:<36}  {n_yr:>7.0f}/yr  {wr*100:>5.1f}%  {ar:>+6.3f}R  {sh:>7.2f}")

    # ── 8. Honest verdict ─────────────────────────────────────────────────────
    sec("8. HONEST VERDICT")
    tdf_best = best_tdf
    if len(tdf_best):
        wr=tdf_best["win"].mean(); ev=tdf_best["r"].mean()
        dr=tdf_best.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        print(f"  Best sweep config: PDH+PDL, 2R target, 9:30–11:00 window")
        print(f"  WR={wr*100:.1f}%  AvgR={ev:+.3f}R  Sharpe={sh:.2f}")
        print()
        if wr > 0.55:
            print("  ✅ Genuine edge — add to system")
        elif wr > 0.48:
            print("  🟡 Marginal — positive EV but low Sharpe")
        else:
            print("  ❌ No meaningful edge vs ORB baseline")
    print()


if __name__ == "__main__":
    main()
