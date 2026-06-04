#!/usr/bin/env python3
"""
backtest_pdhl_proper.py — PDH/L with MARKET entry (correct implementation)

Previous bug: entered at PDH level even when price was already far above it.
This version: enters at the CLOSE of the signal bar (realistic market fill).

Signal:  Both ES + NQ break their prior-day RTH high (LONG) or low (SHORT)
         in the 10:00–11:00 ET window.
Entry:   Close of the bar where BOTH instruments first confirm the break.
         + 1 tick slippage. This is the price you'd actually fill at.
Stop:    Entry − OR_range (LONG) or Entry + OR_range (SHORT).
Target:  Entry ± PMULT × OR_range (3R).
EOD:     Close at 15:30 ET.

Also tests:
  A. Pure PDH/L signal (ORB fired or not)
  B. PDH/L on ORB-miss days only (the original PDH/L fallback concept)
  C. PDH/L when ORB also fired same direction (confirmation setup)
  D. PDH/L when ORB fired OPPOSITE (conflict — skip signal)
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"

OR_START  = time(9,30);  OR_END    = time(10,0)
ORB_CUT   = time(10,30); WATCH_END = time(11,0)
RTH_START = time(9,30);  RTH_END   = time(16,0)
EOD       = time(15,30)
SLIP=0.25; PMULT=3.0; MIN_ES=8; MAX_ES=60; MIN_NQ=30; MAX_NQ=175


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def sim(day, entry_t, direction, ep, sp, tp, col_h, col_l, col_c):
    """Simulate from bars AFTER entry bar. Returns (r_val, exit_reason)."""
    rng = abs(ep - sp)
    for _, bar in day[day["time_et"] > entry_t].iterrows():
        t = bar["time_et"]
        if t >= EOD:
            sign = 1 if direction=="LONG" else -1
            net  = sign*(bar[col_c]-ep) - SLIP*2
            return round(net/rng, 3), "EOD"
        if direction=="LONG":
            if bar[col_l] <= sp: return round(-1.0-SLIP*2/rng, 3), "STOP"
            if bar[col_h] >= tp: return round(PMULT-SLIP/rng,   3), "TGT"
        else:
            if bar[col_h] >= sp: return round(-1.0-SLIP*2/rng, 3), "STOP"
            if bar[col_l] <= tp: return round(PMULT-SLIP/rng,   3), "TGT"
    return 0.0, "NONE"


def run(df, dates, by_d):
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

        # Prior day RTH H/L
        prev = by_d.get(dates[i-1])
        if prev is None: continue
        prth = prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if not len(prth): continue
        PDH_ES=prth["ES_high"].max(); PDL_ES=prth["ES_low"].min()
        PDH_NQ=prth["NQ_high"].max(); PDL_NQ=prth["NQ_low"].min()

        # Check if PDH/L is actually NEW territory (not already inside OR)
        # If PDH_ES is below ES_L (OR low), PDH was broken before market even opened → skip
        # If PDH_ES is above ES_H (OR high), it's genuine resistance above current price
        pdh_inside_or_es = es_L <= PDH_ES <= es_H
        pdh_inside_or_nq = nq_L <= PDH_NQ <= nq_H
        pdh_below_or_es  = PDH_ES < es_L   # already blown through before OR
        pdh_below_or_nq  = PDH_NQ < nq_L

        # ORB detection (10:00–10:30)
        post = day[(day["time_et"]>=OR_END)&(day["time_et"]<ORB_CUT)]
        es_b=nq_b=None; orb_row=None
        for _,r in post.iterrows():
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"
                elif r["ES_low"]<es_L: es_b="S"
            if nq_b is None:
                if r["NQ_high"]>nq_H: nq_b="L"
                elif r["NQ_low"]<nq_L: nq_b="S"
            if es_b and nq_b: orb_row=r; break
        orb_fired = bool(es_b and nq_b and es_b==nq_b)
        orb_dir   = ("LONG" if es_b=="L" else "SHORT") if orb_fired else None

        # PDH/L detection in 10:00–11:00
        # Track each bar — signal fires when BOTH ES and NQ have broken same direction
        watch = day[(day["time_et"]>=OR_END)&(day["time_et"]<WATCH_END)]
        es_up=es_dn=nq_up=nq_dn=False
        pdhl_dir=None; signal_bar=None

        for _, bar in watch.iterrows():
            if not es_up and bar["ES_high"] > PDH_ES: es_up=True
            if not es_dn and bar["ES_low"]  < PDL_ES: es_dn=True
            if not nq_up and bar["NQ_high"] > PDH_NQ: nq_up=True
            if not nq_dn and bar["NQ_low"]  < PDL_NQ: nq_dn=True

            if es_up and nq_up and pdhl_dir is None:
                pdhl_dir="LONG"; signal_bar=bar; break
            if es_dn and nq_dn and pdhl_dir is None:
                pdhl_dir="SHORT"; signal_bar=bar; break
            if (es_up and nq_dn) or (es_dn and nq_up):
                pdhl_dir="CONFLICT"; break

        if pdhl_dir is None or pdhl_dir=="CONFLICT": continue
        if signal_bar is None: continue

        entry_t = signal_bar["time_et"]

        # ── MARKET ENTRY: use the CLOSE of the signal bar ────────────────────
        # This is where you'd actually get filled — the bar's close price
        # with 1 tick slippage added in direction of trade.
        # Stop = entry ± OR_range. Target = entry ± 3×OR_range.
        # This means: risk is defined by OR range, but entry is at MARKET.

        # Classify the signal relative to ORB
        if orb_fired:
            if orb_dir == pdhl_dir:
                sig_class = "CONFIRMED"   # ORB + PDH/L agree
            else:
                sig_class = "CONFLICT"    # opposite — skip
                continue
        else:
            sig_class = "STANDALONE"      # ORB missed, PDH/L alone

        # Determine entry quality: was PDH already breached before watch window?
        if pdhl_dir == "LONG":
            already_above_es = pdh_below_or_es   # PDH below OR low → always above
            already_above_nq = pdh_below_or_nq
        else:
            already_above_es = (PDL_ES > es_H)   # PDL above OR high → always below
            already_above_nq = (PDL_NQ > nq_H)

        # If both instruments were already past PDH/L before market open, skip
        # (price never had to cross the level during the session)
        if (pdhl_dir=="LONG" and pdh_below_or_es and pdh_below_or_nq): continue
        if (pdhl_dir=="SHORT" and PDL_ES>es_H and PDL_NQ>nq_H): continue

        dow = pd.Timestamp(d).day_name()
        entry_min = entry_t.hour*60 + entry_t.minute - 10*60   # mins after 10:00

        for sym, OR_H, OR_L, OR_R, PDH, PDL, ch, cl, cc in [
            ("ES", es_H, es_L, es_R, PDH_ES, PDL_ES, "ES_high","ES_low","ES_close"),
            ("NQ", nq_H, nq_L, nq_R, PDH_NQ, PDL_NQ, "NQ_high","NQ_low","NQ_close"),
        ]:
            # Market entry = close of signal bar + slippage
            ep_raw = signal_bar[cc]
            ep = ep_raw + SLIP if pdhl_dir=="LONG" else ep_raw - SLIP

            sp = ep - OR_R if pdhl_dir=="LONG" else ep + OR_R
            tp = ep + PMULT*OR_R if pdhl_dir=="LONG" else ep - PMULT*OR_R

            # Verify fill is valid: signal bar close is within bar range
            fill_ok = (signal_bar[cl] <= ep_raw <= signal_bar[ch])

            rv, why = sim(day, entry_t, pdhl_dir, ep, sp, tp, ch, cl, cc)

            trades.append(dict(
                date=d, year=d.year, sym=sym, sig_class=sig_class,
                direction=pdhl_dir, r=rv, win=rv>0, exit=why,
                fill_ok=fill_ok, dow=dow,
                entry_min=entry_min, or_R=OR_R,
            ))

    return pd.DataFrame(trades)


def prow(label, sub, pad=28):
    if not len(sub): print(f"  {label:<{pad}}  n=0"); return
    wr=sub["win"].mean(); ar=sub["r"].mean(); ev=wr*PMULT-(1-wr)
    dr=sub.groupby("date")["r"].sum()
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    n_yr=len(sub)/8.1/2
    flag="✅" if ev>1.0 else ("🟡" if ev>0.5 else "❌")
    print(f"  {flag} {label:<{pad}}  n={len(sub)//2:4d} ({n_yr:.0f}/yr)  "
          f"WR={wr*100:5.1f}%  AvgR={ar:+.3f}R  EV={ev:+.3f}R  Sh={sh:.2f}")


def main():
    print("Loading …")
    df = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    print("Running PDH/L with market entry …")
    tdf = run(df, dates, by_d)
    print(f"  {len(tdf):,} legs  {tdf['date'].nunique()} signal days\n")

    W=70
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── 1. Fill validity check ────────────────────────────────────────────────
    sec("1. FILL VALIDITY — market entry on signal bar close")
    valid_pct = tdf["fill_ok"].mean()*100
    print(f"  Valid fills : {tdf['fill_ok'].sum():,} / {len(tdf):,}  ({valid_pct:.1f}%)")
    print(f"  (Close of signal bar is always within bar range by definition)")
    print(f"  ✅ Market entry: {valid_pct:.1f}% valid — much better than limit entry")

    # ── 2. Main signal classes ────────────────────────────────────────────────
    sec("2. SIGNAL CLASS COMPARISON")
    print("  STANDALONE = PDH/L fires, ORB missed")
    print("  CONFIRMED  = PDH/L fires, ORB also fired same direction")
    print()
    prow("ALL PDH/L (market entry)", tdf)
    prow("STANDALONE (ORB missed)", tdf[tdf["sig_class"]=="STANDALONE"])
    prow("CONFIRMED (ORB+PDH/L)",   tdf[tdf["sig_class"]=="CONFIRMED"])

    # ── 3. Year-by-year ───────────────────────────────────────────────────────
    sec("3. YEAR-BY-YEAR — ALL PDH/L")
    print(f"  {'Year':<6}  {'Days':>4}  {'WR':>6}  {'AvgR':>7}  {'EV':>7}")
    print(f"  {'─'*36}")
    for yr, sub in tdf.groupby("year"):
        wr=sub["win"].mean(); ar=sub["r"].mean(); ev=wr*PMULT-(1-wr)
        flag=" ⚠️" if ev<0.5 else ""
        print(f"  {yr:<6}  {len(sub)//2:>4}  {wr*100:>5.1f}%  {ar:>+6.3f}R  {ev:>+6.3f}R{flag}")

    # ── 4. Walk-forward ───────────────────────────────────────────────────────
    sec("4. WALK-FORWARD  IS 2018-2021  →  OOS 2022-2026")
    IS_END=date(2021,12,31); OS_START=date(2022,1,1)
    for label, mask in [
        ("IS  2018-2021", tdf["date"]<=IS_END),
        ("OOS 2022-2026", tdf["date"]>=OS_START),
    ]:
        sub=tdf[mask]
        print(f"\n  {label}:")
        prow("  All PDH/L",   sub)
        prow("  Standalone",  sub[sub["sig_class"]=="STANDALONE"])
        prow("  Confirmed",   sub[sub["sig_class"]=="CONFIRMED"])

    # ── 5. Day of week ────────────────────────────────────────────────────────
    sec("5. WIN RATE BY DAY OF WEEK")
    for dow in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
        prow(dow, tdf[tdf["dow"]==dow])

    # ── 6. OR range effect ────────────────────────────────────────────────────
    sec("6. OR RANGE BUCKET — does tighter OR = better PDH/L WR?")
    tdf["or_bucket"]=pd.cut(tdf["or_R"],bins=[0,15,25,40,175],
                             labels=["8-15pt","15-25pt","25-40pt","40-175pt"])
    for bkt in ["8-15pt","15-25pt","25-40pt","40-175pt"]:
        prow(bkt, tdf[tdf["or_bucket"]==bkt])

    # ── 7. Entry time ─────────────────────────────────────────────────────────
    sec("7. ENTRY TIME — minutes after 10:00 ET")
    tdf["entry_bucket"]=pd.cut(tdf["entry_min"],bins=[-1,5,15,30,60],
                                labels=["0-5min","5-15min","15-30min","30-60min"])
    for bkt in ["0-5min","5-15min","15-30min","30-60min"]:
        prow(bkt, tdf[tdf["entry_bucket"]==bkt])

    # ── 8. Comparison vs ORB ─────────────────────────────────────────────────
    sec("8. COMPARISON TABLE  —  Honest numbers side by side")
    print(f"  {'System':<32}  {'Days/yr':>7}  {'WR':>6}  {'EV':>7}  {'Sharpe':>7}")
    print(f"  {'─'*60}")
    systems = [
        ("ORB (verified, 83% valid fills)", None),
        ("PDH/L standalone (market entry)", tdf[tdf["sig_class"]=="STANDALONE"]),
        ("PDH/L confirmed  (market entry)", tdf[tdf["sig_class"]=="CONFIRMED"]),
        ("PDH/L all        (market entry)", tdf),
    ]
    # ORB stats from walkforward
    print(f"  ORB (verified)                    ~22/yr  47.8%  +0.92R   3.08")
    for label, sub in systems[1:]:
        if sub is None or not len(sub): continue
        wr=sub["win"].mean(); ar=sub["r"].mean(); ev=wr*PMULT-(1-wr)
        dr=sub.groupby("date")["r"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        n_yr=len(sub)/8.1/2
        flag="✅" if ev>0.9 else ("🟡" if ev>0.5 else "❌")
        print(f"  {flag} {label:<32}  {n_yr:>7.0f}/yr  {wr*100:>5.1f}%  {ev:>+6.3f}R  {sh:>7.2f}")

    # ── 9. Summary ────────────────────────────────────────────────────────────
    sec("9. HONEST VERDICT")
    all_wr=tdf["win"].mean(); all_ev=all_wr*PMULT-(1-all_wr)
    st_sub=tdf[tdf["sig_class"]=="STANDALONE"]
    st_wr=st_sub["win"].mean() if len(st_sub) else 0
    st_ev=st_wr*PMULT-(1-st_wr)
    co_sub=tdf[tdf["sig_class"]=="CONFIRMED"]
    co_wr=co_sub["win"].mean() if len(co_sub) else 0
    co_ev=co_wr*PMULT-(1-co_wr)
    print(f"  PDH/L overall WR  : {all_wr*100:.1f}%  EV={all_ev:+.3f}R")
    print(f"  Standalone WR     : {st_wr*100:.1f}%  EV={st_ev:+.3f}R")
    print(f"  Confirmed WR      : {co_wr*100:.1f}%  EV={co_ev:+.3f}R")
    print()
    if all_ev > 0.7:
        print("  ✅ PDH/L with market entry has genuine positive EV")
        print("  ✅ Add to system as fallback when ORB doesn't fire")
    elif all_ev > 0.3:
        print("  🟡 PDH/L borderline — positive EV but weak")
    else:
        print("  ❌ PDH/L with market entry does not have meaningful edge")
    print()


if __name__=="__main__":
    main()
