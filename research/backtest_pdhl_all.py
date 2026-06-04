#!/usr/bin/env python3
"""
backtest_pdhl_all.py — PDH/L on ALL valid-OR days vs ORB on same days

The key question: Is PDH/L's 65.4% WR a real edge, or is it a selection
artefact from only testing it on ORB-miss days?

Test:
  For every valid-OR day (range within filters), run BOTH strategies
  and record outcomes independently. Compare directly.

Segments reported:
  A. ORB-fire days   — what would PDH/L have done on the same days ORB fired?
  B. ORB-miss days   — PDH/L on its natural habitat (already known: 65.4% WR)
  C. ALL days        — PDH/L vs ORB head-to-head across the full universe

Timing note:
  On ORB-fire days, PDH/L may fire BEFORE ORB (if PDH < OR high, level already
  broken), AT THE SAME TIME, or not at all. We track all cases.
"""

import numpy as np
import pandas as pd
from datetime import time
from scipy import stats as sc

DATA = ("/Users/ishanbhardwaj/cme_competition_system/output/mtf/"
        "ES_NQ_1min_aligned.parquet")

OR_START     = time(9, 30);  OR_END       = time(10, 0)
ORB_CUTOFF   = time(10, 30); WATCH_START  = time(10, 0)
ENTRY_CUTOFF = time(11, 0);  EOD_CLOSE    = time(15, 30)
RTH_START    = time(9, 30);  RTH_END      = time(16, 0)
MIN_ES = 8.0; MAX_ES = 60.0; MIN_NQ = 30.0; MAX_NQ = 150.0
PMULT  = 3.0   # competition target
SLIP   = 0.25


def simulate_leg(day, entry_t, direction, col_h, col_l, col_c,
                 entry_px, stop_px, tgt_px, rng):
    after = day[day["time_et"] > entry_t]
    for _, bar in after.iterrows():
        t = bar["time_et"]
        if t >= EOD_CLOSE:
            net = (1 if direction=="LONG" else -1)*(bar[col_c]-entry_px) - SLIP*2
            return net/rng if rng else 0, "EOD"
        if direction == "LONG":
            if bar[col_l] <= stop_px:
                return (stop_px-entry_px-SLIP*2)/rng, "STOP"
            if bar[col_h] >= tgt_px:
                return (tgt_px-entry_px-SLIP)/rng, "TGT"
        else:
            if bar[col_h] >= stop_px:
                return (entry_px-stop_px-SLIP*2)/rng, "STOP"
            if bar[col_l] <= tgt_px:
                return (entry_px-tgt_px-SLIP)/rng, "TGT"
    return None, None


def run():
    print("Loading data …")
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    df = df.sort_values("ts_et").reset_index(drop=True)
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    print(f"  {len(df):,} bars   {dates[0]} → {dates[-1]}\n")

    records = []  # one row per valid-OR day

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue

        # ── Opening range ────────────────────────────────────────────────
        or_b = day[(day["time_et"] >= OR_START) & (day["time_et"] < OR_END)]
        if len(or_b) == 0: continue
        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min()
        nq_H = or_b["NQ_high"].max(); nq_L = or_b["NQ_low"].min()
        es_R = es_H - es_L;           nq_R = nq_H - nq_L
        if not (MIN_ES <= es_R <= MAX_ES and MIN_NQ <= nq_R <= MAX_NQ): continue

        # ── ORB ──────────────────────────────────────────────────────────
        post = day[(day["time_et"] >= OR_END) & (day["time_et"] < ORB_CUTOFF)]
        es_b = nq_b = None; orb_entry_row = None
        for _, r in post.iterrows():
            if es_b is None:
                if r["ES_high"] > es_H: es_b = "L"
                elif r["ES_low"] < es_L: es_b = "S"
            if nq_b is None:
                if r["NQ_high"] > nq_H: nq_b = "L"
                elif r["NQ_low"] < nq_L: nq_b = "S"
            if es_b and nq_b: orb_entry_row = r; break

        orb_fired = (es_b and nq_b and es_b == nq_b)
        orb_dir   = ("LONG" if es_b == "L" else "SHORT") if orb_fired else None

        # Simulate ORB
        orb_r_es = orb_r_nq = None
        if orb_fired:
            et = orb_entry_row["time_et"]
            for sym, h, l, rng, ch, cl, cc in [
                ("ES", es_H, es_L, es_R, "ES_high","ES_low","ES_close"),
                ("NQ", nq_H, nq_L, nq_R, "NQ_high","NQ_low","NQ_close"),
            ]:
                ep = h if orb_dir=="LONG" else l
                sp = l if orb_dir=="LONG" else h
                tp = ep + rng*PMULT if orb_dir=="LONG" else ep - rng*PMULT
                rv, _ = simulate_leg(day, et, orb_dir, ch, cl, cc, ep, sp, tp, rng)
                if sym == "ES": orb_r_es = rv
                else:           orb_r_nq = rv

        # ── PDH/L (run on ALL valid-OR days) ─────────────────────────────
        prev  = by_d.get(dates[i-1])
        pdhl_fired = False; pdhl_dir = None; pdhl_entry_t = None
        pdhl_r_es = pdhl_r_nq = None

        if prev is not None:
            prth = prev[(prev["time_et"] >= RTH_START) & (prev["time_et"] <= RTH_END)]
            if len(prth) > 0:
                es_PDH = prth["ES_high"].max(); es_PDL = prth["ES_low"].min()
                nq_PDH = prth["NQ_high"].max(); nq_PDL = prth["NQ_low"].min()

                watch = day[(day["time_et"] >= WATCH_START) &
                            (day["time_et"] < ENTRY_CUTOFF)]
                eu = ed = nu = nd = False; entry_row = None

                for _, r in watch.iterrows():
                    if not eu and r["ES_high"] > es_PDH: eu = True
                    if not ed and r["ES_low"]  < es_PDL: ed = True
                    if not nu and r["NQ_high"] > nq_PDH: nu = True
                    if not nd and r["NQ_low"]  < nq_PDL: nd = True
                    if eu and nu and pdhl_dir is None:
                        pdhl_dir="LONG";  entry_row=r; break
                    if ed and nd and pdhl_dir is None:
                        pdhl_dir="SHORT"; entry_row=r; break
                    if (eu and nd) or (ed and nu):
                        pdhl_dir="CONFLICT"; break

                if pdhl_dir not in (None, "CONFLICT"):
                    pdhl_fired = True
                    pdhl_entry_t = entry_row["time_et"]

                    # Simulate PDH/L
                    for sym, pdh, pdl, rng, ch, cl, cc in [
                        ("ES", es_PDH, es_PDL, es_R, "ES_high","ES_low","ES_close"),
                        ("NQ", nq_PDH, nq_PDL, nq_R, "NQ_high","NQ_low","NQ_close"),
                    ]:
                        ep = pdh if pdhl_dir=="LONG" else pdl
                        sp = ep - rng if pdhl_dir=="LONG" else ep + rng
                        tp = ep + rng*PMULT if pdhl_dir=="LONG" else ep - rng*PMULT
                        rv, _ = simulate_leg(day, pdhl_entry_t, pdhl_dir,
                                             ch, cl, cc, ep, sp, tp, rng)
                        if sym == "ES": pdhl_r_es = rv
                        else:           pdhl_r_nq = rv

        records.append(dict(
            date        = d,
            orb_fired   = orb_fired,
            pdhl_fired  = pdhl_fired,
            orb_r       = np.mean([r for r in [orb_r_es,  orb_r_nq]  if r is not None])
                          if orb_fired  else None,
            pdhl_r      = np.mean([r for r in [pdhl_r_es, pdhl_r_nq] if r is not None])
                          if pdhl_fired else None,
            orb_dir     = orb_dir,
            pdhl_dir    = pdhl_dir,
            same_dir    = (orb_dir == pdhl_dir) if (orb_fired and pdhl_fired and
                           pdhl_dir not in (None,"CONFLICT")) else None,
        ))

    rdf = pd.DataFrame(records)

    def stats(series, label, pmult=PMULT):
        s  = series.dropna()
        if len(s) == 0:
            return
        wr = (s > 0).mean()
        av = s.mean()
        ev = wr * pmult - (1 - wr)
        z  = (wr - 0.40) / np.sqrt(0.40*0.60/len(s))
        p  = sc.norm.sf(abs(z)) * 2
        sig = "✅ p<0.01" if p<0.01 else "✅ p<0.05" if p<0.05 else "⚠️  NS"
        print(f"  {label:<40} n={len(s):5d}  WR={wr*100:5.1f}%  "
              f"AvgR={av:+6.3f}  EV={ev:+6.3f}  {sig}")

    rdf["orb_fired"]  = rdf["orb_fired"].astype(bool)
    rdf["pdhl_fired"] = rdf["pdhl_fired"].astype(bool)

    total = len(rdf)
    orb_days  = rdf["orb_fired"].sum()
    miss_days = (~rdf["orb_fired"]).sum()
    both_days = (rdf["orb_fired"] & rdf["pdhl_fired"]).sum()
    same_dir  = rdf[rdf["same_dir"] == True].shape[0]

    print("=" * 72)
    print("UNIVERSE BREAKDOWN")
    print("=" * 72)
    print(f"  Total valid-OR days       : {total}")
    print(f"  ORB-fire days             : {orb_days}  ({orb_days/total*100:.1f}%)")
    print(f"  ORB-miss days             : {miss_days}  ({miss_days/total*100:.1f}%)")
    print(f"  Days BOTH fire            : {both_days}  ({both_days/total*100:.1f}%)")
    print(f"  Both fire, same direction : {same_dir}  ({same_dir/both_days*100:.1f}% of overlap)"
          if both_days else "")

    print("\n" + "=" * 72)
    print(f"HEAD-TO-HEAD  (profit target = {PMULT}R for both)")
    print("=" * 72)

    # A. On ALL valid-OR days
    print("\n  A. ALL valid-OR days")
    print(f"  {'─'*68}")
    stats(rdf["orb_r"],  "ORB   (only on days it fires, n=ORB-fire)")
    stats(rdf["pdhl_r"], "PDH/L (only on days it fires, n=PDH/L-fire)")

    # B. ORB-fire days only — what would PDH/L have done?
    orb_fire_df = rdf[rdf["orb_fired"] == True]
    print(f"\n  B. ORB-FIRE DAYS only  (n={len(orb_fire_df)})")
    print(f"  {'─'*68}")
    stats(orb_fire_df["orb_r"],  "ORB on these days")
    stats(orb_fire_df["pdhl_r"], "PDH/L on these SAME days (if it also fired)")

    # C. ORB-miss days only
    miss_df = rdf[rdf["orb_fired"] == False]
    print(f"\n  C. ORB-MISS DAYS only  (n={len(miss_df)})")
    print(f"  {'─'*68}")
    stats(miss_df["pdhl_r"], "PDH/L on ORB-miss days")

    # D. Days where BOTH fired same direction
    both_same = rdf[rdf["same_dir"] == True]
    print(f"\n  D. BOTH fired, SAME direction  (n={len(both_same)}) — which entry is better?")
    print(f"  {'─'*68}")
    stats(both_same["orb_r"],  "ORB  entry (at OR level)")
    stats(both_same["pdhl_r"], "PDH/L entry (at prior-day H/L level)")

    # E. Days where BOTH fired but OPPOSITE directions — conflict
    both_opp = rdf[(rdf["orb_fired"] & rdf["pdhl_fired"]) &
                   (rdf["same_dir"] == False)]
    print(f"\n  E. BOTH fired, OPPOSITE directions  (n={len(both_opp)}) — conflict days")
    print(f"  {'─'*68}")
    if len(both_opp):
        stats(both_opp["orb_r"],  "ORB  result on conflict days")
        stats(both_opp["pdhl_r"], "PDH/L result on conflict days")
    else:
        print("  None found")

    # F. Coverage comparison
    pdhl_all_days = rdf["pdhl_fired"].sum()
    combined_days = (rdf["orb_fired"] | rdf["pdhl_fired"]).sum()
    print("\n" + "=" * 72)
    print("COVERAGE COMPARISON")
    print("=" * 72)
    print(f"  ORB alone            : {orb_days:4d} days  ({orb_days/total*100:.1f}%)")
    print(f"  PDH/L alone          : {pdhl_all_days:4d} days  ({pdhl_all_days/total*100:.1f}%)")
    print(f"  ORB + PDH/L combined : {combined_days:4d} days  ({combined_days/total*100:.1f}%)")

    # G. EV comparison for competition
    orb_ev  = rdf["orb_r"].dropna()
    pdhl_ev = rdf["pdhl_r"].dropna()

    print("\n" + "=" * 72)
    print(f"VERDICT  (competition context, {PMULT}R target)")
    print("=" * 72)
    orb_wr_all  = (orb_ev > 0).mean()
    pdhl_wr_all = (pdhl_ev > 0).mean()
    orb_ev_val  = orb_wr_all * PMULT - (1-orb_wr_all)
    pdhl_ev_val = pdhl_wr_all * PMULT - (1-pdhl_wr_all)

    print(f"\n  ORB   WR={orb_wr_all*100:.1f}%  EV={orb_ev_val:+.3f}R  "
          f"({orb_days} signal days, {orb_days/total*100:.0f}% of valid days)")
    print(f"  PDH/L WR={pdhl_wr_all*100:.1f}%  EV={pdhl_ev_val:+.3f}R  "
          f"({pdhl_all_days} signal days, {pdhl_all_days/total*100:.0f}% of valid days)")

    # WR on ORB-fire days specifically
    pdhl_on_orb_days = orb_fire_df["pdhl_r"].dropna()
    if len(pdhl_on_orb_days):
        wr_orb_days = (pdhl_on_orb_days > 0).mean()
        print(f"\n  PDH/L WR on ORB-FIRE days : {wr_orb_days*100:.1f}%  "
              f"(n={len(pdhl_on_orb_days)}) ← key selection-bias test")
        print(f"  PDH/L WR on ORB-MISS days : "
              f"{(miss_df['pdhl_r'].dropna()>0).mean()*100:.1f}%  "
              f"(n={miss_df['pdhl_r'].dropna().__len__()})")

        delta = (pdhl_on_orb_days>0).mean() - (miss_df["pdhl_r"].dropna()>0).mean()
        if abs(delta) < 0.05:
            print(f"\n  ✅  PDH/L WR consistent across ORB-fire and ORB-miss days "
                  f"(Δ={delta*100:+.1f}pp) — NOT a selection artefact")
            print(f"      PDH/L is a genuine standalone edge")
        elif delta < -0.05:
            print(f"\n  ⚠️   PDH/L WR lower on ORB-fire days (Δ={delta*100:+.1f}pp)")
            print(f"      The 65.4% WR is partly a selection artefact")
            print(f"      Keep PDH/L as FALLBACK only, not primary")
        else:
            print(f"\n  ✅  PDH/L WR HIGHER on ORB-fire days (Δ={delta*100:+.1f}pp)")
            print(f"      Consider running PDH/L on all days instead of ORB")

    print()


if __name__ == "__main__":
    run()
