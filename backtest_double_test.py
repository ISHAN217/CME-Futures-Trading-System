#!/usr/bin/env python3
"""
backtest_double_test.py — Double Test + Rejection at Level

HYPOTHESIS (stated before running):
  A single rejection at PDH/PWH/PMH has WR=30.7% (proven in failed_auction test).
  Reason: single wicks above a level are noise — random price exploration.

  A DOUBLE rejection — price tests the level, fails, pulls back ≥3pts,
  returns to test AGAIN, and fails again — is structurally different:
    1. The first failure removes weak longs (their stops are now hit or avoided)
    2. The pullback confirms sellers were real, not just a random wick
    3. The second test finds LESS buying power (fewer people willing to buy
       a level that already rejected once) but the SAME selling pressure
    4. Second failure = trapped buyers from BOTH tests → accelerated selling

  Expected: WR should be materially higher than single-test 30.7%
  Target threshold for "ADD": WR ≥ 50%, OOS Sharpe ≥ 1.5

PRE-STATED PARAMETERS:
  min_pullback = 3.0pts  (price must fall ≥3pts between test 1 and test 2)
  max_window   = 90min   (both tests within 90-minute window)
  level_zone   = 1.0pt   (second test must reach within 1pt of level)
  reject_min   = 0.5pt   (second test bar must close ≥0.5pt below level)
  target       = 3R      (wider than single-test since more confirmation)
  stop         = level + 1pt

DIRECTIONS tested:
  SHORT: double test of PDH / PWH / PMH (resistance)
  LONG : double test of PDL / PWL / PML (support)

Window: 10:30–14:00 ET; one trade per day (first completed double-test)
"""

import numpy as np
import pandas as pd
from datetime import time, date, datetime, timedelta

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25
RTH_S = time(9, 30); RTH_E = time(16, 0)
OR_S  = time(9, 30); OR_E  = time(10, 0)
EOD   = time(15, 30)
MIN_ES = 8; MAX_ES = 60
ACCT   = 25_000; RISK = 0.02
IS_END = date(2021, 12, 31); OS_START = date(2022, 1, 1)


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def build_levels(by_d, dates):
    weekly = {}; monthly = {}
    for d in dates:
        ts = pd.Timestamp(d); iso = ts.isocalendar()
        wk = (int(iso.year), int(iso.week)); mn = (ts.year, ts.month)
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= RTH_E)]
        if not len(rth): continue
        h = rth["ES_high"].max(); l = rth["ES_low"].min()
        if wk not in weekly: weekly[wk] = [h, l]
        else: weekly[wk][0] = max(weekly[wk][0], h); weekly[wk][1] = min(weekly[wk][1], l)
        if mn not in monthly: monthly[mn] = [h, l]
        else: monthly[mn][0] = max(monthly[mn][0], h); monthly[mn][1] = min(monthly[mn][1], l)
    return weekly, monthly


def sh(dr):
    return dr.mean() / dr.std(ddof=1) * np.sqrt(252) if len(dr) > 1 and dr.std() > 0 else 0.0


def prow(lbl, tdf, pad=46):
    if not len(tdf):
        print(f"  — {lbl:<{pad}}  n=0"); return
    wr  = (tdf["r"] > 0).mean() * 100; ar = tdf["r"].mean()
    dr  = tdf.groupby("date")["r"].sum(); s = sh(dr)
    nyr = tdf["date"].nunique() / 8.1
    wins = tdf[tdf["r"] > 0]; losses = tdf[tdf["r"] <= 0]
    wl = (abs(wins["r"].mean() / losses["r"].mean())
          if len(losses) and losses["r"].mean() != 0 else 99.0)
    flag = "✅" if (ar > 0 and s > 2.5) else ("🟡" if (ar > 0 and s > 1.5) else "❌")
    print(f"  {flag} {lbl:<{pad}}  {nyr:.0f}/yr  WR={wr:5.1f}%  "
          f"Avg={ar:+.4f}  W/L={wl:.2f}x  Sh={s:.2f}")


def yr_row(tdf, yr):
    yt = tdf[tdf["year"] == yr]
    if not len(yt): return
    tag  = "IS " if yr <= 2021 else "OOS"
    wr   = (yt["r"] > 0).mean() * 100; ar = yt["r"].mean()
    dr   = yt.groupby("date")["r"].sum(); s = sh(dr)
    flag = "✅" if (ar > 0 and s > 1.5) else ("🟡" if ar > 0 else "❌")
    print(f"  {flag} {yr}: n={len(yt):3d}  WR={wr:5.1f}%  AvgR={ar:+.4f}  Sh={s:.2f} {tag}")


def time_diff_min(t1, t2):
    """Minutes between two time objects (t2 - t1)."""
    d1 = datetime(2000, 1, 1, t1.hour, t1.minute, t1.second)
    d2 = datetime(2000, 1, 1, t2.hour, t2.minute, t2.second)
    return (d2 - d1).total_seconds() / 60


def run_double_test(df, dates, by_d, weekly, monthly,
                    min_pullback=3.0,
                    max_window_min=90,
                    level_zone=1.0,
                    reject_min=0.5,
                    target_R=3.0):
    """
    Double-test pattern at structural levels.

    State machine per day (runs for both SHORT at resistance, LONG at support):
      WATCHING → first bar tests level and closes below it → FIRST_DONE
      FIRST_DONE → price pulls back ≥ min_pullback pts → PULLED_BACK
                 → timeout (> max_window_min) → WATCHING (reset)
      PULLED_BACK → second bar returns to within level_zone of level
                    AND closes ≥ reject_min below it → FIRE
                  → timeout → WATCHING (reset)

    Only one trade per day. SHORT takes priority (checked first).
    """
    trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 20: continue

        or_b = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] < OR_E)]
        if not len(or_b): continue
        es_R = or_b["ES_high"].max() - or_b["ES_low"].min()
        if not (MIN_ES <= es_R <= MAX_ES): continue

        prev = by_d.get(dates[i - 1])
        if prev is None: continue
        pr = prev[(prev["time_et"] >= RTH_S) & (prev["time_et"] <= RTH_E)]
        if not len(pr): continue
        PDH = pr["ES_high"].max(); PDL = pr["ES_low"].min()

        ts  = pd.Timestamp(d); iso = ts.isocalendar()
        pw  = int(iso.week) - 1; py = int(iso.year)
        if pw == 0: py -= 1; pw = int(pd.Timestamp(f"{py}-12-28").isocalendar().week)
        wk  = weekly.get((py, pw))
        PWH = wk[0] if wk else None; PWL = wk[1] if wk else None
        pm  = ts.month - 1; pmy = ts.year
        if pm == 0: pm = 12; pmy -= 1
        mn  = monthly.get((pmy, pm))
        PMH = mn[0] if mn else None; PML = mn[1] if mn else None

        res_levels = [(PDH, "D1_H")]
        if PWH: res_levels.append((PWH, "W1_H"))
        if PMH: res_levels.append((PMH, "MN_H"))

        sup_levels = [(PDL, "D1_L")]
        if PWL: sup_levels.append((PWL, "W1_L"))
        if PML: sup_levels.append((PML, "MN_L"))

        scan = rth[(rth["time_et"] >= time(10, 30)) & (rth["time_et"] <= time(14, 0))].reset_index(drop=True)

        # ---------- run the state machine for SHORT (resistance) ----------
        def try_short():
            state = "watching"
            first_lv = first_close = first_time_val = None
            lowest_since_first = None

            for _, bar in scan.iterrows():
                t = bar["time_et"]

                if state == "watching":
                    for lv, nm in res_levels:
                        if (bar["ES_high"] >= lv and
                                bar["ES_close"] <= lv - reject_min):
                            state = "first_done"
                            first_lv = lv; first_close = bar["ES_close"]
                            first_time_val = t
                            lowest_since_first = bar["ES_close"]
                            break

                elif state == "first_done":
                    elapsed = time_diff_min(first_time_val, t)
                    if elapsed > max_window_min:
                        state = "watching"; first_lv = first_close = None
                        # Re-check this bar as a new first test
                        for lv, nm in res_levels:
                            if (bar["ES_high"] >= lv and
                                    bar["ES_close"] <= lv - reject_min):
                                state = "first_done"
                                first_lv = lv; first_close = bar["ES_close"]
                                first_time_val = t
                                lowest_since_first = bar["ES_close"]
                                break
                        continue
                    lowest_since_first = min(lowest_since_first, bar["ES_close"])
                    if first_close - lowest_since_first >= min_pullback:
                        state = "pulled_back"

                elif state == "pulled_back":
                    elapsed = time_diff_min(first_time_val, t)
                    if elapsed > max_window_min:
                        state = "watching"; first_lv = first_close = None
                        continue
                    # Check if this bar retests the level
                    if (bar["ES_high"] >= first_lv - level_zone and
                            bar["ES_close"] <= first_lv - reject_min):
                        # FIRE — second test confirmed
                        ep   = bar["ES_close"] - SLIP
                        sp   = first_lv + 1.0 + SLIP
                        risk = abs(ep - sp)
                        if risk < 0.5 or risk > 25:
                            state = "watching"; continue
                        tp = ep - target_R * risk
                        fwd = rth[rth.index > bar.name].reset_index(drop=True)
                        rv = 0.0; why = "NONE"
                        for _, fb in fwd.iterrows():
                            if fb["time_et"] >= EOD:
                                rv = (ep - fb["ES_close"] - SLIP*2) / risk
                                why = "EOD"; break
                            if fb["ES_high"] >= sp:
                                rv = -1.0; why = "STOP"; break
                            if fb["ES_low"] <= tp:
                                rv = target_R - SLIP/risk; why = "TGT"; break
                        return dict(
                            date=d, year=d.year, direction="SHORT", r=rv,
                            usd=round(rv*(ACCT*RISK), 2), win=rv > 0,
                            exit=why, level_type=res_levels[0][1],
                            pullback=round(first_close - lowest_since_first, 1),
                            iso_wk=(int(iso.year), int(iso.week)),
                        )
            return None

        # ---------- run the state machine for LONG (support) ----------
        def try_long():
            state = "watching"
            first_lv = first_close = first_time_val = None
            highest_since_first = None

            for _, bar in scan.iterrows():
                t = bar["time_et"]

                if state == "watching":
                    for lv, nm in sup_levels:
                        if (bar["ES_low"] <= lv and
                                bar["ES_close"] >= lv + reject_min):
                            state = "first_done"
                            first_lv = lv; first_close = bar["ES_close"]
                            first_time_val = t
                            highest_since_first = bar["ES_close"]
                            break

                elif state == "first_done":
                    elapsed = time_diff_min(first_time_val, t)
                    if elapsed > max_window_min:
                        state = "watching"; first_lv = first_close = None
                        for lv, nm in sup_levels:
                            if (bar["ES_low"] <= lv and
                                    bar["ES_close"] >= lv + reject_min):
                                state = "first_done"
                                first_lv = lv; first_close = bar["ES_close"]
                                first_time_val = t
                                highest_since_first = bar["ES_close"]
                                break
                        continue
                    highest_since_first = max(highest_since_first, bar["ES_close"])
                    if highest_since_first - first_close >= min_pullback:
                        state = "pulled_back"

                elif state == "pulled_back":
                    elapsed = time_diff_min(first_time_val, t)
                    if elapsed > max_window_min:
                        state = "watching"; first_lv = first_close = None
                        continue
                    if (bar["ES_low"] <= first_lv + level_zone and
                            bar["ES_close"] >= first_lv + reject_min):
                        ep   = bar["ES_close"] + SLIP
                        sp   = first_lv - 1.0 - SLIP
                        risk = abs(ep - sp)
                        if risk < 0.5 or risk > 25:
                            state = "watching"; continue
                        tp = ep + target_R * risk
                        fwd = rth[rth.index > bar.name].reset_index(drop=True)
                        rv = 0.0; why = "NONE"
                        for _, fb in fwd.iterrows():
                            if fb["time_et"] >= EOD:
                                rv = (fb["ES_close"] - ep - SLIP*2) / risk
                                why = "EOD"; break
                            if fb["ES_low"] <= sp:
                                rv = -1.0; why = "STOP"; break
                            if fb["ES_high"] >= tp:
                                rv = target_R - SLIP/risk; why = "TGT"; break
                        return dict(
                            date=d, year=d.year, direction="LONG", r=rv,
                            usd=round(rv*(ACCT*RISK), 2), win=rv > 0,
                            exit=why, level_type=sup_levels[0][1],
                            pullback=round(highest_since_first - first_close, 1),
                            iso_wk=(int(iso.year), int(iso.week)),
                        )
            return None

        result = try_short() or try_long()
        if result:
            trades.append(result)

    return pd.DataFrame(trades)


def main():
    print("Loading …")
    df    = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    weekly, monthly = build_levels(by_d, dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    W = 72

    # ── STEP 1: Compare pullback sensitivity ─────────────────────────────────
    print("=" * W)
    print("  STEP 1 — PULLBACK SENSITIVITY (pre-committed: 3pt)")
    print("=" * W)
    print("  Hypothesis: larger pullback = more confirmed structure → higher WR\n")
    print(f"  {'pb_min':>6} {'n/yr':>6} {'WR':>7} {'IS_Sh':>7} {'OOS_Sh':>7}")
    print("  " + "─" * 40)

    for pb in [1.0, 2.0, 3.0, 4.0, 5.0]:
        t = run_double_test(df, dates, by_d, weekly, monthly, min_pullback=pb)
        if not len(t):
            print(f"  {pb:>5.1f}pt  n=0"); continue
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        nyr  = len(t) / 8.1
        wr   = (t["r"] > 0).mean() * 100
        s_is = sh(IS_t.groupby("date")["r"].sum()) if len(IS_t) else 0
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        mark = " ← pre-committed" if abs(pb - 3.0) < 0.01 else ""
        print(f"  {pb:>5.1f}pt  {nyr:>5.0f}/yr  {wr:>6.1f}%  "
              f"IS={s_is:>5.2f}  OOS={s_oo:>5.2f}{mark}")

    # ── STEP 2: Full results at pre-committed config ──────────────────────────
    print(f"\n{'=' * W}")
    print("  STEP 2 — FULL RESULTS (pre-committed: pb=3pt, window=90min, tgt=3R)")
    print("=" * W)

    t = run_double_test(df, dates, by_d, weekly, monthly,
                        min_pullback=3.0, max_window_min=90,
                        level_zone=1.0, reject_min=0.5, target_R=3.0)
    IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]

    print(f"  Total trades: {len(t)} ({len(t)/8.1:.0f}/yr)\n")
    prow("Full period  2018-2026", t)
    prow("IS  2018-2021",          IS_t)
    prow("OOS 2022-2026",          OOS_t)

    if len(t):
        print("\n  By direction:")
        prow("  SHORT (resistance double-test)", t[t["direction"] == "SHORT"])
        prow("  LONG  (support double-test)",    t[t["direction"] == "LONG"])

        print("\n  By exit:")
        for ex in ["STOP", "TGT", "EOD"]:
            sub = t[t["exit"] == ex]
            if not len(sub): continue
            wr = (sub["r"] > 0).mean() * 100; ar = sub["r"].mean()
            print(f"    {ex}: n={len(sub):3d} ({len(sub)/len(t)*100:.1f}%)  "
                  f"WR={wr:.1f}%  AvgR={ar:+.3f}")

        print("\n  Pullback size at signal:")
        for lo, hi, lbl in [(3,5,"3–5pt"),(5,8,"5–8pt"),(8,99,">8pt")]:
            sub = t[(t["pullback"] >= lo) & (t["pullback"] < hi)]
            if not len(sub): continue
            wr = (sub["r"] > 0).mean() * 100; ar = sub["r"].mean()
            print(f"    pb {lbl:<7}: n={len(sub):3d}  WR={wr:.1f}%  Avg={ar:+.4f}")

        print("\n  Year-by-year:")
        for yr in sorted(t["year"].unique()): yr_row(t, yr)

    # ── STEP 3: Compare vs single-test (our failed auction result) ────────────
    print(f"\n{'=' * W}")
    print("  STEP 3 — DOUBLE TEST vs SINGLE TEST (our prior result)")
    print("=" * W)
    print(f"  {'Setup':<40} {'n/yr':>6} {'WR':>7} {'OOS_Sh':>8}")
    print("  " + "─" * 64)
    if len(t):
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        nyr  = len(t) / 8.1
        wr   = (t["r"] > 0).mean() * 100
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        print(f"  Single-test (failed auction — prior test)  "
              f"117/yr   30.7%   OOS=-1.41")
        print(f"  Double-test (this test)                   "
              f"{nyr:4.0f}/yr  {wr:5.1f}%   OOS={s_oo:.2f}")

    # ── STEP 4: Target sensitivity ────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  STEP 4 — TARGET SENSITIVITY (2R vs 3R vs 2.5R)")
    print(f"  {'─' * 70}\n")
    for tgt in [2.0, 2.5, 3.0]:
        tx = run_double_test(df, dates, by_d, weekly, monthly,
                             min_pullback=3.0, target_R=tgt)
        if not len(tx): continue
        IS_tx = tx[tx["year"] <= 2021]; OOS_tx = tx[tx["year"] >= 2022]
        nyr   = len(tx) / 8.1
        wr    = (tx["r"] > 0).mean() * 100
        s_is  = sh(IS_tx.groupby("date")["r"].sum()) if len(IS_tx) else 0
        s_oo  = sh(OOS_tx.groupby("date")["r"].sum()) if len(OOS_tx) else 0
        mark  = " ← pre-committed" if abs(tgt - 3.0) < 0.01 else ""
        print(f"  Target {tgt:.1f}R:  {nyr:.0f}/yr  WR={wr:.1f}%  "
              f"IS_Sh={s_is:.2f}  OOS_Sh={s_oo:.2f}{mark}")

    # ── STEP 5: Honest verdict ────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  VERDICT")
    print("=" * W)

    if len(t):
        full_sh = sh(t.groupby("date")["r"].sum())
        IS_t    = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        oos_sh  = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        oos_wr  = (OOS_t["r"] > 0).mean() * 100 if len(OOS_t) else 0
        nyr     = len(t) / 8.1

        wr_improvement = (t["r"] > 0).mean() * 100 - 30.7  # vs single-test

        print(f"\n  WR vs single-test: 30.7% → {(t['r']>0).mean()*100:.1f}% "
              f"({wr_improvement:+.1f}pp)")
        print(f"  Frequency: {nyr:.0f}/yr ({nyr/52:.2f}/wk)")
        print(f"  Full Sh={full_sh:.2f}  IS Sh={sh(IS_t.groupby('date')['r'].sum()):.2f}  "
              f"OOS Sh={oos_sh:.2f}  OOS WR={oos_wr:.1f}%")

        if full_sh > 2.0 and oos_sh > 1.5 and oos_wr > 50 and nyr >= 15:
            verdict = "✅ ADD TO SYSTEM"
            note    = "Double-test adds meaningful edge. Implement in signal_generator.py."
        elif full_sh > 1.5 and oos_sh > 1.2 and nyr >= 10:
            verdict = "🟡 MARGINAL — monitor"
            note    = "Edge exists but marginal. Paper-trade before implementing."
        elif (t["r"] > 0).mean() > 0.30 + 0.05 and oos_sh > 0:
            verdict = "🟡 WEAK IMPROVEMENT vs single-test"
            note    = f"WR improved but Sharpe still low (OOS={oos_sh:.2f}). Not enough."
        else:
            verdict = "❌ SKIP"
            note    = "Double-test doesn't materially improve on single-test WR."

        print(f"\n  {verdict}")
        print(f"  {note}")
    else:
        print("\n  ❌ No trades generated")
    print()


if __name__ == "__main__":
    main()
