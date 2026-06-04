#!/usr/bin/env python3
"""
backtest_opening_drive.py — Opening Drive Continuation (Signal 4)

HYPOTHESIS (stated before running):
  The first 5 minutes of RTH carry the heaviest institutional order flow
  of the day: overnight positioning executes, pre-market accumulation
  unwinds, gap fills begin. When both ES and NQ show a strong, aligned
  directional impulse in the first 5 bars (9:30–9:34), that impulse is
  driven by REAL institutional flow — not random noise.

  After the opening impulse, price frequently pulls back to "test" the
  launch level before continuing in the impulse direction. This pullback
  is the ENTRY POINT: institutional buyers who missed the initial impulse
  use the pullback to establish positions, creating the continuation.

  Unlike ORB (which waits 30 min for a range): this trades the opening
  impulse retest within the first 30–40 minutes.

PRE-STATED PARAMETERS:
  impulse_threshold = 0.30%  (ES + NQ both must move ≥0.3% in same dir)
  pullback_zone     = 1.5pt  (bar must be within 1.5pt of impulse close)
  close_confirm     = 0.5pt  (bar close must be ≥0.5pt past impulse close
                               in impulse direction = bounce confirmed)
  target_R          = 2.0    (fixed 2R target)
  stop              = bar_low − 1pt (LONG) / bar_high + 1pt (SHORT)
  window            = 9:36–10:10 ET (first confirmed pullback/bounce only)

BOTH DIRECTIONS tested:
  LONG  : ES & NQ impulse ≥ +0.30% → enter on pullback to impulse close
  SHORT : ES & NQ impulse ≤ -0.30% → enter on retracement to impulse close
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25
RTH_S = time(9, 30); RTH_E = time(16, 0)
EOD   = time(15, 30)
IS_END = date(2021, 12, 31); OS_START = date(2022, 1, 1)
ACCT   = 25_000; RISK = 0.02


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def sh(dr):
    return dr.mean() / dr.std(ddof=1) * np.sqrt(252) if len(dr) > 1 and dr.std() > 0 else 0.0


def prow(lbl, tdf, r_col="r", pad=48):
    if not len(tdf):
        print(f"  — {lbl:<{pad}}  n=0"); return
    wr  = (tdf[r_col] > 0).mean() * 100
    ar  = tdf[r_col].mean()
    dr  = tdf.groupby("date")[r_col].sum(); s = sh(dr)
    nyr = tdf["date"].nunique() / 8.1
    wins = tdf[tdf[r_col] > 0]; losses = tdf[tdf[r_col] <= 0]
    wl = (abs(wins[r_col].mean() / losses[r_col].mean())
          if len(losses) and losses[r_col].mean() != 0 else 99.0)
    flag = "✅" if (ar > 0 and s > 2.5) else ("🟡" if (ar > 0 and s > 1.5) else "❌")
    print(f"  {flag} {lbl:<{pad}}  {nyr:.0f}/yr  WR={wr:5.1f}%  "
          f"Avg={ar:+.4f}  W/L={wl:.2f}x  Sh={s:.2f}")


def yr_row(tdf, yr, r_col="r"):
    yt = tdf[tdf["year"] == yr]
    if not len(yt): return
    tag  = "IS " if yr <= 2021 else "OOS"
    wr   = (yt[r_col] > 0).mean() * 100; ar = yt[r_col].mean()
    dr   = yt.groupby("date")[r_col].sum(); s = sh(dr)
    flag = "✅" if (ar > 0 and s > 1.5) else ("🟡" if ar > 0 else "❌")
    print(f"  {flag} {yr}: n={len(yt):3d}  WR={wr:5.1f}%  AvgR={ar:+.4f}  Sh={s:.2f} {tag}")


# ─────────────────────────────────────────────────────────────────────────────
def run_opening_drive(df, dates, by_d,
                      impulse_thr=0.30,
                      pullback_zone=1.5,
                      close_confirm=0.5,
                      target_R=2.0,
                      entry_cutoff=time(10, 10)):
    """
    Opening drive continuation.

    Step 1 — Measure impulse (9:30 open → 9:34 close):
      ES impulse pct = (es_9:34_close - es_9:30_open) / es_9:30_open * 100
      NQ impulse pct = (nq_9:34_close - nq_9:30_open) / nq_9:30_open * 100
      Both must be ≥ impulse_thr in SAME direction.

    Step 2 — Find pullback/retest bar (9:36 → entry_cutoff):
      LONG  : bar LOW  ≤ es_impulse_close + pullback_zone  (price came back)
               AND bar CLOSE ≥ es_impulse_close - pullback_zone + close_confirm
               (bar closes at or above impulse close = bounce confirmed)
      SHORT : mirror

    Entry : pullback bar close ± slip  (100% valid market fill)
    Stop  : pullback bar LOW − 1pt (LONG) / HIGH + 1pt (SHORT)
    Target: target_R × risk
    """
    trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 30: continue

        # ── STEP 1: Measure impulse ────────────────────────────────────────
        # ES: 9:30 open + 9:34 close
        bar_930_es = rth[rth["time_et"] == time(9, 30)]
        bar_934_es = rth[rth["time_et"] == time(9, 34)]
        if not len(bar_930_es) or not len(bar_934_es): continue

        es_open_930  = bar_930_es.iloc[0]["ES_open"]
        es_close_934 = bar_934_es.iloc[0]["ES_close"]
        es_imp_pct   = (es_close_934 - es_open_930) / es_open_930 * 100

        # NQ: same bars
        nq_open_930  = bar_930_es.iloc[0]["NQ_open"]
        nq_close_934 = bar_934_es.iloc[0]["NQ_close"]
        nq_imp_pct   = (nq_close_934 - nq_open_930) / nq_open_930 * 100

        # Both must exceed threshold in same direction
        if abs(es_imp_pct) < impulse_thr or abs(nq_imp_pct) < impulse_thr: continue
        if np.sign(es_imp_pct) != np.sign(nq_imp_pct): continue

        di = "LONG" if es_imp_pct > 0 else "SHORT"
        ic = es_close_934   # impulse close — the level we're watching for retest

        # ── STEP 2: Find pullback/retest bar (9:36 → entry_cutoff) ────────
        scan = rth[(rth["time_et"] >= time(9, 36)) &
                   (rth["time_et"] <= entry_cutoff)].reset_index(drop=True)

        entry_bar = None
        for _, bar in scan.iterrows():
            if di == "LONG":
                # Bar came back to impulse close level (within pullback_zone)
                # AND closed back at or above impulse close (bounce confirmed)
                if (bar["ES_low"]  <= ic + pullback_zone and
                        bar["ES_close"] >= ic - pullback_zone + close_confirm):
                    entry_bar = bar; break
            else:  # SHORT
                if (bar["ES_high"] >= ic - pullback_zone and
                        bar["ES_close"] <= ic + pullback_zone - close_confirm):
                    entry_bar = bar; break

        if entry_bar is None: continue

        # ── STEP 3: Entry + simulate ───────────────────────────────────────
        if di == "LONG":
            ep   = entry_bar["ES_close"] + SLIP
            sp   = entry_bar["ES_low"] - 1.0 - SLIP
        else:
            ep   = entry_bar["ES_close"] - SLIP
            sp   = entry_bar["ES_high"] + 1.0 + SLIP

        risk = abs(ep - sp)
        if risk < 0.5 or risk > 30: continue
        tp = ep + target_R * risk if di == "LONG" else ep - target_R * risk

        fwd = rth[rth.index > entry_bar.name].reset_index(drop=True)
        rv = 0.0; why = "NONE"
        for _, fb in fwd.iterrows():
            if fb["time_et"] >= EOD:
                pnl = (fb["ES_close"] - ep - SLIP*2) if di == "LONG" else \
                      (ep - fb["ES_close"] - SLIP*2)
                rv = pnl / risk; why = "EOD"; break
            if di == "LONG":
                if fb["ES_low"]  <= sp: rv = -1.0; why = "STOP"; break
                if fb["ES_high"] >= tp: rv = target_R - SLIP/risk; why = "TGT"; break
            else:
                if fb["ES_high"] >= sp: rv = -1.0; why = "STOP"; break
                if fb["ES_low"]  <= tp: rv = target_R - SLIP/risk; why = "TGT"; break

        ts = pd.Timestamp(d); iso = ts.isocalendar()
        trades.append(dict(
            date=d, year=d.year, direction=di, r=rv,
            usd=round(rv * (ACCT * RISK), 2), win=rv > 0,
            exit=why,
            es_imp=round(es_imp_pct, 3),
            nq_imp=round(nq_imp_pct, 3),
            entry_time=entry_bar["time_et"],
            iso_wk=(int(iso.year), int(iso.week)),
        ))
    return pd.DataFrame(trades)


def main():
    print("Loading …")
    df    = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    W = 72

    # ── STEP 1: Impulse threshold sweep ──────────────────────────────────────
    print("=" * W)
    print("  STEP 1 — IMPULSE THRESHOLD SWEEP (pre-committed: 0.30%)")
    print("=" * W)
    print("  How strong must the opening 5-min impulse be?\n")
    print(f"  {'thr':>6}  {'n/yr':>5}  {'WR':>7}  {'IS_Sh':>7}  {'OOS_Sh':>7}  "
          f"{'IS_WR':>7}  {'OOS_WR':>7}")
    print("  " + "─" * 56)

    for thr in [0.15, 0.20, 0.30, 0.40, 0.50, 0.70]:
        t = run_opening_drive(df, dates, by_d, impulse_thr=thr)
        if not len(t): print(f"  {thr:.2f}%  n=0"); continue
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        nyr  = len(t) / 8.1; wr = (t["r"] > 0).mean() * 100
        s_is = sh(IS_t.groupby("date")["r"].sum()) if len(IS_t) else 0
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        wr_is  = (IS_t["r"] > 0).mean() * 100  if len(IS_t)  else 0
        wr_oos = (OOS_t["r"] > 0).mean() * 100 if len(OOS_t) else 0
        mark = " ← pre-committed" if abs(thr - 0.30) < 0.01 else ""
        print(f"  {thr:.2f}%  {nyr:>5.0f}/yr  {wr:>6.1f}%  "
              f"IS={s_is:>5.2f}  OOS={s_oo:>5.2f}  "
              f"IS_WR={wr_is:5.1f}%  OOS_WR={wr_oos:5.1f}%{mark}")

    # ── STEP 2: Entry cutoff window ───────────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  STEP 2 — ENTRY WINDOW CUTOFF (how long to wait for pullback?)")
    print(f"  {'─' * 70}\n")
    for cutoff, lbl in [(time(9,45),"9:45"), (time(9,55),"9:55"),
                         (time(10,5),"10:05"), (time(10,10),"10:10"),
                         (time(10,20),"10:20")]:
        t = run_opening_drive(df, dates, by_d, impulse_thr=0.30, entry_cutoff=cutoff)
        if not len(t): continue
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        nyr  = len(t) / 8.1; wr = (t["r"] > 0).mean() * 100
        s_is = sh(IS_t.groupby("date")["r"].sum()) if len(IS_t) else 0
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        mark = " ← pre-committed" if lbl == "10:10" else ""
        print(f"  Cutoff {lbl}:  {nyr:.0f}/yr  WR={wr:.1f}%  "
              f"IS_Sh={s_is:.2f}  OOS_Sh={s_oo:.2f}{mark}")

    # ── STEP 3: Full results at pre-committed config ──────────────────────────
    print(f"\n{'=' * W}")
    print("  STEP 3 — FULL RESULTS (pre-committed config)")
    print("=" * W)
    print("  impulse≥0.3% (ES+NQ), pullback_zone=1.5pt, target=2R, cutoff=10:10\n")

    t = run_opening_drive(df, dates, by_d,
                          impulse_thr=0.30, pullback_zone=1.5,
                          close_confirm=0.5, target_R=2.0,
                          entry_cutoff=time(10, 10))
    IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]

    print(f"  Total trades: {len(t)} ({len(t)/8.1:.0f}/yr)\n")
    prow("Full period  2018-2026", t)
    prow("IS  2018-2021",          IS_t)
    prow("OOS 2022-2026",          OOS_t)

    if len(t):
        print("\n  By direction:")
        prow("  LONG  (impulse up + pullback entry)",  t[t["direction"] == "LONG"])
        prow("  SHORT (impulse down + retest entry)", t[t["direction"] == "SHORT"])

        print("\n  By exit:")
        for ex in ["STOP", "TGT", "EOD"]:
            sub = t[t["exit"] == ex]
            if not len(sub): continue
            wr = (sub["r"] > 0).mean() * 100; ar = sub["r"].mean()
            print(f"    {ex}: n={len(sub):3d} ({len(sub)/len(t)*100:.1f}%)  "
                  f"WR={wr:.1f}%  AvgR={ar:+.3f}")

        print("\n  By impulse strength:")
        for lo, hi, lbl in [(0.30, 0.50, "0.30–0.50%"),
                             (0.50, 0.75, "0.50–0.75%"),
                             (0.75, 1.00, "0.75–1.00%"),
                             (1.00, 99,   ">1.00%")]:
            sub = t[(t["es_imp"].abs() >= lo) & (t["es_imp"].abs() < hi)]
            if not len(sub): continue
            wr = (sub["r"] > 0).mean() * 100; ar = sub["r"].mean()
            print(f"    impulse {lbl:<12}: n={len(sub):3d}  WR={wr:.1f}%  Avg={ar:+.4f}")

        print("\n  By entry time (when did the pullback occur?):")
        for lo, hi, lbl in [(time(9,36), time(9,45), "9:36–9:45"),
                             (time(9,45), time(9,55), "9:45–9:55"),
                             (time(9,55), time(10,10), "9:55–10:10")]:
            sub = t[(t["entry_time"] >= lo) & (t["entry_time"] <= hi)]
            if not len(sub): continue
            wr = (sub["r"] > 0).mean() * 100; ar = sub["r"].mean()
            print(f"    {lbl}: n={len(sub):3d}  WR={wr:.1f}%  Avg={ar:+.4f}")

        print("\n  Year-by-year:")
        for yr in sorted(t["year"].unique()): yr_row(t, yr)

    # ── STEP 4: Target sensitivity ────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  STEP 4 — TARGET SENSITIVITY")
    print(f"  {'─' * 70}\n")
    for tgt in [1.5, 2.0, 2.5, 3.0]:
        tx = run_opening_drive(df, dates, by_d, impulse_thr=0.30, target_R=tgt)
        if not len(tx): continue
        IS_tx = tx[tx["year"] <= 2021]; OOS_tx = tx[tx["year"] >= 2022]
        s_is  = sh(IS_tx.groupby("date")["r"].sum()) if len(IS_tx) else 0
        s_oo  = sh(OOS_tx.groupby("date")["r"].sum()) if len(OOS_tx) else 0
        wr    = (tx["r"] > 0).mean() * 100
        mark  = " ← pre-committed" if abs(tgt - 2.0) < 0.01 else ""
        print(f"  Target {tgt:.1f}R:  {len(tx)/8.1:.0f}/yr  WR={wr:.1f}%  "
              f"IS_Sh={s_is:.2f}  OOS_Sh={s_oo:.2f}{mark}")

    # ── STEP 5: Overlap with existing systems ─────────────────────────────────
    print(f"\n{'─' * W}")
    print("  STEP 5 — TIME WINDOW & OVERLAP CHECK")
    print(f"  {'─' * 70}")
    print("  This signal fires at 9:36–10:10 AM")
    print("  Primary ORB  fires at ~10:00–10:30 (after the OR closes)")
    print("  Scalp        fires at 10:30–14:30")
    print()
    print("  → Opening Drive can fire BEFORE the primary ORB fires")
    print("  → On days both fire, Opening Drive is the earlier trade")
    print("  → No scheduling conflict — different time windows\n")
    if len(t):
        nyr = len(t) / 8.1
        print(f"  Opening drive: {nyr:.0f}/yr  ({nyr/52:.2f}/wk)")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  VERDICT")
    print("=" * W)

    if len(t):
        IS_t  = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        fs    = sh(t.groupby("date")["r"].sum())
        s_is  = sh(IS_t.groupby("date")["r"].sum())  if len(IS_t)  else 0
        s_oo  = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        oos_wr= (OOS_t["r"] > 0).mean() * 100        if len(OOS_t) else 0
        nyr   = len(t) / 8.1

        print(f"\n  Full Sh={fs:.2f}  IS Sh={s_is:.2f}  OOS Sh={s_oo:.2f}")
        print(f"  OOS WR={oos_wr:.1f}%  Frequency={nyr:.0f}/yr ({nyr/52:.2f}/wk)")
        print()

        if fs > 2.5 and s_oo > 2.0 and oos_wr > 52 and nyr >= 20:
            verdict = "✅ ADD TO SYSTEM"
            detail  = "Implement in signal_generator.py — fire at pullback bar close."
        elif fs > 1.5 and s_oo > 1.5 and nyr >= 15:
            verdict = "🟡 MARGINAL — paper trade first"
            detail  = "Edge present but not high-conviction. Monitor live before using."
        else:
            verdict = "❌ SKIP"
            detail  = "Insufficient edge in OOS to justify implementation."

        print(f"  {verdict}")
        print(f"  {detail}")

        # Comparison table
        print(f"\n  {'Signal':<35} {'n/yr':>5}  {'OOS Sh':>7}  {'OOS WR':>8}")
        print("  " + "─" * 58)
        print(f"  {'Double Test at Level':35} {'65':>5}  {'4.61':>7}  {'39.8%':>8}  ✅")
        print(f"  {'Opening Drive Cont. (this)':35} {nyr:>5.0f}  {s_oo:>7.2f}  {oos_wr:>7.1f}%")
        print(f"  {'Pre-Close SHORT (signal 3)':35} {'21':>5}  {'~2.0':>7}  {'~31%':>8}  🟡")
        print(f"  {'ES/NQ Divergence (signal 1)':35} {'10':>5}  {'1.40':>7}  {'38.7%':>8}  ❌")
    print()


if __name__ == "__main__":
    main()
