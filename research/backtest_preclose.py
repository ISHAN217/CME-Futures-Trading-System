#!/usr/bin/env python3
"""
backtest_preclose.py — Pre-Close Momentum (Signal 3)

HYPOTHESIS (stated before running):
  The 15:00–15:30 window is dominated by institutional end-of-day flows:
  fund rebalancing, ETF creation/redemption, options gamma hedging.
  When a clear directional trend exists by 3:00 PM AND it is still intact
  (price near session extreme, not mean-reverting), institutional flows
  tend to ACCELERATE that direction into the close.

  Mechanism: Portfolio managers must execute at today's close. When the
  market has trended strongly, their rebalancing flows push in the SAME
  direction (buying into a strong day = increasing equity exposure).
  This creates a self-reinforcing close push.

PRE-STATED PARAMETERS (committed before seeing results):
  min_return     = 0.7%   (ES must have moved ≥0.7% from 9:30 open to 15:00)
  max_from_extr  = 0.3%   (price at 15:00 must be within 0.3% of session H/L)
  fixed_stop     = 5pt    (tight stop — only 55 min of exposure)
  exit           = 15:55 bar close  (ride the close, no price target)
  OR quality     = 8–60pt (same as existing system)

DIRECTIONS:
  LONG : return > +0.7% AND price within 0.3% of session HIGH at 15:00
  SHORT: return < -0.7% AND price within 0.3% of session LOW  at 15:00
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25
RTH_S = time(9, 30); RTH_E = time(16, 0)
OR_S  = time(9, 30); OR_E  = time(10, 0)
EOD   = time(15, 55)
MIN_ES = 8; MAX_ES = 60
ACCT   = 25_000; RISK = 0.02
IS_END = date(2021, 12, 31); OS_START = date(2022, 1, 1)


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def sh(dr):
    return dr.mean() / dr.std(ddof=1) * np.sqrt(252) if len(dr) > 1 and dr.std() > 0 else 0.0


def prow(lbl, tdf, r_col="r", pad=46):
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
def run_preclose(df, dates, by_d,
                 min_return=0.70,
                 max_from_extr=0.30,
                 fixed_stop=5.0,
                 require_or=True):
    """
    Pre-close momentum signal.

    At 15:00 bar close:
      day_return  = (ES_close_15:00 - ES_open_9:30) / ES_open_9:30 * 100
      sess_high   = max ES_high from 9:30 to 15:00
      sess_low    = min ES_low  from 9:30 to 15:00
      from_high   = (sess_high - ES_close_15:00) / sess_high * 100
      from_low    = (ES_close_15:00 - sess_low)  / sess_low  * 100

    LONG  if day_return ≥ +min_return AND from_high ≤ max_from_extr
    SHORT if day_return ≤ -min_return AND from_low  ≤ max_from_extr

    Entry : 15:01 bar close ± slip  (100% valid market fill)
    Stop  : entry ∓ fixed_stop
    Exit  : 15:55 bar close (no price target — ride the institutional flow)
    """
    trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= RTH_E)].reset_index(drop=True)
        if len(rth) < 60: continue

        # OR quality filter (optional)
        if require_or:
            or_b = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] < OR_E)]
            if not len(or_b): continue
            es_R = or_b["ES_high"].max() - or_b["ES_low"].min()
            if not (MIN_ES <= es_R <= MAX_ES): continue

        # RTH open (9:30 first bar)
        open_bar = rth[rth["time_et"] == time(9, 30)]
        if not len(open_bar): continue
        rth_open = open_bar.iloc[0]["ES_open"]

        # Snapshot at exactly 15:00
        bar_1500 = rth[rth["time_et"] == time(15, 0)]
        if not len(bar_1500): continue
        close_1500 = bar_1500.iloc[0]["ES_close"]

        # Session stats from 9:30 up to and including 15:00
        sess = rth[rth["time_et"] <= time(15, 0)]
        sess_high = sess["ES_high"].max()
        sess_low  = sess["ES_low"].min()

        day_return   = (close_1500 - rth_open) / rth_open * 100
        from_high    = (sess_high - close_1500) / sess_high * 100   # % below session high
        from_low     = (close_1500 - sess_low)  / sess_low  * 100   # % above session low

        # Entry bar at 15:01
        entry_bar = rth[rth["time_et"] == time(15, 1)]
        if not len(entry_bar): continue
        ep_raw = entry_bar.iloc[0]["ES_close"]

        di = None
        if day_return >= min_return and from_high <= max_from_extr:
            di = "LONG"
        elif day_return <= -min_return and from_low <= max_from_extr:
            di = "SHORT"

        if di is None: continue

        ep   = ep_raw + SLIP if di == "LONG" else ep_raw - SLIP
        sp   = ep - fixed_stop if di == "LONG" else ep + fixed_stop
        risk = fixed_stop

        # Exit bar at 15:55
        exit_bar = rth[rth["time_et"] == time(15, 55)]
        if not len(exit_bar):
            # fallback to latest bar before 16:00
            exit_bar = rth[rth["time_et"] <= time(15, 59)]
            if not len(exit_bar): continue
        exit_price = exit_bar.iloc[-1]["ES_close"]

        # Check if stop was hit between 15:01 and 15:55
        intra = rth[(rth["time_et"] > time(15, 1)) & (rth["time_et"] <= time(15, 55))]
        rv = 0.0; why = "EOD"
        stopped = False
        for _, fb in intra.iterrows():
            if di == "LONG"  and fb["ES_low"]  <= sp:
                rv = -1.0; why = "STOP"; stopped = True; break
            if di == "SHORT" and fb["ES_high"] >= sp:
                rv = -1.0; why = "STOP"; stopped = True; break

        if not stopped:
            pnl = (exit_price - ep - SLIP * 2) if di == "LONG" else (ep - exit_price - SLIP * 2)
            rv  = pnl / risk

        ts = pd.Timestamp(d); iso = ts.isocalendar()
        trades.append(dict(
            date=d, year=d.year, direction=di, r=rv,
            usd=round(rv * (ACCT * RISK), 2), win=rv > 0,
            exit=why, day_return=round(day_return, 3),
            from_extreme=round(from_high if di == "LONG" else from_low, 3),
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

    # ── STEP 1: Return threshold sweep ───────────────────────────────────────
    print("=" * W)
    print("  STEP 1 — RETURN THRESHOLD SWEEP (pre-committed: 0.7%)")
    print("=" * W)
    print("  How strong must the 9:30→15:00 move be to qualify?\n")
    print(f"  {'thresh':>8}  {'n/yr':>5}  {'WR':>7}  {'IS_Sh':>7}  {'OOS_Sh':>7}  {'IS_WR':>7}  {'OOS_WR':>7}")
    print("  " + "─" * 58)

    for thr in [0.30, 0.50, 0.70, 1.00, 1.30, 1.60]:
        t = run_preclose(df, dates, by_d, min_return=thr, max_from_extr=0.30)
        if not len(t): print(f"  {thr:>7.2f}%  n=0"); continue
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        nyr  = len(t) / 8.1
        wr   = (t["r"] > 0).mean() * 100
        s_is = sh(IS_t.groupby("date")["r"].sum()) if len(IS_t) else 0
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        wr_is  = (IS_t["r"] > 0).mean() * 100  if len(IS_t)  else 0
        wr_oos = (OOS_t["r"] > 0).mean() * 100 if len(OOS_t) else 0
        mark = " ← pre-committed" if abs(thr - 0.70) < 0.01 else ""
        print(f"  {thr:>7.2f}%  {nyr:>5.0f}/yr  {wr:>6.1f}%  "
              f"IS={s_is:>5.2f}  OOS={s_oo:>5.2f}  "
              f"IS_WR={wr_is:5.1f}%  OOS_WR={wr_oos:5.1f}%{mark}")

    # ── STEP 2: "Near extreme" filter impact ─────────────────────────────────
    print(f"\n{'─' * W}")
    print("  STEP 2 — NEAR-EXTREME FILTER (must be within X% of session high/low)")
    print(f"  {'─' * 70}")
    print("  Checks if trend-intact requirement adds edge vs raw momentum alone\n")

    for extr in [99.0, 0.50, 0.30, 0.20, 0.10]:
        t = run_preclose(df, dates, by_d, min_return=0.70, max_from_extr=extr)
        if not len(t): continue
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        nyr  = len(t) / 8.1; wr = (t["r"] > 0).mean() * 100
        s_is = sh(IS_t.groupby("date")["r"].sum()) if len(IS_t) else 0
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        lbl = "No filter (any dist)" if extr > 50 else f"Within {extr:.2f}% of extreme"
        mark = " ← pre-committed" if abs(extr - 0.30) < 0.01 else ""
        print(f"  {lbl:<32}  {nyr:.0f}/yr  WR={wr:.1f}%  "
              f"IS_Sh={s_is:.2f}  OOS_Sh={s_oo:.2f}{mark}")

    # ── STEP 3: Full results at pre-committed config ──────────────────────────
    print(f"\n{'=' * W}")
    print("  STEP 3 — FULL RESULTS (pre-committed: return≥0.7%, within 0.3% of extreme)")
    print("=" * W)

    t = run_preclose(df, dates, by_d, min_return=0.70, max_from_extr=0.30, fixed_stop=5.0)
    IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]

    print(f"  Total trades: {len(t)} ({len(t)/8.1:.0f}/yr)\n")
    prow("Full period  2018-2026", t)
    prow("IS  2018-2021",          IS_t)
    prow("OOS 2022-2026",          OOS_t)

    if len(t):
        print("\n  By direction:")
        prow("  LONG  (strong up day, near session high)", t[t["direction"] == "LONG"])
        prow("  SHORT (strong down day, near session low)", t[t["direction"] == "SHORT"])

        print("\n  By exit:")
        for ex in ["STOP", "EOD"]:
            sub = t[t["exit"] == ex]
            if not len(sub): continue
            wr = (sub["r"] > 0).mean() * 100; ar = sub["r"].mean()
            print(f"    {ex}: n={len(sub):3d} ({len(sub)/len(t)*100:.1f}%)  "
                  f"WR={wr:.1f}%  AvgR={ar:+.3f}")

        print("\n  By return magnitude:")
        for lo, hi, lbl in [(0.7,1.0,"0.7–1.0%"),(1.0,1.5,"1.0–1.5%"),
                             (1.5,2.0,"1.5–2.0%"),(2.0,99,">2.0%")]:
            sub = t[t["day_return"].abs() >= lo][t["day_return"].abs() < hi]
            if not len(sub): continue
            wr = (sub["r"] > 0).mean() * 100; ar = sub["r"].mean()
            print(f"    return {lbl:<10}: n={len(sub):3d}  WR={wr:.1f}%  Avg={ar:+.4f}")

        print("\n  Year-by-year:")
        for yr in sorted(t["year"].unique()): yr_row(t, yr)

    # ── STEP 4: Stop sensitivity ──────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  STEP 4 — STOP DISTANCE SENSITIVITY")
    print(f"  {'─' * 70}\n")
    for stp in [3.0, 4.0, 5.0, 6.0, 8.0]:
        tx = run_preclose(df, dates, by_d, min_return=0.70, max_from_extr=0.30,
                          fixed_stop=stp)
        if not len(tx): continue
        IS_tx = tx[tx["year"] <= 2021]; OOS_tx = tx[tx["year"] >= 2022]
        nyr   = len(tx) / 8.1; wr = (tx["r"] > 0).mean() * 100
        s_is  = sh(IS_tx.groupby("date")["r"].sum()) if len(IS_tx) else 0
        s_oo  = sh(OOS_tx.groupby("date")["r"].sum()) if len(OOS_tx) else 0
        mark  = " ← pre-committed" if abs(stp - 5.0) < 0.01 else ""
        print(f"  Stop {stp:.0f}pt  :  {nyr:.0f}/yr  WR={wr:.1f}%  "
              f"IS_Sh={s_is:.2f}  OOS_Sh={s_oo:.2f}{mark}")

    # ── STEP 5: Without OR quality filter ────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  STEP 5 — WITHOUT OR QUALITY FILTER (fires on any day)")
    print(f"  {'─' * 70}\n")
    t_noOR = run_preclose(df, dates, by_d, min_return=0.70, max_from_extr=0.30,
                          fixed_stop=5.0, require_or=False)
    IS_no = t_noOR[t_noOR["year"] <= 2021]; OOS_no = t_noOR[t_noOR["year"] >= 2022]
    print(f"  Total (no OR filter): {len(t_noOR)} ({len(t_noOR)/8.1:.0f}/yr)")
    prow("No OR filter — full", t_noOR)
    prow("No OR filter — IS",   IS_no)
    prow("No OR filter — OOS",  OOS_no)

    # ── STEP 6: Day-of-week breakdown ─────────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  STEP 6 — DAY OF WEEK BREAKDOWN")
    print(f"  {'─' * 70}\n")
    if len(t):
        t["dow"] = pd.to_datetime(t["date"].astype(str)).dt.day_name()
        for dw in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
            sub = t[t["dow"] == dw]
            if not len(sub): continue
            wr = (sub["r"] > 0).mean() * 100; ar = sub["r"].mean()
            print(f"    {dw:<12}: {len(sub)/8.1:.0f}/yr  WR={wr:.1f}%  Avg={ar:+.4f}")

    # ── STEP 7: Overlap with primary + scalp? ─────────────────────────────────
    print(f"\n{'─' * W}")
    print("  STEP 7 — ADDITIVE CHECK (does it fire on DIFFERENT days?)")
    print(f"  {'─' * 70}")
    print("  Pre-close fires at 15:01 — well after ORB (10:30) and scalp (10:30-14:30)")
    print("  → Can fire on the SAME day as ORB or scalp with zero conflict")
    if len(t):
        nyr = len(t) / 8.1
        print(f"\n  Pre-close fires on {len(t)} days over 8.1yr = {nyr:.0f}/yr")
        print(f"  Combined with existing 145/yr (primary+scalp) → {nyr+145:.0f}/yr potential")
        print(f"  But no overlap constraint needed (different time window)")

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

        if fs > 2.0 and s_oo > 1.5 and oos_wr > 52 and nyr >= 20:
            print("\n  ✅ ADD TO SYSTEM")
            print("  Implement in signal_generator.py: fire at 15:01 on qualifying trend days.")
        elif fs > 1.5 and s_oo > 1.0 and nyr >= 15:
            print("\n  🟡 MARGINAL — monitor live first")
            print("  Edge exists but not strong enough to rely on.")
        else:
            print("\n  ❌ SKIP")
            print("  Not enough consistent edge to add to the system.")
    print()


if __name__ == "__main__":
    main()
