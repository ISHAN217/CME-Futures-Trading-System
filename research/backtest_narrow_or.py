#!/usr/bin/env python3
"""
backtest_narrow_or.py — Narrow-OR Coil Breakout (Signal 5b)

HYPOTHESIS (stated before running):
  Our current system REQUIRES OR range 8–60pt. Days below 8pt are
  excluded as "too tight." But these days are COMPRESSED VOLATILITY —
  the market is coiling. When both ES and NQ have narrow opening ranges
  AND both break out in the same direction, the subsequent move is often
  proportionally LARGER than on wide-OR days.

  Mechanism: Narrow ORs mean institutional participants are waiting for
  a catalyst. When the breakout arrives, all the sidelined orders execute
  simultaneously → explosive, sustained directional move. Think of it
  as a compressed spring releasing.

  Supporting evidence: volatility clustering research shows that
  low-volatility opening periods precede higher-volatility expansions.
  The VIX term structure (backwardation → contango) has the same logic.

PRE-STATED PARAMETERS:
  es_or_max = 7.5pt   (ES OR must be < 8pt — the excluded range)
  nq_or_max = 24pt    (NQ equivalent narrow threshold)
  max_break = 30%     (breakout bar not more than 30% of OR beyond level)
  target_R  = 4.0     (wider target — narrow OR → bigger relative move)
  stop      = full OR range (natural stop for breakout system)
  Both ES + NQ must confirm in same direction (same as primary system)
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25
RTH_S = time(9, 30); RTH_E = time(16, 0)
OR_S  = time(9, 30); OR_E  = time(10, 0); ORB_C = time(10, 30)
EOD   = time(15, 30)
MIN_ES_NARROW = 2.0;  MAX_ES_NARROW = 7.5   # the "excluded" narrow range
MIN_NQ_NARROW = 6.0;  MAX_NQ_NARROW = 24.0
ACCT  = 25_000; RISK = 0.05   # same risk as primary system
IS_END = date(2021, 12, 31); OS_START = date(2022, 1, 1)


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
    wr  = (tdf[r_col] > 0).mean() * 100; ar = tdf[r_col].mean()
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


def sim_orb(day, et, di, ep, sp, tp, es_R):
    """Standard ORB simulation with fixed stop + target."""
    rng = abs(ep - sp)
    if rng < 0.01: return 0.0, "NONE"
    for _, bar in day[day["time_et"] > et].iterrows():
        if bar["time_et"] >= EOD:
            pnl = (bar["ES_close"] - ep - SLIP*2) if di == "LONG" else \
                  (ep - bar["ES_close"] - SLIP*2)
            return round(pnl / rng, 3), "EOD"
        if di == "LONG":
            if bar["ES_low"]  <= sp: return round(-1.0 - SLIP*2/rng, 3), "STOP"
            if bar["ES_high"] >= tp: return round(4.0 - SLIP/rng, 3), "TGT"
        else:
            if bar["ES_high"] >= sp: return round(-1.0 - SLIP*2/rng, 3), "STOP"
            if bar["ES_low"]  <= tp: return round(4.0 - SLIP/rng, 3), "TGT"
    return 0.0, "NONE"


def run_narrow_or(df, dates, by_d,
                  es_max=7.5, nq_max=24.0,
                  es_min=2.0, nq_min=6.0,
                  max_break=0.30, target_R=4.0):
    """
    Narrow-OR coil breakout.
    Same logic as primary ORB but uses the EXCLUDED narrow-OR range.
    Both ES + NQ must confirm same direction.
    Stop = OR range; Target = target_R × OR range.
    """
    trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 20: continue

        or_b = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] < OR_E)]
        if not len(or_b): continue
        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min()
        es_R = es_H - es_L

        nq_or = day[(day["time_et"] >= OR_S) & (day["time_et"] < OR_E)]
        nq_H  = nq_or["NQ_high"].max(); nq_L = nq_or["NQ_low"].min()
        nq_R  = nq_H - nq_L

        # NARROW range only (the excluded zone)
        if not (es_min <= es_R <= es_max): continue
        if not (nq_min <= nq_R <= nq_max): continue

        # ORB: look for ES + NQ breakout in same direction (10:00–10:30)
        post    = rth[(rth["time_et"] >= OR_E) & (rth["time_et"] < ORB_C)]
        nq_post = day[(day["time_et"] >= OR_E) & (day["time_et"] < ORB_C)]

        es_b = nq_b = None; entry_row = None; bd_es = bd_nq = 0
        for _, r in post.iterrows():
            if r["time_et"] < time(10, 2): continue
            if es_b is None:
                if r["ES_high"] > es_H: es_b = "L"; bd_es = (r["ES_high"] - es_H) / es_R
                elif r["ES_low"] < es_L: es_b = "S"; bd_es = (es_L - r["ES_low"]) / es_R
            nqr = nq_post[nq_post["time_et"] == r["time_et"]]
            if nq_b is None and len(nqr):
                nr = nqr.iloc[0]
                if nr["NQ_high"] > nq_H: nq_b = "L"; bd_nq = (nr["NQ_high"] - nq_H) / nq_R
                elif nr["NQ_low"] < nq_L: nq_b = "S"; bd_nq = (nq_L - nr["NQ_low"]) / nq_R
            if es_b and nq_b: entry_row = r; break

        if not (es_b and nq_b and es_b == nq_b): continue
        if bd_es > max_break or bd_nq > max_break: continue

        di = "LONG" if es_b == "L" else "SHORT"
        ep_raw = entry_row["ES_close"]
        ep = ep_raw + SLIP if di == "LONG" else ep_raw - SLIP
        sp = es_L - SLIP   if di == "LONG" else es_H + SLIP
        tp = ep + target_R * es_R if di == "LONG" else ep - target_R * es_R

        rv, why = sim_orb(day, entry_row["time_et"], di, ep, sp, tp, es_R)

        ts = pd.Timestamp(d); iso = ts.isocalendar()
        trades.append(dict(
            date=d, year=d.year, direction=di, r=rv,
            usd=round(rv * (ACCT * RISK), 2), win=rv > 0,
            exit=why, es_R=round(es_R, 1), nq_R=round(nq_R, 1),
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

    # ── STEP 1: ES range bucket comparison ───────────────────────────────────
    print("=" * W)
    print("  STEP 1 — OR RANGE BUCKET COMPARISON")
    print("=" * W)
    print("  How do narrow-OR days compare to the normal range we trade?\n")
    print(f"  {'ES_OR':>10}  {'n/yr':>5}  {'WR':>7}  {'IS_Sh':>7}  {'OOS_Sh':>7}")
    print("  " + "─" * 48)

    # Import primary system data for comparison
    from backtest_final_system import (run_primary, run_scalp,
                                       build_levels, load as load_fs)
    weekly, monthly = build_levels(by_d, dates)

    for es_lo, es_hi, nq_lo, nq_hi, lbl in [
        (2.0,  7.5,  6.0,  24.0, "NARROW  2–7pt  ← excluded"),
        (8.0,  20.0, 30.0, 60.0, "TIGHT   8–20pt"),
        (20.0, 40.0, 60.0,120.0, "NORMAL 20–40pt"),
        (40.0, 60.0,120.0,175.0, "WIDE   40–60pt"),
    ]:
        t = run_narrow_or(df, dates, by_d, es_min=es_lo, es_max=es_hi,
                          nq_min=nq_lo, nq_max=nq_hi)
        if not len(t): print(f"  {lbl:<30}  n=0"); continue
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        nyr  = len(t) / 8.1; wr = (t["r"] > 0).mean() * 100
        s_is = sh(IS_t.groupby("date")["r"].sum()) if len(IS_t) else 0
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        print(f"  {lbl:<30}  {nyr:>5.0f}/yr  {wr:>6.1f}%  "
              f"IS={s_is:>5.2f}  OOS={s_oo:>5.2f}")

    # ── STEP 2: Target sensitivity for narrow OR ──────────────────────────────
    print(f"\n{'─' * W}")
    print("  STEP 2 — TARGET R SENSITIVITY (pre-committed: 4R)")
    print(f"  {'─' * 70}\n")
    for tgt in [2.0, 3.0, 4.0, 5.0, 6.0]:
        t = run_narrow_or(df, dates, by_d, target_R=tgt)
        if not len(t): continue
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        nyr  = len(t) / 8.1; wr = (t["r"] > 0).mean() * 100
        s_is = sh(IS_t.groupby("date")["r"].sum()) if len(IS_t) else 0
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        mark = " ← pre-committed" if abs(tgt - 4.0) < 0.01 else ""
        print(f"  Target {tgt:.0f}R:  {nyr:.0f}/yr  WR={wr:.1f}%  "
              f"IS_Sh={s_is:.2f}  OOS_Sh={s_oo:.2f}{mark}")

    # ── STEP 3: Full results ──────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  STEP 3 — FULL RESULTS (pre-committed: ES 2–7.5pt, NQ 6–24pt, 4R target)")
    print("=" * W)
    t = run_narrow_or(df, dates, by_d, es_max=7.5, nq_max=24.0, target_R=4.0)
    IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
    print(f"  Total trades: {len(t)} ({len(t)/8.1:.0f}/yr)\n")
    prow("Full period  2018-2026", t)
    prow("IS  2018-2021",          IS_t)
    prow("OOS 2022-2026",          OOS_t)

    if len(t):
        print("\n  By direction:")
        prow("  LONG  breakouts", t[t["direction"] == "LONG"])
        prow("  SHORT breakouts", t[t["direction"] == "SHORT"])

        print("\n  By exit:")
        for ex in ["STOP", "TGT", "EOD"]:
            sub = t[t["exit"] == ex]
            if not len(sub): continue
            wr = (sub["r"] > 0).mean() * 100; ar = sub["r"].mean()
            print(f"    {ex}: n={len(sub):3d} ({len(sub)/len(t)*100:.1f}%)  "
                  f"WR={wr:.1f}%  AvgR={ar:+.3f}")

        print("\n  By OR size:")
        for lo, hi, lbl in [(2,4,"2–4pt"),(4,6,"4–6pt"),(6,8,"6–8pt")]:
            sub = t[(t["es_R"] >= lo) & (t["es_R"] < hi)]
            if not len(sub): continue
            wr = (sub["r"] > 0).mean() * 100; ar = sub["r"].mean()
            print(f"    ES OR {lbl}: n={len(sub):3d}  WR={wr:.1f}%  Avg={ar:+.4f}")

        print("\n  Year-by-year:")
        for yr in sorted(t["year"].unique()): yr_row(t, yr)

        # Compare narrow vs normal OR
        print(f"\n{'─' * W}")
        print("  COMPARISON: Narrow-OR vs Normal primary system")
        normal_pri = run_primary(df, dates, by_d)
        print(f"\n  {'System':<40} {'n/yr':>5}  {'WR':>7}  {'OOS Sh':>8}")
        print("  " + "─" * 60)
        IS_n = normal_pri[normal_pri["date"] <= IS_END]
        OOS_n = normal_pri[normal_pri["date"] >= OS_START]
        s_n_oo = sh(OOS_n.groupby("date")["r"].sum()) if len(OOS_n) else 0
        s_t_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        print(f"  {'Primary ORB (8–60pt, asymmetric)':<40} "
              f"{len(normal_pri)/8.1:>5.0f}/yr  "
              f"{(normal_pri['r']>0).mean()*100:>6.1f}%  {s_n_oo:>7.2f}")
        print(f"  {'Narrow-OR Coil (2–7.5pt, 4R target)':<40} "
              f"{len(t)/8.1:>5.0f}/yr  "
              f"{(t['r']>0).mean()*100:>6.1f}%  {s_t_oo:>7.2f}")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  VERDICT — SIGNAL 5b: NARROW-OR COIL BREAKOUT")
    print("=" * W)
    if len(t):
        fs    = sh(t.groupby("date")["r"].sum())
        s_oo  = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        nyr   = len(t) / 8.1
        oos_wr = (OOS_t["r"] > 0).mean() * 100 if len(OOS_t) else 0
        print(f"\n  Full Sh={fs:.2f}  OOS Sh={s_oo:.2f}  "
              f"WR={(t['r']>0).mean()*100:.1f}%  OOS_WR={oos_wr:.1f}%  {nyr:.0f}/yr")
        if fs > 2.5 and s_oo > 2.0: v = "✅ ADD"
        elif fs > 1.5 and s_oo > 1.0: v = "🟡 MARGINAL"
        else: v = "❌ SKIP"
        print(f"  {v}")
    print()


if __name__ == "__main__":
    main()
