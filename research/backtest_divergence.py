#!/usr/bin/env python3
"""
backtest_divergence.py — ES/NQ Divergence Reversal Signal

HYPOTHESIS (stated before running):
  ES and NQ are correlated ~95% of the time. When they diverge during the
  session — ES makes a new session high but NQ is meaningfully below its
  session high — NQ is the leading indicator and ES is about to correct.

  Mechanism: Institutional traders watch relative strength between ES and NQ.
  When ES breaks to a new high but NQ can't confirm, it signals that the
  ES move is running on "thin" participation (mostly ES-specific flows, not
  broad risk-on). The probability of reversal increases.

  Adding a structural level (ES near PDH/PWH/round) as a 3rd condition
  ensures we're catching institutional rejection, not random divergence.

PRE-STATED PARAMETERS (committed before seeing results):
  div_thresh = 0.4%  (NQ must be ≥0.4% below its session high)
  level_zone = 5pts  (ES must be within 5pts of a structural level)
  target = 2R, stop = bar_high + 1pt (SHORT) / bar_low - 1pt (LONG)
  window = 10:30–14:00 ET, max 1 trade per day

BOTH DIRECTIONS tested:
  BEAR: ES new session HIGH, NQ lags → SHORT ES
  BULL: ES new session LOW,  NQ lags → LONG  ES
"""

import numpy as np
import pandas as pd
from datetime import time, date

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
    wr  = (tdf["r"] > 0).mean() * 100
    ar  = tdf["r"].mean()
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
    wr   = (yt["r"] > 0).mean() * 100
    ar   = yt["r"].mean()
    dr   = yt.groupby("date")["r"].sum(); s = sh(dr)
    flag = "✅" if (ar > 0 and s > 1.5) else ("🟡" if ar > 0 else "❌")
    print(f"  {flag} {yr}: n={len(yt):3d}  WR={wr:5.1f}%  AvgR={ar:+.4f}  Sh={s:.2f} {tag}")


# ─────────────────────────────────────────────────────────
def run_divergence(df, dates, by_d, weekly, monthly,
                   div_thresh=0.40,
                   require_level=True,
                   level_zone=5.0,
                   target_R=2.0):
    """
    ES/NQ Divergence Reversal.

    BEAR divergence → SHORT:
      ES bar makes new RTH session high (ES_high > all prior highs since 9:30)
      NQ close is ≥ div_thresh% below NQ session high (NQ lagging)
      [optional] ES close is within level_zone pts of a resistance level (PDH/PWH/PMH/RND)

    BULL divergence → LONG:
      ES bar makes new RTH session low (ES_low < all prior lows since 9:30)
      NQ close is ≥ div_thresh% above NQ session low (NQ lagging)
      [optional] ES close is within level_zone pts of a support level (PDL/PWL/PML/RND)

    Entry : close of signal bar ± slip  (market fill, 100% valid)
    Stop  : bar high + 1pt (SHORT) / bar low − 1pt (LONG)
    Target: target_R × risk
    Window: 10:30–14:00 ET; first signal per day only
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

        # Structural levels
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

        day_mid = (rth["ES_high"].max() + rth["ES_low"].min()) / 2
        rnds    = [round(day_mid / 50) * 50 + k * 50 for k in range(-4, 5)]

        res_levels = [v for v in [PDH, PWH, PMH] + [float(r) for r in rnds] if v is not None]
        sup_levels = [v for v in [PDL, PWL, PML] + [float(r) for r in rnds] if v is not None]

        scan   = rth[(rth["time_et"] >= time(10, 30)) & (rth["time_et"] <= time(14, 0))]
        fired  = False

        # Running session extremes (from RTH start to current bar)
        for idx, bar in scan.iterrows():
            if fired: break

            t = bar["time_et"]

            # Session high/low UP TO (but not including) this bar
            prior = rth[rth.index < idx]
            if not len(prior): continue

            es_sess_high = prior["ES_high"].max()
            es_sess_low  = prior["ES_low"].min()
            nq_sess_high = prior["NQ_high"].max()
            nq_sess_low  = prior["NQ_low"].min()

            # ── BEAR divergence: ES new high, NQ lagging ─────────────────
            if bar["ES_high"] > es_sess_high:
                # NQ divergence check
                nq_lag_pct = (nq_sess_high - bar["NQ_close"]) / nq_sess_high * 100
                if nq_lag_pct >= div_thresh:
                    # Structural level check
                    near_level = True
                    if require_level:
                        near_level = any(abs(bar["ES_close"] - lv) <= level_zone
                                         for lv in res_levels)
                    if near_level:
                        ep   = bar["ES_close"] - SLIP
                        sp   = bar["ES_high"]  + 1.0 + SLIP
                        risk = abs(ep - sp)
                        if 1.0 <= risk <= 20:
                            tp   = ep - target_R * risk
                            fwd  = rth[rth.index > idx].reset_index(drop=True)
                            rv   = 0.0; why = "NONE"
                            for _, fb in fwd.iterrows():
                                if fb["time_et"] >= EOD:
                                    rv = (ep - fb["ES_close"] - SLIP*2) / risk
                                    why = "EOD"; break
                                if fb["ES_high"] >= sp:
                                    rv = -1.0; why = "STOP"; break
                                if fb["ES_low"] <= tp:
                                    rv = target_R - SLIP/risk; why = "TGT"; break
                            trades.append(dict(
                                date=d, year=d.year, direction="SHORT", r=rv,
                                usd=round(rv * (ACCT * RISK), 2), win=rv > 0,
                                exit=why, nq_lag=round(nq_lag_pct, 3),
                                iso_wk=(int(iso.year), int(iso.week)),
                            ))
                            fired = True; continue

            # ── BULL divergence: ES new low, NQ lagging ──────────────────
            if not fired and bar["ES_low"] < es_sess_low:
                nq_lag_pct = (bar["NQ_close"] - nq_sess_low) / nq_sess_low * 100
                if nq_lag_pct >= div_thresh:
                    near_level = True
                    if require_level:
                        near_level = any(abs(bar["ES_close"] - lv) <= level_zone
                                         for lv in sup_levels)
                    if near_level:
                        ep   = bar["ES_close"] + SLIP
                        sp   = bar["ES_low"]   - 1.0 - SLIP
                        risk = abs(ep - sp)
                        if 1.0 <= risk <= 20:
                            tp   = ep + target_R * risk
                            fwd  = rth[rth.index > idx].reset_index(drop=True)
                            rv   = 0.0; why = "NONE"
                            for _, fb in fwd.iterrows():
                                if fb["time_et"] >= EOD:
                                    rv = (fb["ES_close"] - ep - SLIP*2) / risk
                                    why = "EOD"; break
                                if fb["ES_low"] <= sp:
                                    rv = -1.0; why = "STOP"; break
                                if fb["ES_high"] >= tp:
                                    rv = target_R - SLIP/risk; why = "TGT"; break
                            trades.append(dict(
                                date=d, year=d.year, direction="LONG", r=rv,
                                usd=round(rv * (ACCT * RISK), 2), win=rv > 0,
                                exit=why, nq_lag=round(nq_lag_pct, 3),
                                iso_wk=(int(iso.year), int(iso.week)),
                            ))
                            fired = True; continue

    return pd.DataFrame(trades)


def main():
    print("Loading …")
    df    = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    weekly, monthly = build_levels(by_d, dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    W = 72

    # ── STEP 1: Parameter sensitivity (BEFORE looking at full results) ────────
    print("=" * W)
    print("  STEP 1 — PARAMETER SENSITIVITY (div_thresh sweep)")
    print("=" * W)
    print("  Testing div_thresh from 0.2% to 0.8% — pre-commit to 0.4%\n")
    print(f"  {'thresh':>8} {'n/yr':>6} {'WR':>7} {'IS_Sh':>7} {'OOS_Sh':>7} {'IS_WR':>7} {'OOS_WR':>7}")
    print("  " + "─" * 56)

    for thresh in [0.20, 0.30, 0.40, 0.50, 0.60, 0.80]:
        t = run_divergence(df, dates, by_d, weekly, monthly,
                           div_thresh=thresh, require_level=True)
        if not len(t): print(f"  {thresh:>7.2f}%  n=0"); continue
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        nyr  = len(t) / 8.1
        wr   = (t["r"] > 0).mean() * 100
        s_is = sh(IS_t.groupby("date")["r"].sum()) if len(IS_t) else 0
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        wr_is  = (IS_t["r"] > 0).mean() * 100  if len(IS_t)  else 0
        wr_oos = (OOS_t["r"] > 0).mean() * 100 if len(OOS_t) else 0
        mark = " ← pre-committed" if abs(thresh - 0.40) < 0.01 else ""
        print(f"  {thresh:>7.2f}%  {nyr:>5.0f}/yr  {wr:>6.1f}%  "
              f"IS={s_is:>5.2f}  OOS={s_oo:>5.2f}  "
              f"IS_WR={wr_is:5.1f}%  OOS_WR={wr_oos:5.1f}%{mark}")

    # ── STEP 2: Effect of structural level requirement ────────────────────────
    print(f"\n{'─' * W}")
    print("  STEP 2 — STRUCTURAL LEVEL REQUIREMENT (with vs without)")
    print(f"  {'─'*70}\n")

    no_level  = run_divergence(df, dates, by_d, weekly, monthly,
                               div_thresh=0.40, require_level=False)
    with_level = run_divergence(df, dates, by_d, weekly, monthly,
                                div_thresh=0.40, require_level=True)

    for lbl, t in [("Without level filter (div≥0.4%)", no_level),
                   ("With level filter    (div≥0.4%)", with_level)]:
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        nyr  = len(t) / 8.1
        wr   = (t["r"] > 0).mean() * 100
        s_is = sh(IS_t.groupby("date")["r"].sum()) if len(IS_t) else 0
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        print(f"  {lbl:<42} {nyr:.0f}/yr  WR={wr:.1f}%  "
              f"IS_Sh={s_is:.2f}  OOS_Sh={s_oo:.2f}")

    # ── STEP 3: Full results at pre-committed config ──────────────────────────
    print(f"\n{'=' * W}")
    print("  STEP 3 — FULL RESULTS (pre-committed: div≥0.4%, with level)")
    print("=" * W)

    t = with_level
    IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]

    print(f"  Total trades: {len(t)} ({len(t)/8.1:.0f}/yr)\n")
    prow("Full period  2018-2026", t)
    prow("IS  2018-2021",          IS_t)
    prow("OOS 2022-2026",          OOS_t)

    if len(t):
        print("\n  By direction:")
        prow("  BEAR divergence → SHORT", t[t["direction"] == "SHORT"])
        prow("  BULL divergence → LONG",  t[t["direction"] == "LONG"])

        print("\n  By exit type:")
        for ex in ["STOP", "TGT", "EOD"]:
            sub = t[t["exit"] == ex]
            if not len(sub): continue
            wr = (sub["r"] > 0).mean() * 100
            ar = sub["r"].mean()
            print(f"    {ex}: n={len(sub):3d} ({len(sub)/len(t)*100:.1f}%)  "
                  f"WR={wr:.1f}%  AvgR={ar:+.3f}")

        print("\n  NQ lag at signal bar:")
        for lo, hi, lbl in [(0.4, 0.6, "0.4–0.6%"), (0.6, 1.0, "0.6–1.0%"),
                             (1.0, 2.0, "1.0–2.0%"), (2.0, 99, ">2.0%")]:
            sub = t[(t["nq_lag"] >= lo) & (t["nq_lag"] < hi)]
            if not len(sub): continue
            wr = (sub["r"] > 0).mean() * 100; ar = sub["r"].mean()
            print(f"    lag {lbl:<10}: n={len(sub):3d}  WR={wr:.1f}%  Avg={ar:+.4f}")

        print("\n  Year-by-year:")
        for yr in sorted(t["year"].unique()): yr_row(t, yr)

    # ── STEP 4: Fill validity check ───────────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  FILL VALIDITY CHECK")
    print(f"  {'─' * 70}")
    print("  Entry: bar CLOSE ± slip → always reachable (100% valid by design)")
    print("  Stop : bar HIGH + 1pt for SHORT → reachable (next bar can reach it)")
    print("  No limit orders, no fill artefacts\n")
    if len(t):
        avg_risk = t.apply(lambda x: abs(x["r"]), axis=1).mean()
        print(f"  Avg R magnitude: {avg_risk:.3f}R per trade")
        print(f"  At 2% ACCT risk per trade: avg ${t['usd'].mean():+.0f} per trade")

    # ── STEP 5: Without level filter — does removing it help? ─────────────────
    print(f"\n{'=' * W}")
    print("  STEP 5 — WITHOUT STRUCTURAL LEVEL REQUIREMENT")
    print("=" * W)
    nl = no_level
    IS_nl = nl[nl["year"] <= 2021]; OOS_nl = nl[nl["year"] >= 2022]
    print(f"  Total trades: {len(nl)} ({len(nl)/8.1:.0f}/yr)\n")
    prow("Without level — full", nl)
    prow("Without level — IS",   IS_nl)
    prow("Without level — OOS",  OOS_nl)

    if len(nl):
        print("\n  Year-by-year:")
        for yr in sorted(nl["year"].unique()): yr_row(nl, yr)

    # ── STEP 6: Honest verdict ────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  VERDICT")
    print("=" * W)
    if len(t):
        full_sh = sh(t.groupby("date")["r"].sum())
        oos_sh  = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        oos_wr  = (OOS_t["r"] > 0).mean() * 100 if len(OOS_t) else 0
        nyr     = len(t) / 8.1

        if full_sh > 2.0 and oos_sh > 1.5 and oos_wr > 50 and nyr >= 15:
            verdict = "✅ ADD TO SYSTEM"
            note    = "Both IS and OOS show positive edge. Implement in signal_generator.py."
        elif full_sh > 1.5 and oos_sh > 1.0 and nyr >= 10:
            verdict = "🟡 MARGINAL — monitor live"
            note    = "Edge exists but not strong enough to rely on. Paper-trade first."
        else:
            verdict = "❌ SKIP"
            note    = "Edge not confirmed in OOS. Not worth implementing."

        print(f"\n  {verdict}")
        print(f"  {note}")
        print(f"\n  Stats: {nyr:.0f}/yr  WR={(t['r']>0).mean()*100:.1f}%  "
              f"Full_Sh={full_sh:.2f}  OOS_Sh={oos_sh:.2f}  OOS_WR={oos_wr:.1f}%")
    else:
        print("\n  ❌ No trades generated — check parameters")

    print()


if __name__ == "__main__":
    main()
