#!/usr/bin/env python3
"""
backtest_opex_pin.py — Weekly OpEx Pinning (Signal 5a)

HYPOTHESIS (stated before running):
  Every Friday, weekly ES options expire. Market makers who are short
  gamma near major 50-pt strikes (4800, 4850, 4900, 5000, 5050…) must
  delta-hedge: sell ES when price is above the strike, buy when below.
  This creates an "attractor" — price gravitates toward and oscillates
  around the nearest 50-pt strike in the final hours of Friday.

  The pin is strongest when:
    1. The session OR is narrow (market is already indecisive — pin more likely)
    2. ES is within a tight zone of a major 50-pt strike by midday (12:00)
    3. A deviation away from the pin strike occurs → fade it back

  Mechanism: market makers' hedging flows are the largest systematic
  order flow on Friday afternoons. Fading deviations from the pin
  strike aligns WITH the largest participant in the market.

PRE-STATED PARAMETERS:
  pin_zone      = 8pt   (ES must be within 8pt of 50-pt strike at 12:00)
  max_or_range  = 25pt  (narrow OR = indecisive day, pin more likely)
  fade_trigger  = 6pt   (fade when price moves ≥6pt from pin strike)
  stop_dist     = 10pt  (stop if price moves 10pt beyond fade entry)
  target        = within 2pt of pin strike (partial gap close)
  window        = 12:00–15:30 ET (OpEx afternoon only)
  Fridays only
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25
RTH_S = time(9, 30); RTH_E = time(16, 0)
OR_S  = time(9, 30); OR_E  = time(10, 0)
EOD   = time(15, 30)
ACCT  = 25_000; RISK = 0.02
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


def nearest_50_strike(price):
    return round(price / 50) * 50


def run_opex_pin(df, dates, by_d,
                 pin_zone=8.0,
                 max_or_range=25.0,
                 fade_trigger=6.0,
                 stop_dist=10.0):
    """
    Weekly OpEx pin fade — Fridays only.
    At 12:00 PM: find nearest 50-pt strike. If ES is within pin_zone,
    the session OR was narrow (≤ max_or_range), and price subsequently
    moves ≥ fade_trigger pts from the strike → fade it back.
    Target: within 2pt of strike. Stop: fade_entry ± stop_dist.
    One trade per Friday only.
    """
    trades = []
    for i, d in enumerate(dates[1:], 1):
        # Fridays only
        if pd.Timestamp(d).weekday() != 4: continue

        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 30: continue

        # OR range filter
        or_b = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] < OR_E)]
        if not len(or_b): continue
        or_range = or_b["ES_high"].max() - or_b["ES_low"].min()
        if or_range > max_or_range: continue   # only narrow-OR Fridays

        # 12:00 snapshot — find pin strike
        bar_1200 = rth[rth["time_et"] == time(12, 0)]
        if not len(bar_1200): continue
        price_1200 = bar_1200.iloc[0]["ES_close"]
        pin_strike = nearest_50_strike(price_1200)

        # Must be within pin_zone of the strike at noon
        if abs(price_1200 - pin_strike) > pin_zone: continue

        # Scan 12:01 → EOD for fade opportunity (first signal only)
        scan = rth[rth["time_et"] > time(12, 0)].reset_index(drop=True)
        fired = False

        for _, bar in scan.iterrows():
            if fired: break
            c = bar["ES_close"]

            # Fade SHORT: price moved ≥ fade_trigger above pin
            if c >= pin_strike + fade_trigger:
                ep   = c - SLIP
                sp   = ep + stop_dist + SLIP
                tp   = pin_strike + 2.0        # target: back within 2pt of pin
                risk = stop_dist

                fwd = scan[scan.index > bar.name].reset_index(drop=True)
                rv = 0.0; why = "NONE"
                for _, fb in fwd.iterrows():
                    if fb["time_et"] >= EOD:
                        rv = (ep - fb["ES_close"] - SLIP*2) / risk; why = "EOD"; break
                    if fb["ES_high"] >= sp: rv = -1.0; why = "STOP"; break
                    if fb["ES_low"]  <= tp:
                        rv = (ep - tp - SLIP) / risk; why = "TGT"; break

                ts = pd.Timestamp(d); iso = ts.isocalendar()
                trades.append(dict(
                    date=d, year=d.year, direction="SHORT", r=rv,
                    usd=round(rv*(ACCT*RISK), 2), win=rv > 0,
                    exit=why, pin_strike=pin_strike,
                    dist_from_pin=round(c - pin_strike, 1),
                    or_range=round(or_range, 1),
                    iso_wk=(int(iso.year), int(iso.week)),
                ))
                fired = True

            # Fade LONG: price moved ≥ fade_trigger below pin
            elif c <= pin_strike - fade_trigger:
                ep   = c + SLIP
                sp   = ep - stop_dist - SLIP
                tp   = pin_strike - 2.0
                risk = stop_dist

                fwd = scan[scan.index > bar.name].reset_index(drop=True)
                rv = 0.0; why = "NONE"
                for _, fb in fwd.iterrows():
                    if fb["time_et"] >= EOD:
                        rv = (fb["ES_close"] - ep - SLIP*2) / risk; why = "EOD"; break
                    if fb["ES_low"]  <= sp: rv = -1.0; why = "STOP"; break
                    if fb["ES_high"] >= tp:
                        rv = (tp - ep - SLIP) / risk; why = "TGT"; break

                ts = pd.Timestamp(d); iso = ts.isocalendar()
                trades.append(dict(
                    date=d, year=d.year, direction="LONG", r=rv,
                    usd=round(rv*(ACCT*RISK), 2), win=rv > 0,
                    exit=why, pin_strike=pin_strike,
                    dist_from_pin=round(c - pin_strike, 1),
                    or_range=round(or_range, 1),
                    iso_wk=(int(iso.year), int(iso.week)),
                ))
                fired = True

    return pd.DataFrame(trades)


def main():
    print("Loading …")
    df    = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    W = 72

    # ── STEP 1: Parameter sweep ───────────────────────────────────────────────
    print("=" * W)
    print("  STEP 1 — PARAMETER SWEEP")
    print("=" * W)
    print("  Varying pin_zone and fade_trigger (pre-committed: 8pt / 6pt)\n")
    print(f"  {'pin_z':>6} {'fade':>6}  {'n/yr':>5}  {'WR':>7}  {'IS_Sh':>7}  {'OOS_Sh':>7}")
    print("  " + "─" * 48)

    for pz, ft in [(5,4),(5,6),(8,4),(8,6),(8,8),(10,6),(10,8)]:
        t = run_opex_pin(df, dates, by_d, pin_zone=pz, fade_trigger=ft)
        if not len(t): continue
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        nyr  = len(t) / 8.1; wr = (t["r"] > 0).mean() * 100
        s_is = sh(IS_t.groupby("date")["r"].sum()) if len(IS_t) else 0
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        mark = " ←" if (pz == 8 and ft == 6) else ""
        print(f"  {pz:>5}pt  {ft:>5}pt  {nyr:>5.0f}/yr  {wr:>6.1f}%  "
              f"IS={s_is:>5.2f}  OOS={s_oo:>5.2f}{mark}")

    # ── STEP 2: OR range filter impact ───────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  STEP 2 — OR RANGE FILTER (narrow OR = pin more likely)")
    print(f"  {'─' * 70}\n")
    for max_or in [15, 20, 25, 35, 999]:
        t = run_opex_pin(df, dates, by_d, max_or_range=max_or)
        if not len(t): continue
        IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
        nyr  = len(t) / 8.1; wr = (t["r"] > 0).mean() * 100
        s_is = sh(IS_t.groupby("date")["r"].sum()) if len(IS_t) else 0
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        lbl  = f"OR ≤ {max_or}pt" if max_or < 900 else "No OR filter"
        mark = " ← pre-committed" if max_or == 25 else ""
        print(f"  {lbl:<18}: {nyr:.0f}/yr  WR={wr:.1f}%  IS_Sh={s_is:.2f}  OOS_Sh={s_oo:.2f}{mark}")

    # ── STEP 3: Full results ──────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  STEP 3 — FULL RESULTS (pre-committed config)")
    print("=" * W)
    t = run_opex_pin(df, dates, by_d, pin_zone=8, max_or_range=25,
                     fade_trigger=6, stop_dist=10)
    IS_t = t[t["year"] <= 2021]; OOS_t = t[t["year"] >= 2022]
    print(f"  Total trades: {len(t)} ({len(t)/8.1:.0f}/yr)\n")
    prow("Full period  2018-2026", t)
    prow("IS  2018-2021",          IS_t)
    prow("OOS 2022-2026",          OOS_t)

    if len(t):
        print("\n  By direction:")
        prow("  LONG  (fade below pin)", t[t["direction"] == "LONG"])
        prow("  SHORT (fade above pin)", t[t["direction"] == "SHORT"])
        print("\n  By exit:")
        for ex in ["STOP", "TGT", "EOD"]:
            sub = t[t["exit"] == ex]
            if not len(sub): continue
            wr = (sub["r"] > 0).mean() * 100; ar = sub["r"].mean()
            print(f"    {ex}: n={len(sub):3d} ({len(sub)/len(t)*100:.1f}%)  "
                  f"WR={wr:.1f}%  AvgR={ar:+.3f}")
        print("\n  Year-by-year:")
        for yr in sorted(t["year"].unique()): yr_row(t, yr)

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  VERDICT — SIGNAL 5a: WEEKLY OPEX PIN")
    print("=" * W)
    if len(t):
        fs   = sh(t.groupby("date")["r"].sum())
        s_oo = sh(OOS_t.groupby("date")["r"].sum()) if len(OOS_t) else 0
        nyr  = len(t) / 8.1
        wr   = (t["r"] > 0).mean() * 100
        oos_wr = (OOS_t["r"] > 0).mean() * 100 if len(OOS_t) else 0
        print(f"\n  Full Sh={fs:.2f}  OOS Sh={s_oo:.2f}  WR={wr:.1f}%  "
              f"OOS_WR={oos_wr:.1f}%  {nyr:.0f}/yr")
        if fs > 2.5 and s_oo > 2.0: v = "✅ ADD"
        elif fs > 1.5 and s_oo > 1.0: v = "🟡 MARGINAL"
        else: v = "❌ SKIP"
        print(f"  {v}")
    print()


if __name__ == "__main__":
    main()
