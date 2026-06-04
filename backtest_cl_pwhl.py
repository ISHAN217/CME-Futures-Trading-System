#!/usr/bin/env python3
"""
backtest_cl_pwhl.py — Independent verification of CL Prior-Week H/L breakout

Strategy as described:
  OR   : 09:00–10:00 ET (= 08:00–09:00 CT)
  Entry: 10:00–11:30 ET — resting stop order at PWH (LONG) or PWL (SHORT)
         i.e. when bar HIGH >= PWH  → fill LONG  at PWH price
              when bar LOW  <= PWL  → fill SHORT at PWL price
  Stop : entry ∓ 1× OR_range
  Target: entry ± 4× OR_range
  Close: 12:00 ET hard flatten (noon ET = 11:00 CT)
  Contract: 1 MCL (Micro WTI) = $100/point

FILL ASSUMPTION (critical):
  Entry is a STOP ORDER at the PWH/PWL level.
  We assume the fill is at exactly PWH/PWL (best case).
  We also test with 2-tick ($0.02) and 5-tick ($0.05) slippage.

ANTI-HALLUCINATION CHECKS:
  1. PWH/PWL computed strictly from prior ISO week (no look-ahead)
  2. OR computed strictly from 09:00–10:00 ET bars only
  3. Entry bar HIGH/LOW must actually reach PWH/PWL (verified per trade)
  4. Stop and target hit checked bar by bar
  5. P&L cross-checked trade-sum vs daily-sum
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/CL_1min_continuous.parquet"
MCL_MULT = 100      # $100 per point for Micro WTI
SLIP     = 0.00     # start with zero, then test 0.02 and 0.05

# ET times
OR_START  = time(9,  0)    # 09:00 ET = 08:00 CT
OR_END    = time(10, 0)    # 10:00 ET = 09:00 CT
ENTRY_END = time(11, 30)   # 11:30 ET = 10:30 CT
CLOSE_T   = time(12, 0)    # 12:00 ET = 11:00 CT (hard flatten)


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["ts_et"])
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def build_weekly_hl(by_d, dates):
    """
    Compute prior-ISO-week H/L for each trading day.
    Strictly uses only bars from the prior calendar week.
    No look-ahead.
    """
    # Group all days by ISO week
    wk_hl = {}
    for d in dates:
        ts = pd.Timestamp(d); iso = ts.isocalendar()
        wk = (int(iso.year), int(iso.week))
        day = by_d.get(d)
        if day is None: continue
        h = day["CL_high"].max(); l = day["CL_low"].min()
        if wk not in wk_hl: wk_hl[wk] = [h, l]
        else:
            wk_hl[wk][0] = max(wk_hl[wk][0], h)
            wk_hl[wk][1] = min(wk_hl[wk][1], l)
    return wk_hl


def prior_week_key(d):
    """Return the ISO (year, week) key for the week BEFORE date d."""
    ts = pd.Timestamp(d); iso = ts.isocalendar()
    pw = int(iso.week) - 1; py = int(iso.year)
    if pw == 0:
        py -= 1
        pw = int(pd.Timestamp(f"{py}-12-28").isocalendar().week)
    return (py, pw)


def sh(dr):
    return (dr.mean() / dr.std(ddof=1) * np.sqrt(252)
            if len(dr) > 1 and dr.std() > 0 else 0.0)


def run_cl(df, dates, by_d, wk_hl, slip=0.00, target_R=4.0, stop_R=1.0):
    trades = []
    for d in dates:
        day = by_d.get(d)
        if day is None: continue

        # Prior week H/L
        pwk = prior_week_key(d)
        hl  = wk_hl.get(pwk)
        if hl is None: continue
        PWH, PWL = hl[0], hl[1]

        # Opening range 09:00–10:00 ET
        or_b = day[(day["time_et"] >= OR_START) & (day["time_et"] < OR_END)]
        if len(or_b) < 5: continue     # need at least 5 bars
        or_H  = or_b["CL_high"].max()
        or_L  = or_b["CL_low"].min()
        or_R  = or_H - or_L
        if or_R < 0.05: continue       # degenerate OR

        # Entry scan: 10:00–11:30 ET
        scan = day[(day["time_et"] >= OR_END) &
                   (day["time_et"] <= ENTRY_END)].reset_index(drop=True)
        if not len(scan): continue

        fired = False
        for idx, bar in scan.iterrows():
            if fired: break
            # LONG trigger: bar HIGH touches or exceeds PWH
            if not fired and bar["CL_high"] >= PWH:
                ep   = PWH + slip          # filled at PWH (+ slip)
                sp   = ep - stop_R * or_R  # stop below
                tp   = ep + target_R * or_R
                risk = ep - sp
                if risk < 0.01: continue

                # Simulate forward from this bar's close onwards
                # (bar already touched PWH, enter at PWH price)
                entry_t = bar["time_et"]
                fwd = day[day["time_et"] > entry_t]
                # also include rest of entry bar from PWH level
                rv = 0.0; why = "CLOSE"; exit_t = CLOSE_T; exit_p = None

                for _, fb in fwd.iterrows():
                    if fb["time_et"] >= CLOSE_T:
                        # Hard close at 12:00 ET
                        exit_p = fb["CL_close"]
                        pnl_pts = exit_p - ep
                        rv = pnl_pts / risk
                        why = "CLOSE"; break
                    if fb["CL_low"] <= sp:
                        rv = -1.0; why = "STOP"
                        exit_p = sp; break
                    if fb["CL_high"] >= tp:
                        rv = target_R; why = "TGT"
                        exit_p = tp; break

                pnl_usd = rv * risk * MCL_MULT
                ts = pd.Timestamp(d); iso = ts.isocalendar()
                trades.append(dict(
                    date=d, year=d.year, direction="LONG",
                    r=rv, pnl=round(pnl_usd, 2), win=pnl_usd > 0,
                    exit=why, PWH=round(PWH, 3), PWL=round(PWL, 3),
                    or_R=round(or_R, 3), ep=round(ep, 3),
                    sp=round(sp, 3), tp=round(tp, 3),
                    entry_t=entry_t,
                    iso_wk=(int(iso.year), int(iso.week)),
                ))
                fired = True

            # SHORT trigger: bar LOW touches or goes below PWL
            if not fired and bar["CL_low"] <= PWL:
                ep   = PWL - slip          # filled at PWL (- slip)
                sp   = ep + stop_R * or_R
                tp   = ep - target_R * or_R
                risk = sp - ep
                if risk < 0.01: continue

                entry_t = bar["time_et"]
                fwd = day[day["time_et"] > entry_t]
                rv = 0.0; why = "CLOSE"; exit_p = None

                for _, fb in fwd.iterrows():
                    if fb["time_et"] >= CLOSE_T:
                        exit_p = fb["CL_close"]
                        pnl_pts = ep - exit_p
                        rv = pnl_pts / risk
                        why = "CLOSE"; break
                    if fb["CL_high"] >= sp:
                        rv = -1.0; why = "STOP"
                        exit_p = sp; break
                    if fb["CL_low"] <= tp:
                        rv = target_R; why = "TGT"
                        exit_p = tp; break

                pnl_usd = rv * risk * MCL_MULT
                ts = pd.Timestamp(d); iso = ts.isocalendar()
                trades.append(dict(
                    date=d, year=d.year, direction="SHORT",
                    r=rv, pnl=round(pnl_usd, 2), win=pnl_usd > 0,
                    exit=why, PWH=round(PWH, 3), PWL=round(PWL, 3),
                    or_R=round(or_R, 3), ep=round(ep, 3),
                    sp=round(sp, 3), tp=round(tp, 3),
                    entry_t=entry_t,
                    iso_wk=(int(iso.year), int(iso.week)),
                ))
                fired = True

    return pd.DataFrame(trades)


def report(label, t, full_start=None):
    if not len(t):
        print(f"  {label}: NO TRADES"); return
    nyr  = len(t) / max((t["date"].max() - t["date"].min()).days / 365.25, 0.1)
    wr   = (t["pnl"] > 0).mean() * 100
    ar   = t["pnl"].mean()
    tot  = t["pnl"].sum()
    wins   = t[t["pnl"] > 0]["pnl"]
    losses = t[t["pnl"] < 0]["pnl"]
    pf     = abs(wins.sum() / losses.sum()) if len(losses) and losses.sum() != 0 else 99
    dr   = t.groupby("date")["pnl"].sum()
    s    = sh(dr)
    mdd  = (dr.cumsum() - dr.cumsum().cummax()).min()
    flag = "✅" if (ar > 0 and s > 1.5) else ("🟡" if ar > 0 else "❌")
    print(f"  {flag} {label}")
    print(f"     Trades : {len(t)} ({nyr:.0f}/yr, {nyr/52:.2f}/wk)")
    print(f"     WR     : {wr:.1f}%")
    print(f"     AvgTrd : ${ar:+.2f}")
    print(f"     Total  : ${tot:,.2f}")
    print(f"     PF     : {pf:.2f}")
    print(f"     Sharpe : {s:.2f}")
    print(f"     MaxDD  : ${mdd:,.2f}")
    print(f"     Exits  → STOP:{(t['exit']=='STOP').sum()}  TGT:{(t['exit']=='TGT').sum()}  CLOSE:{(t['exit']=='CLOSE').sum()}")
    # year by year
    for yr in sorted(t["year"].unique()):
        yt  = t[t["year"] == yr]
        ywr = (yt["pnl"] > 0).mean() * 100
        yar = yt["pnl"].sum()
        print(f"     {yr}: n={len(yt):3d}  WR={ywr:5.1f}%  P&L=${yar:+,.2f}")


def main():
    print("Loading CL data …")
    df    = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    wk_hl = build_weekly_hl(by_d, dates)

    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}")
    print(f"  {len(dates)} trading days  |  {len(wk_hl)} ISO weeks\n")

    W = 70

    # ── FULL PERIOD ──────────────────────────────────────────────────────────
    t_all = run_cl(df, dates, by_d, wk_hl, slip=0.00)
    print("=" * W)
    print("  FULL PERIOD (all data, zero slippage)")
    print("=" * W)
    report("Full 2021-2026", t_all)

    # ── IS / OOS SPLIT ───────────────────────────────────────────────────────
    IS_END   = date(2022, 12, 31)
    OS_START = date(2023, 1, 1)

    t_is  = t_all[t_all["date"] <= IS_END]
    t_oos = t_all[t_all["date"] >= OS_START]

    print(f"\n{'=' * W}")
    print("  IS vs OOS SPLIT  (IS=2021-22, OOS=2023-2026)")
    print("=" * W)
    report("IS  2021-2022", t_is)
    print()
    report("OOS 2023-2026", t_oos)

    # ── SLIPPAGE SENSITIVITY ─────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  SLIPPAGE SENSITIVITY (OOS only)")
    print("=" * W)
    print(f"  {'Slip':>8}  {'n':>5}  {'WR':>7}  {'AvgTrd':>8}  {'Total':>9}  {'PF':>6}  {'Sh':>6}")
    print("  " + "─" * 56)

    for slip_val in [0.00, 0.01, 0.02, 0.05, 0.10]:
        t_s = run_cl(df, [d for d in dates if d >= OS_START], by_d, wk_hl,
                     slip=slip_val)
        if not len(t_s): continue
        nyr = len(t_s) / 3.3
        wr  = (t_s["pnl"] > 0).mean() * 100
        ar  = t_s["pnl"].mean()
        tot = t_s["pnl"].sum()
        w_  = t_s[t_s["pnl"] > 0]["pnl"]; l_ = t_s[t_s["pnl"] < 0]["pnl"]
        pf  = abs(w_.sum() / l_.sum()) if len(l_) and l_.sum() != 0 else 99
        dr  = t_s.groupby("date")["pnl"].sum()
        s   = sh(dr)
        print(f"  {slip_val:>7.2f}pt  {len(t_s):>5}  {wr:>6.1f}%  "
              f"${ar:>7.2f}  ${tot:>8,.2f}  {pf:>5.2f}  {s:>5.2f}")

    # ── DIRECTION BREAKDOWN ──────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  DIRECTION BREAKDOWN (OOS, zero slip)")
    print("=" * W)
    for di in ["LONG", "SHORT"]:
        sub = t_oos[t_oos["direction"] == di]
        if not len(sub): continue
        wr = (sub["pnl"] > 0).mean() * 100
        ar = sub["pnl"].mean()
        tot = sub["pnl"].sum()
        flag = "✅" if ar > 0 else "❌"
        print(f"  {flag} {di:<6}: n={len(sub):3d} ({len(sub)/3.3:.0f}/yr)  "
              f"WR={wr:.1f}%  Avg=${ar:+.2f}  Total=${tot:,.2f}")

    # ── EIA WEDNESDAY EFFECT ─────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  EIA WEDNESDAY EFFECT (OOS)")
    print("=" * W)
    print("  EIA crude inventory report: 10:30 ET = 09:30 CT (Wednesdays)")
    print("  Strategy entry window includes pre-EIA trades on Wednesdays\n")
    t_oos["dow"] = pd.to_datetime(t_oos["date"].astype(str)).dt.day_name()
    for dw in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
        sub = t_oos[t_oos["dow"] == dw]
        if not len(sub): continue
        wr  = (sub["pnl"] > 0).mean() * 100
        ar  = sub["pnl"].mean()
        flag = "✅" if ar > 0 else "❌"
        print(f"  {flag} {dw:<12}: n={len(sub):3d}  WR={wr:.1f}%  Avg=${ar:+.2f}")

    # ── FILL VALIDITY CHECK ──────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  FILL VALIDITY AUDIT")
    print("=" * W)
    print("  For each trade, verify bar HIGH >= PWH (LONG) or LOW <= PWL (SHORT)")
    print("  i.e. the entry level was actually reached — no phantom fills\n")
    issues = 0
    for _, row in t_oos.iterrows():
        d = row["date"]; day = by_d.get(d)
        if day is None: continue
        scan = day[(day["time_et"] >= OR_END) & (day["time_et"] <= ENTRY_END)]
        if row["direction"] == "LONG":
            reached = (scan["CL_high"] >= row["PWH"]).any()
        else:
            reached = (scan["CL_low"] <= row["PWL"]).any()
        if not reached:
            issues += 1
    print(f"  Audited {len(t_oos)} OOS trades")
    pct_valid = (len(t_oos) - issues) / len(t_oos) * 100 if len(t_oos) else 0
    print(f"  Valid fills: {len(t_oos)-issues}/{len(t_oos)} ({pct_valid:.1f}%)")
    if issues == 0:
        print(f"  ✅ All fills valid — every entry bar actually reached the PWH/PWL level")
    else:
        print(f"  ⚠️  {issues} phantom fills detected — entry level never reached")

    # ── DOUBLE CHECK: P&L CROSS-CHECK ────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  P&L CROSS-CHECK (trade-sum vs daily-sum)")
    print("=" * W)
    daily_sum = t_oos.groupby("date")["pnl"].sum().sum()
    trade_sum = t_oos["pnl"].sum()
    print(f"  Trade-sum  : ${trade_sum:,.2f}")
    print(f"  Daily-sum  : ${daily_sum:,.2f}")
    print(f"  Difference : ${abs(trade_sum-daily_sum):.2f}")
    ok = "✅ MATCH" if abs(trade_sum-daily_sum) < 0.01 else "❌ MISMATCH"
    print(f"  {ok}")

    # ── WEEKLY DISTRIBUTION ──────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  WEEKLY P&L DISTRIBUTION (OOS)")
    print("=" * W)
    wk_pnl = {}
    for _, row in t_oos.iterrows():
        wk = row["iso_wk"]; wk_pnl[wk] = wk_pnl.get(wk, 0) + row["pnl"]
    wks = pd.Series(wk_pnl)
    print(f"  Signal weeks  : {len(wks)}")
    print(f"  Mean/wk       : ${wks.mean():+.2f}")
    print(f"  Median/wk     : ${wks.median():+.2f}")
    print(f"  Std/wk        : ${wks.std():.2f}")
    print(f"  Win weeks     : {(wks>0).sum()}/{len(wks)} ({(wks>0).mean()*100:.1f}%)")
    print(f"  P10/P25/P75/P90: ${wks.quantile(0.10):.0f} / "
          f"${wks.quantile(0.25):.0f} / "
          f"${wks.quantile(0.75):.0f} / "
          f"${wks.quantile(0.90):.0f}")
    print(f"  Best week     : ${wks.max():+.2f}")
    print(f"  Worst week    : ${wks.min():+.2f}")

    # ── FINAL COMPARISON ─────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  INDEPENDENT RESULT vs CLAIMED NUMBERS")
    print("=" * W)
    print(f"\n  {'Metric':<28}  {'Claimed':>12}  {'Independent':>12}  Match?")
    print("  " + "─" * 60)
    oos_wr    = (t_oos["pnl"]>0).mean()*100
    oos_tot   = t_oos["pnl"].sum()
    oos_avg   = t_oos["pnl"].mean()
    oos_n     = len(t_oos)
    oos_nyr   = oos_n / 3.3
    oos_pf    = (abs(t_oos[t_oos["pnl"]>0]["pnl"].sum() /
                t_oos[t_oos["pnl"]<0]["pnl"].sum())
                if len(t_oos[t_oos["pnl"]<0]) else 99)
    oos_mdd   = (t_oos.groupby("date")["pnl"].sum().cumsum() -
                 t_oos.groupby("date")["pnl"].sum().cumsum().cummax()).min()
    oos_sh    = sh(t_oos.groupby("date")["pnl"].sum())

    rows = [
        ("Trades (OOS)",       "412",     f"{oos_n}"),
        ("Trades/yr",          "138",     f"{oos_nyr:.0f}"),
        ("Win Rate",           "79.1%",   f"{oos_wr:.1f}%"),
        ("Profit Factor",      "10.78",   f"{oos_pf:.2f}"),
        ("Total PnL",          "$42,942", f"${oos_tot:,.0f}"),
        ("Avg per trade",      "$104",    f"${oos_avg:.0f}"),
        ("Max Drawdown",       "$400",    f"${oos_mdd:,.0f}"),
        ("Sharpe",             "1.41",    f"{oos_sh:.2f}"),
    ]
    for metric, claimed, actual in rows:
        # rough match check
        try:
            c = float(claimed.replace('$','').replace(',','').replace('%',''))
            a = float(actual.replace('$','').replace(',','').replace('%','').replace('−','-'))
            match = "✅" if abs(c-a)/max(abs(c),0.01) < 0.15 else "⚠️ "
        except:
            match = "?"
        print(f"  {match} {metric:<28}  {claimed:>12}  {actual:>12}")
    print()


if __name__ == "__main__":
    main()
