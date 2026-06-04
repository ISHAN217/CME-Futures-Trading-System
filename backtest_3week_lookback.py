#!/usr/bin/env python3
"""
backtest_3week_lookback.py — Live 3-week performance review

Shows exactly what signals fired on each trading day over the last 3 weeks
(May 12 – May 30, 2026) and what the P&L was, using the fully verified
joint system (ES×4 + CL).

Every signal entry is market-fill (bar close). Fill validity checked.
"""
import pandas as pd, numpy as np
from datetime import time, date
from backtest_verify_deep import (load, build_levels,
    run_primary, run_scalp, run_double_test, run_gap_orb)
from backtest_joint_cl import (load_cl, build_cl_weekly, run_cl,
                                prior_wk, build_cl_weekly)

CUTOFF_START = date(2026, 5, 12)   # 3 weeks back from competition
CUTOFF_END   = date(2026, 5, 30)
ACCT = 25_000

def main():
    print("Loading updated data …")
    df_es = load()
    dates_es = sorted(df_es["date_et"].unique())
    by_d_es  = {d: g.reset_index(drop=True) for d,g in df_es.groupby("date_et")}
    weekly, monthly = build_levels(by_d_es, dates_es)

    df_cl    = load_cl()
    dates_cl = sorted(df_cl["date_et"].unique())
    by_d_cl  = {d: g.reset_index(drop=True) for d,g in df_cl.groupby("date_et")}
    wk_hl    = build_cl_weekly(by_d_cl, dates_cl)

    print(f"  ES/NQ: {dates_es[-1]}  |  CL: {dates_cl[-1]}\n")

    print("Running all 5 systems (full history for context) …")
    pri = run_primary(df_es, dates_es, by_d_es)
    scl = run_scalp(df_es, dates_es, by_d_es, weekly, monthly)
    dbt = run_double_test(df_es, dates_es, by_d_es, weekly, monthly)
    gap = run_gap_orb(df_es, dates_es, by_d_es)
    cl  = run_cl(by_d_cl, dates_cl, wk_hl, n_mcl=5)
    print()

    # Filter to 3-week window
    def window(t, col="date"):
        return t[(t[col] >= CUTOFF_START) & (t[col] <= CUTOFF_END)]

    pri3 = window(pri); scl3 = window(scl)
    dbt3 = window(dbt); gap3 = window(gap); cl3 = window(cl)

    # Build daily joint P&L for window
    all_d = sorted(set(
        list(pri3["date"]) + list(scl3["date"]) +
        list(dbt3["date"]) + list(gap3["date"]) + list(cl3["date"])
    ))

    d_pri = pri3.groupby("date")["usd"].sum()
    d_scl = scl3.groupby("date")["usd"].sum()
    d_dbt = dbt3.groupby("date")["usd"].sum()
    d_gap = gap3.groupby("date")["usd"].sum()
    d_cl  = cl3.groupby("date")["pnl"].sum()

    W = 78
    print("=" * W)
    print(f"  LAST 3 WEEKS: {CUTOFF_START} → {CUTOFF_END}")
    print("=" * W)
    print(f"\n  {'Date':<12} {'Day':<4} {'Primary':>9} {'Scalp':>8} {'DblTest':>8} {'Gap+ORB':>8} {'CL 5MCL':>8}  {'DAY P&L':>9}  signals")
    print("  " + "─" * 76)

    total = 0; wk_pnl = {}; wk_sigs = {}
    for d in sorted(set(list(df_es["date_et"].unique()) +
                        list(df_cl["date_et"].unique()))):
        if d < CUTOFF_START or d > CUTOFF_END: continue
        p = d_pri.get(d, 0); s = d_scl.get(d, 0)
        b = d_dbt.get(d, 0); g = d_gap.get(d, 0); c = d_cl.get(d, 0)
        day_total = p + s + b + g + c
        total += day_total
        dow = pd.Timestamp(d).strftime("%a")

        ts = pd.Timestamp(d); iso = ts.isocalendar()
        wk = (int(iso.year), int(iso.week))
        wk_pnl[wk]  = wk_pnl.get(wk, 0) + day_total
        wk_sigs[wk] = wk_sigs.get(wk, 0) + (1 if day_total != 0 else 0)

        sigs = []
        if p != 0: sigs.append(f"PRI({'L' if (pri3[pri3['date']==d]['direction']=='LONG').any() else 'S'} {'W' if p>0 else 'L'})")
        if s != 0: sigs.append(f"SCL({'W' if s>0 else 'L'})")
        if b != 0: sigs.append(f"DT({'W' if b>0 else 'L'})")
        if g != 0: sigs.append(f"GAP({'W' if g>0 else 'L'})")
        if c != 0: sigs.append(f"CL({'W' if c>0 else 'L'})")

        flag = "✅" if day_total > 0 else ("❌" if day_total < 0 else "  ")
        if day_total != 0 or sigs:
            print(f"  {flag} {d}  {dow}  "
                  f"{'$'+str(int(p)):>9} {'$'+str(int(s)):>8} {'$'+str(int(b)):>8} "
                  f"{'$'+str(int(g)):>8} {'$'+str(int(c)):>8}  "
                  f"{'$'+str(int(day_total)):>9}  {' '.join(sigs)}")
        else:
            print(f"     {d}  {dow}  {'—':>9} {'—':>8} {'—':>8} {'—':>8} {'—':>8}  {'—':>9}  no signal")

    # Weekly summary
    print("\n  " + "─" * 76)
    print(f"\n  WEEKLY SUMMARY:")
    running = 0
    for wk in sorted(wk_pnl.keys()):
        pnl = wk_pnl[wk]; running += pnl
        flag = "✅" if pnl > 0 else "❌"
        print(f"  {flag} Week {wk[1]:02d} (ISO):  ${pnl:+,.0f}  "
              f"({wk_sigs.get(wk,0)} signal days)  running: ${running:+,.0f}")

    print(f"\n  3-WEEK TOTAL: ${total:+,.0f}  ({total/ACCT*100:+.1f}% on $25k account)")

    # Quick stats
    print(f"\n  SIGNAL COUNT (3 weeks):")
    for nm, t3 in [("Primary",pri3),("Scalp",scl3),("Double Test",dbt3),
                    ("Gap+ORB",gap3),("CL 5MCL",cl3)]:
        col = "pnl" if nm=="CL 5MCL" else "usd"
        if not len(t3): print(f"    {nm:<14}: 0 trades"); continue
        wr = (t3[col]>0).mean()*100
        tot = t3[col].sum()
        print(f"    {nm:<14}: {len(t3):3d} trades  WR={wr:.0f}%  Total=${tot:+,.0f}")

    print(f"\n  JOINT TOTAL:  ${total:+,.0f}")

    # Compare to OOS expectations
    print(f"\n{'=' * W}")
    print("  CONTEXT: 3-WEEK RESULT vs HISTORICAL EXPECTATION")
    print("=" * W)
    # OOS weekly avg
    pass  # already imported above
    pri_oos = pri[pri["date"]>=date(2022,1,1)]
    scl_oos = scl[scl["date"]>=date(2022,1,1)]
    dbt_oos = dbt[dbt["date"]>=date(2022,1,1)]
    gap_oos = gap[gap["date"]>=date(2022,1,1)]
    cl5_oos = cl[cl["date"]>=date(2023,1,1)]

    d_pri_oos=pri_oos.groupby("date")["usd"].sum()
    d_scl_oos=scl_oos.groupby("date")["usd"].sum()
    d_dbt_oos=dbt_oos.groupby("date")["usd"].sum()
    d_gap_oos=gap_oos.groupby("date")["usd"].sum()
    d_cl_oos =cl5_oos.groupby("date")["pnl"].sum()

    all_oos=sorted(set(list(pri_oos["date"])+list(scl_oos["date"])+
                        list(dbt_oos["date"])+list(gap_oos["date"])+
                        list(cl5_oos["date"])))
    daily_oos={d: d_pri_oos.get(d,0)+d_scl_oos.get(d,0)+
                  d_dbt_oos.get(d,0)+d_gap_oos.get(d,0)+
                  (d_cl_oos.get(d,0) if d>=date(2023,1,1) else 0)
               for d in all_oos}

    wk_oos={}
    for d,v in daily_oos.items():
        ts=pd.Timestamp(d); iso=ts.isocalendar()
        k=(int(iso.year),int(iso.week)); wk_oos[k]=wk_oos.get(k,0)+v
    wks_oos=pd.Series(wk_oos)

    exp_3wk = wks_oos.mean() * 3
    p25_3wk = wks_oos.quantile(0.25) * 3
    p75_3wk = wks_oos.quantile(0.75) * 3
    n_wks = len(wk_pnl)
    n_win  = sum(1 for v in wk_pnl.values() if v>0)

    print(f"\n  Historical OOS median week  : ${wks_oos.median():+,.0f}")
    print(f"  Expected 3-week P&L (mean×3): ${exp_3wk:+,.0f}")
    print(f"  Typical 3-week range (P25-P75): ${p25_3wk:+,.0f} to ${p75_3wk:+,.0f}")
    print(f"\n  Actual 3-week result        : ${total:+,.0f}")
    pct = (total - exp_3wk) / abs(exp_3wk) * 100 if exp_3wk != 0 else 0
    beat = "above" if total > exp_3wk else "below"
    print(f"  vs expectation              : {beat} by ${abs(total-exp_3wk):,.0f} ({abs(pct):.0f}%)")
    print(f"  Win weeks: {n_win}/{n_wks}")
    print()


if __name__ == "__main__":
    main()
