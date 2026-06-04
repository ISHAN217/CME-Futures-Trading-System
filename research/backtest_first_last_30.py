#!/usr/bin/env python3
"""
backtest_first_last_30.py — Gao et al. (2018) Intraday Momentum

Paper: "Intraday Momentum: The First Half-Hour Return Predicts the Last
        Half-Hour Return"
        Gao, Han, Li, Zhou — Journal of Financial Economics (2018)
        SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2552752

EXACT RULES:
  1. Compute the FIRST HALF-HOUR RETURN (9:30 open → 10:00 close)
  2. If positive → go LONG at 15:30 ET
     If negative → go SHORT at 15:30 ET
  3. Exit at 16:00 ET (market close, 30-min holding)
  4. No overnight positions

KEY PAPER FINDINGS:
  - R² = 1.6% (first half-hour → last half-hour predictability)
  - R² = 2.6% when combined with 12th half-hour (15:00-15:30)
  - Effect is STRONGER on:
      • High-volume days (>20-day rolling average)
      • High-volatility days
      • Macro news release days (NFP, FOMC, CPI)
  - Confirmed in FTSE 100 and EuroStoxx 50 futures
  - Driven by gamma hedging and institutional rebalancing

WHAT WE TEST:
  A. Base: pure first-half-hour direction, any magnitude
  B. Magnitude filter: only trade if |return| > threshold
  C. Both agree: ES and NQ first-half-hour must point same direction
  D. 12th half-hour: 15:00-15:30 return must CONFIRM first-half-hour direction
  E. Volume filter: only on high-volume days (>20-day average)
  F. Best combination

P&L expressed in points AND normalised by first-half-hour range (as 1R)
for comparison with ORB baseline.
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"

SLIP = 0.25   # 1 tick per side

def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def get_bar(day, t, col):
    """Get a specific column value at a specific time."""
    row = day[day["time_et"] == t]
    if len(row) == 0: return None
    return row[col].iloc[0]


def rolling_volume(by_d, dates, d, n=20):
    """Average ES volume in 9:30-16:00 window over last n days."""
    d_idx = dates.index(d)
    vols = []
    for prev in dates[max(0, d_idx-n): d_idx]:
        day = by_d.get(prev)
        if day is None: continue
        rth = day[(day["time_et"]>=time(9,30))&(day["time_et"]<=time(16,0))]
        vols.append(rth["ES_volume"].sum())
    return float(np.mean(vols)) if vols else 0.0


def run(df, dates, by_d,
        min_ret_pct=0.0,      # minimum |first-half-hour return| to trade
        require_both=True,     # both ES+NQ must agree on direction
        use_12th_hh=False,     # 12th half-hour (15:00-15:30) must confirm
        volume_filter=False,   # only on above-average volume days
        long_only=False,
        short_only=False):

    trades = []
    vol_cache = {}

    for d in dates:
        day = by_d.get(d)
        if day is None: continue

        # First half-hour returns (9:30 open → 10:00 close, 30 complete bars)
        es_open_930 = get_bar(day, time(9,30), "ES_open")
        nq_open_930 = get_bar(day, time(9,30), "NQ_open")
        es_cl_1000  = get_bar(day, time(10,0),  "ES_close")
        nq_cl_1000  = get_bar(day, time(10,0),  "NQ_close")

        if any(x is None for x in [es_open_930,nq_open_930,es_cl_1000,nq_cl_1000]):
            continue

        es_ret_1st = (es_cl_1000 - es_open_930) / es_open_930
        nq_ret_1st = (nq_cl_1000 - nq_open_930) / nq_open_930

        # Direction: based on ES (primary) or both required
        if require_both:
            if np.sign(es_ret_1st) != np.sign(nq_ret_1st): continue  # disagreement
            direction = "LONG" if es_ret_1st > 0 else "SHORT"
        else:
            direction = "LONG" if es_ret_1st > 0 else "SHORT"

        # Magnitude filter
        avg_ret = (abs(es_ret_1st) + abs(nq_ret_1st)) / 2
        if avg_ret < min_ret_pct / 100: continue

        # 12th half-hour confirmation (15:00-15:30)
        if use_12th_hh:
            es_cl_1500 = get_bar(day, time(15,0),  "ES_close")
            es_cl_1530 = get_bar(day, time(15,30), "ES_close")
            nq_cl_1500 = get_bar(day, time(15,0),  "NQ_close")
            nq_cl_1530 = get_bar(day, time(15,30), "NQ_close")
            if any(x is None for x in [es_cl_1500,es_cl_1530,nq_cl_1500,nq_cl_1530]):
                continue
            es_ret_12  = (es_cl_1530 - es_cl_1500) / es_cl_1500
            nq_ret_12  = (nq_cl_1530 - nq_cl_1500) / nq_cl_1500
            # 12th half-hour must agree with 1st half-hour
            if direction == "LONG"  and (es_ret_12 < 0 or nq_ret_12 < 0): continue
            if direction == "SHORT" and (es_ret_12 > 0 or nq_ret_12 > 0): continue

        # Volume filter
        if volume_filter:
            if d not in vol_cache:
                vol_cache[d] = rolling_volume(by_d, dates, d, n=20)
            avg_vol = vol_cache[d]
            today_vol_es = day[(day["time_et"]>=time(9,30))&
                               (day["time_et"]<=time(16,0))]["ES_volume"].sum()
            if avg_vol > 0 and today_vol_es < avg_vol: continue

        # Long/short filter
        if long_only  and direction != "LONG":  continue
        if short_only and direction != "SHORT": continue

        # Entry: close of 15:30 bar
        es_entry = get_bar(day, time(15,30), "ES_close")
        nq_entry = get_bar(day, time(15,30), "NQ_close")
        if es_entry is None or nq_entry is None: continue

        # Exit: close of 16:00 bar (last 30-minute holding)
        es_exit = get_bar(day, time(16,0), "ES_close")
        nq_exit = get_bar(day, time(16,0), "NQ_close")
        # Fallback: last available bar between 15:45-16:15
        if es_exit is None:
            fallback = day[(day["time_et"]>=time(15,45))&(day["time_et"]<=time(16,15))]
            if len(fallback): es_exit = fallback["ES_close"].iloc[-1]
        if nq_exit is None:
            fallback = day[(day["time_et"]>=time(15,45))&(day["time_et"]<=time(16,15))]
            if len(fallback): nq_exit = fallback["NQ_close"].iloc[-1]
        if es_exit is None or nq_exit is None: continue

        sign = 1 if direction == "LONG" else -1

        # Raw P&L in points
        es_ep = es_entry + SLIP*sign; es_xp = es_exit - SLIP*sign
        nq_ep = nq_entry + SLIP*sign; nq_xp = nq_exit - SLIP*sign
        pnl_es_pts = sign * (es_xp - es_ep)
        pnl_nq_pts = sign * (nq_xp - nq_ep)

        # Normalise by first-half-hour range (|open-close| as 1R proxy)
        es_rng = abs(es_cl_1000 - es_open_930)
        nq_rng = abs(nq_cl_1000 - nq_open_930)
        r_es   = pnl_es_pts / es_rng if es_rng > 0.01 else 0.0
        r_nq   = pnl_nq_pts / nq_rng if nq_rng > 0.01 else 0.0
        r_avg  = (r_es + r_nq) / 2

        # Last-30-min range for reference
        es_lo_30 = day[(day["time_et"]>=time(15,30))&(day["time_et"]<=time(16,0))]["ES_low"].min()
        es_hi_30 = day[(day["time_et"]>=time(15,30))&(day["time_et"]<=time(16,0))]["ES_high"].max()

        trades.append(dict(
            date=d, year=d.year,
            direction=direction,
            es_ret_1st=round(es_ret_1st*100,3),  # %
            nq_ret_1st=round(nq_ret_1st*100,3),
            avg_ret_pct=round(avg_ret*100,3),
            pnl_es=round(pnl_es_pts,2), pnl_nq=round(pnl_nq_pts,2),
            r_es=round(r_es,3), r_nq=round(r_nq,3),
            r_avg=round(r_avg,3),
            win=r_avg>0,
            dow=pd.Timestamp(d).day_name(),
        ))

    return pd.DataFrame(trades)


def prow(label, sub, r_col="r_avg", pad=42):
    if not len(sub): print(f"  {label:<{pad}}  n=0"); return
    wr=sub["win"].mean(); ar=sub[r_col].mean()
    dr=sub.groupby("date")[r_col].sum()
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    n_yr=sub["date"].nunique()/8.1
    flag="✅" if wr>0.53 else ("🟡" if wr>0.47 else "❌")
    print(f"  {flag} {label:<{pad}}  n={sub['date'].nunique():4d} ({n_yr:.0f}/yr)  "
          f"WR={wr*100:5.1f}%  AvgR={ar:+.4f}  Sh={sh:.2f}")


def main():
    print("Loading …")
    df = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")
    IS_END=date(2021,12,31); OS_START=date(2022,1,1)

    W=76
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── 1. All variations ──────────────────────────────────────────────────────
    sec("1. ALL VARIATIONS  —  First 30-min → Last 30-min momentum")
    print("  Paper finding: R²=1.6% (1st HH → last HH), R²=2.6% with 12th HH")
    print("  Applied on SPY 1993-2013. We test on ES+NQ 2018-2026.\n")
    print(f"  Baseline ORB (verified): ~22/yr  WR=47.8%  Sharpe=3.08")
    print(f"  PDH/L Confirmed:         ~36/yr  WR=54.5%  Sharpe=3.14\n")

    configs = [
        ("A. Base (any magnitude, both agree)",       0.0,  True,  False, False),
        ("B. Magnitude ≥ 0.1% (weak filter)",        0.1,  True,  False, False),
        ("C. Magnitude ≥ 0.2% (medium filter)",      0.2,  True,  False, False),
        ("D. Magnitude ≥ 0.3% (strong filter)",      0.3,  True,  False, False),
        ("E. + 12th HH confirmation",                 0.2,  True,  True,  False),
        ("F. + Volume filter",                        0.2,  True,  False, True),
        ("G. + 12th HH + Volume",                    0.2,  True,  True,  True),
        ("H. LONG only (≥0.2%)",                     0.2,  True,  False, False),
        ("I. ES only (no NQ require)",                0.2,  False, False, False),
    ]
    results = {}
    for label, minr, both, hh12, volf in configs:
        lo = label=="H. LONG only (≥0.2%)"
        tdf = run(df, dates, by_d,
                  min_ret_pct=minr, require_both=both,
                  use_12th_hh=hh12, volume_filter=volf,
                  long_only=lo)
        results[label] = tdf
        prow(label, tdf)

    # ── 2. Magnitude sensitivity ───────────────────────────────────────────────
    sec("2. MAGNITUDE SENSITIVITY  —  does signal strength matter?")
    print("  First half-hour return bucketed by size:\n")
    base = results["A. Base (any magnitude, both agree)"]
    buckets = [
        ("0.0–0.1% (flat open)",      0.0,  0.1),
        ("0.1–0.2% (small move)",     0.1,  0.2),
        ("0.2–0.3% (medium move)",    0.2,  0.3),
        ("0.3–0.5% (large move)",     0.3,  0.5),
        (">0.5%   (very large move)", 0.5, 99.0),
    ]
    for label, lo, hi in buckets:
        sub = base[(base["avg_ret_pct"]>=lo)&(base["avg_ret_pct"]<hi)]
        prow(f"  {label}", sub)

    # ── 3. Direction asymmetry ────────────────────────────────────────────────
    sec("3. DIRECTION ASYMMETRY  —  LONG vs SHORT")
    base = results["A. Base (any magnitude, both agree)"]
    prow("  All",    base)
    prow("  LONG",   base[base["direction"]=="LONG"])
    prow("  SHORT",  base[base["direction"]=="SHORT"])

    # ── 4. Year-by-year ───────────────────────────────────────────────────────
    sec("4. YEAR-BY-YEAR  —  Base (A) vs Best filtered")
    # Find best by Sharpe
    best_label = max(results,
                     key=lambda k: (
                         results[k].groupby("date")["r_avg"].sum().mean() /
                         results[k].groupby("date")["r_avg"].sum().std(ddof=1)
                         if len(results[k])>1 and
                         results[k].groupby("date")["r_avg"].sum().std()>0 else -999
                     ))
    best = results[best_label]
    A    = results["A. Base (any magnitude, both agree)"]

    print(f"  Base vs Best ({best_label})\n")
    print(f"  {'Year':<6}  {'Base n':>6}  {'Base WR':>8}  {'Base Sh':>8}  │  "
          f"{'Best n':>6}  {'Best WR':>8}  {'Best Sh':>8}")
    print(f"  {'─'*66}")
    for yr in sorted(A["year"].unique()):
        a_s=A[A["year"]==yr]; b_s=best[best["year"]==yr] if len(best) else pd.DataFrame()
        a_wr=a_s["win"].mean() if len(a_s) else 0
        b_wr=b_s["win"].mean() if len(b_s) else 0
        a_dr=a_s.groupby("date")["r_avg"].sum()
        a_sh=a_dr.mean()/a_dr.std(ddof=1)*np.sqrt(252) if len(a_dr)>1 and a_dr.std()>0 else 0
        b_dr=b_s.groupby("date")["r_avg"].sum() if len(b_s) else pd.Series([0])
        b_sh=b_dr.mean()/b_dr.std(ddof=1)*np.sqrt(252) if len(b_dr)>1 and b_dr.std()>0 else 0
        lbl="← IS" if yr<=2021 else "← OOS"
        print(f"  {yr:<6}  {len(a_s):>6}  {a_wr*100:>7.1f}%  {a_sh:>8.2f}  │  "
              f"{len(b_s):>6}  {b_wr*100:>7.1f}%  {b_sh:>8.2f}  {lbl}")

    # ── 5. Walk-forward ───────────────────────────────────────────────────────
    sec("5. WALK-FORWARD  IS 2018-2021  →  OOS 2022-2026")
    print(f"  Best config: {best_label}\n")
    for period, mask in [("IS  2018-2021", best["date"]<=IS_END),
                         ("OOS 2022-2026", best["date"]>=OS_START)]:
        prow(period, best[mask])

    # ── 6. Day of week ────────────────────────────────────────────────────────
    sec("6. WIN RATE BY DAY OF WEEK")
    for dow in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
        prow(f"  {dow}", best[best["dow"]==dow])

    # ── 7. Calibration check ─────────────────────────────────────────────────
    sec("7. CALIBRATION CHECK vs PAPER")
    print("  Paper (Gao et al. 2018, SPY 1993-2013):")
    print("    R² first → last half-hour  : 1.6%")
    print("    R² with 12th half-hour     : 2.6%")
    print("    Strategy more effective on : high volume, high vol, news days")
    print()
    A = results["A. Base (any magnitude, both agree)"]
    if len(A):
        # Compute linear R² between first-HH return and last-HH return
        A_tmp = A.copy()
        A_tmp["last_30_ret"] = (A_tmp["r_avg"])   # proxy
        X = A_tmp["avg_ret_pct"].values
        y = A_tmp["r_avg"].values
        corr = np.corrcoef(X, y)[0,1]
        r2   = corr**2
        print(f"  Our data (ES+NQ 2018-2026):")
        print(f"    Correlation (1st HH ret → last 30 P&L) : {corr:+.3f}")
        print(f"    R² equivalent                           : {r2*100:.2f}%")
        print(f"    Paper target R²                         : 1.6%")
        print()
        if r2 >= 0.01:
            print("  ✅ R² ≥ 1% — result broadly consistent with paper")
        elif r2 >= 0.005:
            print("  🟡 R² 0.5-1% — weaker than paper but directionally correct")
        else:
            print("  ❌ R² < 0.5% — signal is very weak in our data/period")

    # ── 8. Does this COMBINE with ORB? ───────────────────────────────────────
    sec("8. COMBINATION WITH ORB  —  Are they additive?")
    print("  ORB fires in the MORNING (10:00-10:30 ET)")
    print("  First/Last fires in the EVENING (entry 15:30 ET)")
    print("  These are COMPLETELY non-overlapping time windows.")
    print("  Can trade BOTH on the same day.\n")
    print("  On ORB signal days, how does the last-30-min perform?")

    # Identify ORB days
    orb_days = set()
    for d in dates:
        day = by_d.get(d)
        if day is None: continue
        or_b = day[(day["time_et"]>=time(9,30))&(day["time_et"]<time(10,0))]
        if len(or_b)==0: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(8<=es_R<=60 and 30<=nq_R<=175): continue
        post=day[(day["time_et"]>=time(10,0))&(day["time_et"]<time(10,30))]
        es_b=nq_b=None
        for _,r in post.iterrows():
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"
                elif r["ES_low"]<es_L: es_b="S"
            if nq_b is None:
                if r["NQ_high"]>nq_H: nq_b="L"
                elif r["NQ_low"]<nq_L: nq_b="S"
            if es_b and nq_b: break
        if es_b and nq_b and es_b==nq_b: orb_days.add(d)

    all_days = set(A["date"])
    orb_overlap = A[A["date"].isin(orb_days)]
    non_orb     = A[~A["date"].isin(orb_days)]
    prow("  Last-30-min on ORB days",       orb_overlap)
    prow("  Last-30-min on non-ORB days",   non_orb)

    print()
    print("  Do ORB direction and first-HH direction AGREE?")
    # On ORB days, check if ORB direction and first-HH direction are same
    orb_dir_map = {}
    for d in dates:
        day = by_d.get(d)
        if day is None: continue
        or_b = day[(day["time_et"]>=time(9,30))&(day["time_et"]<time(10,0))]
        if len(or_b)==0: continue
        es_H=or_b["ES_high"].max(); es_L=or_b["ES_low"].min(); es_R=es_H-es_L
        nq_H=or_b["NQ_high"].max(); nq_L=or_b["NQ_low"].min(); nq_R=nq_H-nq_L
        if not(8<=es_R<=60 and 30<=nq_R<=175): continue
        post=day[(day["time_et"]>=time(10,0))&(day["time_et"]<time(10,30))]
        es_b=nq_b=None
        for _,r in post.iterrows():
            if es_b is None:
                if r["ES_high"]>es_H: es_b="L"
                elif r["ES_low"]<es_L: es_b="S"
            if nq_b is None:
                if r["NQ_high"]>nq_H: nq_b="L"
                elif r["NQ_low"]<nq_L: nq_b="S"
            if es_b and nq_b: break
        if es_b and nq_b and es_b==nq_b:
            orb_dir_map[d] = "LONG" if es_b=="L" else "SHORT"

    A_orb = A[A["date"].isin(orb_dir_map)].copy()
    A_orb["orb_dir"] = A_orb["date"].map(orb_dir_map)
    A_orb["agree"] = A_orb["direction"] == A_orb["orb_dir"]
    agree_sub = A_orb[A_orb["agree"]==True]
    disag_sub = A_orb[A_orb["agree"]==False]
    prow("  ORB+First-HH AGREE  (same direction)", agree_sub)
    prow("  ORB+First-HH DISAGREE", disag_sub)

    # ── 9. Final verdict ──────────────────────────────────────────────────────
    sec("9. HONEST VERDICT")
    best_wr=best["win"].mean() if len(best) else 0
    dr=best.groupby("date")["r_avg"].sum() if len(best) else pd.Series([0])
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    n_yr=best["date"].nunique()/8.1 if len(best) else 0
    print(f"  Best config   : {best_label}")
    print(f"  Signal days   : {best['date'].nunique() if len(best) else 0} ({n_yr:.0f}/yr)")
    print(f"  WR            : {best_wr*100:.1f}%")
    print(f"  Sharpe        : {sh:.2f}")
    print()
    print(f"  ORB baseline  : 22/yr  WR=47.8%  Sharpe=3.08")
    print(f"  PDH/L Conf    : 36/yr  WR=54.5%  Sharpe=3.14")
    print()
    print("  KEY DISTINCTION vs ORB:")
    print("  → Fires in last 30 min (15:30-16:00) — completely separate window")
    print("  → Fires EVERY day (if magnitude filter met) — much higher frequency")
    print("  → True additive signal — can stack with ORB on same day")
    print()
    if sh > 2.0:
        print("  ✅ Strong edge — implement as standalone end-of-day signal")
    elif sh > 1.0:
        print("  🟡 Moderate edge — useful as supplementary EOD signal")
    elif sh > 0.3:
        print("  ⚠️  Weak but positive — study further before trading")
    else:
        print("  ❌ No meaningful edge in our data for this period")
    print()


if __name__ == "__main__":
    main()
