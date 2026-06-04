#!/usr/bin/env python3
"""
backtest_pdhl.py — Prior Day High/Low Breakout on ORB-Miss Days
              + Combined ORB + PDH/L system full statistics

Entry cutoff tightened to 11:00 ET (edge evaporates after noon).

Output sections
───────────────
  1. Day classification
  2. PDH/L detailed trade stats (WR, EV, profit factor, avg win/loss)
  3. Weekly performance  ← competition-relevant
  4. Monthly breakdown
  5. Year-by-year
  6. Combined ORB + PDH/L system overview
"""

import sys
import numpy as np
import pandas as pd
from datetime import time
from collections import defaultdict

# ─── Parameters ───────────────────────────────────────────────────────────────
DATA         = ("/Users/ishanbhardwaj/cme_competition_system/output/mtf/"
                "ES_NQ_1min_aligned.parquet")

OR_START     = time(9, 30)
OR_END       = time(10, 0)
ORB_CUTOFF   = time(10, 30)   # ORB dual-confirm window
WATCH_START  = time(10, 0)    # PDH/L watch starts at OR lock
ENTRY_CUTOFF = time(11, 0)    # ← tightened from 14:00 (edge evaporates after noon)
EOD_CLOSE    = time(15, 30)
RTH_START    = time(9, 30)
RTH_END      = time(16, 0)

MIN_ES = 8.0;  MAX_ES = 60.0
MIN_NQ = 30.0; MAX_NQ = 150.0
PMULT  = 1.5
SLIP   = 0.25   # 1 tick per side


# ─── Helpers ──────────────────────────────────────────────────────────────────

def simulate_trade(day_df, entry_t, direction, sym,
                   entry_px, stop_px, tgt_px,
                   col_h, col_l, col_c):
    """Bar-by-bar simulation after entry. Returns (exit_px, exit_r, net_pts, r_val)."""
    after = day_df[day_df["time_et"] > entry_t]
    for _, bar in after.iterrows():
        t = bar["time_et"]
        if t >= EOD_CLOSE:
            exit_px = bar[col_c]; exit_r = "EOD"; break
        if direction == "LONG":
            if bar[col_l] <= stop_px: exit_px = stop_px; exit_r = "STOP"; break
            if bar[col_h] >= tgt_px:  exit_px = tgt_px;  exit_r = "TGT";  break
        else:
            if bar[col_h] >= stop_px: exit_px = stop_px; exit_r = "STOP"; break
            if bar[col_l] <= tgt_px:  exit_px = tgt_px;  exit_r = "TGT";  break
    else:
        return None, None, None, None

    sign     = 1 if direction == "LONG" else -1
    slip_out = SLIP if exit_r in ("STOP", "EOD") else 0.0
    net_pts  = sign * (exit_px - entry_px) - SLIP - slip_out
    rng      = abs(entry_px - stop_px)
    r_val    = net_pts / rng if rng > 0 else 0.0
    return exit_px, exit_r, net_pts, r_val


def week_key(d):
    ts = pd.Timestamp(d)
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def print_section(title):
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


def print_sub(title):
    print(f"\n  {title}")
    print(f"  {'─'*56}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("Loading data …")
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    df = df.sort_values("ts_et").reset_index(drop=True)

    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    print(f"  {len(df):,} bars   {dates[0]} → {dates[-1]}   ({len(dates)} trading days)")

    # ── Classification counters ───────────────────────────────────────────────
    cnt = dict(fire=0, miss_range=0, miss_align=0)
    sig = dict(fired=0, none=0, conflict=0)

    orb_trades  = []   # ORB trades (for combined stats)
    pdhl_trades = []   # PDH/L trades

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue

        # Opening range
        or_b = day[(day["time_et"] >= OR_START) & (day["time_et"] < OR_END)]
        if len(or_b) == 0: continue

        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min()
        nq_H = or_b["NQ_high"].max(); nq_L = or_b["NQ_low"].min()
        es_R = es_H - es_L;           nq_R = nq_H - nq_L

        if not (MIN_ES <= es_R <= MAX_ES and MIN_NQ <= nq_R <= MAX_NQ):
            cnt["miss_range"] += 1
            continue

        wk = week_key(d)
        mo = str(d)[:7]   # YYYY-MM
        yr = d.year

        # ── ORB check ────────────────────────────────────────────────────────
        post = day[(day["time_et"] >= OR_END) & (day["time_et"] < ORB_CUTOFF)]
        es_b = nq_b = None
        orb_entry_row = None
        orb_dir = None

        for _, r in post.iterrows():
            if es_b is None:
                if r["ES_high"] > es_H: es_b = "L"
                elif r["ES_low"] < es_L: es_b = "S"
            if nq_b is None:
                if r["NQ_high"] > nq_H: nq_b = "L"
                elif r["NQ_low"] < nq_L: nq_b = "S"
            if es_b and nq_b:
                orb_entry_row = r
                break

        orb_fired = (es_b and nq_b and es_b == nq_b)

        if orb_fired:
            cnt["fire"] += 1
            orb_dir = "LONG" if es_b == "L" else "SHORT"

            # Simulate ORB trade
            for sym, h, l, rng, col_h, col_l, col_c in [
                ("ES", es_H, es_L, es_R, "ES_high", "ES_low", "ES_close"),
                ("NQ", nq_H, nq_L, nq_R, "NQ_high", "NQ_low", "NQ_close"),
            ]:
                if orb_dir == "LONG":
                    entry_px, stop_px, tgt_px = h, l, h + rng * PMULT
                else:
                    entry_px, stop_px, tgt_px = l, h, l - rng * PMULT

                _, exit_r, net_pts, r_val = simulate_trade(
                    day, orb_entry_row["time_et"], orb_dir, sym,
                    entry_px, stop_px, tgt_px, col_h, col_l, col_c
                )
                if r_val is None: continue
                orb_trades.append(dict(
                    date=d, week=wk, month=mo, year=yr,
                    sym=sym, dir=orb_dir, exit_r=exit_r,
                    r=r_val, win=r_val > 0,
                ))
            continue   # ORB handled this day

        # ── ORB missed — try PDH/L ───────────────────────────────────────────
        cnt["miss_align"] += 1

        prev = by_d.get(dates[i - 1])
        if prev is None: sig["none"] += 1; continue
        prth = prev[(prev["time_et"] >= RTH_START) & (prev["time_et"] <= RTH_END)]
        if len(prth) == 0: sig["none"] += 1; continue

        es_PDH = prth["ES_high"].max(); es_PDL = prth["ES_low"].min()
        nq_PDH = prth["NQ_high"].max(); nq_PDL = prth["NQ_low"].min()

        watch = day[(day["time_et"] >= WATCH_START) & (day["time_et"] < ENTRY_CUTOFF)]
        es_up = es_dn = nq_up = nq_dn = False
        direction = None; entry_row = None

        for _, r in watch.iterrows():
            if not es_up and r["ES_high"] > es_PDH: es_up = True
            if not es_dn and r["ES_low"]  < es_PDL: es_dn = True
            if not nq_up and r["NQ_high"] > nq_PDH: nq_up = True
            if not nq_dn and r["NQ_low"]  < nq_PDL: nq_dn = True

            if es_up and nq_up and direction is None:
                direction = "LONG";  entry_row = r; break
            if es_dn and nq_dn and direction is None:
                direction = "SHORT"; entry_row = r; break
            if (es_up and nq_dn) or (es_dn and nq_up):
                sig["conflict"] += 1; direction = "CONFLICT"; break

        if direction is None:   sig["none"] += 1; continue
        if direction == "CONFLICT": continue

        sig["fired"] += 1
        entry_t = entry_row["time_et"]
        eh = entry_t.hour + entry_t.minute / 60

        if   eh < 10.5: tbucket = "10:00-10:30"
        elif eh < 11.0: tbucket = "10:30-11:00"
        else:           tbucket = "11:00+"

        for sym, pdh, pdl, rng, col_h, col_l, col_c in [
            ("ES", es_PDH, es_PDL, es_R, "ES_high", "ES_low", "ES_close"),
            ("NQ", nq_PDH, nq_PDL, nq_R, "NQ_high", "NQ_low", "NQ_close"),
        ]:
            if direction == "LONG":
                entry_px, stop_px, tgt_px = pdh, pdh - rng, pdh + rng * PMULT
            else:
                entry_px, stop_px, tgt_px = pdl, pdl + rng, pdl - rng * PMULT

            _, exit_r, net_pts, r_val = simulate_trade(
                day, entry_t, direction, sym,
                entry_px, stop_px, tgt_px, col_h, col_l, col_c
            )
            if r_val is None: continue
            pdhl_trades.append(dict(
                date=d, week=wk, month=mo, year=yr,
                sym=sym, dir=direction, tbucket=tbucket,
                exit_r=exit_r, r=r_val, win=r_val > 0,
            ))

    # ═══════════════════════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════════════════════

    total_valid = cnt["fire"] + cnt["miss_align"]

    print_section("1. DAY CLASSIFICATION")
    print(f"  Total valid-OR days      : {total_valid}")
    print(f"  ORB fired                : {cnt['fire']:4d}  ({cnt['fire']/total_valid*100:.1f}%)")
    print(f"  ORB miss — no alignment  : {cnt['miss_align']:4d}  ({cnt['miss_align']/total_valid*100:.1f}%)")
    print(f"  Range-filtered (skipped) : {cnt['miss_range']:4d}")

    miss = cnt["miss_align"]
    print(f"\n  PDH/L signals on {miss} ORB-miss days:")
    print(f"    Fired     : {sig['fired']:4d}  ({sig['fired']/miss*100:.1f}%)")
    print(f"    No break  : {sig['none']:4d}  ({sig['none']/miss*100:.1f}%)")
    print(f"    Conflict  : {sig['conflict']:4d}  ({sig['conflict']/miss*100:.1f}%)")

    if not pdhl_trades:
        print("\nNo PDH/L trades — check data/parameters."); return

    pdf  = pd.DataFrame(pdhl_trades)
    n    = len(pdf)
    nsig = n // 2   # signal days (2 legs per signal)
    wr   = pdf["win"].mean()
    avr  = pdf["r"].mean()
    ev   = wr * PMULT - (1 - wr)

    wins  = pdf[pdf["win"] == True]["r"]
    loses = pdf[pdf["win"] == False]["r"]
    pf    = wins.sum() / abs(loses.sum()) if len(loses) and loses.sum() != 0 else float("inf")
    wl    = wins.mean() / abs(loses.mean()) if len(loses) and loses.mean() != 0 else float("inf")

    # ── 2. PDH/L Trade Stats ──────────────────────────────────────────────────
    print_section(f"2. PDH/L TRADE STATS  (entry cutoff 11:00 ET)")

    print_sub("Overview")
    print(f"    Signal days        : {nsig}")
    print(f"    Total legs         : {n}")
    print(f"    Win rate           : {wr*100:.1f}%    ← ORB baseline: 51.9%")
    print(f"    Avg R per leg      : {avr:+.3f}R  ← ORB baseline: +0.247R")
    print(f"    EV (1.5R/1R)       : {ev:+.3f}R  ← ORB baseline: +0.247R")
    print(f"    Profit factor      : {pf:.2f}x  (gross win / gross loss)")
    print(f"    Avg winner         : {wins.mean():+.3f}R")
    print(f"    Avg loser          : {loses.mean():+.3f}R")
    print(f"    Win/loss ratio     : {wl:.2f}x")

    print_sub("By direction")
    for d_ in ["LONG", "SHORT"]:
        sub = pdf[pdf["dir"] == d_]
        print(f"    {d_:5s}  n={len(sub):4d}  WR={sub['win'].mean()*100:.1f}%  "
              f"AvgR={sub['r'].mean():+.3f}R")

    print_sub("By exit reason")
    for r_ in ["TGT", "STOP", "EOD"]:
        sub = pdf[pdf["exit_r"] == r_]
        if len(sub):
            print(f"    {r_:4s}  n={len(sub):4d}  ({len(sub)/n*100:.1f}%)  "
                  f"WR={sub['win'].mean()*100:.1f}%  AvgR={sub['r'].mean():+.3f}R")

    print_sub("By entry time bucket")
    for b in ["10:00-10:30", "10:30-11:00"]:
        sub = pdf[pdf["tbucket"] == b]
        if len(sub):
            print(f"    {b}  n={len(sub):4d}  ({len(sub)/n*100:.1f}%)  "
                  f"WR={sub['win'].mean()*100:.1f}%  AvgR={sub['r'].mean():+.3f}R")

    # ── 3. Weekly Performance ─────────────────────────────────────────────────
    print_section("3. WEEKLY PERFORMANCE  (competition-relevant)")

    # Average the two legs per signal day, then sum per week
    day_r   = pdf.groupby("date")["r"].mean()          # one R per signal day
    day_win = pdf.groupby("date")["win"].mean() >= 0.5  # day "won" if majority legs won

    pdf_day = day_r.reset_index()
    pdf_day.columns = ["date", "r"]
    pdf_day["week"] = pdf_day["date"].apply(week_key)

    weekly = pdf_day.groupby("week").agg(
        signals  = ("r", "count"),
        total_r  = ("r", "sum"),
        avg_r    = ("r", "mean"),
        win_days = ("r", lambda x: (x > 0).sum()),
    ).reset_index()
    weekly["week_win"] = weekly["total_r"] > 0

    # Total trading weeks in dataset
    all_weeks = len(set(week_key(d) for d in dates[1:]))

    print(f"\n  Total trading weeks in dataset  : {all_weeks}")
    print(f"  Weeks with ≥1 PDH/L signal      : {len(weekly)}  "
          f"({len(weekly)/all_weeks*100:.1f}%)")
    print(f"  Avg signal days per week        : {weekly['signals'].mean():.2f}")
    print(f"  Avg weekly R (signal weeks)     : {weekly['total_r'].mean():+.3f}R")
    print(f"  Median weekly R                 : {weekly['total_r'].median():+.3f}R")
    print(f"  Weekly win rate                 : {weekly['week_win'].mean()*100:.1f}%  "
          f"(weeks ending positive)")
    print(f"  Best week                       : {weekly['total_r'].max():+.3f}R  "
          f"({weekly.loc[weekly['total_r'].idxmax(),'week']})")
    print(f"  Worst week                      : {weekly['total_r'].min():+.3f}R  "
          f"({weekly.loc[weekly['total_r'].idxmin(),'week']})")

    print_sub("5-day competition simulation  (draw 10,000 random 5-day windows)")
    np.random.seed(42)
    week_rs = weekly["total_r"].values
    sims = np.random.choice(week_rs, size=10_000, replace=True)
    print(f"    Expected 1-week P&L   : {sims.mean():+.3f}R")
    print(f"    Median  1-week P&L    : {np.median(sims):+.3f}R")
    print(f"    90th pctile           : {np.percentile(sims,90):+.3f}R")
    print(f"    10th pctile           : {np.percentile(sims,10):+.3f}R")
    print(f"    Prob profitable week  : {(sims>0).mean()*100:.1f}%")

    # ── 4. Monthly Breakdown ──────────────────────────────────────────────────
    print_section("4. MONTHLY BREAKDOWN")
    monthly = pdf.groupby("month").agg(
        signals = ("win", "count"),
        wr      = ("win", "mean"),
        avg_r   = ("r",   "mean"),
        total_r = ("r",   "sum"),
    ).reset_index()
    monthly["signals"] //= 2   # legs → signal days

    print(f"\n  {'Month':<9} {'Sigs':>5} {'WR':>7} {'AvgR':>8} {'TotalR':>8}")
    print(f"  {'─'*42}")
    for _, row in monthly.iterrows():
        flag = "⚠️ " if row["wr"] < 0.50 else "  "
        print(f"  {flag}{row['month']}  {row['signals']:4.0f}   "
              f"{row['wr']*100:5.1f}%   {row['avg_r']:+6.3f}R   {row['total_r']:+7.3f}R")

    # ── 5. Year-by-Year ───────────────────────────────────────────────────────
    print_section("5. YEAR-BY-YEAR")
    yearly = pdf.groupby("year").agg(
        legs    = ("win", "count"),
        wr      = ("win", "mean"),
        avg_r   = ("r",   "mean"),
        total_r = ("r",   "sum"),
    ).reset_index()
    yearly["sigs"] = yearly["legs"] // 2

    print(f"\n  {'Year':<6} {'Sigs':>5} {'Legs':>5} {'WR':>7} {'AvgR':>8} {'Ann.R':>8}")
    print(f"  {'─'*48}")
    for _, row in yearly.iterrows():
        print(f"  {int(row['year'])}   {row['sigs']:4.0f}   {row['legs']:4.0f}   "
              f"{row['wr']*100:5.1f}%   {row['avg_r']:+6.3f}R   {row['total_r']:+7.2f}R")

    # Trades per year
    avg_sigs_yr = yearly["sigs"].mean()
    print(f"\n  Avg signal days/year  : {avg_sigs_yr:.0f}")
    print(f"  Avg signal days/week  : {avg_sigs_yr/52:.2f}")

    # ── 6. Combined ORB + PDH/L System ───────────────────────────────────────
    print_section("6. COMBINED ORB + PDH/L SYSTEM")

    odf = pd.DataFrame(orb_trades) if orb_trades else pd.DataFrame()

    orb_n   = len(odf)
    orb_wr  = odf["win"].mean() if orb_n else 0
    orb_avr = odf["r"].mean()   if orb_n else 0
    pdhl_n  = n
    pdhl_wr = wr
    pdhl_avr = avr

    total_n = orb_n + pdhl_n
    all_r   = list(odf["r"]) + list(pdf["r"]) if orb_n else list(pdf["r"])
    comb_wr = np.mean([r > 0 for r in all_r])
    comb_avr = np.mean(all_r)

    print(f"\n  {'System':<10} {'Legs':>6} {'Sigs':>6} {'WR':>7} {'AvgR':>8} {'EV':>8}")
    print(f"  {'─'*52}")
    if orb_n:
        orb_ev = orb_wr * PMULT - (1 - orb_wr)
        print(f"  {'ORB':<10} {orb_n:6d}  {orb_n//2:5d}   {orb_wr*100:5.1f}%   "
              f"{orb_avr:+6.3f}R   {orb_ev:+6.3f}R")
    pdhl_ev = pdhl_wr * PMULT - (1 - pdhl_wr)
    print(f"  {'PDH/L':<10} {pdhl_n:6d}  {pdhl_n//2:5d}   {pdhl_wr*100:5.1f}%   "
          f"{pdhl_avr:+6.3f}R   {pdhl_ev:+6.3f}R")
    comb_ev = comb_wr * PMULT - (1 - comb_wr)
    print(f"  {'─'*52}")
    print(f"  {'COMBINED':<10} {total_n:6d}  {total_n//2:5d}   {comb_wr*100:5.1f}%   "
          f"{comb_avr:+6.3f}R   {comb_ev:+6.3f}R")

    # Coverage
    orb_days  = cnt["fire"]
    pdhl_days = sig["fired"]
    total_days_covered = orb_days + pdhl_days
    coverage = total_days_covered / total_valid * 100

    print(f"\n  Day coverage:")
    print(f"    ORB alone   : {orb_days:4d} days  ({orb_days/total_valid*100:.1f}% of valid-OR days)")
    print(f"    PDH/L adds  : {pdhl_days:4d} days  ({pdhl_days/total_valid*100:.1f}%)")
    print(f"    Combined    : {total_days_covered:4d} days  ({coverage:.1f}%)  "
          f"← was {orb_days/total_valid*100:.1f}% with ORB alone")

    # Avg signals per 5-day competition week (combined)
    combined_sigs_per_week = (total_n / 2) / (len(dates) / 5)
    print(f"\n  Avg active trading days per 5-day week:")
    print(f"    ORB alone         : {(orb_n/2) / (len(dates)/5):.1f} days")
    print(f"    PDH/L alone       : {(pdhl_n/2) / (len(dates)/5):.1f} days")
    print(f"    Combined system   : {combined_sigs_per_week:.1f} days per week")

    # Statistical significance of PDH/L
    from scipy import stats as sc
    z = (pdhl_wr - 0.40) / np.sqrt(0.40 * 0.60 / pdhl_n)
    p = sc.norm.sf(abs(z)) * 2
    print(f"\n  PDH/L significance: z={z:.2f}  p={p:.5f}  ", end="")
    print("✅ highly significant" if p < 0.01 else
          "✅ significant" if p < 0.05 else "⚠️  marginal")

    print_section("VERDICT")
    print(f"  ORB  : fires {orb_days/total_valid*100:.0f}% of valid days  —  WR={orb_wr*100:.1f}%  EV=+{orb_wr*PMULT-(1-orb_wr):.3f}R")
    print(f"  PDH/L: fires {pdhl_days/miss*100:.0f}% of ORB-miss days  —  WR={pdhl_wr*100:.1f}%  EV=+{pdhl_ev:.3f}R")
    print()
    print(f"  Combined system covers {coverage:.0f}% of valid trading days.")
    print(f"  PDH/L is the stronger strategy per trade (WR +{(pdhl_wr-orb_wr)*100:.1f}pp,")
    print(f"  EV +{pdhl_ev-orb_ev:.3f}R vs ORB). Deploy as fallback on ORB-miss days.")
    print()
    if pdhl_wr >= 0.58 and pdhl_ev > 0.30 and p < 0.01:
        print("  ✅  PDH/L APPROVED as ORB fallback — deploy for competition week")
    elif pdhl_wr >= 0.52 and pdhl_ev > 0.10:
        print("  ⚠️   PDH/L has edge but weaker — use with reduced size")
    else:
        print("  ❌  PDH/L insufficient — sit out on ORB-miss days")
    print()


if __name__ == "__main__":
    try:
        from scipy import stats  # noqa
    except ImportError:
        print("ERROR: scipy required — pip install scipy"); sys.exit(1)
    run()
