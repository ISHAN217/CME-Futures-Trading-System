#!/usr/bin/env python3
"""
backtest_combined_final.py — Full combined system P&L from $25k

Combined system logic:
  Priority 1: Both ORB + PDH/L fire SAME direction → PDH/L entry (WR=72.5%)
  Priority 2: Conflict (opposite directions)         → SKIP both
  Priority 3: Only ORB fires                        → ORB entry
  Priority 4: Only PDH/L fires (ORB-miss day)       → PDH/L entry
  Priority 5: Neither fires                          → no trade

Two runs reported:
  BENCHMARK   — 1% risk, micro contracts ($5/pt ES, $2/pt NQ), 1.5R target
                (apples-to-apples vs original ORB-only $25k→$369k benchmark)
  COMPETITION — 5% risk, full contracts ($50/pt ES, $20/pt NQ), 3.0R target
                (June 8-12 competition configuration)
"""

import numpy as np
import pandas as pd
from datetime import time

DATA = ("/Users/ishanbhardwaj/cme_competition_system/output/mtf/"
        "ES_NQ_1min_aligned.parquet")

OR_START     = time(9, 30);  OR_END       = time(10, 0)
ORB_CUTOFF   = time(10, 30); WATCH_START  = time(10, 0)
ENTRY_CUTOFF = time(11, 0);  EOD_CLOSE    = time(15, 30)
RTH_START    = time(9, 30);  RTH_END      = time(16, 0)
MIN_ES = 8.0; MAX_ES = 60.0; MIN_NQ = 30.0; MAX_NQ = 150.0
SLIP   = 0.25   # 1 tick per side


def simulate_trade(day, entry_t, direction,
                   col_h, col_l, col_c,
                   entry_px, stop_px, tgt_px, pmult):
    """Returns (r_val, exit_reason) where r_val is P&L in units of 1R."""
    after = day[day["time_et"] > entry_t]
    for _, bar in after.iterrows():
        t = bar["time_et"]
        if t >= EOD_CLOSE:
            sign = 1 if direction == "LONG" else -1
            net  = sign * (bar[col_c] - entry_px) - SLIP * 2
            rng  = abs(entry_px - stop_px)
            return (net / rng) if rng else 0.0, "EOD"
        if direction == "LONG":
            if bar[col_l] <= stop_px:
                return -1.0 - SLIP * 2 / abs(entry_px - stop_px), "STOP"
            if bar[col_h] >= tgt_px:
                return pmult - SLIP / abs(entry_px - stop_px), "TGT"
        else:
            if bar[col_h] >= stop_px:
                return -1.0 - SLIP * 2 / abs(entry_px - stop_px), "STOP"
            if bar[col_l] <= tgt_px:
                return pmult - SLIP / abs(entry_px - stop_px), "TGT"
    return 0.0, "NONE"


def run_combined(df, pmult, risk_pct, es_pt, nq_pt, label, start_capital=25_000.0):
    """Run full combined system and return results."""
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}

    account  = start_capital
    trades   = []
    day_rets = {}

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue

        # Opening range
        or_b = day[(day["time_et"] >= OR_START) & (day["time_et"] < OR_END)]
        if len(or_b) == 0: continue
        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min()
        nq_H = or_b["NQ_high"].max(); nq_L = or_b["NQ_low"].min()
        es_R = es_H - es_L;           nq_R = nq_H - nq_L
        if not (MIN_ES <= es_R <= MAX_ES and MIN_NQ <= nq_R <= MAX_NQ): continue

        # ── ORB signal ────────────────────────────────────────────────────
        post = day[(day["time_et"] >= OR_END) & (day["time_et"] < ORB_CUTOFF)]
        es_b = nq_b = None; orb_row = None
        for _, r in post.iterrows():
            if es_b is None:
                if r["ES_high"] > es_H: es_b = "L"
                elif r["ES_low"] < es_L: es_b = "S"
            if nq_b is None:
                if r["NQ_high"] > nq_H: nq_b = "L"
                elif r["NQ_low"] < nq_L: nq_b = "S"
            if es_b and nq_b: orb_row = r; break

        orb_fired = bool(es_b and nq_b and es_b == nq_b)
        orb_dir   = ("LONG" if es_b == "L" else "SHORT") if orb_fired else None

        # ── PDH/L signal ──────────────────────────────────────────────────
        prev  = by_d.get(dates[i - 1])
        pdhl_fired = False; pdhl_dir = None; pdhl_row = None
        es_PDH = es_PDL = nq_PDH = nq_PDL = None

        if prev is not None:
            prth = prev[(prev["time_et"] >= RTH_START) & (prev["time_et"] <= RTH_END)]
            if len(prth):
                es_PDH = prth["ES_high"].max(); es_PDL = prth["ES_low"].min()
                nq_PDH = prth["NQ_high"].max(); nq_PDL = prth["NQ_low"].min()
                watch  = day[(day["time_et"] >= WATCH_START) &
                             (day["time_et"] < ENTRY_CUTOFF)]
                eu = ed = nu = nd = False

                for _, r in watch.iterrows():
                    if not eu and r["ES_high"] > es_PDH: eu = True
                    if not ed and r["ES_low"]  < es_PDL: ed = True
                    if not nu and r["NQ_high"] > nq_PDH: nu = True
                    if not nd and r["NQ_low"]  < nq_PDL: nd = True
                    if eu and nu and pdhl_dir is None:
                        pdhl_dir = "LONG";  pdhl_row = r; break
                    if ed and nd and pdhl_dir is None:
                        pdhl_dir = "SHORT"; pdhl_row = r; break
                    if (eu and nd) or (ed and nu):
                        pdhl_dir = "CONFLICT"; break

                pdhl_fired = (pdhl_dir not in (None, "CONFLICT"))

        # ── Priority logic ────────────────────────────────────────────────
        conflict = (orb_fired and pdhl_fired and orb_dir != pdhl_dir)

        if conflict:
            trade_type = "CONFLICT"   # skip
            entry_dir = None
        elif orb_fired and pdhl_fired and orb_dir == pdhl_dir:
            trade_type = "PDHL_CONFIRMED"   # both agree → PDH/L entry
            entry_dir  = pdhl_dir
        elif orb_fired:
            trade_type = "ORB_ONLY"
            entry_dir  = orb_dir
        elif pdhl_fired:
            trade_type = "PDHL_ONLY"
            entry_dir  = pdhl_dir
        else:
            trade_type = "NONE"
            entry_dir  = None

        if entry_dir is None:
            continue    # no trade today

        # ── Simulate the chosen entry ─────────────────────────────────────
        use_pdhl = trade_type in ("PDHL_CONFIRMED", "PDHL_ONLY")
        entry_row_used = pdhl_row if use_pdhl else orb_row
        entry_t = entry_row_used["time_et"]

        day_pnl = 0.0
        for sym, pdh, pdl, h_or, l_or, rng, col_h, col_l, col_c, pt_val in [
            ("ES", es_PDH, es_PDL, es_H, es_L, es_R,
             "ES_high", "ES_low", "ES_close", es_pt),
            ("NQ", nq_PDH, nq_PDL, nq_H, nq_L, nq_R,
             "NQ_high", "NQ_low", "NQ_close", nq_pt),
        ]:
            if use_pdhl and pdh is not None:
                ep = pdh if entry_dir == "LONG" else pdl
            else:
                ep = h_or if entry_dir == "LONG" else l_or

            sp  = ep - rng if entry_dir == "LONG" else ep + rng
            tp  = ep + rng * pmult if entry_dir == "LONG" else ep - rng * pmult
            contracts = max(1, round(account * risk_pct / (rng * pt_val)))

            r_val, exit_r = simulate_trade(day, entry_t, entry_dir,
                                           col_h, col_l, col_c,
                                           ep, sp, tp, pmult)
            pnl_pts  = r_val * rng
            pnl_usd  = pnl_pts * contracts * pt_val - 0.74 * contracts
            pnl_pct  = pnl_usd / account
            day_pnl += pnl_pct

            trades.append(dict(
                date=d, sym=sym, type=trade_type, dir=entry_dir,
                contracts=contracts, r=r_val, exit=exit_r,
                pnl_usd=pnl_usd, pnl_pct=pnl_pct, win=pnl_usd > 0,
            ))

        account *= (1 + day_pnl)
        if day_pnl != 0:
            day_rets[str(d)] = day_pnl

    # ── Stats ─────────────────────────────────────────────────────────────
    tdf = pd.DataFrame(trades)
    dr  = pd.Series(day_rets, dtype=float)
    dr.index = pd.to_datetime(dr.index)
    dr = dr.sort_index()

    if len(dr) > 1:
        idx = pd.bdate_range(dr.index.min(), dr.index.max())
        dr  = dr.reindex(idx, fill_value=0.0)

    n_years = (dr.index.max() - dr.index.min()).days / 365.25 if len(dr) > 1 else 1
    cum_ret = (1 + dr).prod() - 1
    ann_ret = (1 + cum_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe  = dr.mean() / dr.std(ddof=1) * np.sqrt(252) if dr.std() > 0 else 0
    equity  = (1 + dr).cumprod()
    peak    = equity.cummax()
    max_dd  = ((equity - peak) / peak).min() * 100

    return dict(
        label         = label,
        account_start = start_capital,
        account_end   = round(account, 0),
        cum_ret_pct   = round(cum_ret * 100, 1),
        ann_ret_pct   = round(ann_ret * 100, 1),
        sharpe        = round(sharpe, 3),
        max_dd        = round(max_dd, 1),
        total_trades  = len(tdf),
        signal_days   = len(tdf) // 2,
        win_rate      = round(float((tdf["win"]).mean()) * 100, 1),
        trades_per_yr = round(len(tdf) / n_years, 0),
        n_years       = round(n_years, 1),
        tdf           = tdf,
        dr            = dr,
    )


def print_results(r):
    print(f"\n{'═'*62}")
    print(f"  {r['label']}")
    print(f"{'═'*62}")
    print(f"  Period          : {r['dr'].index.min().date()} → "
          f"{r['dr'].index.max().date()}  ({r['n_years']:.1f} years)")
    print(f"  Starting capital: ${r['account_start']:,.0f}")
    print(f"  Final capital   : ${r['account_end']:,.0f}")
    print(f"  Total return    : {r['cum_ret_pct']:+.1f}%")
    print(f"  Annual return   : {r['ann_ret_pct']:+.1f}% / year")
    print(f"  Sharpe ratio    : {r['sharpe']:.3f}")
    print(f"  Max drawdown    : {r['max_dd']:.1f}%")
    print(f"  Signal days     : {r['signal_days']:,}  "
          f"({r['signal_days']/r['n_years']:.0f}/year)")
    print(f"  Total legs      : {r['total_trades']:,}  "
          f"({r['trades_per_yr']:.0f}/year)")
    print(f"  Win rate        : {r['win_rate']:.1f}%")

    tdf = r["tdf"]
    print(f"\n  By trade type:")
    for tt in ["ORB_ONLY", "PDHL_ONLY", "PDHL_CONFIRMED"]:
        sub = tdf[tdf["type"] == tt]
        if len(sub):
            wr  = (sub["win"]).mean() * 100
            avg = sub["r"].mean()
            print(f"    {tt:<18}  n={len(sub)//2:4d} days  "
                  f"WR={wr:.1f}%  AvgR={avg:+.3f}R")

    print(f"\n  By exit reason:")
    for ex in ["TGT", "STOP", "EOD"]:
        sub = tdf[tdf["exit"] == ex]
        if len(sub):
            pct = len(sub) / len(tdf) * 100
            print(f"    {ex:<6}  {len(sub):5d} legs  ({pct:.1f}%)")

    print(f"\n  Year-by-year:")
    tdf["year"] = pd.to_datetime(tdf["date"].astype(str)).dt.year
    for yr, sub in tdf.groupby("year"):
        sig = len(sub) // 2
        wr  = (sub["win"]).mean() * 100
        yr_pnl = sub.groupby("date")["pnl_pct"].sum().sum() * 100
        print(f"    {int(yr)}  {sig:3d} days  WR={wr:.1f}%  P&L={yr_pnl:+.1f}%")


def main():
    print("Loading data …")
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    df = df.sort_values("ts_et").reset_index(drop=True)
    print(f"  {len(df):,} bars   "
          f"{sorted(df['date_et'].unique())[0]} → "
          f"{sorted(df['date_et'].unique())[-1]}\n")

    # ── Run 1: Benchmark (apples-to-apples vs original ORB-only) ─────────────
    print("Running BENCHMARK (1% risk, micro, 1.5R) …")
    bench = run_combined(df,
                         pmult      = 1.5,
                         risk_pct   = 0.01,
                         es_pt      = 5.0,    # MES = $5/pt
                         nq_pt      = 2.0,    # MNQ = $2/pt
                         label      = "COMBINED SYSTEM — Benchmark "
                                      "(1% risk / micro / 1.5R)",
                         start_capital = 25_000)

    # ── Run 2: Competition (5% risk, full contracts, 3R) ─────────────────────
    print("Running COMPETITION (5% risk, full contracts, 3.0R) …")
    comp  = run_combined(df,
                         pmult      = 3.0,
                         risk_pct   = 0.05,
                         es_pt      = 50.0,   # ES = $50/pt
                         nq_pt      = 20.0,   # NQ = $20/pt
                         label      = "COMBINED SYSTEM — Competition "
                                      "(5% risk / full contracts / 3.0R)",
                         start_capital = 25_000)

    # ── Print both ────────────────────────────────────────────────────────────
    print_results(bench)
    print_results(comp)

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print("  COMPARISON SUMMARY")
    print(f"{'═'*62}")
    print(f"  {'Metric':<24} {'Benchmark':>14} {'Competition':>14}")
    print(f"  {'─'*54}")
    rows = [
        ("Starting capital",  f"$25,000",               f"$25,000"),
        ("Final capital",
         f"${bench['account_end']:,.0f}",
         f"${comp['account_end']:,.0f}"),
        ("Total return",
         f"{bench['cum_ret_pct']:+.1f}%",
         f"{comp['cum_ret_pct']:+.1f}%"),
        ("Annual return",
         f"{bench['ann_ret_pct']:+.1f}%/yr",
         f"{comp['ann_ret_pct']:+.1f}%/yr"),
        ("Sharpe",
         f"{bench['sharpe']:.3f}",
         f"{comp['sharpe']:.3f}"),
        ("Max drawdown",
         f"{bench['max_dd']:.1f}%",
         f"{comp['max_dd']:.1f}%"),
        ("Signal days",
         f"{bench['signal_days']:,}",
         f"{comp['signal_days']:,}"),
        ("Win rate",
         f"{bench['win_rate']:.1f}%",
         f"{comp['win_rate']:.1f}%"),
    ]
    for name, b, c in rows:
        print(f"  {name:<24} {b:>14} {c:>14}")

    # ── ORB-only baseline for comparison ─────────────────────────────────────
    print(f"\n  ORB-only benchmark (original):  $25k → $369k  "
          f"Ann=+37%  Sh=1.326")
    print(f"  Combined benchmark (this run):  "
          f"$25k → ${bench['account_end']:,.0f}  "
          f"Ann={bench['ann_ret_pct']:+.0f}%  "
          f"Sh={bench['sharpe']:.3f}")
    print()


if __name__ == "__main__":
    main()
