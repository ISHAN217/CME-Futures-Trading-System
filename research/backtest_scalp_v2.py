#!/usr/bin/env python3
"""
backtest_scalp_v2.py — Three improvements to the S&R zone scalp

Baseline problem: WR=74.7% but AvgPnL=-0.21pts (stops are 5.9x wins).
Root cause: entering anywhere in zone with stop at zone boundary = bad R:R.

IDEA 1 — Rejection Candle Entry
  Wait for a bar that tests the level AND closes back below it.
  Enter SHORT on the NEXT bar (after rejection confirmed).
  Stop: above rejection candle's high (tight — 1-3pts only).
  Target: tested at 3, 4, 5, 6pts.

IDEA 2 — Context Alignment Filter
  Only fade resistance (PDH, OR_H) when gap is DOWN (bearish day context).
  Only fade support (PDL, OR_L) when gap is UP (bullish day context).
  Rationale: breakouts happen most often against the gap bias.

IDEA 3 — Confluence Zones (Double Wall)
  Only trade when two structural levels coincide within N points.
  PDH + round number within 3pts = double friction zone.
  Expected: fewer trades, tighter stops, higher success rate.

COMBINED — All three together.

Each tested with IS/OOS split and comparison to baseline.
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25
ZONE_WIDTH = 5
N_TIME = 10   # max bars to hold

RTH_START = time(9,30); RTH_END = time(16,0)
OR_START  = time(9,30); OR_END  = time(10,0)
SCALP_START = time(10,30); SCALP_END = time(15,0)


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def weekly_hl(by_d, dates):
    w = {}
    for d in dates:
        ts=pd.Timestamp(d); iso=ts.isocalendar()
        key=(int(iso.year),int(iso.week))
        day=by_d.get(d)
        if day is None: continue
        rth=day[(day["time_et"]>=RTH_START)&(day["time_et"]<=RTH_END)]
        if not len(rth): continue
        h=rth["ES_high"].max(); l=rth["ES_low"].min()
        if key not in w: w[key]=[h,l]
        else: w[key][0]=max(w[key][0],h); w[key][1]=min(w[key][1],l)
    return w


def get_pwhl(d, w):
    ts=pd.Timestamp(d); iso=ts.isocalendar()
    pw=int(iso.week)-1; yr=int(iso.year)
    if pw==0: yr-=1; pw=int(pd.Timestamp(f"{yr}-12-28").isocalendar().week)
    key=(yr,pw)
    return (w[key][0],w[key][1]) if key in w else (None,None)


def simulate_trade(fwd_opens, fwd_closes, direction, ep, stop_px, tgt_px):
    """Simulate from bars after entry. Enter at fwd_opens[0]."""
    sign = 1 if direction == "LONG" else -1
    for i in range(len(fwd_closes)):
        if direction == "SHORT":
            if fwd_closes[i] >= stop_px: return -(stop_px - ep) - SLIP*2, "STOP"
            if fwd_closes[i] <= tgt_px:  return (ep - tgt_px) - SLIP*2,   "TARGET"
        else:
            if fwd_closes[i] <= stop_px: return -(ep - stop_px) - SLIP*2, "STOP"
            if fwd_closes[i] >= tgt_px:  return (tgt_px - ep) - SLIP*2,   "TARGET"
    # Time stop: exit at last close
    last = fwd_closes[-1] if len(fwd_closes) else ep
    return sign * (last - ep) - SLIP*2, "TIME"


def run_scalps(df, dates, by_d, wkly,
               idea1_rejection=False,  rejection_thresh=1.0,
               idea2_context=False,    gap_thresh=0.10,
               idea3_confluence=False, confluence_dist=3.0,
               target_pts=5.0):
    """
    Run the scalp system with selected improvements.
    Returns DataFrame of trades.
    """
    trades = []

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"]>=RTH_START)&(day["time_et"]<=RTH_END)].reset_index(drop=True)
        if len(rth) < 20: continue

        opens  = rth["ES_open"].values
        closes = rth["ES_close"].values
        highs  = rth["ES_high"].values
        lows   = rth["ES_low"].values
        times  = rth["time_et"].values

        # --- Build levels ---
        or_b = rth[(rth["time_et"]>=OR_START)&(rth["time_et"]<OR_END)]
        if not len(or_b): continue
        OR_H = or_b["ES_high"].max(); OR_L = or_b["ES_low"].min()

        prev = by_d.get(dates[i-1])
        if prev is None: continue
        prev_rth = prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if not len(prev_rth): continue
        PDH = prev_rth["ES_high"].max(); PDL = prev_rth["ES_low"].min()
        prev_close = prev_rth["ES_close"].iloc[-1]

        PWH, PWL = get_pwhl(d, wkly)

        # Today's open
        today_open = opens[0] if len(opens) else None
        if today_open is None: continue

        # Round numbers near today's range
        day_mid = (rth["ES_high"].max() + rth["ES_low"].min()) / 2
        round_lvls = [round(day_mid/50)*50 + k*50 for k in range(-2,3)]

        # --- IDEA 2: Gap context ---
        gap_pct = (today_open - prev_close) / prev_close * 100
        if idea2_context:
            bearish_context = gap_pct <= -gap_thresh
            bullish_context = gap_pct >= gap_thresh
            neutral_context = not bearish_context and not bullish_context
        else:
            bearish_context = bullish_context = neutral_context = True

        # --- IDEA 3: Confluence check helper ---
        def has_confluence(level, level_type):
            if not idea3_confluence: return True
            all_lvls = []
            if PDH: all_lvls.append(("PDH", PDH))
            if PDL: all_lvls.append(("PDL", PDL))
            if PWH: all_lvls.append(("PWH", PWH))
            if PWL: all_lvls.append(("PWL", PWL))
            all_lvls.append(("OR_H", OR_H))
            all_lvls.append(("OR_L", OR_L))
            for rl in round_lvls: all_lvls.append(("ROUND", float(rl)))
            # Check if any OTHER level is within confluence_dist
            for other_type, other_lvl in all_lvls:
                if other_type == level_type: continue
                if abs(other_lvl - level) <= confluence_dist:
                    return True
            return False

        # --- Levels to trade ---
        # resistance (fade SHORT), support (fade LONG)
        resist_levels = []
        support_levels = []
        if bearish_context or neutral_context:
            resist_levels += [("PDH", PDH), ("OR_H", OR_H)]
            if PWH: resist_levels.append(("PWH", PWH))
            for rl in round_lvls: resist_levels.append(("ROUND", float(rl)))
        if bullish_context or neutral_context:
            support_levels += [("PDL", PDL), ("OR_L", OR_L)]
            if PWL: support_levels.append(("PWL", PWL))
            for rl in round_lvls: support_levels.append(("ROUND", float(rl)))

        # --- Scan for scalp setups ---
        def scan_level(level_type, level, fade_dir):
            """
            fade_dir: 'SHORT' (resistance) or 'LONG' (support)
            Returns one trade record or None.
            """
            for j in range(1, len(closes) - N_TIME - 2):
                if times[j] < SCALP_START: continue
                if times[j] >= SCALP_END: break

                if fade_dir == "SHORT":
                    if closes[j-1] >= level - ZONE_WIDTH: continue  # must approach from below
                    in_zone = (closes[j] >= level - ZONE_WIDTH) and (closes[j] <= level + ZONE_WIDTH)
                    if not in_zone: continue

                    if idea1_rejection:
                        # Require rejection candle: high > level AND close < level - thresh
                        if not (highs[j] >= level and closes[j] <= level - rejection_thresh):
                            continue
                        # Entry on NEXT bar
                        entry_j = j + 1
                        if entry_j >= len(closes) - N_TIME: break
                        ep = opens[entry_j] + SLIP   # short at next bar open
                        stop_px = highs[j] + SLIP*2  # stop above rejection wick
                    else:
                        # Baseline: enter at close of zone-entry bar
                        ep = closes[j] + SLIP
                        stop_px = level + ZONE_WIDTH + SLIP
                        entry_j = j

                    tgt_px = ep - target_pts
                    fwd_closes = closes[entry_j+1:entry_j+1+N_TIME]
                    fwd_opens  = opens[entry_j+1:entry_j+1+N_TIME]
                    if len(fwd_closes) < 2: break

                    pnl, outcome = simulate_trade(
                        fwd_opens, fwd_closes, "SHORT", ep, stop_px, tgt_px)

                    stop_dist = stop_px - ep
                    return dict(
                        date=d, year=d.year,
                        level_type=level_type, fade_dir=fade_dir,
                        level=round(level,2), entry=round(ep,2),
                        stop=round(stop_px,2), target=round(tgt_px,2),
                        stop_dist=round(stop_dist,2),
                        pnl=round(pnl,2), win=pnl>0, outcome=outcome,
                        gap_pct=round(gap_pct,3),
                        dow=pd.Timestamp(d).day_name(),
                    )

                else:  # LONG (support)
                    if closes[j-1] <= level + ZONE_WIDTH: continue
                    in_zone = (closes[j] >= level - ZONE_WIDTH) and (closes[j] <= level + ZONE_WIDTH)
                    if not in_zone: continue

                    if idea1_rejection:
                        if not (lows[j] <= level and closes[j] >= level + rejection_thresh):
                            continue
                        entry_j = j + 1
                        if entry_j >= len(closes) - N_TIME: break
                        ep = opens[entry_j] - SLIP
                        stop_px = lows[j] - SLIP*2
                    else:
                        ep = closes[j] - SLIP
                        stop_px = level - ZONE_WIDTH - SLIP
                        entry_j = j

                    tgt_px = ep + target_pts
                    fwd_closes = closes[entry_j+1:entry_j+1+N_TIME]
                    fwd_opens  = opens[entry_j+1:entry_j+1+N_TIME]
                    if len(fwd_closes) < 2: break

                    pnl, outcome = simulate_trade(
                        fwd_opens, fwd_closes, "LONG", ep, stop_px, tgt_px)

                    stop_dist = ep - stop_px
                    return dict(
                        date=d, year=d.year,
                        level_type=level_type, fade_dir=fade_dir,
                        level=round(level,2), entry=round(ep,2),
                        stop=round(stop_px,2), target=round(tgt_px,2),
                        stop_dist=round(stop_dist,2),
                        pnl=round(pnl,2), win=pnl>0, outcome=outcome,
                        gap_pct=round(gap_pct,3),
                        dow=pd.Timestamp(d).day_name(),
                    )
            return None

        # One trade per level per day
        for lt, lv in resist_levels:
            if not has_confluence(lv, lt): continue
            rec = scan_level(lt, lv, "SHORT")
            if rec: trades.append(rec)

        for lt, lv in support_levels:
            if not has_confluence(lv, lt): continue
            rec = scan_level(lt, lv, "LONG")
            if rec: trades.append(rec)

    return pd.DataFrame(trades)


def prow(label, sub, pad=44):
    if not len(sub): print(f"  {label:<{pad}}  n=0"); return
    wr=sub["win"].mean(); ar=sub["pnl"].mean()
    dr=sub.groupby("date")["pnl"].sum()
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    n_yr=sub["date"].nunique()/8.1
    flag="✅" if (ar>0 and sh>1.0) else ("🟡" if ar>0 else "❌")
    print(f"  {flag} {label:<{pad}}  n={sub['date'].nunique():4d} ({n_yr:.0f}/yr)  "
          f"WR={wr*100:5.1f}%  AvgPnL={ar:+.3f}pts  Sh={sh:.2f}")


def main():
    print("Loading …")
    df = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    wkly  = weekly_hl(by_d, dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    IS_END=date(2021,12,31); OS_START=date(2022,1,1)
    W=76
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── BASELINE (no improvements) ────────────────────────────────────────────
    sec("BASELINE  —  Original scalp (zone entry, zone stop)")
    print("  Enters anywhere in zone, stops at far zone boundary")
    baseline = run_scalps(df, dates, by_d, wkly, target_pts=5.0)
    prow("Baseline (no improvements)", baseline)
    print(f"  Reference: AvgPnL=-0.21pts  Sh=-1.71  (from previous backtest)\n")

    # ── IDEA 1: Rejection candle — sweep of target distances ─────────────────
    sec("IDEA 1: REJECTION CANDLE ENTRY  —  Target distance sensitivity")
    print("  Entry only after bar that tested level AND closed back below/above")
    print("  Stop: above rejection bar wick (tight)\n")
    idea1_results = {}
    for tgt in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        tdf = run_scalps(df, dates, by_d, wkly,
                         idea1_rejection=True, rejection_thresh=1.0,
                         target_pts=tgt)
        idea1_results[tgt] = tdf
        prow(f"  Rejection candle  target={tgt}pt", tdf)

    # Find best
    best_tgt = max(idea1_results, key=lambda k: (
        idea1_results[k].groupby("date")["pnl"].sum().mean() /
        idea1_results[k].groupby("date")["pnl"].sum().std(ddof=1)
        if len(idea1_results[k])>1 and
        idea1_results[k].groupby("date")["pnl"].sum().std()>0 else -999
    ))
    print(f"\n  Best target: {best_tgt}pt")
    best1 = idea1_results[best_tgt]

    # Stop distribution for rejection candle
    if len(best1):
        print(f"\n  Stop distance distribution (Idea 1, {best_tgt}pt target):")
        print(f"    Mean stop dist : {best1['stop_dist'].mean():.2f}pts")
        print(f"    Median stop    : {best1['stop_dist'].median():.2f}pts")
        print(f"    Max stop       : {best1['stop_dist'].max():.2f}pts")
        print(f"  vs Baseline stop: ~7.42pts avg")
        print()
        for out in ["TARGET","STOP","TIME"]:
            s=best1[best1["outcome"]==out]
            if len(s):
                print(f"    {out:<8}  {len(s):4d} ({len(s)/len(best1)*100:.1f}%)  "
                      f"WR={(s['pnl']>0).mean()*100:.1f}%  AvgPnL={s['pnl'].mean():+.2f}pts")

    # ── IDEA 2: Context alignment ─────────────────────────────────────────────
    sec("IDEA 2: CONTEXT ALIGNMENT FILTER  —  Gap direction bias")
    print("  Only fade resistance on down-gap days, support on up-gap days\n")
    for gap_t in [0.05, 0.10, 0.15, 0.20, 0.30]:
        tdf = run_scalps(df, dates, by_d, wkly,
                         idea2_context=True, gap_thresh=gap_t,
                         target_pts=5.0)
        prow(f"  Context filter  gap≥{gap_t:.2f}%", tdf)

    print()
    print("  Gap context + best rejection (combined ideas 1+2):")
    best2_combined = run_scalps(df, dates, by_d, wkly,
                                idea1_rejection=True,  rejection_thresh=1.0,
                                idea2_context=True,    gap_thresh=0.10,
                                target_pts=best_tgt)
    prow(f"  Rejection + context gap≥0.10%", best2_combined)

    # ── IDEA 3: Confluence zones ──────────────────────────────────────────────
    sec("IDEA 3: CONFLUENCE ZONES  —  Two levels within N points")
    print("  Only trade when another structural level is nearby\n")
    for dist in [2.0, 3.0, 5.0, 8.0]:
        tdf = run_scalps(df, dates, by_d, wkly,
                         idea3_confluence=True, confluence_dist=dist,
                         target_pts=5.0)
        prow(f"  Confluence  dist≤{dist}pt", tdf)

    print()
    print("  All three ideas combined:")
    all3 = run_scalps(df, dates, by_d, wkly,
                      idea1_rejection=True,  rejection_thresh=1.0,
                      idea2_context=True,    gap_thresh=0.10,
                      idea3_confluence=True, confluence_dist=5.0,
                      target_pts=best_tgt)
    prow(f"  All 3 combined (best target={best_tgt}pt)", all3)

    # ── YEAR-BY-YEAR on best config ───────────────────────────────────────────
    # Find the overall best
    best_overall = max(
        [("Idea 1", best1),
         ("Idea 1+2", best2_combined),
         ("All 3", all3)],
        key=lambda x: (
            x[1].groupby("date")["pnl"].sum().mean() /
            x[1].groupby("date")["pnl"].sum().std(ddof=1)
            if len(x[1])>1 and x[1].groupby("date")["pnl"].sum().std()>0 else -999
        )
    )
    best_name, best_df = best_overall

    sec(f"YEAR-BY-YEAR  —  Best config: {best_name}")
    print(f"  {'Year':<6}  {'n':>4}  {'WR':>6}  {'AvgPnL':>8}  {'Sharpe':>7}  IS/OOS")
    print(f"  {'─'*48}")
    for yr, sub in best_df.groupby("year"):
        wr=(sub["pnl"]>0).mean()*100; ar=sub["pnl"].mean()
        dr=sub.groupby("date")["pnl"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        lbl="← IS" if yr<=2021 else "← OOS"
        flag="✅" if ar>0 else "❌"
        print(f"  {flag} {yr:<6}  {len(sub):>4}  {wr:>5.1f}%  {ar:>+7.3f}pts  "
              f"{sh:>7.2f}  {lbl}")

    # ── WALK-FORWARD ──────────────────────────────────────────────────────────
    sec("WALK-FORWARD  IS 2018-2021  →  OOS 2022-2026")
    print("  Baseline vs All ideas:")
    print()
    prow("Baseline IS",       baseline[baseline["date"]<=IS_END])
    prow("Baseline OOS",      baseline[baseline["date"]>=OS_START])
    print()
    if len(best_df):
        prow(f"{best_name} IS",  best_df[best_df["date"]<=IS_END])
        prow(f"{best_name} OOS", best_df[best_df["date"]>=OS_START])

    # ── FINAL COMPARISON ──────────────────────────────────────────────────────
    sec("FINAL COMPARISON  —  All systems")
    print(f"  {'System':<46}  {'n/yr':>5}  {'WR':>6}  {'AvgPnL':>8}  {'Sharpe':>7}")
    print(f"  {'─'*72}")

    systems = [
        ("Baseline (zone entry, zone stop, 5pt tgt)", baseline),
        (f"Idea 1: Rejection candle ({best_tgt}pt tgt)",         best1),
        (f"Idea 1+2: +Context align",                             best2_combined),
        (f"All 3: +Confluence",                                   all3),
    ]
    for label, tdf in systems:
        if not len(tdf): print(f"  ❌ {label:<46}  n=0"); continue
        wr=(tdf["pnl"]>0).mean(); ar=tdf["pnl"].mean()
        dr=tdf.groupby("date")["pnl"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        n_yr=tdf["date"].nunique()/8.1
        flag="✅" if (ar>0 and sh>1) else ("🟡" if ar>0 else "❌")
        print(f"  {flag} {label:<46}  {n_yr:>5.0f}/yr  {wr*100:>5.1f}%  "
              f"{ar:>+7.3f}pts  {sh:>7.2f}")

    print()
    print("  Reference — verified primary system:")
    print("  ORB baseline:      22/yr  WR=47.8%  Sharpe=3.08")
    print("  PDH/L Confirmed:   36/yr  WR=54.5%  Sharpe=3.14")
    print()

    # ── HONEST VERDICT ────────────────────────────────────────────────────────
    sec("HONEST VERDICT")
    if len(best_df):
        ar=best_df["pnl"].mean()
        dr=best_df.groupby("date")["pnl"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
        n_yr=best_df["date"].nunique()/8.1
        print(f"  Best config ({best_name}):")
        print(f"    AvgPnL/trade : {ar:+.3f} pts")
        print(f"    Sharpe       : {sh:.2f}")
        print(f"    Signal days  : {n_yr:.0f}/yr")
        print()
        if ar > 0 and sh > 1.5:
            print("  ✅ Improvements WORK — scalp is now positive EV and tradeable")
        elif ar > 0 and sh > 0.5:
            print("  🟡 Improvements help but edge is thin — implement at very small size")
        elif ar > 0:
            print("  ⚠️  Marginally positive but Sharpe too low to trade confidently")
        else:
            print("  ❌ Improvements insufficient — scalp concept needs rethinking")
    print()


if __name__=="__main__":
    main()
