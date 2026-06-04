#!/usr/bin/env python3
"""
backtest_sr_zones.py — Supply & Resistance Zone Respect Test

Tests how often ES/NQ respects structural S&R ZONES vs breaks through.
Zone = ±Z points around a level. Compared against a random baseline.

Definition:
  Approach = price enters zone from outside (first bar)
  Respect  = within 10 bars, price closes back OUTSIDE zone same side
  Break    = within 10 bars, price closes BEYOND zone far side
  Inconclusive = neither within 10 bars

S&R levels tested:
  PDH/PDL  — Prior Day High/Low
  PWH/PWL  — Prior Week High/Low
  OR_H/L   — Opening Range High/Low (current day)
  ROUND    — Nearest 50-point psychological levels

Zone widths tested: 2, 5, 10, 20 points
Compared vs random baseline throughout.
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
RTH_START = time(9,30); RTH_END = time(16,0)
OR_START  = time(9,30); OR_END  = time(10,0)
ZONE_WIDTHS = [2, 5, 10, 20]
N_BARS = 10


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


def classify_zone_touch(closes, highs, lows, approach_dir, zone_lo, zone_hi):
    """
    Vectorised: given arrays of closes/highs/lows for the N bars after entry,
    return first outcome ('RESPECT', 'BREAK', 'INCONCLUSIVE').
    """
    for i in range(len(closes)):
        if approach_dir == 'below':
            if closes[i] > zone_hi: return 'BREAK'
            if closes[i] < zone_lo: return 'RESPECT'
        else:
            if closes[i] < zone_lo: return 'BREAK'
            if closes[i] > zone_hi: return 'RESPECT'
    return 'INCONCLUSIVE'


def run_sr_test(df, dates, by_d, wkly):
    events = []

    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"]>=RTH_START)&(day["time_et"]<=RTH_END)].reset_index(drop=True)
        if len(rth) < N_BARS + 5: continue

        closes = rth["ES_close"].values
        highs  = rth["ES_high"].values
        lows   = rth["ES_low"].values
        times  = rth["time_et"].values

        # Compute levels
        or_b = rth[(rth["time_et"]>=OR_START)&(rth["time_et"]<OR_END)]
        if not len(or_b): continue
        OR_H = or_b["ES_high"].max(); OR_L = or_b["ES_low"].min()

        prev = by_d.get(dates[i-1])
        if prev is None: continue
        prev_rth = prev[(prev["time_et"]>=RTH_START)&(prev["time_et"]<=RTH_END)]
        if not len(prev_rth): continue
        PDH = prev_rth["ES_high"].max(); PDL = prev_rth["ES_low"].min()

        PWH, PWL = get_pwhl(d, wkly)

        # Round numbers near today's range
        day_mid = (rth["ES_high"].max() + rth["ES_low"].min()) / 2
        round_base = round(day_mid / 50) * 50
        round_levels = [round_base + k*50 for k in range(-2, 3)]

        # All levels
        all_levels = [
            ("PDH", PDH), ("PDL", PDL),
            ("OR_H", OR_H), ("OR_L", OR_L),
        ]
        if PWH: all_levels += [("PWH", PWH), ("PWL", PWL)]
        for rl in round_levels:
            all_levels.append(("ROUND", float(rl)))

        # For each level and zone width, find FIRST zone touch
        for level_type, level in all_levels:
            if level is None: continue

            for zw in ZONE_WIDTHS:
                zlo = level - zw; zhi = level + zw
                in_zone = (closes >= zlo) & (closes <= zhi)

                # Find first transition: outside → inside
                first_touch_idx = None
                for j in range(1, len(closes) - N_BARS - 1):
                    if in_zone[j] and not in_zone[j-1]:
                        # Determine approach direction
                        if closes[j-1] < zlo:
                            approach_dir = 'below'
                        elif closes[j-1] > zhi:
                            approach_dir = 'above'
                        else:
                            continue
                        first_touch_idx = j
                        break

                if first_touch_idx is None: continue
                j = first_touch_idx

                # Check outcome over next N_BARS
                fwd_closes = closes[j+1:j+1+N_BARS]
                fwd_highs  = highs[j+1:j+1+N_BARS]
                fwd_lows   = lows[j+1:j+1+N_BARS]
                if len(fwd_closes) < 3: continue

                outcome = classify_zone_touch(
                    fwd_closes, fwd_highs, fwd_lows,
                    approach_dir if approach_dir == 'below' else 'above',
                    zlo, zhi)

                t = times[j]
                t_mins = t.hour*60 + t.minute - 9*60 - 30   # minutes from open
                events.append(dict(
                    date=d, year=d.year,
                    level_type=level_type,
                    zone_width=zw,
                    approach_dir=approach_dir,
                    outcome=outcome,
                    session_mins=t_mins,
                    session_pct=round(t_mins/(6.5*60)*100, 1),
                ))

    return pd.DataFrame(events)


def random_baseline(df, dates, by_d, n_sample=8000):
    """Random reference: what's the base respect rate at arbitrary points?"""
    np.random.seed(42)
    recs = []
    all_days = []
    for d in dates:
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"]>=RTH_START)&(day["time_et"]<time(15,0))].reset_index(drop=True)
        if len(rth) < N_BARS + 5: continue
        all_days.append(rth)

    for _ in range(n_sample):
        rth = all_days[np.random.randint(len(all_days))]
        closes = rth["ES_close"].values
        j = np.random.randint(2, len(closes)-N_BARS-2)
        for zw in ZONE_WIDTHS:
            level = closes[j]
            if abs(closes[j-1] - level) < zw: continue
            approach_dir = 'below' if closes[j-1] < level - zw else 'above'
            zlo = level - zw; zhi = level + zw
            fwd = closes[j+1:j+1+N_BARS]
            if len(fwd) < 3: continue
            outcome = classify_zone_touch(fwd, fwd, fwd, approach_dir, zlo, zhi)
            recs.append(dict(zone_width=zw, outcome=outcome))
    return pd.DataFrame(recs)


def main():
    print("Loading …")
    df = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    wkly  = weekly_hl(by_d, dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    print("Running S&R zone test …")
    ev = run_sr_test(df, dates, by_d, wkly)
    print(f"  {len(ev):,} zone approach events\n")

    print("Computing random baseline …")
    bl = random_baseline(df, dates, by_d, n_sample=8000)

    # Baseline respect rates per zone width
    bl_rates = {zw: (bl[bl["zone_width"]==zw]["outcome"]=="RESPECT").mean()*100
                for zw in ZONE_WIDTHS}

    W = 74
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── 1. Main table ──────────────────────────────────────────────────────────
    sec("1. RESPECT RATE BY LEVEL TYPE AND ZONE WIDTH")
    print("  ✅ = respect rate ≥5pp above random baseline")
    print("  🟡 = 2-5pp above random")
    print("  ❌ = <2pp above random (no meaningful edge)\n")
    print(f"  {'Level':<8}  {'ZW':>4}  {'n':>5}  {'Respect%':>9}  "
          f"{'Break%':>7}  {'Rand%':>7}  {'Alpha':>7}")
    print(f"  {'─'*60}")

    level_order = ["PDH","PDL","PWH","PWL","OR_H","OR_L","ROUND"]
    for lt in level_order:
        sub = ev[ev["level_type"]==lt]
        if not len(sub): continue
        for zw in ZONE_WIDTHS:
            s = sub[sub["zone_width"]==zw]
            if len(s) < 30: continue
            r = (s["outcome"]=="RESPECT").mean()*100
            b = (s["outcome"]=="BREAK").mean()*100
            rand = bl_rates.get(zw, 50)
            alpha = r - rand
            flag = "✅" if alpha>=5 else ("🟡" if alpha>=2 else "❌")
            print(f"  {flag} {lt:<8}  {zw:>3}pt  {len(s):>5}  "
                  f"{r:>8.1f}%  {b:>6.1f}%  {rand:>6.1f}%  {alpha:>+6.1f}pp")
        print()

    # ── 2. Random baseline ────────────────────────────────────────────────────
    sec("2. RANDOM BASELINE  —  what % respect at any random price point?")
    print()
    for zw in ZONE_WIDTHS:
        b = bl[bl["zone_width"]==zw]
        r=(b["outcome"]=="RESPECT").mean()*100
        br=(b["outcome"]=="BREAK").mean()*100
        inc=(b["outcome"]=="INCONCLUSIVE").mean()*100
        print(f"  Zone ±{zw:2d}pt   n={len(b):5d}   "
              f"Respect={r:.1f}%   Break={br:.1f}%   "
              f"Inconclusive={inc:.1f}%")

    # ── 3. Best level+zone combination ───────────────────────────────────────
    sec("3. BEST LEVEL+ZONE COMBINATION  —  sorted by alpha over random")
    combos = []
    for lt in level_order:
        sub = ev[ev["level_type"]==lt]
        for zw in ZONE_WIDTHS:
            s = sub[sub["zone_width"]==zw]
            if len(s) < 30: continue
            r = (s["outcome"]=="RESPECT").mean()*100
            rand = bl_rates.get(zw, 50)
            combos.append((lt, zw, r, r-rand, len(s)))
    combos.sort(key=lambda x: -x[3])
    print(f"\n  {'Level':<8}  {'ZW':>4}  {'Respect%':>9}  {'Alpha':>8}  {'n':>5}")
    print(f"  {'─'*42}")
    for lt, zw, r, alpha, n in combos[:15]:
        flag="✅" if alpha>=5 else ("🟡" if alpha>=2 else "❌")
        print(f"  {flag} {lt:<8}  {zw:>3}pt  {r:>8.1f}%  {alpha:>+6.1f}pp  {n:>5}")

    # ── 4. First touch vs repeated touches ────────────────────────────────────
    sec("4. FIRST TOUCH vs REPEATED TOUCHES  —  does freshness matter?")
    key_levels = ev[ev["level_type"].isin(["PDH","PDL","PWH","PWL"])&
                    (ev["zone_width"]==5)].copy()
    key_levels = key_levels.sort_values(["date","level_type","session_mins"])
    key_levels["touch_num"] = key_levels.groupby(["date","level_type"]).cumcount()+1
    rand5 = bl_rates.get(5, 50)
    for tn in [1, 2, 3]:
        s = key_levels[key_levels["touch_num"]==tn]
        if not len(s): continue
        r=(s["outcome"]=="RESPECT").mean()*100
        b=(s["outcome"]=="BREAK").mean()*100
        flag="✅" if r-rand5>=5 else ("🟡" if r-rand5>=2 else "❌")
        print(f"  {flag} Touch #{tn}   n={len(s):4d}   "
              f"Respect={r:.1f}%   Break={b:.1f}%   "
              f"alpha={r-rand5:+.1f}pp")

    # ── 5. Time of day ─────────────────────────────────────────────────────────
    sec("5. TIME OF DAY  —  which session period shows most S&R respect?")
    key_levels5 = ev[ev["level_type"].isin(["PDH","PDL","PWH","PWL"])&
                     (ev["zone_width"]==5)]
    buckets = [
        ("9:30–10:00 (opening)",   0,  30),
        ("10:00–11:00 (post-OR)",  30,  90),
        ("11:00–13:00 (mid AM)",   90, 210),
        ("13:00–14:30 (lunch)",   210, 300),
        ("14:30–16:00 (close)",   300, 999),
    ]
    rand5 = bl_rates.get(5, 50)
    for label, lo, hi in buckets:
        s = key_levels5[(key_levels5["session_mins"]>=lo)&(key_levels5["session_mins"]<hi)]
        if not len(s): continue
        r=(s["outcome"]=="RESPECT").mean()*100
        b=(s["outcome"]=="BREAK").mean()*100
        flag="✅" if r-rand5>=5 else ("🟡" if r-rand5>=2 else "❌")
        print(f"  {flag} {label:<30}  n={len(s):4d}  "
              f"Respect={r:.1f}%  Break={b:.1f}%  "
              f"alpha={r-rand5:+.1f}pp")

    # ── 6. Approach direction ─────────────────────────────────────────────────
    sec("6. APPROACH DIRECTION  —  approaching from below vs above")
    for lt in ["PDH","PDL","PWH","PWL"]:
        sub = ev[(ev["level_type"]==lt)&(ev["zone_width"]==5)]
        if not len(sub): continue
        rand5 = bl_rates.get(5, 50)
        for direction in ["below","above"]:
            s = sub[sub["approach_dir"]==direction]
            if not len(s): continue
            r=(s["outcome"]=="RESPECT").mean()*100
            flag="✅" if r-rand5>=5 else ("🟡" if r-rand5>=2 else "")
            print(f"  {flag} {lt:<5} from {direction:<6}  n={len(s):4d}  "
                  f"Respect={r:.1f}%  alpha={r-rand5:+.1f}pp")
        print()

    # ── 7. Honest summary ─────────────────────────────────────────────────────
    sec("7. HONEST SUMMARY")
    print(f"  Random baseline (arbitrary price): "
          f"~{np.mean(list(bl_rates.values())):.0f}% respect at any point\n")
    print("  Level      Best Zone  Respect%   Alpha    Verdict")
    print(f"  {'─'*54}")
    for lt in level_order:
        sub = ev[ev["level_type"]==lt]
        if not len(sub): continue
        best_alpha=-999; best_zw=None; best_r=0
        for zw in ZONE_WIDTHS:
            s=sub[sub["zone_width"]==zw]
            if len(s)<30: continue
            r=(s["outcome"]=="RESPECT").mean()*100
            rand=bl_rates.get(zw,50)
            if r-rand>best_alpha:
                best_alpha=r-rand; best_zw=zw; best_r=r
        if best_zw is None: continue
        flag="✅ REAL EDGE" if best_alpha>=5 else ("🟡 MARGINAL" if best_alpha>=2 else "❌ NO EDGE")
        print(f"  {lt:<10}  {best_zw:>4}pt      {best_r:>5.1f}%  "
              f"{best_alpha:>+5.1f}pp   {flag}")
    print()


if __name__=="__main__":
    main()
