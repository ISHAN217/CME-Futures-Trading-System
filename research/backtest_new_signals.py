#!/usr/bin/env python3
"""
backtest_new_signals.py — Five New Signal Types (honest, fill-verified)

SIGNAL 1: Gap Fade         — fade overnight gap >0.3% when OR confirms rejection
SIGNAL 2: Failed Auction   — bar briefly exceeds PDH/PWH/PMH, closes back → same-bar entry
SIGNAL 3: FOMC Pre-Drift   — LONG 24h before FOMC announcement (Lucca & Moench 2015)
SIGNAL 4: OR Midpoint Return — price returns to OR midpoint and bounces
SIGNAL 5: Post-NFP Momentum — ORB in same direction as NFP data surprise

Each signal is independent; all use market entry (close of signal bar) — no limit fills.
"""

import numpy as np
import pandas as pd
from datetime import time, date, timedelta

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25
RTH_S = time(9, 30); RTH_E = time(16, 0)
OR_S  = time(9, 30); OR_E  = time(10, 0); ORB_C = time(10, 30)
EOD   = time(15, 30)
MIN_ES = 8; MAX_ES = 60
ACCT   = 25_000
RISK   = 0.02   # 2% per new signal trade (vs 5% primary, 1% scalp)
IS_END = date(2021, 12, 31); OS_START = date(2022, 1, 1)


# ─────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────
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
        if wk not in weekly:
            weekly[wk] = [h, l]
        else:
            weekly[wk][0] = max(weekly[wk][0], h); weekly[wk][1] = min(weekly[wk][1], l)
        if mn not in monthly:
            monthly[mn] = [h, l]
        else:
            monthly[mn][0] = max(monthly[mn][0], h); monthly[mn][1] = min(monthly[mn][1], l)
    return weekly, monthly


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def sh(dr):
    return dr.mean() / dr.std(ddof=1) * np.sqrt(252) if len(dr) > 1 and dr.std() > 0 else 0.0


def prow(lbl, tdf, r_col="r", pad=42):
    if not len(tdf):
        print(f"  {'—'} {lbl:<{pad}}  n=0"); return
    wr = (tdf[r_col] > 0).mean() * 100
    ar = tdf[r_col].mean()
    dr = tdf.groupby("date")[r_col].sum()
    s  = sh(dr)
    nyr = tdf["date"].nunique() / 8.1
    wins   = tdf[tdf[r_col] > 0]
    losses = tdf[tdf[r_col] <= 0]
    wl = abs(wins[r_col].mean() / losses[r_col].mean()) if (len(losses) and losses[r_col].mean() != 0) else 99.0
    flag = "✅" if (ar > 0 and s > 2.0) else ("🟡" if (ar > 0 and s > 1.0) else "❌")
    print(f"  {flag} {lbl:<{pad}}  {nyr:.0f}/yr  WR={wr:5.1f}%  "
          f"Avg={ar:+.4f}  W/L={wl:.2f}x  Sh={s:.2f}")


def yr_row(tdf, yr, r_col="r"):
    yt = tdf[tdf["year"] == yr]
    if not len(yt): return
    tag  = "IS " if yr <= 2021 else "OOS"
    wr   = (yt[r_col] > 0).mean() * 100
    ar   = yt[r_col].mean()
    dr   = yt.groupby("date")[r_col].sum()
    s    = sh(dr)
    flag = "✅" if (ar > 0 and s > 1.5) else ("🟡" if ar > 0 else "❌")
    print(f"  {flag} {yr}: n={len(yt):3d}  WR={wr:5.1f}%  AvgR={ar:+.4f}  Sh={s:.2f} {tag}")


# ─────────────────────────────────────────────────────────
# SIGNAL 1 — GAP FADE
# ─────────────────────────────────────────────────────────
def run_gap_fade(df, dates, by_d, min_gap=0.30, max_gap=2.0):
    """
    Overnight gap fade.
      Gap  = (9:30 RTH open) / (prev RTH 4pm close) − 1  (in %)
      Gap up   > min_gap → fade SHORT; target = prev close; stop = OR_H + 2pt
      Gap down < −min_gap → fade LONG;  target = prev close; stop = OR_L − 2pt
      Entry : 10:00 close (end of OR) only if gap NOT already filled.
      Valid fills: entry bar is always a market fill (close of bar) — 100% reachable.
    """
    trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 30: continue

        or_b = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] < OR_E)]
        if not len(or_b): continue
        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min(); es_R = es_H - es_L
        if not (MIN_ES <= es_R <= MAX_ES): continue

        # Previous RTH close
        prev = by_d.get(dates[i - 1])
        if prev is None: continue
        pr = prev[(prev["time_et"] >= RTH_S) & (prev["time_et"] <= RTH_E)]
        if not len(pr): continue
        prev_close = pr["ES_close"].iloc[-1]

        rth_open = or_b["ES_open"].iloc[0]
        gap_pct  = (rth_open - prev_close) / prev_close * 100

        if abs(gap_pct) < min_gap or abs(gap_pct) > max_gap:
            continue

        di     = "SHORT" if gap_pct > 0 else "LONG"
        target = prev_close

        # Entry at 10:00 close
        ep_raw = or_b["ES_close"].iloc[-1]

        # Skip if gap already filled
        if di == "SHORT" and ep_raw <= prev_close: continue
        if di == "LONG"  and ep_raw >= prev_close: continue

        ep = ep_raw - SLIP if di == "SHORT" else ep_raw + SLIP
        sp = es_H + 2.0 + SLIP if di == "SHORT" else es_L - 2.0 - SLIP
        risk = abs(ep - sp)
        if risk < 1.0: continue

        # Simulate forward from 10:00
        fwd = rth[rth["time_et"] >= OR_E].reset_index(drop=True)
        rv = 0.0; why = "NONE"
        for _, bar in fwd.iterrows():
            if bar["time_et"] >= EOD:
                pnl = (ep - bar["ES_close"] - SLIP * 2) if di == "SHORT" else (bar["ES_close"] - ep - SLIP * 2)
                rv = pnl / risk; why = "EOD"; break
            if di == "SHORT":
                if bar["ES_low"]  <= target: rv = (ep - target) / risk; why = "TGT"; break
                if bar["ES_high"] >= sp:     rv = -1.0;                  why = "STOP"; break
            else:
                if bar["ES_high"] >= target: rv = (target - ep) / risk;  why = "TGT"; break
                if bar["ES_low"]  <= sp:     rv = -1.0;                   why = "STOP"; break

        ts = pd.Timestamp(d); iso = ts.isocalendar()
        trades.append(dict(
            date=d, year=d.year, direction=di, r=rv,
            usd=round(rv * (ACCT * RISK), 2), win=rv > 0,
            exit=why, gap_pct=round(gap_pct, 3),
            iso_wk=(int(iso.year), int(iso.week)),
        ))
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────
# SIGNAL 2 — FAILED AUCTION
# ─────────────────────────────────────────────────────────
def run_failed_auction(df, dates, by_d, weekly, monthly, reject_min=0.5):
    """
    Failed auction at structural levels.
    SHORT: bar HIGH >= level AND bar CLOSE <= level − reject_min  (SAME bar)
    LONG : bar LOW  <= level AND bar CLOSE >= level + reject_min  (SAME bar)
    Entry: CLOSE of signal bar ± slip  (market fill, 100% valid)
    Stop : bar HIGH/LOW ± slip
    Target: 2R fixed
    Window: 10:30–14:30
    Only ONE trade per day (first signal).
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

        # Resistance (for SHORT failed auction)
        res_lvls = [(PDH, "D1_H")]
        if PWH: res_lvls.append((PWH, "W1_H"))
        if PMH: res_lvls.append((PMH, "MN_H"))

        # Support (for LONG failed auction)
        sup_lvls = [(PDL, "D1_L")]
        if PWL: sup_lvls.append((PWL, "W1_L"))
        if PML: sup_lvls.append((PML, "MN_L"))

        scan   = rth[(rth["time_et"] >= time(10, 30)) & (rth["time_et"] <= time(14, 30))]
        fired  = False

        for _, bar in scan.iterrows():
            if fired: break

            # Check SHORT (failed breakout above resistance)
            for lv, nm in res_lvls:
                if bar["ES_high"] >= lv and bar["ES_close"] <= lv - reject_min:
                    ep   = bar["ES_close"] - SLIP
                    sp   = bar["ES_high"]  + SLIP
                    risk = abs(ep - sp)
                    if risk < 0.5 or risk > 25: continue
                    tp   = ep - 2 * risk

                    fwd  = rth[rth["time_et"] > bar["time_et"]].reset_index(drop=True)
                    rv   = 0.0; why = "NONE"
                    for _, fb in fwd.iterrows():
                        if fb["time_et"] >= EOD:
                            rv = (ep - fb["ES_close"] - SLIP * 2) / risk; why = "EOD"; break
                        if fb["ES_high"] >= sp:  rv = -1.0;               why = "STOP"; break
                        if fb["ES_low"]  <= tp:  rv = 2.0 - SLIP / risk;  why = "TGT";  break

                    trades.append(dict(
                        date=d, year=d.year, direction="SHORT", r=rv,
                        usd=round(rv * (ACCT * RISK), 2), win=rv > 0,
                        exit=why, level_type=nm,
                        iso_wk=(int(iso.year), int(iso.week)),
                    ))
                    fired = True; break

            if fired: break

            # Check LONG (failed breakdown below support)
            for lv, nm in sup_lvls:
                if bar["ES_low"] <= lv and bar["ES_close"] >= lv + reject_min:
                    ep   = bar["ES_close"] + SLIP
                    sp   = bar["ES_low"]   - SLIP
                    risk = abs(ep - sp)
                    if risk < 0.5 or risk > 25: continue
                    tp   = ep + 2 * risk

                    fwd  = rth[rth["time_et"] > bar["time_et"]].reset_index(drop=True)
                    rv   = 0.0; why = "NONE"
                    for _, fb in fwd.iterrows():
                        if fb["time_et"] >= EOD:
                            rv = (fb["ES_close"] - ep - SLIP * 2) / risk; why = "EOD"; break
                        if fb["ES_low"]  <= sp:  rv = -1.0;               why = "STOP"; break
                        if fb["ES_high"] >= tp:  rv = 2.0 - SLIP / risk;  why = "TGT";  break

                    trades.append(dict(
                        date=d, year=d.year, direction="LONG", r=rv,
                        usd=round(rv * (ACCT * RISK), 2), win=rv > 0,
                        exit=why, level_type=nm,
                        iso_wk=(int(iso.year), int(iso.week)),
                    ))
                    fired = True; break

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────
# SIGNAL 3 — FOMC PRE-ANNOUNCEMENT DRIFT
# ─────────────────────────────────────────────────────────
FOMC_DATES = {
    # 2018
    date(2018, 1, 31), date(2018, 3, 21), date(2018, 5, 2),  date(2018, 6, 13),
    date(2018, 8, 1),  date(2018, 9, 26), date(2018, 11, 8), date(2018, 12, 19),
    # 2019
    date(2019, 1, 30), date(2019, 3, 20), date(2019, 5, 1),  date(2019, 6, 19),
    date(2019, 7, 31), date(2019, 9, 18), date(2019, 10, 30),date(2019, 12, 11),
    # 2020
    date(2020, 1, 29), date(2020, 3, 3),  date(2020, 3, 15), date(2020, 4, 29),
    date(2020, 6, 10), date(2020, 7, 29), date(2020, 9, 16), date(2020, 11, 5),
    date(2020, 12, 16),
    # 2021
    date(2021, 1, 27), date(2021, 3, 17), date(2021, 4, 28), date(2021, 6, 16),
    date(2021, 7, 28), date(2021, 9, 22), date(2021, 11, 3), date(2021, 12, 15),
    # 2022
    date(2022, 1, 26), date(2022, 3, 16), date(2022, 5, 4),  date(2022, 6, 15),
    date(2022, 7, 27), date(2022, 9, 21), date(2022, 11, 2), date(2022, 12, 14),
    # 2023
    date(2023, 2, 1),  date(2023, 3, 22), date(2023, 5, 3),  date(2023, 6, 14),
    date(2023, 7, 26), date(2023, 9, 20), date(2023, 11, 1), date(2023, 12, 13),
    # 2024
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),  date(2024, 6, 12),
    date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    # 2025
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),  date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 11, 5), date(2025, 12, 10),
    # 2026
    date(2026, 1, 28), date(2026, 3, 18),
}


def run_fomc_drift(df, dates, by_d):
    """
    Pre-FOMC long (Lucca & Moench 2015 JF paper).
    Entry : 14:00 ET on the trading day BEFORE FOMC announcement.
    Exit  : 13:55 ET on FOMC announcement day.
    Stop  : 0.5% below entry (wide intentional — drift is a carry, not intraday).
    Entry is always a close-of-bar fill — 100% valid.
    """
    dates_set = set(dates)
    trades    = []

    for fomc_day in sorted(FOMC_DATES):
        if fomc_day not in dates_set: continue
        # Previous trading day
        prev_day = None
        for d in reversed(dates):
            if d < fomc_day: prev_day = d; break
        if prev_day is None: continue

        entry_day = by_d.get(prev_day)
        if entry_day is None: continue

        eb = entry_day[entry_day["time_et"] >= time(14, 0)]
        if not len(eb): continue
        ep_raw = eb.iloc[0]["ES_close"]
        ep     = ep_raw + SLIP
        sp     = ep * (1 - 0.005) - SLIP   # 0.5% stop
        risk   = abs(ep - sp)
        if risk < 1.0: continue

        # Forward: rest of prev_day (after 14:00) + fomc_day up to 13:55
        fomc_data = by_d.get(fomc_day)
        if fomc_data is None: continue
        fwd_prev  = entry_day[entry_day["time_et"] > time(14, 0)]
        fwd_fomc  = fomc_data[fomc_data["time_et"] <= time(13, 55)]

        rv = 0.0; why = "NONE"; stopped = False
        for rows in [fwd_prev, fwd_fomc]:
            if stopped: break
            for _, bar in rows.iterrows():
                if bar["ES_low"] <= sp:
                    rv = -1.0; why = "STOP"; stopped = True; break

        if not stopped and len(fwd_fomc):
            exit_p = fwd_fomc.iloc[-1]["ES_close"]
            rv     = (exit_p - ep - SLIP * 2) / risk
            why    = "EOD"

        ts = pd.Timestamp(prev_day); iso = ts.isocalendar()
        trades.append(dict(
            date=prev_day, year=prev_day.year, direction="LONG", r=rv,
            usd=round(rv * (ACCT * RISK), 2), win=rv > 0,
            exit=why, fomc_date=fomc_day,
            iso_wk=(int(iso.year), int(iso.week)),
        ))
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────
# SIGNAL 4 — OR MIDPOINT RETURN
# ─────────────────────────────────────────────────────────
def run_or_midpoint(df, dates, by_d, bounce_req=2.0, min_move=1.0):
    """
    OR midpoint support / resistance.
    OR_MID = (OR_H + OR_L) / 2 after OR locks at 10:00.

    LONG : prev bar close > OR_MID + bounce_req, bar LOW touches OR_MID,
           bar CLOSE > OR_MID + min_move  → bounce confirmed
    SHORT: prev bar close < OR_MID − bounce_req, bar HIGH touches OR_MID,
           bar CLOSE < OR_MID − min_move  → bounce confirmed

    Entry : bar close ± slip  (100% valid)
    Stop  : bar low/high ± slip
    Target: 2R
    Window: 10:30–14:00 (first signal only)
    """
    trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 20: continue

        or_b = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] < OR_E)]
        if not len(or_b): continue
        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min(); es_R = es_H - es_L
        if not (MIN_ES <= es_R <= MAX_ES): continue
        OR_MID = (es_H + es_L) / 2.0

        scan   = rth[(rth["time_et"] >= time(10, 30)) & (rth["time_et"] <= time(14, 0))]
        prev_c = None; fired = False

        for _, bar in scan.iterrows():
            if fired: break
            if prev_c is None: prev_c = bar["ES_close"]; continue

            # LONG: price was above midpoint, dips to it, closes back up
            if (prev_c > OR_MID + bounce_req and
                    bar["ES_low"] <= OR_MID and
                    bar["ES_close"] >= OR_MID + min_move):
                ep   = bar["ES_close"] + SLIP
                sp   = bar["ES_low"]   - SLIP
                risk = abs(ep - sp)
                if 0.5 <= risk <= 20:
                    tp  = ep + 2 * risk
                    fwd = rth[rth["time_et"] > bar["time_et"]].reset_index(drop=True)
                    rv  = 0.0; why = "NONE"
                    for _, fb in fwd.iterrows():
                        if fb["time_et"] >= EOD:
                            rv = (fb["ES_close"] - ep - SLIP * 2) / risk; why = "EOD"; break
                        if fb["ES_low"]  <= sp: rv = -1.0;              why = "STOP"; break
                        if fb["ES_high"] >= tp: rv = 2.0 - SLIP / risk; why = "TGT";  break
                    ts = pd.Timestamp(d); iso = ts.isocalendar()
                    trades.append(dict(
                        date=d, year=d.year, direction="LONG", r=rv,
                        usd=round(rv * (ACCT * RISK), 2), win=rv > 0,
                        exit=why, iso_wk=(int(iso.year), int(iso.week)),
                    ))
                    fired = True

            # SHORT: price was below midpoint, rises to it, closes back down
            elif (prev_c < OR_MID - bounce_req and
                  bar["ES_high"] >= OR_MID and
                  bar["ES_close"] <= OR_MID - min_move):
                ep   = bar["ES_close"] - SLIP
                sp   = bar["ES_high"]  + SLIP
                risk = abs(ep - sp)
                if 0.5 <= risk <= 20:
                    tp  = ep - 2 * risk
                    fwd = rth[rth["time_et"] > bar["time_et"]].reset_index(drop=True)
                    rv  = 0.0; why = "NONE"
                    for _, fb in fwd.iterrows():
                        if fb["time_et"] >= EOD:
                            rv = (ep - fb["ES_close"] - SLIP * 2) / risk; why = "EOD"; break
                        if fb["ES_high"] >= sp: rv = -1.0;              why = "STOP"; break
                        if fb["ES_low"]  <= tp: rv = 2.0 - SLIP / risk; why = "TGT";  break
                    ts = pd.Timestamp(d); iso = ts.isocalendar()
                    trades.append(dict(
                        date=d, year=d.year, direction="SHORT", r=rv,
                        usd=round(rv * (ACCT * RISK), 2), win=rv > 0,
                        exit=why, iso_wk=(int(iso.year), int(iso.week)),
                    ))
                    fired = True

            prev_c = bar["ES_close"]
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────
# SIGNAL 5 — POST-NFP DATA MOMENTUM
# ─────────────────────────────────────────────────────────
def get_nfp_dates():
    """NFP = first Friday of each month (8:30 ET release)."""
    nfp = set()
    for yr in range(2018, 2027):
        for mo in range(1, 13):
            for day in range(1, 8):
                try:
                    d = date(yr, mo, day)
                    if d.weekday() == 4:   # Friday
                        nfp.add(d); break
                except ValueError:
                    pass
    return nfp


def run_post_data(df, dates, by_d, min_move_pct=0.50):
    """
    Post-NFP momentum ORB.
    On NFP Fridays: if ES moves > min_move_pct% from pre-data level (9:29 close)
    by 10:00 AM, trade the ORB in SAME direction as the data surprise.
    Entry : breakout bar close ± slip  (market fill)
    Stop  : OR range; Target: 3R
    """
    nfp_dates = get_nfp_dates()
    dates_set  = set(dates)
    trades     = []

    for i, d in enumerate(dates[1:], 1):
        if d not in nfp_dates: continue
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 20: continue

        # Pre-data level = last bar at or before 9:29
        pre = day[day["time_et"] <= time(9, 29)]
        if not len(pre): continue
        pre_level = pre.iloc[-1]["ES_close"]

        or_b = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] < OR_E)]
        if not len(or_b): continue
        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min(); es_R = es_H - es_L
        if not (MIN_ES <= es_R <= MAX_ES): continue

        or_close   = or_b["ES_close"].iloc[-1]
        data_move  = (or_close - pre_level) / pre_level * 100

        if abs(data_move) < min_move_pct: continue

        data_di = "LONG" if data_move > 0 else "SHORT"

        # ORB in same direction as data move
        post = rth[(rth["time_et"] >= OR_E) & (rth["time_et"] < ORB_C)]
        entry_row = None
        for _, r in post.iterrows():
            if r["time_et"] < time(10, 2): continue
            if data_di == "LONG"  and r["ES_high"] > es_H: entry_row = r; break
            if data_di == "SHORT" and r["ES_low"]  < es_L: entry_row = r; break

        if entry_row is None: continue

        ep_raw = entry_row["ES_close"]
        ep     = ep_raw + SLIP if data_di == "LONG" else ep_raw - SLIP
        sp     = es_L - SLIP   if data_di == "LONG" else es_H + SLIP
        tp     = ep + 3 * es_R if data_di == "LONG" else ep - 3 * es_R
        risk   = abs(ep - sp)
        if risk < 1.0: continue

        fwd = rth[rth["time_et"] > entry_row["time_et"]].reset_index(drop=True)
        rv  = 0.0; why = "NONE"
        for _, bar in fwd.iterrows():
            if bar["time_et"] >= EOD:
                pnl = (bar["ES_close"] - ep - SLIP * 2) if data_di == "LONG" else (ep - bar["ES_close"] - SLIP * 2)
                rv  = pnl / risk; why = "EOD"; break
            if data_di == "LONG":
                if bar["ES_low"]  <= sp: rv = -1.0;               why = "STOP"; break
                if bar["ES_high"] >= tp: rv = 3.0 - SLIP / risk;  why = "TGT";  break
            else:
                if bar["ES_high"] >= sp: rv = -1.0;               why = "STOP"; break
                if bar["ES_low"]  <= tp: rv = 3.0 - SLIP / risk;  why = "TGT";  break

        ts = pd.Timestamp(d); iso = ts.isocalendar()
        trades.append(dict(
            date=d, year=d.year, direction=data_di, r=rv,
            usd=round(rv * (ACCT * RISK), 2), win=rv > 0,
            exit=why, data_move=round(data_move, 3),
            iso_wk=(int(iso.year), int(iso.week)),
        ))
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("Loading …")
    df     = load()
    dates  = sorted(df["date_et"].unique())
    by_d   = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    weekly, monthly = build_levels(by_d, dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    W = 60

    # ── SIGNAL 1: GAP FADE ────────────────────────────────────────────────────
    print("=" * W)
    print("  SIGNAL 1: GAP FADE")
    print("=" * W)
    print("  Gap overnight > 0.3%; fade at 10:00 close; target = prev close")
    print("  Entry: 10:00 bar close (100% valid); Stop: OR extreme + 2pt\n")

    gf = run_gap_fade(df, dates, by_d)
    print(f"  Total trades: {len(gf)} ({len(gf)/8.1:.0f}/yr)\n")
    prow("Full period  2018-2026", gf)
    prow("IS  2018-2021",          gf[gf["year"] <= 2021])
    prow("OOS 2022-2026",          gf[gf["year"] >= 2022])

    if len(gf):
        print("\n  By gap direction:")
        prow("  Gap UP  → SHORT", gf[gf["direction"] == "SHORT"])
        prow("  Gap DN  → LONG",  gf[gf["direction"] == "LONG"])
        print("\n  By gap magnitude:")
        prow("  0.3 – 0.5%", gf[(gf["gap_pct"].abs() >= 0.30) & (gf["gap_pct"].abs() < 0.50)])
        prow("  0.5 – 1.0%", gf[(gf["gap_pct"].abs() >= 0.50) & (gf["gap_pct"].abs() < 1.00)])
        prow("  > 1.0%",     gf[gf["gap_pct"].abs() >= 1.00])
        print("\n  Year-by-year:")
        for yr in sorted(gf["year"].unique()): yr_row(gf, yr)

    # ── SIGNAL 2: FAILED AUCTION ──────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  SIGNAL 2: FAILED AUCTION AT STRUCTURAL LEVELS")
    print("=" * W)
    print("  Same bar: HIGH >= PDH/PWH/PMH, CLOSE < level−0.5 → SHORT entry at CLOSE")
    print("            LOW  <= PDL/PWL/PML, CLOSE > level+0.5 → LONG  entry at CLOSE")
    print("  Entry: SAME bar close ± slip (100% valid); Stop: bar H/L; Target: 2R\n")

    fa = run_failed_auction(df, dates, by_d, weekly, monthly)
    print(f"  Total trades: {len(fa)} ({len(fa)/8.1:.0f}/yr)\n")
    prow("Full period  2018-2026", fa)
    prow("IS  2018-2021",          fa[fa["year"] <= 2021])
    prow("OOS 2022-2026",          fa[fa["year"] >= 2022])

    if len(fa):
        print("\n  By direction:")
        prow("  SHORT (failed breakout)", fa[fa["direction"] == "SHORT"])
        prow("  LONG  (failed breakdown)", fa[fa["direction"] == "LONG"])
        print("\n  By level type:")
        for lt in sorted(fa["level_type"].unique()):
            prow(f"  {lt}", fa[fa["level_type"] == lt])
        print("\n  Year-by-year:")
        for yr in sorted(fa["year"].unique()): yr_row(fa, yr)

    # ── SIGNAL 3: FOMC PRE-DRIFT ──────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  SIGNAL 3: FOMC PRE-ANNOUNCEMENT DRIFT")
    print("=" * W)
    print("  Academic: Lucca & Moench (2015, JF) — ES drifts +0.4% in 24h before FOMC")
    print("  Entry: 14:00 day before FOMC; Exit: 13:55 announcement day")
    print("  Stop: 0.5% below entry; ~8 trades/year\n")

    fm = run_fomc_drift(df, dates, by_d)
    print(f"  Total trades: {len(fm)} ({len(fm)/8.1:.0f}/yr)\n")
    prow("Full period  2018-2026", fm)
    prow("IS  2018-2021",          fm[fm["year"] <= 2021])
    prow("OOS 2022-2026",          fm[fm["year"] >= 2022])

    if len(fm):
        print("\n  Year-by-year:")
        for yr in sorted(fm["year"].unique()): yr_row(fm, yr)
        wins   = fm[fm["r"] > 0]
        losses = fm[fm["r"] <= 0]
        print(f"\n  Exit breakdown: STOP={len(fm[fm['exit']=='STOP'])}  "
              f"EOD={len(fm[fm['exit']=='EOD'])}")
        if len(wins):
            print(f"  Avg win  = {wins['r'].mean():+.3f}R")
        if len(losses):
            print(f"  Avg loss = {losses['r'].mean():+.3f}R")

    # ── SIGNAL 4: OR MIDPOINT ─────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  SIGNAL 4: OR MIDPOINT RETURN")
    print("=" * W)
    print("  Price must be 2+ pts from OR_MID, then touch it, then close away → bounce")
    print("  Entry: signal bar close ± slip (100% valid); Stop: bar H/L; Target: 2R\n")

    orm = run_or_midpoint(df, dates, by_d)
    print(f"  Total trades: {len(orm)} ({len(orm)/8.1:.0f}/yr)\n")
    prow("Full period  2018-2026", orm)
    prow("IS  2018-2021",          orm[orm["year"] <= 2021])
    prow("OOS 2022-2026",          orm[orm["year"] >= 2022])

    if len(orm):
        print("\n  By direction:")
        prow("  LONG  (bounce from above)", orm[orm["direction"] == "LONG"])
        prow("  SHORT (bounce from below)", orm[orm["direction"] == "SHORT"])
        print("\n  Year-by-year:")
        for yr in sorted(orm["year"].unique()): yr_row(orm, yr)

    # ── SIGNAL 5: POST-NFP MOMENTUM ───────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  SIGNAL 5: POST-NFP DATA MOMENTUM")
    print("=" * W)
    print("  NFP Fridays: if ES moves > 0.5% from pre-data level by 10:00 AM")
    print("  → trade ORB breakout in SAME direction; Target: 3R; Stop: OR range\n")

    nfp = run_post_data(df, dates, by_d)
    print(f"  Total trades: {len(nfp)} ({len(nfp)/8.1:.0f}/yr)\n")
    prow("Full period  2018-2026", nfp)
    prow("IS  2018-2021",          nfp[nfp["year"] <= 2021])
    prow("OOS 2022-2026",          nfp[nfp["year"] >= 2022])

    if len(nfp):
        print("\n  By direction:")
        prow("  LONG  (positive surprise)", nfp[nfp["direction"] == "LONG"])
        prow("  SHORT (negative surprise)", nfp[nfp["direction"] == "SHORT"])
        print("\n  Year-by-year:")
        for yr in sorted(nfp["year"].unique()): yr_row(nfp, yr)

    # ── FINAL RANKING ─────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  FINAL RANKING — ALL 5 NEW SIGNALS (full period, honest results)")
    print("=" * W)
    print(f"\n  {'Signal':<35} {'n/yr':>6} {'WR':>7} {'AvgR':>8} {'W/L':>6} {'Sh':>6}")
    print("  " + "─" * 68)

    all_sigs = [
        ("Gap Fade",              gf),
        ("Failed Auction",        fa),
        ("FOMC Pre-Drift",        fm),
        ("OR Midpoint Return",    orm),
        ("Post-NFP Momentum",     nfp),
    ]

    results = []
    for name, tdf in all_sigs:
        if not len(tdf): results.append((name, 0, 0, 0, 0, 0, "❌")); continue
        nyr  = tdf["date"].nunique() / 8.1
        wr   = (tdf["r"] > 0).mean() * 100
        ar   = tdf["r"].mean()
        dr   = tdf.groupby("date")["r"].sum()
        s    = sh(dr)
        wins   = tdf[tdf["r"] > 0]
        losses = tdf[tdf["r"] <= 0]
        wl = abs(wins["r"].mean() / losses["r"].mean()) if (len(losses) and losses["r"].mean() != 0) else 99.0
        flag = "✅" if (ar > 0 and s > 2) else ("🟡" if (ar > 0 and s > 1) else "❌")
        results.append((name, nyr, wr, ar, wl, s, flag))

    results_sorted = sorted(results, key=lambda x: -x[5])
    for name, nyr, wr, ar, wl, s, flag in results_sorted:
        print(f"  {flag} {name:<35} {nyr:>5.0f}/yr  {wr:5.1f}%  {ar:+.4f}  {wl:.2f}x  Sh={s:.2f}")

    # ── ADDITIVE SUMMARY (what to add to existing system) ──────────────────────
    print(f"\n{'=' * W}")
    print("  ADDITIVE SIGNAL SUMMARY")
    print(f"  {'─' * 58}")
    print("  Existing system: Primary 18/yr + Scalp 127/yr → 2.8 trades/week")
    print()
    total_new = sum(len(tdf) for _, tdf in all_sigs)
    total_new_yr = total_new / 8.1
    print(f"  New signals total: {total_new_yr:.0f}/yr")
    good_new = sum(len(tdf) for _, tdf in all_sigs
                   if len(tdf) and tdf["r"].mean() > 0)
    print()

    for name, nyr, wr, ar, wl, s, flag in results_sorted:
        add = "ADD" if (ar > 0 and s > 1.5) else ("MARGINAL" if ar > 0 else "SKIP")
        print(f"  {flag} {name:<30} → {add}  ({nyr:.0f}/yr, WR={wr:.1f}%, Sh={s:.2f})")

    print()


if __name__ == "__main__":
    main()
