#!/usr/bin/env python3
"""
backtest_joint_full.py — Complete Joint System Test

THREE INDEPENDENT SYSTEMS, fully self-contained (no external imports):

  SYSTEM 1 — ASYMMETRIC PRIMARY (ORB + PDH/L + 1H Momentum)
    Risk per trade : 5% of $25k = $1,250
    Target         : 3R, Stop: OR range

  SYSTEM 2 — MTF SCALP (rejection at S/R levels)
    Risk per trade : 1% of $25k = $250 (variable MES sizing)
    Target         : trailing 3pt stop

  SYSTEM 3 — DOUBLE TEST AT LEVEL (two failed tests = reversal)
    Risk per trade : 2% of $25k = $500
    Target         : 3R, Stop: level + 1pt

All three run INDEPENDENTLY — can fire on the same day without conflict
(different entry times: Primary 10:00–10:30, Scalp 10:30–14:30, Double
Test 10:30–14:00). Daily P&L is summed across all active systems.

VERIFICATION SECTION: Prints individual system stats and flags if they
differ from prior known values by more than ±5%.
"""

import numpy as np
import pandas as pd
from datetime import time, date, datetime

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
SLIP  = 0.25
RTH_S = time(9, 30); RTH_E = time(16, 0)
OR_S  = time(9, 30); OR_E  = time(10, 0); ORB_C = time(10, 30)
EOD   = time(15, 30)
MIN_ES = 8;  MAX_ES = 60
MIN_NQ = 30; MAX_NQ = 175
TRAIL = 3.0; BE = 3.0
ACCT  = 25_000
PRI_RISK = 0.05   # System 1: 5%
SC_RISK  = 0.01   # System 2: 1%
DT_RISK  = 0.02   # System 3: 2%
IS_END   = date(2021, 12, 31)
OS_START = date(2022, 1, 1)

# Known reference values from individual backtests (for verification)
KNOWN = {
    "primary_wr":    60.6,   # %
    "primary_sh":    4.06,
    "primary_nyr":   18,
    "scalp_wr":      44.7,   # %
    "scalp_sh":      2.28,
    "scalp_nyr":     127,
    "dt_wr":         38.7,   # %
    "dt_sh":         4.23,
    "dt_nyr":        65,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATORS
# ─────────────────────────────────────────────────────────────────────────────
def sim_orb(day, et, di, ep, sp, tp):
    rng = abs(ep - sp)
    if rng < 0.01: return 0.0, "NONE"
    for _, bar in day[day["time_et"] > et].iterrows():
        if bar["time_et"] >= EOD:
            pnl = (bar["ES_close"] - ep - SLIP*2) if di == "LONG" else (ep - bar["ES_close"] - SLIP*2)
            return round(pnl / rng, 3), "EOD"
        if di == "LONG":
            if bar["ES_low"]  <= sp: return round(-1.0 - SLIP*2/rng, 3), "STOP"
            if bar["ES_high"] >= tp: return round(3.0 - SLIP/rng, 3),    "TGT"
        else:
            if bar["ES_high"] >= sp: return round(-1.0 - SLIP*2/rng, 3), "STOP"
            if bar["ES_low"]  <= tp: return round(3.0 - SLIP/rng, 3),    "TGT"
    return 0.0, "NONE"


def sim_trail(fwd, di, ep, s0):
    stop = s0; best = ep
    for c in fwd:
        if di == "SHORT": best = min(best, c); u = ep - c - SLIP*2
        else:             best = max(best, c); u = c - ep - SLIP*2
        if u >= BE:
            if di == "SHORT": stop = min(stop, best + TRAIL)
            else:             stop = max(stop, best - TRAIL)
        if di == "SHORT":
            if c >= stop: return ep - stop - SLIP*2
        else:
            if c <= stop: return stop - ep - SLIP*2
    last = fwd[-1] if len(fwd) else ep
    return ep - last - SLIP*2 if di == "SHORT" else last - ep - SLIP*2


def time_diff_min(t1, t2):
    d1 = datetime(2000,1,1,t1.hour,t1.minute); d2 = datetime(2000,1,1,t2.hour,t2.minute)
    return (d2-d1).total_seconds()/60


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM 1 — ASYMMETRIC PRIMARY
# ─────────────────────────────────────────────────────────────────────────────
def run_primary(df, dates, by_d,
                long_min_1h=0.10, short_max_1h=-0.20,
                short_bearish_prev=True, max_break=0.20, max_pdh=2.0):
    trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 20: continue
        or_b = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] < OR_E)]
        if not len(or_b): continue
        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min(); es_R = es_H - es_L
        nq_or = day[(day["time_et"] >= OR_S) & (day["time_et"] < OR_E)]
        nq_H = nq_or["NQ_high"].max(); nq_L = nq_or["NQ_low"].min(); nq_R = nq_H - nq_L
        if not (MIN_ES <= es_R <= MAX_ES and MIN_NQ <= nq_R <= MAX_NQ): continue
        or_ret = (or_b["ES_close"].iloc[-1] - or_b["ES_open"].iloc[0]) / or_b["ES_open"].iloc[0] * 100
        prev = by_d.get(dates[i-1])
        if prev is None: continue
        pr = prev[(prev["time_et"] >= RTH_S) & (prev["time_et"] <= RTH_E)]
        if not len(pr): continue
        PDH = pr["ES_high"].max(); PDL = pr["ES_low"].min()
        prev_bearish = (pr["ES_close"].iloc[-1] < pr["ES_open"].iloc[0])
        post = rth[(rth["time_et"] >= OR_E) & (rth["time_et"] < ORB_C)]
        nq_post = day[(day["time_et"] >= OR_E) & (day["time_et"] < ORB_C)]
        es_b = nq_b = None; entry_row = None; bd_es = bd_nq = 0
        for _, r in post.iterrows():
            if r["time_et"] < time(10, 2): continue
            if es_b is None:
                if r["ES_high"] > es_H: es_b = "L"; bd_es = (r["ES_high"]-es_H)/es_R
                elif r["ES_low"] < es_L: es_b = "S"; bd_es = (es_L-r["ES_low"])/es_R
            nqr = nq_post[nq_post["time_et"] == r["time_et"]]
            if nq_b is None and len(nqr):
                nr = nqr.iloc[0]
                if nr["NQ_high"] > nq_H: nq_b = "L"; bd_nq = (nr["NQ_high"]-nq_H)/nq_R
                elif nr["NQ_low"] < nq_L: nq_b = "S"; bd_nq = (nq_L-nr["NQ_low"])/nq_R
            if es_b and nq_b: entry_row = r; break
        if not (es_b and nq_b and es_b == nq_b): continue
        if bd_es > max_break or bd_nq > max_break: continue
        di = "LONG" if es_b == "L" else "SHORT"
        if di == "LONG"  and or_ret < long_min_1h: continue
        if di == "SHORT" and or_ret > short_max_1h: continue
        if di == "SHORT" and short_bearish_prev and not prev_bearish: continue
        watch = rth[(rth["time_et"] >= OR_E) & (rth["time_et"] < time(11, 0))]
        pdh_dir = None; eu = ed = False
        for _, r in watch.iterrows():
            if not eu and r["ES_high"] > PDH: eu = True
            if not ed and r["ES_low"]  < PDL: ed = True
            if eu and not ed and pdh_dir is None: pdh_dir = "LONG";  break
            if ed and not eu and pdh_dir is None: pdh_dir = "SHORT"; break
        if pdh_dir != di: continue
        ep_i = entry_row["ES_close"]
        pdh_d = abs(PDH-ep_i)/es_R if di == "LONG" else abs(PDL-ep_i)/es_R
        if pdh_d > max_pdh: continue
        ep = ep_i + SLIP if di == "LONG" else ep_i - SLIP
        sp = es_L - SLIP  if di == "LONG" else es_H + SLIP
        tp = ep + 3*es_R  if di == "LONG" else ep - 3*es_R
        rv, why = sim_orb(day, entry_row["time_et"], di, ep, sp, tp)
        ts = pd.Timestamp(d); iso = ts.isocalendar()
        trades.append(dict(
            date=d, year=d.year, system="primary", direction=di,
            r=rv, usd=round(rv*(ACCT*PRI_RISK), 2),
            win=rv > 0, exit=why, or_R=es_R,
            iso_wk=(int(iso.year), int(iso.week)),
        ))
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM 2 — MTF SCALP
# ─────────────────────────────────────────────────────────────────────────────
def run_scalp(df, dates, by_d, weekly, monthly):
    trades = []
    for i, d in enumerate(dates[1:], 1):
        day = by_d.get(d)
        if day is None: continue
        rth = day[(day["time_et"] >= RTH_S) & (day["time_et"] <= EOD)].reset_index(drop=True)
        if len(rth) < 20: continue
        opens  = rth["ES_open"].values;  closes = rth["ES_close"].values
        highs  = rth["ES_high"].values;  lows   = rth["ES_low"].values
        times  = rth["time_et"].values
        or_b = rth[(rth["time_et"] >= OR_S) & (rth["time_et"] < OR_E)]
        if not len(or_b): continue
        es_H = or_b["ES_high"].max(); es_L = or_b["ES_low"].min(); es_R = es_H - es_L
        if not (MIN_ES <= es_R <= MAX_ES): continue
        prev = by_d.get(dates[i-1])
        if prev is None: continue
        pr = prev[(prev["time_et"] >= RTH_S) & (prev["time_et"] <= RTH_E)]
        if not len(pr): continue
        PDH = pr["ES_high"].max(); PDL = pr["ES_low"].min()
        ts = pd.Timestamp(d); iso = ts.isocalendar()
        pw = int(iso.week)-1; py = int(iso.year)
        if pw == 0: py -= 1; pw = int(pd.Timestamp(f"{py}-12-28").isocalendar().week)
        wk  = weekly.get((py, pw)); PWH = wk[0] if wk else None; PWL = wk[1] if wk else None
        pm  = ts.month - 1; pmy = ts.year
        if pm == 0: pm = 12; pmy -= 1
        mn  = monthly.get((pmy, pm)); PMH = mn[0] if mn else None; PML = mn[1] if mn else None
        day_mid = (rth["ES_high"].max() + rth["ES_low"].min()) / 2
        rnds = [round(day_mid/50)*50 + k*50 for k in range(-3, 4)]
        lvl_r = [(a,b) for a,b in [("D1_H",PDH),("W1_H",PWH),("MN_H",PMH)] +
                 [("RND",float(r)) for r in rnds] if b is not None]
        lvl_s = [(a,b) for a,b in [("D1_L",PDL),("W1_L",PWL),("MN_L",PML)] +
                 [("RND",float(r)) for r in rnds] if b is not None]
        def conf(lv, ll): return sum(1 for (_,v) in ll if 0 < abs(v-lv) <= 8) + 1
        prev_c = None
        for j in range(1, len(closes)-15):
            if times[j] < time(10, 30): prev_c = closes[j]; continue
            if times[j] > time(14, 30): break
            if prev_c is None: prev_c = closes[j]; continue
            fired = False
            for fd, lvls in [("SHORT", lvl_r), ("LONG", lvl_s)]:
                best_c = 0; best_pnl = best_risk = best_nm = None
                for nm, lv in sorted(lvls, key=lambda x: -conf(x[1], lvls)):
                    c2 = conf(lv, lvls)
                    if c2 <= best_c: continue
                    if fd == "SHORT":
                        if not (prev_c < lv-SLIP and highs[j] >= lv and closes[j] <= lv-1.0): continue
                        if j+1 >= len(opens): break
                        ep2 = opens[j+1] + SLIP; s0 = highs[j] + SLIP*2
                    else:
                        if not (prev_c > lv+SLIP and lows[j] <= lv and closes[j] >= lv+1.0): continue
                        if j+1 >= len(opens): break
                        ep2 = opens[j+1] - SLIP; s0 = lows[j] - SLIP*2
                    if abs(ep2-s0) < 0.5: continue
                    best_c = c2; best_pnl = sim_trail(closes[j+1:], fd, ep2, s0)
                    best_risk = abs(ep2-s0); best_nm = nm; break
                if best_pnl is not None:
                    n_mes = max(1, round((ACCT*SC_RISK)/(best_risk*5)))
                    mm    = 1.5 if best_nm == "MN_L" else 1.0
                    usd   = best_pnl * 5 * n_mes * mm
                    trades.append(dict(
                        date=d, year=d.year, system="scalp",
                        pnl=best_pnl, usd=round(usd, 2),
                        win=best_pnl > 0, level_type=best_nm,
                        iso_wk=(int(iso.year), int(iso.week)),
                    ))
                    fired = True; break
            prev_c = closes[j]
            if fired: break
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM 3 — DOUBLE TEST AT LEVEL
# ─────────────────────────────────────────────────────────────────────────────
def run_double_test(df, dates, by_d, weekly, monthly,
                    min_pullback=3.0, max_window_min=90,
                    level_zone=1.0, reject_min=0.5, target_R=3.0):
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
        prev = by_d.get(dates[i-1])
        if prev is None: continue
        pr = prev[(prev["time_et"] >= RTH_S) & (prev["time_et"] <= RTH_E)]
        if not len(pr): continue
        PDH = pr["ES_high"].max(); PDL = pr["ES_low"].min()
        ts = pd.Timestamp(d); iso = ts.isocalendar()
        pw = int(iso.week)-1; py = int(iso.year)
        if pw == 0: py -= 1; pw = int(pd.Timestamp(f"{py}-12-28").isocalendar().week)
        wk  = weekly.get((py, pw)); PWH = wk[0] if wk else None; PWL = wk[1] if wk else None
        pm  = ts.month - 1; pmy = ts.year
        if pm == 0: pm = 12; pmy -= 1
        mn  = monthly.get((pmy, pm)); PMH = mn[0] if mn else None; PML = mn[1] if mn else None
        res_levels = [(PDH,"D1_H")]
        if PWH: res_levels.append((PWH,"W1_H"))
        if PMH: res_levels.append((PMH,"MN_H"))
        sup_levels = [(PDL,"D1_L")]
        if PWL: sup_levels.append((PWL,"W1_L"))
        if PML: sup_levels.append((PML,"MN_L"))
        scan = rth[(rth["time_et"] >= time(10,30)) & (rth["time_et"] <= time(14,0))].reset_index(drop=True)

        def try_dir(is_short):
            lvls = res_levels if is_short else sup_levels
            state = "watching"
            first_lv = first_close = first_t = None
            best_since = None
            for _, bar in scan.iterrows():
                t = bar["time_et"]
                if state == "watching":
                    for lv, nm in lvls:
                        cond = (bar["ES_high"] >= lv and bar["ES_close"] <= lv - reject_min) if is_short \
                               else (bar["ES_low"] <= lv and bar["ES_close"] >= lv + reject_min)
                        if cond:
                            state = "first_done"; first_lv = lv
                            first_close = bar["ES_close"]; first_t = t
                            best_since = bar["ES_close"]; break
                elif state == "first_done":
                    if time_diff_min(first_t, t) > max_window_min:
                        state = "watching"; first_lv = first_close = first_t = None
                        continue
                    best_since = min(best_since, bar["ES_close"]) if is_short \
                                 else max(best_since, bar["ES_close"])
                    gap = (first_close - best_since) if is_short else (best_since - first_close)
                    if gap >= min_pullback: state = "pulled_back"
                elif state == "pulled_back":
                    if time_diff_min(first_t, t) > max_window_min:
                        state = "watching"; continue
                    cond2 = (bar["ES_high"] >= first_lv - level_zone and bar["ES_close"] <= first_lv - reject_min) \
                            if is_short \
                            else (bar["ES_low"] <= first_lv + level_zone and bar["ES_close"] >= first_lv + reject_min)
                    if cond2:
                        ep = bar["ES_close"] - SLIP if is_short else bar["ES_close"] + SLIP
                        sp = first_lv + 1.0 + SLIP  if is_short else first_lv - 1.0 - SLIP
                        risk = abs(ep - sp)
                        if 0.5 <= risk <= 25:
                            tp = ep - target_R*risk if is_short else ep + target_R*risk
                            fwd = rth[rth.index > bar.name].reset_index(drop=True)
                            rv = 0.0; why = "NONE"
                            for _, fb in fwd.iterrows():
                                if fb["time_et"] >= EOD:
                                    rv = (ep-fb["ES_close"]-SLIP*2)/risk if is_short \
                                         else (fb["ES_close"]-ep-SLIP*2)/risk
                                    why = "EOD"; break
                                if is_short:
                                    if fb["ES_high"] >= sp: rv=-1.0; why="STOP"; break
                                    if fb["ES_low"]  <= tp: rv=target_R-SLIP/risk; why="TGT"; break
                                else:
                                    if fb["ES_low"]  <= sp: rv=-1.0; why="STOP"; break
                                    if fb["ES_high"] >= tp: rv=target_R-SLIP/risk; why="TGT"; break
                            return dict(
                                date=d, year=d.year, system="double_test",
                                direction="SHORT" if is_short else "LONG",
                                r=rv, usd=round(rv*(ACCT*DT_RISK),2), win=rv>0,
                                exit=why, iso_wk=(int(iso.year),int(iso.week)),
                            )
            return None

        result = try_dir(True) or try_dir(False)
        if result: trades.append(result)
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# STATS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sh(dr):
    return dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0.0


def max_drawdown(equity_curve):
    """Max peak-to-trough drawdown in same units as equity_curve."""
    peak = equity_curve.cummax()
    dd   = equity_curve - peak
    return dd.min()


def prow(lbl, tdf, r_col="usd", pad=44):
    if not len(tdf): print(f"  — {lbl:<{pad}}  n=0"); return
    wr  = (tdf[r_col] > 0).mean() * 100
    ar  = tdf[r_col].mean()
    dr  = tdf.groupby("date")[r_col].sum()
    s   = sh(dr / ACCT * 252**0.5 / 252**0.5)   # Sharpe on USD daily series
    # recompute properly
    daily_ret = dr / ACCT
    s = daily_ret.mean()/daily_ret.std(ddof=1)*np.sqrt(252) if daily_ret.std()>0 else 0
    nyr = tdf["date"].nunique() / 8.1
    wins = tdf[tdf[r_col]>0]; losses = tdf[tdf[r_col]<=0]
    wl = abs(wins[r_col].mean()/losses[r_col].mean()) if len(losses) and losses[r_col].mean()!=0 else 99
    flag = "✅" if (ar>0 and s>2.5) else ("🟡" if (ar>0 and s>1.5) else "❌")
    print(f"  {flag} {lbl:<{pad}}  {nyr:.0f}/yr  WR={wr:5.1f}%  "
          f"Avg=${ar:+.0f}  W/L={wl:.2f}x  Sh={s:.2f}")


def verify(name, actual_wr, actual_sh, actual_nyr, known_wr, known_sh, known_nyr):
    ok_wr  = abs(actual_wr - known_wr) / known_wr < 0.05
    ok_sh  = abs(actual_sh - known_sh) / (abs(known_sh)+0.01) < 0.10
    ok_nyr = abs(actual_nyr - known_nyr) / known_nyr < 0.05
    status = "✅" if (ok_wr and ok_sh and ok_nyr) else "⚠️ "
    print(f"  {status} {name:<14}  "
          f"WR: {actual_wr:.1f}% (exp {known_wr:.1f}%)  "
          f"Sh: {actual_sh:.2f} (exp {known_sh:.2f})  "
          f"n/yr: {actual_nyr:.0f} (exp {known_nyr})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading …")
    df    = load()
    dates = sorted(df["date_et"].unique())
    by_d  = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    weekly, monthly = build_levels(by_d, dates)
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    print("Running System 1 — Asymmetric Primary …")
    pri = run_primary(df, dates, by_d)
    print(f"  → {len(pri)} trades  ({len(pri)/8.1:.0f}/yr)")

    print("Running System 2 — MTF Scalp …")
    scl = run_scalp(df, dates, by_d, weekly, monthly)
    print(f"  → {len(scl)} trades  ({len(scl)/8.1:.0f}/yr)")

    print("Running System 3 — Double Test …")
    dbt = run_double_test(df, dates, by_d, weekly, monthly)
    print(f"  → {len(dbt)} trades  ({len(dbt)/8.1:.0f}/yr)\n")

    W = 76

    # ── STEP 0: VERIFICATION ─────────────────────────────────────────────────
    print("=" * W)
    print("  STEP 0 — VERIFICATION (actual vs prior reported numbers)")
    print("=" * W)
    print("  Tolerance: ±5% WR, ±10% Sharpe, ±5% frequency\n")

    pri_wr  = (pri["r"]>0).mean()*100
    pri_sh  = sh(pri.groupby("date")["r"].sum())
    pri_nyr = len(pri)/8.1

    scl_wr  = (scl["pnl"]>0).mean()*100
    scl_sh  = sh(scl.groupby("date")["pnl"].sum())
    scl_nyr = len(scl)/8.1

    dbt_wr  = (dbt["r"]>0).mean()*100
    dbt_sh  = sh(dbt.groupby("date")["r"].sum())
    dbt_nyr = len(dbt)/8.1

    verify("Primary",     pri_wr, pri_sh, pri_nyr,
           KNOWN["primary_wr"], KNOWN["primary_sh"], KNOWN["primary_nyr"])
    verify("Scalp",       scl_wr, scl_sh, scl_nyr,
           KNOWN["scalp_wr"], KNOWN["scalp_sh"], KNOWN["scalp_nyr"])
    verify("Double Test", dbt_wr, dbt_sh, dbt_nyr,
           KNOWN["dt_wr"], KNOWN["dt_sh"], KNOWN["dt_nyr"])

    # ── STEP 1: INDIVIDUAL SYSTEM DETAIL ─────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  STEP 1 — INDIVIDUAL SYSTEMS (USD P&L basis, 2% daily return Sharpe)")
    print("=" * W)

    print("\n  — SYSTEM 1: ASYMMETRIC PRIMARY (5% risk / trade = $1,250)")
    pri_IS  = pri[pri["date"] <= IS_END]
    pri_OOS = pri[pri["date"] >= OS_START]
    prow("Full 2018-2026", pri, r_col="usd")
    prow("IS  2018-2021",  pri_IS, r_col="usd")
    prow("OOS 2022-2026",  pri_OOS, r_col="usd")

    print("\n  — SYSTEM 2: MTF SCALP (1% risk / trade ≈ $250)")
    scl_IS  = scl[scl["date"] <= IS_END]
    scl_OOS = scl[scl["date"] >= OS_START]
    prow("Full 2018-2026", scl, r_col="usd")
    prow("IS  2018-2021",  scl_IS, r_col="usd")
    prow("OOS 2022-2026",  scl_OOS, r_col="usd")

    print("\n  — SYSTEM 3: DOUBLE TEST AT LEVEL (2% risk / trade = $500)")
    dbt_IS  = dbt[dbt["date"] <= IS_END]
    dbt_OOS = dbt[dbt["date"] >= OS_START]
    prow("Full 2018-2026", dbt, r_col="usd")
    prow("IS  2018-2021",  dbt_IS, r_col="usd")
    prow("OOS 2022-2026",  dbt_OOS, r_col="usd")

    # ── STEP 2: JOINT SYSTEM ─────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  STEP 2 — JOINT SYSTEM (all three combined)")
    print("=" * W)

    # Build daily USD P&L for each system
    all_dates = sorted(set(
        list(pri["date"].unique()) +
        list(scl["date"].unique()) +
        list(dbt["date"].unique())
    ))

    daily_pri = pri.groupby("date")["usd"].sum()
    daily_scl = scl.groupby("date")["usd"].sum()
    daily_dbt = dbt.groupby("date")["usd"].sum()

    daily_total = {}
    for d in all_dates:
        p = daily_pri.get(d, 0)
        s = daily_scl.get(d, 0)
        b = daily_dbt.get(d, 0)
        daily_total[d] = p + s + b

    dr = pd.Series(daily_total)
    dr.index = pd.to_datetime(dr.index)

    # Weekly P&L
    wk_pnl = {}
    for d, v in daily_total.items():
        ts = pd.Timestamp(d); iso = ts.isocalendar()
        wk = (int(iso.year), int(iso.week))
        wk_pnl[wk] = wk_pnl.get(wk, 0) + v

    wks = pd.Series(wk_pnl)
    total_trading_wks = int(8.1 * 52)

    # Equity curve
    equity = dr.cumsum()

    # Core stats
    daily_ret = dr / ACCT
    joint_sh  = daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(252)

    dr_IS  = dr[dr.index.date <= IS_END  ]  # type: ignore
    dr_OOS = dr[dr.index.date >= OS_START]  # type: ignore

    sh_is  = (dr_IS/ACCT).mean()  / (dr_IS/ACCT).std(ddof=1)  * np.sqrt(252) if len(dr_IS)>1  else 0
    sh_oos = (dr_OOS/ACCT).mean() / (dr_OOS/ACCT).std(ddof=1) * np.sqrt(252) if len(dr_OOS)>1 else 0

    ann_usd    = dr.sum() / 8.1
    ann_ret    = ann_usd / ACCT * 100
    max_dd_usd = max_drawdown(equity)
    max_dd_pct = max_dd_usd / ACCT * 100

    sig_wks   = len(wks)
    win_wks   = (wks > 0).sum()
    zero_wks  = total_trading_wks - sig_wks

    trade_days_yr = len(all_dates) / 8.1
    total_trades_yr = (len(pri) + len(scl) + len(dbt)) / 8.1

    print(f"\n  SIGNAL COVERAGE")
    print(f"  {'─'*60}")
    print(f"  Primary days/yr       : {len(pri)/8.1:5.1f}  ({len(pri)/8.1/52:.2f}/wk)")
    print(f"  Scalp days/yr         : {len(scl)/8.1:5.1f}  ({len(scl)/8.1/52:.2f}/wk)")
    print(f"  Double Test days/yr   : {len(dbt)/8.1:5.1f}  ({len(dbt)/8.1/52:.2f}/wk)")
    print(f"  Total unique days/yr  : {trade_days_yr:5.1f}  ({trade_days_yr/52:.2f}/wk)")
    print(f"  Total signals/yr      : {total_trades_yr:5.1f}  ({total_trades_yr/52:.2f}/wk)")
    print(f"  Weeks with ≥1 signal  : {sig_wks:4d}/{total_trading_wks}  ({sig_wks/total_trading_wks*100:.0f}%)")
    print(f"  Zero-signal weeks     : {zero_wks:4d}  ({zero_wks/total_trading_wks*100:.0f}%)")

    print(f"\n  PERFORMANCE (full 2018-2026)")
    print(f"  {'─'*60}")
    print(f"  Joint Sharpe          : {joint_sh:+.2f}")
    print(f"  IS  Sharpe (2018-21)  : {sh_is:+.2f}")
    print(f"  OOS Sharpe (2022-26)  : {sh_oos:+.2f}")
    print(f"  Annual P&L            : ${ann_usd:+,.0f}  ({ann_ret:+.1f}% on $25k)")
    print(f"  Max Drawdown          : ${max_dd_usd:,.0f}  ({max_dd_pct:.1f}% of account)")
    print(f"  Win weeks (of active) : {win_wks}/{sig_wks}  ({win_wks/sig_wks*100:.0f}%)")
    print(f"  Avg P&L per signal wk : ${wks.mean():+.0f}")
    print(f"  Avg P&L per signal day: ${dr.mean():+.0f}")

    print(f"\n  RISK SIZING BREAKDOWN (max single-day risk)")
    print(f"  {'─'*60}")
    print(f"  Primary  (5%)         : $1,250 / trade")
    print(f"  Scalp    (1%)         : $250 / trade")
    print(f"  Dbl Test (2%)         : $500 / trade")
    print(f"  Max combined/day      : $2,000  (8% account) if all 3 fire")
    print(f"  Typical combined/day  : ~$750–1,250 (most days 1–2 systems)")

    # ── STEP 3: YEAR-BY-YEAR ─────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  STEP 3 — YEAR-BY-YEAR BREAKDOWN")
    print("=" * W)
    print(f"\n  {'Year':<6}  {'P signal':>9}  {'S signal':>9}  {'D signal':>9}  "
          f"{'Tot/wk':>7}  {'P&L':>9}  {'Ret%':>6}  {'Sh':>6}  tag")
    print("  " + "─" * 76)

    for yr in sorted(set(list(pri["year"].unique()) +
                         list(scl["year"].unique()) +
                         list(dbt["year"].unique()))):
        py = pri[pri["year"]==yr]; sy = scl[scl["year"]==yr]; dy = dbt[dbt["year"]==yr]
        all_d_yr = sorted(set(
            list(py["date"].unique()) + list(sy["date"].unique()) + list(dy["date"].unique())
        ))
        yr_usd = (py["usd"].sum() + sy["usd"].sum() + dy["usd"].sum())
        yr_ret = yr_usd / ACCT * 100
        dr_y = {}
        for d in all_d_yr:
            dr_y[d] = (daily_pri.get(d,0) + daily_scl.get(d,0) + daily_dbt.get(d,0))
        ds_y = pd.Series(dr_y)
        sh_y = (ds_y/ACCT).mean()/(ds_y/ACCT).std(ddof=1)*np.sqrt(252) if len(ds_y)>1 else 0
        tag = "IS " if yr <= 2021 else "OOS"
        flag = "✅" if yr_usd > 0 else "❌"
        n_wks = 52 if yr < 2026 else 22  # 2026 partial year
        p_pw = len(py)/n_wks; s_pw = len(sy)/n_wks; d_pw = len(dy)/n_wks
        tot_pw = p_pw + s_pw + d_pw
        print(f"  {flag} {yr}  "
              f"{p_pw:>7.2f}/wk  {s_pw:>7.2f}/wk  {d_pw:>7.2f}/wk  "
              f"{tot_pw:>5.2f}/wk  ${yr_usd:>7,.0f}  {yr_ret:>5.1f}%  {sh_y:>5.2f}  {tag}")

    # ── STEP 4: IS vs OOS COMPARISON ─────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  STEP 4 — IS vs OOS COMPARISON (overfitting check)")
    print("=" * W)

    def period_stats(dr_sub, label):
        if not len(dr_sub): return
        ret = dr_sub / ACCT
        ann  = dr_sub.sum() / (len(dr_sub)/252) / ACCT * 100
        shr  = ret.mean()/ret.std(ddof=1)*np.sqrt(252) if ret.std()>0 else 0
        mdd  = max_drawdown(dr_sub.cumsum()) / ACCT * 100
        pos  = (dr_sub > 0).mean() * 100
        print(f"  {label:<22}  Ann={ann:+5.1f}%  Sh={shr:+.2f}  "
              f"MaxDD={mdd:.1f}%  Win_days={pos:.0f}%  n_days={len(dr_sub)}")

    period_stats(dr[dr.index.date <= IS_END],   "IS  2018-2021")
    period_stats(dr[dr.index.date >= OS_START], "OOS 2022-2026")
    period_stats(dr, "Full 2018-2026")

    overfitting_flag = ""
    if sh_oos > sh_is * 0.85:
        overfitting_flag = "✅ OOS within 15% of IS — no overfitting detected"
    elif sh_oos > sh_is * 0.70:
        overfitting_flag = "🟡 OOS ~70-85% of IS — mild degradation, normal"
    else:
        overfitting_flag = "⚠️  OOS < 70% of IS — possible overfitting"
    print(f"\n  {overfitting_flag}")

    # ── STEP 5: CONTRIBUTION BREAKDOWN ───────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  STEP 5 — CONTRIBUTION BREAKDOWN (each system's share)")
    print("=" * W)
    total_pnl = pri["usd"].sum() + scl["usd"].sum() + dbt["usd"].sum()
    print(f"\n  {'System':<20}  {'Trades':>7}  {'Total $':>9}  {'Share':>7}  "
          f"{'Avg $/trade':>12}  {'Sharpe':>7}")
    print("  " + "─" * 68)
    for name, tdf, r_col in [("Primary", pri,"usd"), ("Scalp",scl,"usd"), ("Double Test",dbt,"usd")]:
        if not len(tdf): continue
        total  = tdf[r_col].sum()
        share  = total / total_pnl * 100 if total_pnl != 0 else 0
        avg_t  = tdf[r_col].mean()
        dr_t   = tdf.groupby("date")[r_col].sum()
        sh_t   = (dr_t/ACCT).mean()/(dr_t/ACCT).std(ddof=1)*np.sqrt(252) if dr_t.std()>0 else 0
        print(f"  {name:<20}  {len(tdf):>7,d}  ${total:>8,.0f}  {share:>6.1f}%  "
              f"${avg_t:>10,.0f}  {sh_t:>6.2f}")
    print(f"  {'TOTAL':<20}  {len(pri)+len(scl)+len(dbt):>7,d}  "
          f"${total_pnl:>8,.0f}  {'100.0%':>7}")

    # ── STEP 6: FINAL SUMMARY ────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  FINAL SUMMARY — JOINT SYSTEM vs EXISTING (Primary + Scalp only)")
    print("=" * W)

    # Existing system (Primary + Scalp only)
    ex_daily = {}
    for d in sorted(set(list(pri["date"].unique()) + list(scl["date"].unique()))):
        ex_daily[d] = daily_pri.get(d,0) + daily_scl.get(d,0)
    ex_dr = pd.Series(ex_daily)
    ex_ret = ex_dr / ACCT
    ex_sh = ex_ret.mean()/ex_ret.std(ddof=1)*np.sqrt(252) if ex_ret.std()>0 else 0
    ex_ann = ex_dr.sum() / 8.1
    ex_wks = {}
    for d, v in ex_daily.items():
        ts = pd.Timestamp(d); iso = ts.isocalendar()
        wk = (int(iso.year),int(iso.week)); ex_wks[wk] = ex_wks.get(wk,0)+v
    ex_wks_s = pd.Series(ex_wks)
    ex_sig_wks = len(ex_wks_s)

    print(f"\n  {'Metric':<35}  {'Existing (P+S)':>16}  {'Joint (P+S+DT)':>16}")
    print("  " + "─" * 70)
    for metric, ex_val, jt_val in [
        ("Signals/week",            f"{(len(pri)+len(scl))/8.1/52:.2f}",
                                    f"{total_trades_yr/52:.2f}"),
        ("Unique signal days/week",  f"{len(set(list(pri['date'])+list(scl['date'])))/8.1/52:.2f}",
                                    f"{trade_days_yr/52:.2f}"),
        ("Annual P&L",              f"${ex_ann:,.0f}",
                                    f"${ann_usd:,.0f}"),
        ("Annual return on $25k",   f"{ex_ann/ACCT*100:.1f}%",
                                    f"{ann_ret:.1f}%"),
        ("Joint Sharpe",            f"{ex_sh:.2f}",
                                    f"{joint_sh:.2f}"),
        ("OOS Sharpe",              f"{sh(ex_dr[ex_dr.index >= OS_START]/ACCT):.2f}",
                                    f"{sh_oos:.2f}"),
        ("Max Drawdown",            f"${max_drawdown(ex_dr.cumsum()):,.0f}",
                                    f"${max_dd_usd:,.0f}"),
        ("Win weeks (of active)",   f"{(ex_wks_s>0).sum()}/{ex_sig_wks} ({(ex_wks_s>0).mean()*100:.0f}%)",
                                    f"{win_wks}/{sig_wks} ({win_wks/sig_wks*100:.0f}%)"),
        ("Weeks with ≥1 signal",    f"{ex_sig_wks}/{total_trading_wks} ({ex_sig_wks/total_trading_wks*100:.0f}%)",
                                    f"{sig_wks}/{total_trading_wks} ({sig_wks/total_trading_wks*100:.0f}%)"),
    ]:
        print(f"  {metric:<35}  {ex_val:>16}  {jt_val:>16}")

    # competition week projection
    print(f"\n  COMPETITION WEEK PROJECTION (June 9–13, 2026 — 5 days)")
    print(f"  {'─'*60}")
    p_pw_2026 = len(pri[pri["year"]==2026])/22  if len(pri[pri["year"]==2026]) else 0
    s_pw_2026 = len(scl[scl["year"]==2026])/22  if len(scl[scl["year"]==2026]) else 0
    d_pw_2026 = len(dbt[dbt["year"]==2026])/22  if len(dbt[dbt["year"]==2026]) else 0
    avg_pri_win = pri[pri["win"]==True]["usd"].mean() if len(pri[pri["win"]]) else 0
    avg_pri_los = pri[pri["win"]==False]["usd"].mean() if len(pri[pri["win"]==False]) else 0
    avg_scl_win = scl[scl["win"]==True]["usd"].mean() if len(scl[scl["win"]]) else 0
    avg_scl_los = scl[scl["win"]==False]["usd"].mean() if len(scl[scl["win"]==False]) else 0
    avg_dbt_win = dbt[dbt["win"]==True]["usd"].mean() if len(dbt[dbt["win"]]) else 0
    avg_dbt_los = dbt[dbt["win"]==False]["usd"].mean() if len(dbt[dbt["win"]==False]) else 0
    exp_pri = (p_pw_2026 * (pri_wr/100 * avg_pri_win + (1-pri_wr/100) * avg_pri_los))
    exp_scl = (s_pw_2026 * (scl_wr/100 * avg_scl_win + (1-scl_wr/100) * avg_scl_los))
    exp_dbt = (d_pw_2026 * (dbt_wr/100 * avg_dbt_win + (1-dbt_wr/100) * avg_dbt_los))
    print(f"  Expected primary signals/wk  : {p_pw_2026:.2f}  (2026 YTD rate)")
    print(f"  Expected scalp signals/wk    : {s_pw_2026:.2f}  (2026 YTD rate)")
    print(f"  Expected dbl-test signals/wk : {d_pw_2026:.2f}  (2026 YTD rate)")
    print(f"  Expected P&L for competition : ${exp_pri+exp_scl+exp_dbt:+,.0f}  "
          f"(primary ${exp_pri:+.0f} + scalp ${exp_scl:+.0f} + dbl ${exp_dbt:+.0f})")
    print()


if __name__ == "__main__":
    main()
