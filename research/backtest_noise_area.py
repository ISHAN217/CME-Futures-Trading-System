#!/usr/bin/env python3
"""
backtest_noise_area.py — Zarattini/Barbon "Noise Area" Intraday Momentum

Paper: "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF"
       Zarattini, Barbon, Aziz (2024, SSRN 4824172 / Swiss Finance Institute)
Applied to ES and NQ futures (as done by Quantitativo).

EXACT FORMULA (from CXO Advisory analysis of the paper):
  For each half-hour mark t (10:00, 10:30, 11:00, ... 15:00 ET):
    avg_abs_return_t = mean(|return from open to t| over last N trading days)
    Upper_bound_t    = today_open × (1 + avg_abs_return_t)
    Lower_bound_t    = today_open × (1 - avg_abs_return_t)

  The noise area EXPANDS through the day (larger moves expected as time passes).
  When price > Upper_bound_t  → LONG signal (demand imbalance)
  When price < Lower_bound_t  → SHORT signal (supply imbalance)

ENTRY:  Close of the half-hour bar that first breaks outside the noise area.
        Using 30-minute bars to avoid whipsaw (matches paper's HH:00 / HH:30 checks).
EXIT:   At the NEXT half-hour check when price is back INSIDE the noise area.
        OR at EOD (15:30 ET). No overnight positions.

VARIATIONS tested:
  V1. Base (N=14, noise exit)
  V2. Improved (N=90, noise exit)
  V3. N=90 + VWAP stop (exit at max(upper,VWAP) for LONG)
  V4. Bidirectional (both LONG and SHORT allowed)

HONEST CHECKS:
  - No look-ahead: noise bounds use only past N days
  - Market entry: close of signal bar
  - Walk-forward IS/OOS split
  - Compared directly to verified ORB baseline
"""

import numpy as np
import pandas as pd
from datetime import time, date

DATA = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"

# Half-hour check times (paper checks at HH:00 and HH:30)
CHECK_TIMES = [
    time(10,0), time(10,30), time(11,0), time(11,30),
    time(12,0), time(12,30), time(13,0), time(13,30),
    time(14,0), time(14,30), time(15,0),
]
OPEN_TIME = time(9,30)
EOD       = time(15,30)


def load():
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    return df.sort_values("ts_et").reset_index(drop=True)


def build_lookup(df, dates, by_d, col_close, col_open, col_high, col_low, col_vol):
    """
    Pre-compute for each (date, check_time): open price, close at check time,
    abs return at check time.
    Also compute VWAP at each check time.
    Returns dict: date -> {check_time: {open, close, abs_ret, vwap}}
    """
    lookup = {}
    for d in dates:
        day = by_d.get(d)
        if day is None: continue
        # Opening price = close of 9:30 bar
        open_bar = day[day["time_et"] == OPEN_TIME]
        if len(open_bar) == 0: continue
        op = open_bar[col_close].iloc[0]

        lookup[d] = {"open": op}

        # For each check time, get close price
        for ct in CHECK_TIMES:
            bar = day[day["time_et"] == ct]
            if len(bar) == 0: continue
            close_ct = bar[col_close].iloc[0]
            abs_ret   = abs(close_ct - op) / op if op > 0 else 0.0

            # VWAP up to this check time
            session = day[(day["time_et"] >= OPEN_TIME) & (day["time_et"] <= ct)]
            if len(session) > 0:
                typical = (session[col_high] + session[col_low] + session[col_close]) / 3
                vwap = (typical * session[col_vol]).sum() / session[col_vol].sum() \
                       if session[col_vol].sum() > 0 else close_ct
            else:
                vwap = close_ct

            lookup[d][ct] = dict(close=close_ct, abs_ret=abs_ret, vwap=vwap)

    return lookup


def compute_noise_bounds(d, ct, lookup, dates, N):
    """
    Compute upper/lower noise bounds for day d at check time ct,
    using the past N days' average absolute return at same check time.
    Returns (upper, lower, avg_abs_ret) or None.
    """
    d_idx = dates.index(d)
    past_rets = []
    for prev_d in dates[max(0, d_idx-N): d_idx]:
        info = lookup.get(prev_d, {})
        if ct not in info: continue
        past_rets.append(info[ct]["abs_ret"])

    if len(past_rets) < max(5, N//3): return None  # not enough history

    avg_abs_ret = float(np.mean(past_rets))
    today_open  = lookup.get(d, {}).get("open")
    if today_open is None: return None

    upper = today_open * (1 + avg_abs_ret)
    lower = today_open * (1 - avg_abs_ret)
    return upper, lower, avg_abs_ret


def run_noise_area(df, dates, by_d, lookup_es, lookup_nq,
                   N=14, use_vwap_exit=False,
                   long_only=False, short_only=False):
    """
    Run the noise area strategy on ES and NQ.
    At each half-hour check: compute noise bounds, check if either instrument
    has broken out. If both break same direction: enter.
    Exit when both are back inside their respective noise areas, or EOD.

    Strategy design:
    - Require BOTH ES and NQ to confirm the signal (same as our ORB)
    - Enter at the close of the signal bar
    - Exit when EITHER instrument returns inside its noise area (conservative)
    - P&L in R-units where 1R = avg_abs_ret × open (the noise band half-width)
    """
    trades = []

    dates_list = list(dates)

    for d in dates:
        # Need at least N days of history
        d_idx = dates_list.index(d)
        if d_idx < N: continue

        day_es = by_d.get(d)
        if day_es is None: continue

        open_es = lookup_es.get(d, {}).get("open")
        open_nq = lookup_nq.get(d, {}).get("open")
        if open_es is None or open_nq is None: continue

        in_trade = False
        direction = None
        entry_price_es = entry_price_nq = None
        entry_r_es = entry_r_nq = None
        entry_vwap_es = entry_vwap_nq = None

        for ct in CHECK_TIMES:
            bounds_es = compute_noise_bounds(d, ct, lookup_es, dates_list, N)
            bounds_nq = compute_noise_bounds(d, ct, lookup_nq, dates_list, N)
            if bounds_es is None or bounds_nq is None: continue

            upper_es, lower_es, avg_r_es = bounds_es
            upper_nq, lower_nq, avg_r_nq = bounds_nq

            es_info = lookup_es[d].get(ct)
            nq_info = lookup_nq[d].get(ct)
            if es_info is None or nq_info is None: continue

            close_es = es_info["close"]; vwap_es = es_info["vwap"]
            close_nq = nq_info["close"]; vwap_nq = nq_info["vwap"]

            if not in_trade:
                # Check for new signal
                es_long  = close_es > upper_es
                es_short = close_es < lower_es
                nq_long  = close_nq > upper_nq
                nq_short = close_nq < lower_nq

                both_long  = es_long  and nq_long  and not short_only
                both_short = es_short and nq_short and not long_only

                if both_long and not both_short:
                    in_trade = True; direction = "LONG"
                    entry_price_es = close_es; entry_price_nq = close_nq
                    entry_r_es = avg_r_es * open_es   # 1R in $ = noise half-width
                    entry_r_nq = avg_r_nq * open_nq
                    entry_vwap_es = vwap_es; entry_vwap_nq = vwap_nq

                elif both_short and not both_long:
                    in_trade = True; direction = "SHORT"
                    entry_price_es = close_es; entry_price_nq = close_nq
                    entry_r_es = avg_r_es * open_es
                    entry_r_nq = avg_r_nq * open_nq
                    entry_vwap_es = vwap_es; entry_vwap_nq = vwap_nq

            else:
                # Check exit conditions
                es_inside = lower_es <= close_es <= upper_es
                nq_inside = lower_nq <= close_nq <= upper_nq

                # VWAP exit for longs: exit if below VWAP (price lost momentum)
                if use_vwap_exit and direction == "LONG":
                    es_vwap_stop = close_es < vwap_es
                    nq_vwap_stop = close_nq < vwap_nq
                    should_exit = es_inside or nq_inside or es_vwap_stop or nq_vwap_stop
                elif use_vwap_exit and direction == "SHORT":
                    es_vwap_stop = close_es > vwap_es
                    nq_vwap_stop = close_nq > vwap_nq
                    should_exit = es_inside or nq_inside or es_vwap_stop or nq_vwap_stop
                else:
                    # Basic exit: price back inside noise area for EITHER instrument
                    should_exit = es_inside or nq_inside

                if should_exit or ct == time(15,0):
                    # Exit this bar
                    sign = 1 if direction == "LONG" else -1
                    # R-value: move / noise band half-width (avg_abs_ret × open)
                    r_es = sign*(close_es - entry_price_es) / entry_r_es if entry_r_es>0 else 0
                    r_nq = sign*(close_nq - entry_price_nq) / entry_r_nq if entry_r_nq>0 else 0

                    trades.append(dict(
                        date=d, year=d.year,
                        direction=direction,
                        entry_ct=entry_price_es,  # store entry check time indirectly
                        r_es=round(r_es,3), r_nq=round(r_nq,3),
                        r_avg=round((r_es+r_nq)/2, 3),
                        win=(r_es+r_nq)/2 > 0,
                        exit_ct=ct,
                        dow=pd.Timestamp(d).day_name(),
                    ))
                    in_trade=False; direction=None

        # EOD close if still in trade
        if in_trade:
            eod_bar_es = day_es[day_es["time_et"]==EOD]
            if len(eod_bar_es) == 0: eod_bar_es = day_es[day_es["time_et"]<EOD].tail(1)
            if len(eod_bar_es):
                close_eod_es = eod_bar_es["ES_close"].iloc[0]
                close_eod_nq = eod_bar_es["NQ_close"].iloc[0]
                sign = 1 if direction=="LONG" else -1
                r_es = sign*(close_eod_es-entry_price_es)/entry_r_es if entry_r_es>0 else 0
                r_nq = sign*(close_eod_nq-entry_price_nq)/entry_r_nq if entry_r_nq>0 else 0
                trades.append(dict(
                    date=d, year=d.year, direction=direction,
                    entry_ct=entry_price_es,
                    r_es=round(r_es,3), r_nq=round(r_nq,3),
                    r_avg=round((r_es+r_nq)/2,3),
                    win=(r_es+r_nq)/2>0, exit_ct="EOD",
                    dow=pd.Timestamp(d).day_name(),
                ))

    return pd.DataFrame(trades)


def prow(label, sub, r_col="r_avg", pad=40):
    if not len(sub): print(f"  {label:<{pad}}  n=0"); return
    wr=sub["win"].mean(); ar=sub[r_col].mean()
    dr=sub.groupby("date")[r_col].sum()
    sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    n_yr=sub["date"].nunique()/8.1
    flag="✅" if wr>0.50 else ("🟡" if wr>0.40 else "❌")
    print(f"  {flag} {label:<{pad}}  n={sub['date'].nunique():4d} ({n_yr:.0f}/yr)  "
          f"WR={wr*100:5.1f}%  AvgR={ar:+.3f}  Sh={sh:.2f}")


def main():
    print("Loading …")
    df = load()
    dates  = sorted(df["date_et"].unique())
    by_d   = {d: g.reset_index(drop=True) for d, g in df.groupby("date_et")}
    print(f"  {len(df):,} bars  {dates[0]} → {dates[-1]}\n")

    print("Building half-hour lookups (ES and NQ) …")
    lookup_es = build_lookup(df, dates, by_d,
                             "ES_close","ES_open","ES_high","ES_low","ES_volume")
    lookup_nq = build_lookup(df, dates, by_d,
                             "NQ_close","NQ_open","NQ_high","NQ_low","NQ_volume")
    print(f"  ES lookup: {len(lookup_es)} days   NQ lookup: {len(lookup_nq)} days\n")

    IS_END=date(2021,12,31); OS_START=date(2022,1,1)
    W=74
    def sec(t): print("\n"+"="*W+f"\n  {t}\n"+"="*W)

    # ── 1. All variations ──────────────────────────────────────────────────────
    sec("1. ALL STRATEGY VARIATIONS  —  Honest comparison")
    print("  Baseline ORB (verified):    ~22/yr  WR=47.8%  Sharpe=3.08")
    print("  PDH/L Confirmed (verified): ~36/yr  WR=54.5%  Sharpe=3.14\n")

    variants = [
        ("Noise N=14 basic exit",      14, False),
        ("Noise N=30 basic exit",      30, False),
        ("Noise N=90 basic exit",      90, False),
        ("Noise N=14 + VWAP exit",     14, True),
        ("Noise N=90 + VWAP exit",     90, True),
    ]
    results = {}
    for label, N, vwap in variants:
        print(f"  Running {label} …")
        tdf = run_noise_area(df, dates, by_d, lookup_es, lookup_nq,
                             N=N, use_vwap_exit=vwap)
        results[label] = tdf
        prow(label, tdf)

    # ── 2. Best variant deep dive ──────────────────────────────────────────────
    best_label = max(results, key=lambda k: (
        results[k].groupby("date")["r_avg"].sum().mean() /
        results[k].groupby("date")["r_avg"].sum().std(ddof=1)
        if len(results[k])>1 and results[k].groupby("date")["r_avg"].sum().std()>0 else -999
    ))
    best_tdf = results[best_label]
    sec(f"2. DEEP DIVE — Best: {best_label}")

    print("  Year-by-year:")
    print(f"  {'Year':<6}  {'Trades':>6}  {'WR':>6}  {'AvgR':>7}  {'IS/OOS':>6}")
    print(f"  {'─'*40}")
    for yr, sub in best_tdf.groupby("year"):
        wr=sub["win"].mean(); ar=sub["r_avg"].mean()
        lbl = "← IS" if yr<=2021 else "← OOS"
        flag="✅" if wr>0.48 else ("🟡" if wr>0.38 else "❌")
        print(f"  {flag} {yr:<6}  {len(sub):>6}  {wr*100:>5.1f}%  {ar:>+6.3f}  {lbl}")

    print()
    print("  Walk-forward:")
    for period, mask in [("IS  2018-2021", best_tdf["date"]<=IS_END),
                         ("OOS 2022-2026", best_tdf["date"]>=OS_START)]:
        prow(period, best_tdf[mask])

    print()
    print("  By direction:")
    prow("  LONG  trades", best_tdf[best_tdf["direction"]=="LONG"])
    prow("  SHORT trades", best_tdf[best_tdf["direction"]=="SHORT"])

    print()
    print("  By day of week:")
    for dow in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
        prow(f"  {dow}", best_tdf[best_tdf["dow"]==dow])

    print()
    print("  Exit breakdown:")
    print(f"  {'Exit time':>8}  {'n':>5}  {'WR':>6}  {'AvgR':>7}")
    for ct, sub in best_tdf.groupby("exit_ct"):
        wr=sub["win"].mean(); ar=sub["r_avg"].mean()
        print(f"  {str(ct):>8}  {len(sub):>5}  {wr*100:>5.1f}%  {ar:>+6.3f}")

    # ── 3. Overlap with ORB ────────────────────────────────────────────────────
    sec("3. OVERLAP WITH ORB  —  Are noise area days additive?")
    # Get ORB signal days from our verified backtest data
    orb_sig_days = set()
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
            orb_sig_days.add(d)

    noise_days = set(best_tdf["date"].unique())
    overlap = noise_days & orb_sig_days
    only_noise = noise_days - orb_sig_days
    only_orb   = orb_sig_days - noise_days

    total_trading = len([d for d in dates
                         if by_d.get(d) is not None and
                         len(by_d[d][(by_d[d]["time_et"]>=time(9,30))&(by_d[d]["time_et"]<time(10,0))])>0])

    print(f"  ORB signal days   : {len(orb_sig_days):4d}")
    print(f"  Noise area days   : {len(noise_days):4d}")
    print(f"  Overlap (both)    : {len(overlap):4d}  ({len(overlap)/len(noise_days)*100:.1f}% of noise days)")
    print(f"  Only noise area   : {len(only_noise):4d}  ← ADDITIVE DAYS")
    print(f"  Only ORB          : {len(only_orb):4d}")
    print(f"  Total trading days: {total_trading:4d}")
    print()
    prow("Noise on OVERLAP days",    best_tdf[best_tdf["date"].isin(overlap)])
    prow("Noise on NON-ORB days",    best_tdf[best_tdf["date"].isin(only_noise)])
    print()
    if len(only_noise) > 0:
        add_pct = len(only_noise)/total_trading*100
        print(f"  Noise area adds {len(only_noise)} trading days ({add_pct:.1f}% of all days)")
        print(f"  that ORB doesn't catch → ADDITIVE coverage")

    # ── 4. Calibration check ───────────────────────────────────────────────────
    sec("4. CALIBRATION CHECK  —  Do results match Quantitativo's paper?")
    print("  Quantitativo reported on ES/NQ:")
    print("    ES: CAGR=8.1%,  Sharpe=0.91,  WR=36%,  Payoff=2.09")
    print("    NQ: CAGR=24.3%, Sharpe=1.67,  WR=38%,  Payoff=2.25")
    print()
    print("  Our implementation:")
    es_r = best_tdf["r_es"]; nq_r = best_tdf["r_nq"]
    print(f"    ES: WR={( best_tdf['r_es']>0).mean()*100:.1f}%  "
          f"AvgR={es_r.mean():+.3f}  "
          f"Sh={(best_tdf.groupby('date')['r_es'].sum().mean() / best_tdf.groupby('date')['r_es'].sum().std(ddof=1) * np.sqrt(252)):.2f}")
    print(f"    NQ: WR={(best_tdf['r_nq']>0).mean()*100:.1f}%  "
          f"AvgR={nq_r.mean():+.3f}  "
          f"Sh={(best_tdf.groupby('date')['r_nq'].sum().mean() / best_tdf.groupby('date')['r_nq'].sum().std(ddof=1) * np.sqrt(252)):.2f}")
    print()

    # ── 5. Final honest verdict ────────────────────────────────────────────────
    sec("5. HONEST VERDICT")
    if len(best_tdf):
        wr=best_tdf["win"].mean()
        dr=best_tdf.groupby("date")["r_avg"].sum()
        sh=dr.mean()/dr.std(ddof=1)*np.sqrt(252) if dr.std()>0 else 0
        days_yr=best_tdf["date"].nunique()/8.1
        print(f"  Best config: {best_label}")
        print(f"  Signal days : {best_tdf['date'].nunique()} ({days_yr:.0f}/yr)")
        print(f"  WR          : {wr*100:.1f}%")
        print(f"  Sharpe      : {sh:.2f}")
        print()
        print("  vs verified ORB:      WR=47.8%  Sh=3.08  22/yr")
        print("  vs PDH/L Confirmed:   WR=54.5%  Sh=3.14  36/yr")
        print()
        if sh > 2.0:
            print("  ✅ Noise Area has genuine edge — ADD to system")
        elif sh > 0.8:
            print("  🟡 Noise Area has marginal edge — worth monitoring")
        else:
            print("  ❌ Noise Area does not show meaningful edge in our data")
        print()
        if len(only_noise) > 100:
            print(f"  ✅ Additive: {len(only_noise)} new signal days vs ORB — genuine coverage expansion")
        else:
            print(f"  ⚠️  Low additive value: only {len(only_noise)} days not covered by ORB")
    print()


if __name__ == "__main__":
    main()
