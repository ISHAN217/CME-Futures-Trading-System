#!/usr/bin/env python3
"""Live diagnostic for today's session — June 3, 2026."""
import pandas as pd
from datetime import date, time

TODAY = date(2026, 6, 3)
PREV  = date(2026, 6, 2)


def main():
    # ── ES/NQ ─────────────────────────────────────────────────────────────────
    df = pd.read_parquet("output/mtf/ES_NQ_1min_aligned.parquet")
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time

    today_es = df[df["date_et"] == TODAY].copy()
    prev_es  = df[df["date_et"] == PREV].copy()

    pr_rth = prev_es[(prev_es["time_et"] >= time(9,30)) &
                     (prev_es["time_et"] <= time(16,0))]
    PDH = pr_rth["ES_high"].max(); PDL = pr_rth["ES_low"].min()
    prev_close = pr_rth["ES_close"].iloc[-1]
    prev_open  = pr_rth["ES_open"].iloc[0]
    prev_bear  = prev_close < prev_open

    or_b = today_es[(today_es["time_et"] >= time(9,30)) &
                    (today_es["time_et"] <  time(10,0))]
    OR_H = or_b["ES_high"].max(); OR_L = or_b["ES_low"].min(); OR_R = OR_H - OR_L

    rth_now = today_es[today_es["time_et"] >= time(9,30)]
    cur      = rth_now.iloc[-1] if len(rth_now) else None
    sess_H   = rth_now["ES_high"].max()  if len(rth_now) else 0
    sess_L   = rth_now["ES_low"].min()   if len(rth_now) else 0
    last_t   = cur["ts_et"].strftime("%H:%M ET") if cur is not None else "—"
    cur_p    = cur["ES_close"] if cur is not None else 0

    # Prior week levels
    df["iso_wk"] = df["ts_et"].apply(lambda x: (int(x.isocalendar().year),
                                                  int(x.isocalendar().week)))
    pw22 = df[(df["ts_et"].dt.isocalendar().week == 22) &
              (df["ts_et"].dt.year == 2026)]
    PWH  = pw22["ES_high"].max() if len(pw22) else None
    PWL  = pw22["ES_low"].min()  if len(pw22) else None

    W = 66
    print("=" * W)
    print(f"  TODAY'S DIAGNOSTIC — Wednesday June 3, 2026  ({last_t})")
    print("=" * W)

    print(f"\n  ── ES KEY LEVELS ──────────────────────────────────────")
    print(f"  PDH        : {PDH:.2f}  (+{cur_p-PDH:+.2f}pt from current)")
    print(f"  PDL        : {PDL:.2f}  ({cur_p-PDL:+.2f}pt from current)")
    print(f"  PWH        : {PWH:.2f}  (+{cur_p-PWH:+.2f}pt from current)" if PWH else "  PWH: —")
    print(f"  PWL        : {PWL:.2f}  ({cur_p-PWL:+.2f}pt from current)" if PWL else "  PWL: —")
    print(f"  OR High    : {OR_H:.2f}  ({cur_p-OR_H:+.2f}pt)")
    print(f"  OR Low     : {OR_L:.2f}  ({cur_p-OR_L:+.2f}pt)")
    print(f"  Current    : {cur_p:.2f}  (sess H={sess_H:.2f}  L={sess_L:.2f})")
    print(f"  Prev close : {prev_close:.2f}  ({'BEARISH' if prev_bear else 'BULLISH'} prev day)")

    print(f"\n  ── SIGNAL STATUS ──────────────────────────────────────")

    # 1. Primary ORB
    nq_or = today_es[(today_es["time_et"] >= time(9,30)) &
                     (today_es["time_et"] <  time(10,0))]
    NQ_R = nq_or["NQ_high"].max() - nq_or["NQ_low"].min() if len(nq_or) else 0
    print(f"\n  1. PRIMARY ORB")
    print(f"     ES OR range: {OR_R:.1f}pt  (need 8-60pt) → {'✅' if 8<=OR_R<=60 else '❌'}")
    print(f"     NQ OR range: {NQ_R:.1f}pt  (need 30-175pt) → {'✅' if 30<=NQ_R<=175 else '❌ TOO WIDE'}")
    print(f"     → STATUS: ⛔ NO TRADE (NQ {NQ_R:.0f}pt > 175pt max)")

    # 2. Scalp — check if any levels were touched 10:30-now
    scalp_w = today_es[(today_es["time_et"] >= time(10,30)) &
                       (today_es["time_et"] <= time(14,30))]

    print(f"\n  2. MTF SCALP (10:30–14:30 ET)")
    if not len(scalp_w):
        print(f"     Window not yet open. Levels to watch:")
        print(f"     SHORT at resistance: PDH {PDH:.2f} | PWH {PWH:.2f}" if PWH else
              f"     SHORT at resistance: PDH {PDH:.2f}")
        print(f"     LONG  at support   : PDL {PDL:.2f} | PWL {PWL:.2f}" if PWL else
              f"     LONG  at support   : PDL {PDL:.2f}")
    else:
        print(f"     {len(scalp_w)} bars scanned so far")
        # Check each level
        lvls_res = [(PDH,"PDH")] + ([(PWH,"PWH")] if PWH else [])
        lvls_sup = [(PDL,"PDL")] + ([(PWL,"PWL")] if PWL else [])
        fired = False
        prev_c = None
        for _, bar in scalp_w.iterrows():
            if prev_c is None: prev_c = bar["ES_close"]; continue
            for lv, nm in lvls_res:
                if prev_c < lv-0.25 and bar["ES_high"] >= lv and bar["ES_close"] <= lv-1.0:
                    ep = bar["ES_close"]
                    print(f"     ✅ SCALP SHORT fired at {bar['time_et']}!")
                    print(f"        Level: {nm} = {lv:.2f}")
                    print(f"        Entry: ~{ep:.2f}  Stop: {bar['ES_high']+0.25:.2f}")
                    # Sim result
                    fwd = scalp_w[scalp_w["time_et"] > bar["time_et"]]
                    stop = bar["ES_high"] + 0.25 + 3.0
                    best = ep; trail_stop = stop
                    result = "open"
                    for _, fb in fwd.iterrows():
                        best = min(best, fb["ES_close"])
                        gain = ep - best - 0.5
                        if gain >= 3.0: trail_stop = min(trail_stop, best + 3.0)
                        if fb["ES_high"] >= trail_stop:
                            result = f"EXIT at {trail_stop:.2f} = {ep-trail_stop-0.5:+.2f}pt"
                            break
                    print(f"        Result so far: {result}")
                    fired = True; break
            if fired: break
            for lv, nm in lvls_sup:
                if prev_c > lv+0.25 and bar["ES_low"] <= lv and bar["ES_close"] >= lv+1.0:
                    ep = bar["ES_close"]
                    print(f"     ✅ SCALP LONG fired at {bar['time_et']}!")
                    print(f"        Level: {nm} = {lv:.2f}")
                    print(f"        Entry: ~{ep:.2f}  Stop: {bar['ES_low']-0.25:.2f}")
                    fired = True; break
            prev_c = bar["ES_close"]
            if fired: break
        if not fired:
            print(f"     No scalp signal yet. Monitoring:")
            print(f"     SHORT watch: PDH {PDH:.2f} (cur {cur_p:.2f}, "
                  f"{PDH-cur_p:.1f}pt away)")
            print(f"     LONG  watch: PDL {PDL:.2f} (cur {cur_p:.2f}, "
                  f"{cur_p-PDL:.1f}pt away)")

    # 3. Double Test
    print(f"\n  3. DOUBLE TEST (10:30–14:00 ET)")
    if not len(scalp_w):
        print(f"     Window not yet open")
    else:
        # Look for first rejection at PDH or PDL
        first_test_res = None; first_test_sup = None
        prev_c = None
        for _, bar in scalp_w.iterrows():
            if prev_c is None: prev_c = bar["ES_close"]; continue
            if first_test_res is None:
                if bar["ES_high"] >= PDH and bar["ES_close"] <= PDH - 0.5:
                    first_test_res = (bar["time_et"], bar["ES_close"])
                    print(f"     First rejection at PDH {PDH:.2f}: "
                          f"{bar['time_et']}  close={bar['ES_close']:.2f}")
            if first_test_sup is None:
                if bar["ES_low"] <= PDL and bar["ES_close"] >= PDL + 0.5:
                    first_test_sup = (bar["time_et"], bar["ES_close"])
                    print(f"     First rejection at PDL {PDL:.2f}: "
                          f"{bar['time_et']}  close={bar['ES_close']:.2f}")
            prev_c = bar["ES_close"]
        if first_test_res is None and first_test_sup is None:
            print(f"     No first rejection yet. Watching PDH {PDH:.2f} & PDL {PDL:.2f}")

    # 4. Gap + ORB
    print(f"\n  4. GAP + ORB")
    gap_pct = (or_b.iloc[0]["ES_open"] - prev_close) / prev_close * 100 if len(or_b) else 0
    print(f"     Overnight gap: {gap_pct:+.2f}% ({('DOWN' if gap_pct<0 else 'UP')})")
    print(f"     NQ too wide → Gap+ORB also ⛔ NO TRADE")

    # 5. CL
    print(f"\n  5. CL PWH/L (10:00–11:30 ET)")
    try:
        df_cl = pd.read_parquet("output/mtf/CL_1min_continuous.parquet")
        df_cl["ts_et"]   = pd.to_datetime(df_cl["ts_et"])
        df_cl["date_et"] = df_cl["ts_et"].dt.date
        df_cl["time_et"] = df_cl["ts_et"].dt.time

        # Prior week CL (week 22 = May 26-30)
        pw22_cl = df_cl[(df_cl["ts_et"].dt.isocalendar().week == 22) &
                        (df_cl["ts_et"].dt.year == 2026)]
        CL_PWH = pw22_cl["CL_high"].max(); CL_PWL = pw22_cl["CL_low"].min()

        cl_today = df_cl[df_cl["date_et"] == TODAY]
        cl_or_t  = cl_today[cl_today["time_et"].between(time(9,0), time(9,59))]
        cl_entry = cl_today[cl_today["time_et"].between(time(10,0), time(11,30))]
        cl_close = cl_today[cl_today["time_et"].between(time(10,0), time(12,0))]

        if len(cl_or_t):
            CL_OR_H = cl_or_t["CL_high"].max(); CL_OR_L = cl_or_t["CL_low"].min()
            CL_OR_R = CL_OR_H - CL_OR_L
            print(f"     CL PWH (prior wk): {CL_PWH:.2f}  PWL: {CL_PWL:.2f}")
            print(f"     CL OR (9-10 ET):   H={CL_OR_H:.2f}  L={CL_OR_L:.2f}  R={CL_OR_R:.3f}")
        else:
            print(f"     CL PWH: {CL_PWH:.2f}  PWL: {CL_PWL:.2f}")
            print(f"     CL OR: no data for 9-10 ET today")
            CL_OR_R = 1.0  # fallback

        if len(cl_entry):
            touched_H = cl_entry["CL_high"] >= CL_PWH
            touched_L = cl_entry["CL_low"]  <= CL_PWL
            cur_cl    = cl_entry.iloc[-1]["CL_close"]
            last_cl_t = cl_entry.iloc[-1]["ts_et"].strftime("%H:%M ET")

            if touched_H.any():
                first_h = cl_entry[touched_H].iloc[0]
                ep = CL_PWH
                tp = ep + 4*CL_OR_R; sp = ep - CL_OR_R
                # Simulate
                fwd_cl = cl_close[cl_close["time_et"] > first_h["time_et"]]
                pnl_pts = 0.0; result = "open"
                for _, fb in fwd_cl.iterrows():
                    if fb["time_et"] >= time(12,0):
                        pnl_pts = fb["CL_close"] - ep; result = "CLOSE 12:00"; break
                    if fb["CL_low"] <= sp: pnl_pts = sp - ep; result = "STOP"; break
                    if fb["CL_high"] >= tp: pnl_pts = tp - ep; result = "TARGET 4R"; break
                else:
                    pnl_pts = cur_cl - ep; result = "still open"
                pnl_5mcl = pnl_pts * 100 * 5
                flag = "✅" if pnl_pts > 0 else "❌"
                print(f"     {flag} CL LONG signal fired at {first_h['ts_et'].strftime('%H:%M ET')}!")
                print(f"        Entry: {ep:.2f}  Stop: {sp:.2f}  Target: {tp:.2f}")
                print(f"        Result: {result}  P&L: {pnl_pts:+.3f}pt "
                      f"= ${pnl_5mcl:+.0f} (5 MCL)")
            elif touched_L.any():
                first_l = cl_entry[touched_L].iloc[0]
                ep = CL_PWL; tp = ep - 4*CL_OR_R; sp = ep + CL_OR_R
                fwd_cl = cl_close[cl_close["time_et"] > first_l["time_et"]]
                pnl_pts = 0.0; result = "open"
                for _, fb in fwd_cl.iterrows():
                    if fb["time_et"] >= time(12,0):
                        pnl_pts = ep - fb["CL_close"]; result = "CLOSE 12:00"; break
                    if fb["CL_high"] >= sp: pnl_pts = ep - sp; result = "STOP"; break
                    if fb["CL_low"] <= tp: pnl_pts = ep - tp; result = "TARGET 4R"; break
                else:
                    pnl_pts = ep - cur_cl; result = "still open"
                pnl_5mcl = pnl_pts * 100 * 5
                flag = "✅" if pnl_pts > 0 else "❌"
                print(f"     {flag} CL SHORT signal fired at {first_l['ts_et'].strftime('%H:%M ET')}!")
                print(f"        Entry: {ep:.2f}  Stop: {sp:.2f}  Target: {tp:.2f}")
                print(f"        Result: {result}  P&L: {pnl_pts:+.3f}pt "
                      f"= ${pnl_5mcl:+.0f} (5 MCL)")
            else:
                print(f"     Entry window scanned ({len(cl_entry)} bars), "
                      f"no PWH/PWL touch. CL at {cur_cl:.2f} ({last_cl_t})")
                print(f"     PWH={CL_PWH:.2f} (need price ≥ that)  "
                      f"PWL={CL_PWL:.2f} (need price ≤ that)")
        else:
            print(f"     Entry window (10-11:30 ET) data not yet available")
            if len(cl_today):
                lat = cl_today.iloc[-1]
                print(f"     Latest CL bar: {lat['CL_close']:.2f} at "
                      f"{lat['ts_et'].strftime('%H:%M ET')}")
    except Exception as e:
        print(f"     CL check error: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  TODAY'S SUMMARY ({last_t})")
    print("=" * W)
    print(f"""
  ORB Primary  : ⛔ NO TRADE  (NQ {NQ_R:.0f}pt, too wide)
  Gap + ORB    : ⛔ NO TRADE  (requires valid NQ OR)
  MTF Scalp    : {'monitoring' if len(scalp_w) else 'window opens 10:30 ET'}
  Double Test  : {'monitoring' if len(scalp_w) else 'window opens 10:30 ET'}
  CL PWH/L     : see above
  """)


if __name__ == "__main__":
    main()
