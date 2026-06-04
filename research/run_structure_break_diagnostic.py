"""
run_structure_break_diagnostic.py
==================================
Core research question:
  After how many structural breaks does a trend ACTUALLY reverse?

Method:
  For every TREND trade in the backtest:
  1. Identify the max structure_break_count reached during the hold period
  2. Identify when the first break occurred (day N of hold)
  3. Compute outcome: win/loss, final PnL, and PnL AT the moment of each break level
  4. Compute forward returns from each break level (1, 2, 3, 4+)
  5. Separately: for ALL rows in df, compute next-5d and next-10d return by
     current structure_break_count (unconditional forward return by break count)

This answers:
  - Does break_count=1 predict reversal? Or does the trend usually resume?
  - Does break_count=2 mark the actual reversal?
  - What is the expected return from the day of each break?
  - Are structure breaks in TREND regime different from breaks in TRANSITION?
"""

import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd

import config
from data_loader        import load_daily, load_4h
from features           import compute_features
from momentum_engine    import compute_momentum
from weekly_bias        import compute_weekly_bias
from regime_engine      import compute_regime
from volatility_engine  import compute_volatility_features
from intraday_4h_engine import compute_4h_features, aggregate_4h_to_daily, merge_4h_into_daily
from mean_reversion     import compute_mean_reversion
from stat_arb           import compute_stat_arb
from sentiment_layer    import compute_sentiment
from signal_engine      import compute_signals
from signal_quality_engine          import compute_signal_quality
from trend_continuation_edge_engine import compute_trend_continuation_edge
from trend_structure_engine         import compute_trend_structure
from backtest           import run_backtest


def build_df():
    df = load_daily(); df_4h_raw = load_4h()
    df = compute_features(df); df = compute_momentum(df); df = compute_weekly_bias(df)
    df = compute_regime(df); df = compute_volatility_features(df)
    for col, val in [("cwt_chop_label","CWT_UNCLEAR"),("cwt_trade_allowed",1),
                     ("cwt_size_multiplier",1.0),("cwt_confidence",0.0),
                     ("cwt_total_energy",0.0),("cwt_entropy",0.5),("cwt_energy_z",0.0),
                     ("cwt_compression_flag",0),("cwt_expansion_flag",0),
                     ("cwt_high_energy_ratio",1/3),("cwt_mid_energy_ratio",1/3),
                     ("cwt_slow_energy_ratio",1/3)]:
        if col not in df.columns: df[col] = val
    df_4h_daily = None
    if df_4h_raw is not None:
        df_4h_daily = aggregate_4h_to_daily(compute_4h_features(df_4h_raw))
    df = merge_4h_into_daily(df, df_4h_daily)
    df = compute_mean_reversion(df); df = compute_stat_arb(df); df = compute_sentiment(df)
    df = compute_signals(df); df = compute_signal_quality(df)
    df = compute_trend_continuation_edge(df)
    df = compute_trend_structure(df)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_forward_returns(df):
    """Add fwd_5d, fwd_10d per row (lookahead — research only)."""
    df = df.copy()
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date")
        cl = g["close"].values
        N  = len(cl)
        idx = g.index
        for n, col in [(5, "fwd_5d"), (10, "fwd_10d"), (20, "fwd_20d")]:
            fwd = np.full(N, np.nan)
            for i in range(N - n):
                fwd[i] = cl[i + n] / cl[i] - 1
            df.loc[idx, col] = fwd
    return df


def main():
    print("Building pipeline...")
    df = build_df()
    df = add_forward_returns(df)

    # ── Run baseline backtest ────────────────────────────────────────────────
    res = run_backtest(df)
    tl  = res["trade_log"]
    tl["entry_date"] = pd.to_datetime(tl["entry_date"])
    tl["exit_date"]  = pd.to_datetime(tl["exit_date"])

    # ── Build fast index: (symbol, date) → structure_break_count etc. ────────
    df_idx = df.set_index(["symbol", "date"])
    struct_cols = ["structure_break_count", "structure_exit_warning",
                   "structure_exit_confirmed", "break_of_structure_down",
                   "lower_low_flag", "lower_high_flag",
                   "close", "fwd_5d", "fwd_10d", "fwd_20d",
                   "portfolio_regime", "ret_5d"]

    # ── For each trade, extract hold-period structure break profile ───────────
    records = []
    for _, trade in tl.iterrows():
        sym        = trade["symbol"]
        entry_date = trade["entry_date"]
        exit_date  = trade["exit_date"]
        pnl        = trade["pnl"]
        win        = trade["win"]
        strat      = trade["strategy"]
        n_hold     = trade["holding_days"]

        # Get hold-period rows
        try:
            hold_rows = df_idx.loc[sym].loc[
                (df_idx.loc[sym].index > entry_date) &
                (df_idx.loc[sym].index <= exit_date),
                struct_cols
            ].sort_index()
        except Exception:
            continue

        if hold_rows.empty:
            records.append(dict(
                symbol=sym, strategy=strat, entry_date=entry_date,
                exit_date=exit_date, pnl=pnl, win=win, n_hold=n_hold,
                max_break_count=0,
                day_of_first_break=np.nan,
                day_of_second_break=np.nan,
                pnl_at_first_break=np.nan,
                pnl_at_second_break=np.nan,
                exit_was_structural=False,
                entry_regime=trade.get("entry_regime",""),
            ))
            continue

        bk = hold_rows["structure_break_count"].fillna(0).values
        cl = hold_rows["close"].values
        entry_price = df_idx.loc[sym, entry_date]["close"] if (sym, entry_date) in df_idx.index else np.nan

        max_bk = float(np.max(bk)) if len(bk) > 0 else 0

        # Day of first break (break_count goes from 0 → 1)
        first_bk_day = np.nan
        second_bk_day = np.nan
        pnl_at_first = np.nan
        pnl_at_second = np.nan

        for day_i, bk_val in enumerate(bk):
            if not np.isnan(first_bk_day) and not np.isnan(second_bk_day):
                break
            if np.isnan(first_bk_day) and bk_val >= 1:
                first_bk_day = day_i + 1
                if not np.isnan(entry_price) and day_i < len(cl):
                    pnl_at_first = cl[day_i] / entry_price - 1
            if not np.isnan(first_bk_day) and np.isnan(second_bk_day) and bk_val >= 2:
                second_bk_day = day_i + 1
                if not np.isnan(entry_price) and day_i < len(cl):
                    pnl_at_second = cl[day_i] / entry_price - 1

        records.append(dict(
            symbol=sym, strategy=strat, entry_date=entry_date,
            exit_date=exit_date, pnl=pnl, win=win, n_hold=n_hold,
            max_break_count=max_bk,
            day_of_first_break=first_bk_day,
            day_of_second_break=second_bk_day,
            pnl_at_first_break=pnl_at_first,
            pnl_at_second_break=pnl_at_second,
            exit_was_structural=(trade.get("exit_reason","") == "STRUCTURAL"),
            entry_regime=trade.get("entry_regime",""),
        ))

    tr = pd.DataFrame(records)
    trend_tr = tr[tr["strategy"] == "TREND"].copy()

    # ── Q1: Win rate and avg PnL by max_break_count during hold ──────────────
    print("\n" + "="*70)
    print("Q1: TRADE OUTCOMES BY MAX STRUCTURE BREAKS REACHED DURING HOLD")
    print("="*70)
    print("(All TREND trades, n={})".format(len(trend_tr)))

    bk_bins = [0, 1, 2, 3, 4, 999]
    bk_labs = ["0 (no break)", "1 break", "2 breaks", "3 breaks", "4+ breaks"]
    trend_tr["bk_bucket"] = pd.cut(
        trend_tr["max_break_count"], bins=bk_bins, labels=bk_labs,
        right=False, include_lowest=True)

    gb = trend_tr.groupby("bk_bucket", observed=True)
    summary = gb.agg(
        n          = ("pnl", "count"),
        win_rate   = ("win", "mean"),
        avg_pnl    = ("pnl", "mean"),
        med_pnl    = ("pnl", "median"),
        pct_pos    = ("pnl", lambda x: (x > 0).mean()),
        avg_hold   = ("n_hold", "mean"),
    ).round(4)
    print(summary.to_string())

    # ── Q2: PnL AT the moment of the Nth break ────────────────────────────────
    print("\n" + "="*70)
    print("Q2: WHAT WAS THE TRADE PnL AT THE MOMENT OF EACH BREAK?")
    print("    (Positive = position was winning when break occurred)")
    print("="*70)

    had_1st = trend_tr[trend_tr["max_break_count"] >= 1].copy()
    had_2nd = trend_tr[trend_tr["max_break_count"] >= 2].copy()

    print(f"\nAt 1st structural break (n={len(had_1st)}):")
    print(f"  Avg PnL at break:   {had_1st['pnl_at_first_break'].mean():.4f}  ({had_1st['pnl_at_first_break'].mean()*100:.2f}%)")
    print(f"  Pct profitable:     {(had_1st['pnl_at_first_break'] > 0).mean():.1%}")
    print(f"  Avg final PnL:      {had_1st['pnl'].mean():.4f}")
    print(f"  Win rate at end:    {had_1st['win'].mean():.1%}")
    print(f"  Day of 1st break:   {had_1st['day_of_first_break'].mean():.1f}d into trade")

    print(f"\nAt 2nd structural break (n={len(had_2nd)}):")
    print(f"  Avg PnL at break:   {had_2nd['pnl_at_second_break'].mean():.4f}  ({had_2nd['pnl_at_second_break'].mean()*100:.2f}%)")
    print(f"  Pct profitable:     {(had_2nd['pnl_at_second_break'] > 0).mean():.1%}")
    print(f"  Avg final PnL:      {had_2nd['pnl'].mean():.4f}")
    print(f"  Win rate at end:    {had_2nd['win'].mean():.1%}")
    print(f"  Day of 2nd break:   {had_2nd['day_of_second_break'].mean():.1f}d into trade")

    # ── Q3: After the 1st break — did the trade recover or worsen? ────────────
    print("\n" + "="*70)
    print("Q3: AFTER 1ST BREAK — DID TRADES RECOVER OR CONTINUE TO LOSE?")
    print("="*70)
    had_1st["recovered"] = (had_1st["pnl"] > had_1st["pnl_at_first_break"]) & (had_1st["win"])
    had_1st["deteriorated"] = had_1st["pnl"] < had_1st["pnl_at_first_break"]
    had_1st["win_from_break"] = had_1st["pnl"] > 0

    print(f"  Recovered further after 1st break:  {had_1st['recovered'].mean():.1%}  (trade ended profitable AND gained vs break level)")
    print(f"  Ended profitable overall:            {had_1st['win_from_break'].mean():.1%}")
    print(f"  Deteriorated vs break level:         {had_1st['deteriorated'].mean():.1%}")

    print(f"\n  By 1st-break profit state:")
    had_1st["was_profitable_at_break"] = had_1st["pnl_at_first_break"] > 0
    for is_prof, label in [(True,"Profitable at break"), (False,"Losing at break")]:
        sub = had_1st[had_1st["was_profitable_at_break"] == is_prof]
        if len(sub) == 0: continue
        print(f"    {label} (n={len(sub)}):")
        print(f"      Final win rate:        {sub['win'].mean():.1%}")
        print(f"      Avg final PnL:         {sub['pnl'].mean():.4f}")
        print(f"      % that recovered:      {sub['recovered'].mean():.1%}")
        print(f"      % that deteriorated:   {sub['deteriorated'].mean():.1%}")

    # ── Q4: Unconditional forward returns by current break_count (all rows) ───
    print("\n" + "="*70)
    print("Q4: UNCONDITIONAL FORWARD RETURNS BY CURRENT structure_break_count")
    print("    (All df rows in TREND regime with LONG bias — research lookahead)")
    print("="*70)

    trend_rows = df[
        (df["portfolio_regime"] == "TREND") &
        (df["momentum_direction"] == "LONG") &
        (df["weekly_bias"] == "LONG")
    ].copy()

    trend_rows["bk_bucket"] = pd.cut(
        trend_rows["structure_break_count"].fillna(0),
        bins=[0,1,2,3,4,999], labels=["0","1","2","3","4+"],
        right=False, include_lowest=True)

    fwd_gb = trend_rows.groupby(["symbol","bk_bucket"], observed=True).agg(
        n           = ("fwd_5d", "count"),
        fwd_5d_mean = ("fwd_5d", "mean"),
        fwd_10d_mean= ("fwd_10d","mean"),
        fwd_20d_mean= ("fwd_20d","mean"),
        hit_rate_5d = ("fwd_5d", lambda x: (x > 0).mean()),
        hit_rate_10d= ("fwd_10d",lambda x: (x > 0).mean()),
    ).round(4)
    print(fwd_gb.to_string())

    # ── Q5: By symbol ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("Q5: TRADE OUTCOMES BY BREAK COUNT AND SYMBOL")
    print("="*70)
    sym_gb = trend_tr.groupby(["symbol","bk_bucket"], observed=True).agg(
        n        = ("pnl","count"),
        win_rate = ("win","mean"),
        avg_pnl  = ("pnl","mean"),
    ).round(4)
    print(sym_gb.to_string())

    # ── Q6: Within a trend, at what break count did trades actually end badly? -
    print("\n" + "="*70)
    print("Q6: BREAK COUNT AT WHICH LOSING TRADES (PnL < 0) PEAKED")
    print("    (What was the max break count during the hold of a losing trade?)")
    print("="*70)
    losing = trend_tr[trend_tr["win"] == False]
    winning = trend_tr[trend_tr["win"] == True]

    print(f"\nLosing trades (n={len(losing)}):")
    print(losing["max_break_count"].value_counts().sort_index().to_string())
    print(f"  Mean max_break_count: {losing['max_break_count'].mean():.2f}")

    print(f"\nWinning trades (n={len(winning)}):")
    print(winning["max_break_count"].value_counts().sort_index().to_string())
    print(f"  Mean max_break_count: {winning['max_break_count'].mean():.2f}")

    # ── Q7: What is the OPTIMAL exit threshold? ──────────────────────────────
    print("\n" + "="*70)
    print("Q7: OPTIMAL EXIT — SIMULATED PnL IF WE EXITED AT BREAK COUNT N")
    print("    (Compared to actual final PnL)")
    print("="*70)

    actual_pnl = trend_tr["pnl"].sum()
    print(f"  Actual total PnL (no structural exit): {actual_pnl:.4f}")

    for exit_at in [1, 2, 3, 4]:
        sim_pnl = 0.0
        for _, row in trend_tr.iterrows():
            if row["max_break_count"] >= exit_at:
                # Would have exited at that break: use pnl_at_first or second break
                if exit_at == 1:
                    sim_pnl += row["pnl_at_first_break"] if not np.isnan(row["pnl_at_first_break"]) else row["pnl"]
                elif exit_at == 2:
                    sim_pnl += row["pnl_at_second_break"] if not np.isnan(row["pnl_at_second_break"]) else row["pnl"]
                else:
                    sim_pnl += row["pnl"]  # approximation
            else:
                sim_pnl += row["pnl"]
        print(f"  Exit at break >= {exit_at}: sim PnL = {sim_pnl:.4f}  "
              f"({'BETTER' if sim_pnl > actual_pnl else 'WORSE'} by {abs(sim_pnl-actual_pnl):.4f})")

    print("\n" + "="*70)
    print("SUMMARY ANSWER")
    print("="*70)
    # Compute key answer
    wr_by_bk = trend_tr.groupby("bk_bucket", observed=True)["win"].mean()
    pnl_by_bk = trend_tr.groupby("bk_bucket", observed=True)["pnl"].mean()
    n_by_bk = trend_tr.groupby("bk_bucket", observed=True)["pnl"].count()
    for bk in wr_by_bk.index:
        print(f"  {bk:14s}: n={n_by_bk[bk]:3d}  WR={wr_by_bk[bk]:.1%}  AvgPnL={pnl_by_bk[bk]:.4f}")


if __name__ == "__main__":
    main()
