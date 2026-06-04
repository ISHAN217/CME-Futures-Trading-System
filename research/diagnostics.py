"""
diagnostics.py — Competition system reports, charts, and console summary.

V3 additions (marked [V3]):
    - Stop loss / trail stop / fast exit breakdown in console
    - Extension count and extended win rate
    - Add-to-winner count in console
    - exit_reason_report.csv    [V3] P&L and win rate by exit reason
    - trade_extension_report.csv [V3] TREND trades that ran past day 3
    - winner_add_report.csv      [V3] trades that got an add-to-winner
    - competition_week_report.csv [V3] rolling 5-day and 7-day windows
    - exit_reason_returns.png    [V3] box plot of returns by exit reason
    - trade_hold_distribution.png [V3] histogram of holding days
    - winner_extension_returns.png [V3] extended vs non-extended TREND
    - rolling_5day_returns.png   [V3] 5-day rolling return (competition view)

CSVs produced:
    trade_log.csv               every completed trade
    strategy_performance.csv    win rates and P&L by strategy
    component_accuracy.csv      signal component predictive accuracy
    signal_reason_report.csv    P&L and win rate per signal_reason
    quality_score_report.csv    win rate by h4_quality level
    momentum_accuracy.csv       momentum signal forward accuracy
    symbol_performance.csv      ES and NQ breakdown
    exit_reason_report.csv      [V3] P&L by exit mechanism
    trade_extension_report.csv  [V3] TREND extension trades
    winner_add_report.csv       [V3] add-to-winner trades
    competition_week_report.csv [V3] 5/7-day rolling returns

Charts produced:
    equity_curve.png            portfolio cumulative return
    drawdown.png                drawdown over time
    strategy_contribution.png   cumulative P&L by strategy (4 lines)
    trades_per_strategy.png     trade counts and win rates
    signal_reason.png           top signal reasons by P&L
    spread_zscore.png           ES/NQ spread with SA entries
    weekly_bias_accuracy.png    bias vs forward return
    quality_win_rate.png        entry quality vs win rate
    rolling_sharpe.png          30-day rolling Sharpe
    rolling_5day_returns.png    [V3] 5-day rolling return
    trade_hold_distribution.png [V3] holding days histogram
    winner_extension_returns.png [V3] extended vs non-extended P&L
    exit_reason_returns.png     [V3] P&L by exit reason
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _print_cwt_console_summary(trade_log: pd.DataFrame, df_signals: pd.DataFrame) -> None:
    """Print CWT filter performance summary to console."""
    if trade_log is None or len(trade_log) == 0:
        return
    # Merge cwt label from df_signals if not in trade_log
    tl = trade_log.copy()
    if "cwt_chop_label" not in tl.columns and df_signals is not None:
        sig = df_signals[["date", "symbol", "cwt_chop_label"]].rename(columns={"date": "entry_date"})
        tl = tl.merge(sig, on=["entry_date", "symbol"], how="left")
    tl["cwt_chop_label"] = tl.get("cwt_chop_label", pd.Series(dtype=str)).fillna("CWT_UNCLEAR")

    print("\n  ── CWT Chop Filter Summary ───────────────────────────────")
    for label in sorted(tl["cwt_chop_label"].unique()):
        grp = tl[tl["cwt_chop_label"] == label]
        wr  = grp["win"].mean() * 100 if "win" in grp.columns else float("nan")
        tot = grp["pnl"].sum() * 100
        sl  = (grp["exit_reason"] == "STOP_LOSS").sum() if "exit_reason" in grp.columns else 0
        print(f"  {label:<30} n={len(grp):3d}  win={wr:5.1f}%  pnl={tot:+.3f}%  SL={sl}")

    if df_signals is not None and "signal_reason" in df_signals.columns:
        n_blocked = df_signals["signal_reason"].str.contains("BLOCKED_CWT", na=False).sum()
        print(f"  Signals blocked by CWT: {n_blocked}")


def _save_cwt_diagnostics(trade_log: pd.DataFrame, df_signals: pd.DataFrame) -> None:
    """Save CWT diagnostic CSVs."""
    if trade_log is None or len(trade_log) == 0:
        return

    tl = trade_log.copy()
    if "cwt_chop_label" not in tl.columns and df_signals is not None and "cwt_chop_label" in df_signals.columns:
        sig_cols = ["date", "symbol", "cwt_chop_label", "cwt_size_multiplier", "cwt_entropy", "cwt_energy_z", "cwt_confidence"]
        sig_cols = [c for c in sig_cols if c in df_signals.columns]
        sig = df_signals[sig_cols].rename(columns={"date": "entry_date"})
        tl = tl.merge(sig, on=["entry_date", "symbol"], how="left")
    if "cwt_chop_label" not in tl.columns:
        tl["cwt_chop_label"] = "CWT_UNCLEAR"
    tl["cwt_chop_label"] = tl["cwt_chop_label"].fillna("CWT_UNCLEAR")

    out = Path(config.OUTPUT_DIR)

    # 1. Label distribution
    ld = tl.groupby("cwt_chop_label").agg(
        trade_count=("pnl", "count"),
        win_rate   =("win", "mean"),
        total_pnl  =("pnl", "sum"),
        avg_pnl    =("pnl", "mean"),
        stop_losses=("exit_reason", lambda x: (x == "STOP_LOSS").sum()),
    ).round(6)
    ld.to_csv(out / "cwt_label_distribution.csv")
    logger.info("Saved cwt_label_distribution.csv")

    # 2. Performance by label (same as ld but with more columns if available)
    ld.to_csv(out / "cwt_trade_performance_by_label.csv")
    logger.info("Saved cwt_trade_performance_by_label.csv")

    # 3. Strategy x label
    if "strategy" in tl.columns:
        sl = tl.groupby(["strategy", "cwt_chop_label"]).agg(
            trades   =("pnl", "count"),
            win_rate =("win", "mean"),
            total_pnl=("pnl", "sum"),
            avg_pnl  =("pnl", "mean"),
        ).round(6)
        sl.to_csv(out / "cwt_strategy_label_breakdown.csv")
        logger.info("Saved cwt_strategy_label_breakdown.csv")

    # 4. Blocked trades
    if df_signals is not None and "signal_reason" in df_signals.columns:
        blocked = df_signals[df_signals["signal_reason"].str.contains("BLOCKED_CWT", na=False)].copy()
        if len(blocked):
            b_rpt = blocked.groupby("signal_reason").size().reset_index(name="count")
            b_rpt["note"] = "Counterfactual P&L requires unfiltered baseline run"
            b_rpt.to_csv(out / "cwt_blocked_trade_report.csv", index=False)
        else:
            pd.DataFrame({"message": ["No trades blocked"]}).to_csv(out / "cwt_blocked_trade_report.csv", index=False)
        logger.info("Saved cwt_blocked_trade_report.csv")

    # 5. CHOP / TRANSITION report
    if "entry_regime" in tl.columns:
        ct = tl[tl["entry_regime"].isin(["CHOP", "TRANSITION"])]
        if len(ct):
            ctr = ct.groupby(["entry_regime", "cwt_chop_label"]).agg(
                trades   =("pnl", "count"),
                win_rate =("win", "mean"),
                total_pnl=("pnl", "sum"),
                avg_pnl  =("pnl", "mean"),
            ).round(6)
            ctr.to_csv(out / "cwt_chop_transition_report.csv")
            logger.info("Saved cwt_chop_transition_report.csv")


def run_diagnostics(
    portfolio_returns,
    symbol_returns,
    strategy_returns,
    positions,
    df_signals,
    trade_log,
    pair_trades,
    stats,
    daily_records=None,   # [V3] per-day position detail DataFrame
) -> None:
    out = Path(config.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    _print_console_summary(stats, strategy_returns, portfolio_returns)

    _save_trade_log(trade_log, out)
    _save_strategy_performance(trade_log, out)
    _save_component_accuracy(df_signals, trade_log, out)
    _save_signal_reason_report(trade_log, out)
    _save_quality_score_report(trade_log, out)
    _save_momentum_accuracy(df_signals, out)
    _save_symbol_performance(trade_log, out)

    # [V3] New reports
    _save_exit_reason_report(trade_log, out)
    _save_trade_extension_report(trade_log, out)
    _save_winner_add_report(trade_log, out)
    _save_competition_week_report(portfolio_returns, out)

    _plot_equity(portfolio_returns, out)
    _plot_drawdown(portfolio_returns, out)
    _plot_rolling_sharpe(portfolio_returns, out)
    _plot_strategy_contribution(strategy_returns, out)
    _plot_trades_per_strategy(trade_log, out)
    _plot_signal_reasons(trade_log, out)
    _plot_spread_zscore(df_signals, trade_log, out)
    _plot_weekly_bias_accuracy(df_signals, out)
    _plot_quality_win_rate(trade_log, out)

    # [V3] New charts
    _plot_rolling_5day_returns(portfolio_returns, out)
    _plot_hold_distribution(trade_log, out)
    _plot_winner_extension_returns(trade_log, out)
    _plot_exit_reason_returns(trade_log, out)

    # CWT diagnostics [NEW]
    if getattr(config, "USE_CWT_CHOP_FILTER", False):
        _save_cwt_diagnostics(trade_log, df_signals)
        _print_cwt_console_summary(trade_log, df_signals)


# ── Console summary ───────────────────────────────────────────────────────────

def _print_console_summary(stats, strategy_returns, portfolio_returns) -> None:
    print("\n" + "="*70)
    print("CME COMPETITION SYSTEM — BACKTEST SUMMARY")
    print("="*70)
    print(f"  Period:          {portfolio_returns.index[0].date()} → {portfolio_returns.index[-1].date()}")
    print(f"  Trading days:    {stats['n_days']}")
    print()
    print(f"  Total return:    {stats['total_return_pct']:+.2f}%")
    print(f"  Ann. return:     {stats['annualized_return_pct']:+.2f}%")
    print(f"  Ann. volatility: {stats['annualized_vol_pct']:.2f}%")
    print(f"  Sharpe ratio:    {stats['sharpe_ratio']:.3f}")
    print(f"  Max drawdown:    {stats['max_drawdown_pct']:.2f}%")
    print(f"  Profit factor:   {stats['profit_factor']:.2f}")
    print(f"  Hit rate:        {stats['hit_rate']*100:.1f}%")
    print(f"  Time in market:  {stats['pct_time_in_market']*100:.1f}%")
    print(f"  Recent 30d ret:  {stats['recent_30d_return_pct']:+.2f}%")
    print(f"  Recent 10d ret:  {stats['recent_10d_return_pct']:+.2f}%")
    print(f"  Avg rolling SR:  {stats['avg_rolling_30d_sharpe']:.3f}")
    print()

    print("  ── By strategy ────────────────────────────────────────────────")
    for strat in ["TREND", "PULLBACK", "MEAN_REVERT", "STAT_ARB"]:
        pnl = stats["strategy_totals"].get(strat, 0)
        wr  = stats["win_rate_by_strategy"].get(strat, 0)
        cnt = stats["trades_by_strategy"].get(strat, 0)
        print(f"  {strat:<14} {pnl*100:+.2f}%  win={wr*100:.1f}%  trades={cnt}")

    print()
    print("  ── By symbol ──────────────────────────────────────────────────")
    for sym, pnl in stats["symbol_contribution"].items():
        print(f"  {sym:<6} {pnl*100:+.2f}%")

    print()
    print("  ── Trade quality ──────────────────────────────────────────────")
    print(f"  Total trades:    {stats['total_trades']}")
    print(f"  Long:  {stats['trades_long']}  win={stats['long_win_rate']*100:.1f}%")
    print(f"  Short: {stats['trades_short']}  win={stats['short_win_rate']*100:.1f}%")
    print(f"  Avg hold:        {stats['avg_holding_days']:.1f} days")
    print(f"  Median hold:     {stats['median_holding_days']:.1f} days")
    print(f"  High-conv ({stats['high_conv_count']} trades): win={stats['high_conv_win_rate']*100:.1f}%")
    print(f"  Quality ≥4:      win={stats['quality_4plus_win_rate']*100:.1f}%")
    print()
    print("  ── V3 risk & extension ────────────────────────────────────────")
    print(f"  Stop losses:     {stats.get('stop_loss_count', 0)}")
    print(f"  Trail stops:     {stats.get('trail_stop_count', 0)}  (arm at +1.0%, fire on -0.7% from peak)")
    print(f"  Fast exits:      {stats.get('fast_exit_count', 0)}   (protect ≥+0.8% gains on conviction fade)")
    ext_cnt = stats.get('extension_count', 0)
    ext_wr  = stats.get('extended_win_rate', 0.0)
    print(f"  TREND extended:  {ext_cnt} trades past day-3  win={ext_wr*100:.1f}%")
    print(f"  Add-to-winner:   {stats.get('add_to_winner_count', 0)} positions received an add")
    print("="*70 + "\n")


# ── CSV exports ───────────────────────────────────────────────────────────────

def _save_trade_log(tl, out):
    if tl is None or tl.empty:
        logger.warning("No trades to save")
        return
    tl.to_csv(out / "trade_log.csv", index=False)
    logger.info("Saved trade_log.csv (%d trades)", len(tl))


def _save_strategy_performance(tl, out):
    if tl is None or tl.empty:
        return
    rows = []
    for strat, g in tl.groupby("strategy"):
        long_  = g[g["direction"] == "LONG"]
        short_ = g[g["direction"] == "SHORT"]
        rows.append({
            "strategy":       strat,
            "n_trades":       len(g),
            "win_rate":       round(g["win"].mean(),        4),
            "avg_pnl":        round(g["pnl"].mean(),        6),
            "total_pnl":      round(g["pnl"].sum(),         6),
            "best_trade":     round(g["pnl"].max(),         6),
            "worst_trade":    round(g["pnl"].min(),         6),
            "avg_hold_days":  round(g["holding_days"].mean(), 1),
            "n_long":         len(long_),
            "n_short":        len(short_),
            "long_win_rate":  round(long_["win"].mean(),   4) if len(long_)  else 0.0,
            "short_win_rate": round(short_["win"].mean(),  4) if len(short_) else 0.0,
        })
    pd.DataFrame(rows).to_csv(out / "strategy_performance.csv", index=False)


def _save_component_accuracy(df, tl, out):
    rows = []

    # Weekly bias accuracy
    for sym in df["symbol"].unique():
        g = df[df["symbol"] == sym].sort_values("date").copy()
        if "weekly_bias" not in g.columns:
            continue
        g["fwd_5d"] = g["ret_1d"].shift(-1).rolling(5).sum().shift(-4)
        for bias_val in ["LONG", "NEUTRAL"]:
            sub = g[g["weekly_bias"] == bias_val].dropna(subset=["fwd_5d"])
            if len(sub) < 5:
                continue
            correct = (bias_val == "LONG") & (sub["fwd_5d"] > 0)
            rows.append({
                "component":      f"weekly_bias_{bias_val}",
                "symbol":         sym,
                "n":              len(sub),
                "accuracy":       round(correct.mean(), 4) if bias_val == "LONG" else np.nan,
                "avg_fwd_return": round(sub["fwd_5d"].mean(), 5),
            })

    # 4H execution signal accuracy
    if "h4_exec_signal" in df.columns:
        for sym in df["symbol"].unique():
            g = df[df["symbol"] == sym].sort_values("date").copy()
            g["next_ret"] = g["ret_1d"].shift(-1)
            for sig in ["MOMENTUM", "PULLBACK", "BREAKOUT", "WAIT"]:
                sub = g[g["h4_exec_signal"] == sig].dropna(subset=["next_ret"])
                if len(sub) < 5:
                    continue
                rows.append({
                    "component":      f"h4_{sig.lower()}",
                    "symbol":         sym,
                    "n":              len(sub),
                    "accuracy":       round((sub["next_ret"] > 0).mean(), 4),
                    "avg_fwd_return": round(sub["next_ret"].mean(), 5),
                })

    # Regime distribution
    if "portfolio_regime" in df.columns:
        for sym in df["symbol"].unique():
            g = df[df["symbol"] == sym].sort_values("date").copy()
            g["next_ret"] = g["ret_1d"].shift(-1)
            for regime in ["TREND", "TRANSITION", "CHOP", "SHOCK"]:
                sub = g[g["portfolio_regime"] == regime].dropna(subset=["next_ret"])
                if len(sub) < 5:
                    continue
                rows.append({
                    "component":      f"regime_{regime.lower()}",
                    "symbol":         sym,
                    "n":              len(sub),
                    "accuracy":       round((sub["next_ret"] > 0).mean(), 4),
                    "avg_fwd_return": round(sub["next_ret"].mean(), 5),
                })

    # SA convergence rate
    if tl is not None and not tl.empty:
        sa = tl[tl["strategy"] == "STAT_ARB"]
        if len(sa) > 0:
            converged = sa[sa["exit_reason"] == "SIGNAL_ZERO"]
            rows.append({
                "component":      "SA_convergence",
                "symbol":         "ALL",
                "n":              len(sa),
                "accuracy":       round(len(converged) / len(sa), 4),
                "avg_fwd_return": round(sa["pnl"].mean(), 6),
            })

    pd.DataFrame(rows).to_csv(out / "component_accuracy.csv", index=False)


def _save_signal_reason_report(tl, out):
    if tl is None or tl.empty or "entry_reason" not in tl.columns:
        return
    rows = []
    for reason, g in tl.groupby("entry_reason"):
        rows.append({
            "signal_reason": reason,
            "n_trades":      len(g),
            "win_rate":      round(g["win"].mean(), 4),
            "avg_pnl":       round(g["pnl"].mean(), 6),
            "total_pnl":     round(g["pnl"].sum(),  6),
            "avg_hold":      round(g["holding_days"].mean(), 1),
        })
    pd.DataFrame(rows).sort_values("total_pnl", ascending=False).to_csv(
        out / "signal_reason_report.csv", index=False
    )


def _save_quality_score_report(tl, out):
    """Win rate and PnL by h4_quality level (1-5). [NEW]"""
    if tl is None or tl.empty or "entry_quality" not in tl.columns:
        return
    rows = []
    for q, g in tl.groupby("entry_quality"):
        rows.append({
            "quality_score": q,
            "n_trades":      len(g),
            "win_rate":      round(g["win"].mean(), 4),
            "avg_pnl":       round(g["pnl"].mean(), 6),
            "total_pnl":     round(g["pnl"].sum(),  6),
            "avg_hold":      round(g["holding_days"].mean(), 1),
        })
    pd.DataFrame(rows).to_csv(out / "quality_score_report.csv", index=False)
    logger.info("Saved quality_score_report.csv")


def _save_momentum_accuracy(df, out):
    """Momentum direction predictive accuracy vs forward returns. [NEW]"""
    if "momentum_direction" not in df.columns:
        return
    rows = []
    for sym in df["symbol"].unique():
        g = df[df["symbol"] == sym].sort_values("date").copy()
        g["fwd_5d"] = g["ret_1d"].shift(-1).rolling(5).sum().shift(-4)
        for direction in ["LONG", "NEUTRAL"]:
            sub = g[g["momentum_direction"] == direction].dropna(subset=["fwd_5d"])
            if len(sub) < 10:
                continue
            correct = (direction == "LONG") & (sub["fwd_5d"] > 0)
            rows.append({
                "symbol":           sym,
                "momentum_direction": direction,
                "n":                len(sub),
                "accuracy":         round(correct.mean(), 4) if direction == "LONG" else np.nan,
                "avg_fwd_5d":       round(sub["fwd_5d"].mean(), 5),
                "avg_mom_strength": round(sub["momentum_strength"].mean(), 4)
                                    if "momentum_strength" in sub.columns else np.nan,
            })
    pd.DataFrame(rows).to_csv(out / "momentum_accuracy.csv", index=False)
    logger.info("Saved momentum_accuracy.csv")


def _save_symbol_performance(tl, out):
    if tl is None or tl.empty:
        return
    rows = []
    for sym, g in tl.groupby("symbol"):
        rows.append({
            "symbol":    sym,
            "n_trades":  len(g),
            "win_rate":  round(g["win"].mean(), 4),
            "total_pnl": round(g["pnl"].sum(),  6),
            "avg_pnl":   round(g["pnl"].mean(), 6),
            "avg_hold":  round(g["holding_days"].mean(), 1),
        })
    pd.DataFrame(rows).to_csv(out / "symbol_performance.csv", index=False)


# ── V3 CSV reports ────────────────────────────────────────────────────────────

def _save_exit_reason_report(tl, out):
    """[V3] P&L and win rate broken down by exit mechanism."""
    if tl is None or tl.empty or "exit_reason" not in tl.columns:
        return
    rows = []
    for reason, g in tl.groupby("exit_reason"):
        rows.append({
            "exit_reason":   reason,
            "n_trades":      len(g),
            "win_rate":      round(g["win"].mean(), 4),
            "avg_pnl":       round(g["pnl"].mean(), 6),
            "total_pnl":     round(g["pnl"].sum(),  6),
            "avg_hold":      round(g["holding_days"].mean(), 1),
        })
    (pd.DataFrame(rows)
        .sort_values("total_pnl", ascending=False)
        .to_csv(out / "exit_reason_report.csv", index=False))
    logger.info("Saved exit_reason_report.csv")


def _save_trade_extension_report(tl, out):
    """[V3] Detail on TREND trades that extended past day 3."""
    if tl is None or tl.empty or "extension_used" not in tl.columns:
        return
    ext = tl[(tl["strategy"] == "TREND") & (tl["extension_used"] == True)].copy()
    if ext.empty:
        logger.info("No extended TREND trades to report")
        return
    ext.to_csv(out / "trade_extension_report.csv", index=False)
    logger.info(
        "Saved trade_extension_report.csv (%d extended trades, %.1f%% win rate)",
        len(ext), 100 * ext["win"].mean(),
    )


def _save_winner_add_report(tl, out):
    """[V3] Detail on trades that received an add-to-winner."""
    if tl is None or tl.empty or "add_count" not in tl.columns:
        return
    added = tl[tl["add_count"] > 0].copy()
    if added.empty:
        logger.info("No add-to-winner trades to report")
        return
    added.to_csv(out / "winner_add_report.csv", index=False)
    logger.info(
        "Saved winner_add_report.csv (%d trades with add, %.1f%% win rate)",
        len(added), 100 * added["win"].mean(),
    )


def _save_competition_week_report(portfolio_returns, out):
    """[V3] Rolling 5-day and 7-day return windows (competition window view)."""
    if portfolio_returns is None or len(portfolio_returns) < 5:
        return
    df = pd.DataFrame({"daily_ret": portfolio_returns})
    df["rolling_5d"]  = (1 + df["daily_ret"]).rolling(5).apply(lambda x: x.prod() - 1)
    df["rolling_7d"]  = (1 + df["daily_ret"]).rolling(7).apply(lambda x: x.prod() - 1)
    df["cumulative"]  = (1 + df["daily_ret"]).cumprod() - 1
    df = df.dropna(subset=["rolling_5d"])
    df.to_csv(out / "competition_week_report.csv")
    logger.info(
        "Saved competition_week_report.csv  "
        "avg_5d=%.2f%%  best_5d=%.2f%%  worst_5d=%.2f%%",
        df["rolling_5d"].mean() * 100,
        df["rolling_5d"].max()  * 100,
        df["rolling_5d"].min()  * 100,
    )


# ── Charts ────────────────────────────────────────────────────────────────────

def _plot_equity(portfolio_returns, out):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cumret = (1 + portfolio_returns).cumprod()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(cumret.index, (cumret - 1) * 100, color="royalblue", linewidth=1.5)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title("Competition System — Portfolio Equity Curve", fontsize=13, fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "equity_curve.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot equity: %s", e)


def _plot_drawdown(portfolio_returns, out):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cumret = (1 + portfolio_returns).cumprod()
        dd     = (cumret - cumret.cummax()) / cumret.cummax() * 100

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(dd.index, dd, 0, color="crimson", alpha=0.5)
        ax.set_title("Portfolio Drawdown", fontsize=13, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "drawdown.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot drawdown: %s", e)


def _plot_rolling_sharpe(portfolio_returns, out):
    """Rolling 30-day Sharpe — key competition consistency metric. [NEW]"""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ann = config.TRADING_DAYS_PER_YEAR
        rolling_ret = portfolio_returns.rolling(30).mean() * ann
        rolling_vol = portfolio_returns.rolling(30).std()  * np.sqrt(ann)
        rolling_sr  = (rolling_vol > 1e-9) * (rolling_ret / rolling_vol.replace(0, np.nan))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(rolling_sr.index, rolling_sr, color="darkorchid", linewidth=1.2, alpha=0.9)
        ax.axhline(0,   color="black", linewidth=0.8)
        ax.axhline(1.0, color="seagreen", linestyle="--", linewidth=0.8, alpha=0.7, label="SR=1.0")
        ax.fill_between(rolling_sr.index, rolling_sr, 0,
                        where=(rolling_sr > 0), alpha=0.2, color="seagreen")
        ax.fill_between(rolling_sr.index, rolling_sr, 0,
                        where=(rolling_sr < 0), alpha=0.2, color="crimson")
        ax.set_title("Rolling 30-day Sharpe Ratio (annualised)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "rolling_sharpe.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot rolling Sharpe: %s", e)


def _plot_strategy_contribution(strategy_returns, out):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cumret = strategy_returns.cumsum() * 100
        colors = {
            "TREND":       "royalblue",
            "PULLBACK":    "deepskyblue",
            "MEAN_REVERT": "seagreen",
            "STAT_ARB":    "darkorange",
        }

        fig, ax = plt.subplots(figsize=(12, 5))
        for col in cumret.columns:
            ax.plot(cumret.index, cumret[col], label=col,
                    color=colors.get(col, "grey"), linewidth=1.3)
        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
        ax.set_title("Cumulative P&L by Strategy", fontsize=13, fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "strategy_contribution.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot strategy contribution: %s", e)


def _plot_trades_per_strategy(tl, out):
    if tl is None or tl.empty:
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        strats = tl.groupby("strategy").agg(n=("win", "count"), wr=("win", "mean"))
        colors = ["royalblue", "deepskyblue", "seagreen", "darkorange"][:len(strats)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        strats["n"].plot(kind="bar", ax=ax1, color=colors, rot=20)
        ax1.set_title("Trade Count by Strategy"); ax1.set_ylabel("Trades")

        (strats["wr"] * 100).plot(kind="bar", ax=ax2, color=colors, rot=20)
        ax2.axhline(50, color="grey", linestyle="--", linewidth=0.8)
        ax2.set_title("Win Rate by Strategy (%)"); ax2.set_ylabel("Win Rate (%)")

        plt.tight_layout()
        plt.savefig(out / "trades_per_strategy.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot trades per strategy: %s", e)


def _plot_signal_reasons(tl, out):
    if tl is None or tl.empty or "entry_reason" not in tl.columns:
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        by_reason = (tl.groupby("entry_reason")["pnl"].sum() * 100).sort_values().tail(15)
        colors = ["crimson" if v < 0 else "seagreen" for v in by_reason]

        fig, ax = plt.subplots(figsize=(10, 6))
        by_reason.plot(kind="barh", ax=ax, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Total P&L by Signal Reason (top 15)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Total P&L (%)")
        plt.tight_layout()
        plt.savefig(out / "signal_reason.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot signal reasons: %s", e)


def _plot_spread_zscore(df, tl, out):
    if "spread_zscore" not in df.columns:
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        es = df[df["symbol"] == "ES"].sort_values("date")
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(es["date"], es["spread_zscore"], color="navy", linewidth=0.8, alpha=0.8)
        ax.axhline( config.SA_ENTRY_THRESHOLD, color="crimson",  linestyle="--",
                    linewidth=0.8, label=f"+{config.SA_ENTRY_THRESHOLD}")
        ax.axhline(-config.SA_ENTRY_THRESHOLD, color="seagreen", linestyle="--",
                    linewidth=0.8, label=f"-{config.SA_ENTRY_THRESHOLD}")
        ax.axhline(0, color="grey", linewidth=0.5)

        if tl is not None and not tl.empty:
            sa = tl[(tl["strategy"] == "STAT_ARB") & (tl["symbol"] == "NQ")]
            for _, row in sa.iterrows():
                ax.axvline(row["entry_date"], color="darkorange", alpha=0.5, linewidth=0.6)

        ax.set_title("ES/NQ Spread Z-score with SA NQ Entries", fontsize=12, fontweight="bold")
        ax.set_ylabel("Spread Z-score")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "spread_zscore.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot spread zscore: %s", e)


def _plot_weekly_bias_accuracy(df, out):
    if "weekly_bias" not in df.columns:
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = []
        for sym in df["symbol"].unique():
            g = df[df["symbol"] == sym].sort_values("date").copy()
            g["fwd_5d"] = g["ret_1d"].shift(-1).rolling(5).sum().shift(-4)
            for bias_val in ["LONG", "NEUTRAL"]:
                sub = g[g["weekly_bias"] == bias_val].dropna(subset=["fwd_5d"])
                if len(sub) < 5:
                    continue
                rows.append({
                    "label":   f"{sym} {bias_val}",
                    "avg_fwd": sub["fwd_5d"].mean() * 100,
                    "n":       len(sub),
                })

        if not rows:
            return

        rdf    = pd.DataFrame(rows)
        colors = ["seagreen" if v > 0 else "crimson" for v in rdf["avg_fwd"]]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(rdf["label"], rdf["avg_fwd"], color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Avg Forward 5-day Return by Weekly Bias", fontsize=12, fontweight="bold")
        ax.set_ylabel("Avg 5-day return (%)")
        plt.tight_layout()
        plt.savefig(out / "weekly_bias_accuracy.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot weekly bias accuracy: %s", e)


def _plot_quality_win_rate(tl, out):
    """Entry quality score (1-5) vs win rate. [NEW]"""
    if tl is None or tl.empty or "entry_quality" not in tl.columns:
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        qstats = tl.groupby("entry_quality")["win"].agg(["mean", "count"]).reset_index()
        qstats.columns = ["quality", "win_rate", "count"]

        fig, ax1 = plt.subplots(figsize=(8, 4))
        colors = ["#d62728" if wr < 0.5 else "#2ca02c" for wr in qstats["win_rate"]]
        bars = ax1.bar(qstats["quality"].astype(str), qstats["win_rate"] * 100,
                       color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax1.axhline(50, color="black", linewidth=0.8, linestyle="--")
        ax1.set_xlabel("Entry Quality Score (1–5)")
        ax1.set_ylabel("Win Rate (%)")
        ax1.set_title("Win Rate by Entry Quality Score", fontsize=12, fontweight="bold")

        for bar, cnt in zip(bars, qstats["count"]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"n={cnt}", ha="center", va="bottom", fontsize=9)

        ax1.set_ylim(0, 100)
        ax1.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(out / "quality_win_rate.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot quality win rate: %s", e)


# ── V3 Charts ─────────────────────────────────────────────────────────────────

def _plot_rolling_5day_returns(portfolio_returns, out):
    """[V3] Rolling 5-day and 7-day returns — competition window view."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r5 = (1 + portfolio_returns).rolling(5).apply(lambda x: x.prod() - 1) * 100
        r7 = (1 + portfolio_returns).rolling(7).apply(lambda x: x.prod() - 1) * 100

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(r5.index, r5, color="royalblue",  linewidth=1.0, alpha=0.85, label="5-day rolling")
        ax.plot(r7.index, r7, color="darkorange", linewidth=1.0, alpha=0.85, label="7-day rolling")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.fill_between(r5.index, r5, 0, where=(r5 > 0), alpha=0.12, color="royalblue")
        ax.fill_between(r5.index, r5, 0, where=(r5 < 0), alpha=0.15, color="crimson")
        ax.set_title("Rolling 5-day & 7-day Returns (competition window view)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Rolling Return (%)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "rolling_5day_returns.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot rolling 5-day returns: %s", e)


def _plot_hold_distribution(tl, out):
    """[V3] Histogram of holding days by strategy."""
    if tl is None or tl.empty:
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {
            "TREND":       "royalblue",
            "PULLBACK":    "deepskyblue",
            "MEAN_REVERT": "seagreen",
            "STAT_ARB":    "darkorange",
        }
        fig, ax = plt.subplots(figsize=(10, 4))
        for strat, g in tl.groupby("strategy"):
            ax.hist(g["holding_days"], bins=range(1, 12), alpha=0.6,
                    label=strat, color=colors.get(strat, "grey"), edgecolor="white")
        ax.set_xlabel("Holding Days")
        ax.set_ylabel("Trade Count")
        ax.set_title("Trade Holding Period Distribution by Strategy",
                     fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(out / "trade_hold_distribution.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot hold distribution: %s", e)


def _plot_winner_extension_returns(tl, out):
    """[V3] P&L comparison: extended TREND trades vs non-extended."""
    if tl is None or tl.empty or "extension_used" not in tl.columns:
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        trend = tl[tl["strategy"] == "TREND"].copy()
        if trend.empty:
            return

        ext     = trend[trend["extension_used"] == True]["pnl"]
        non_ext = trend[trend["extension_used"] == False]["pnl"]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # Box plot
        ax1 = axes[0]
        data = [non_ext.values, ext.values] if len(ext) > 0 else [non_ext.values]
        labels = ["Standard\n(≤3 days)", "Extended\n(4-7 days)"] if len(ext) > 0 else ["Standard"]
        ax1.boxplot(data, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor="lightblue", color="navy"),
                    medianprops=dict(color="crimson", linewidth=2))
        ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax1.set_ylabel("Trade P&L")
        ax1.set_title("TREND Trade P&L: Extended vs Standard")
        ax1.grid(alpha=0.3, axis="y")

        # Win rates
        ax2 = axes[1]
        groups = {"Standard\n(≤3d)": non_ext}
        if len(ext) > 0:
            groups["Extended\n(4-7d)"] = ext
        wr_vals  = [(v > 0).mean() * 100 for v in groups.values()]
        wr_cnts  = [len(v) for v in groups.values()]
        wr_bars  = ax2.bar(list(groups.keys()), wr_vals,
                           color=["royalblue", "seagreen"][:len(groups)], alpha=0.8)
        ax2.axhline(50, color="black", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("Win Rate (%)")
        ax2.set_title("Win Rate: Extended vs Standard TREND")
        ax2.set_ylim(0, 100)
        for bar, cnt in zip(wr_bars, wr_cnts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"n={cnt}", ha="center", fontsize=9)
        ax2.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(out / "winner_extension_returns.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot winner extension returns: %s", e)


def _plot_exit_reason_returns(tl, out):
    """[V3] Average P&L and trade count by exit reason."""
    if tl is None or tl.empty or "exit_reason" not in tl.columns:
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        agg = (tl.groupby("exit_reason")["pnl"]
               .agg(["mean", "count", "sum"])
               .sort_values("sum", ascending=False))
        agg = agg.head(12)  # top 12 by total P&L

        colors = ["crimson" if v < 0 else "seagreen" for v in agg["mean"]]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(agg.index, agg["mean"] * 100, color=colors, alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)

        for bar, cnt in zip(bars, agg["count"]):
            x = bar.get_width()
            ax.text(x + (0.001 if x >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f"n={int(cnt)}", va="center", fontsize=8,
                    ha="left" if x >= 0 else "right")

        ax.set_xlabel("Average P&L per Trade (%)")
        ax.set_title("Average P&L by Exit Reason (V3)", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(out / "exit_reason_returns.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning("Could not plot exit reason returns: %s", e)
