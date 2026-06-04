"""
main.py — CME Competition System pipeline orchestrator. [CHANGED V3]

Pipeline (13 steps — portfolio.py removed, sizing now inside backtest):

    1.  Fetch / load daily data
    2.  Fetch / load 4H intraday data
    3.  Compute daily features (returns, EWMA vol, ATR, spread, rel returns)
    4.  Compute multi-timeframe momentum
    5.  Compute weekly bias (LONG/NEUTRAL only — no SHORT)
    6.  Compute daily regime (TREND / TRANSITION / CHOP / SHOCK)
    7.  Compute volatility features (EWMA vol, vol_regime, vol_scalar)
    8.  Compute 4H execution features (breakout score, quality scoring)
    9.  Compute mean reversion + TREND pullback signals
    10. Compute stat arb signals (NQ-only, faster exit)
    11. Compute sentiment / event layer
    12. Run stateful signal engine (TRANSITION regime, quality gating)
    13. Run V3 backtest + generate diagnostics
        [V3] Sizing via position_sizing.py inside backtest loop
        [V3] Winner extension, trailing stops, stop losses, add-to-winner

Usage:
    python main.py --fetch               # download data first
    python main.py                       # run with cached data
    python main.py --verbose             # debug logging
    python main.py --start 2022-01-01   # custom start date
    python main.py --end   2024-12-31   # custom end date

Output:
    Console summary + output/ directory with CSVs and charts.
"""

import argparse
import logging
import sys
from pathlib import Path

import config
from data_loader        import fetch_and_save, load_daily, load_4h
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
from backtest           import run_backtest
from diagnostics        import run_diagnostics


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args():
    p = argparse.ArgumentParser(
        description="CME Competition System — competition-optimised multi-strategy trading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fetch",   action="store_true", help="Re-download data from yfinance")
    p.add_argument("--start",   default=config.START_DATE, help="Backtest start date")
    p.add_argument("--end",     default=None,               help="Backtest end date (default: today)")
    p.add_argument("--verbose", action="store_true",        help="Debug logging")
    return p.parse_args()


def run(
    fetch: bool = False,
    start: str  = None,
    end:   str  = None,
) -> dict:
    log = logging.getLogger("main")

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    if fetch:
        log.info("Step  0 — Fetching data from yfinance")
        fetch_and_save(start=start or config.START_DATE, end=end)

    log.info("Step  1 — Loading daily data")
    df = load_daily()

    log.info("Step  2 — Loading 4H intraday data")
    df_4h_raw = load_4h()

    # ── Step 3: Daily features ────────────────────────────────────────────────
    log.info("Step  3 — Daily features (returns, EWMA vol, ATR, spread)")
    df = compute_features(df)

    # ── Step 4: Multi-timeframe momentum [NEW] ────────────────────────────────
    log.info("Step  4 — Multi-timeframe momentum (5d/20d/60d)")
    df = compute_momentum(df)

    # ── Step 5: Weekly bias ───────────────────────────────────────────────────
    log.info("Step  5 — Weekly bias (LONG/NEUTRAL only)")
    df = compute_weekly_bias(df)

    # ── Step 6: Regime classification ─────────────────────────────────────────
    log.info("Step  6 — Regime classification (TREND/TRANSITION/CHOP/SHOCK)")
    df = compute_regime(df)

    # ── Step 7: Volatility features [NEW] ────────────────────────────────────
    log.info("Step  7 — Volatility features (vol_regime, vol_scalar, atr_pct)")
    df = compute_volatility_features(df)

    # ── Step 7b: CWT chop filter [NEW] ─────────────────────────────────────
    log.info("Step 7b — CWT chop filter (wavelet-based chop classification)")
    if config.USE_CWT_CHOP_FILTER:
        from cwt_chop_filter import compute_cwt_chop_features
        df = compute_cwt_chop_features(df)
    else:
        _cwt_neutral = {
            "cwt_chop_label": "CWT_UNCLEAR", "cwt_trade_allowed": 1,
            "cwt_size_multiplier": 1.0, "cwt_confidence": 0.0,
            "cwt_total_energy": 0.0, "cwt_entropy": 0.5, "cwt_energy_z": 0.0,
            "cwt_compression_flag": 0, "cwt_expansion_flag": 0,
            "cwt_high_energy_ratio": 1/3, "cwt_mid_energy_ratio": 1/3, "cwt_slow_energy_ratio": 1/3,
        }
        for col, val in _cwt_neutral.items():
            if col not in df.columns:
                df[col] = val

    # ── Step 8: 4H intraday features ──────────────────────────────────────────
    log.info("Step  8 — 4H execution features (breakout, quality scoring)")
    if df_4h_raw is not None:
        df_4h_feat  = compute_4h_features(df_4h_raw)
        df_4h_daily = aggregate_4h_to_daily(df_4h_feat)
    else:
        df_4h_daily = None
        log.warning("No 4H data — falling back to daily z-score for MR signals")

    df = merge_4h_into_daily(df, df_4h_daily)

    # ── Step 9: Mean reversion + TREND pullback ───────────────────────────────
    log.info("Step  9 — Mean reversion signals (CHOP MR + TREND pullback)")
    df = compute_mean_reversion(df)

    # ── Step 10: Stat arb ─────────────────────────────────────────────────────
    log.info("Step 10 — Stat arb signals (NQ-only, exit at z=0.40)")
    df = compute_stat_arb(df)

    # ── Step 11: Sentiment ────────────────────────────────────────────────────
    log.info("Step 11 — Sentiment / event layer")
    df = compute_sentiment(df)

    # ── Step 12: Signal engine ────────────────────────────────────────────────
    log.info("Step 12 — Signal engine (TRANSITION regime, quality gating, can_extend)")
    df = compute_signals(df)

    # ── Step 12b: Signal quality scores [NEW] ────────────────────────────────
    log.info("Step 12b — Signal quality scores (trend + stat_arb quality buckets)")
    from signal_quality_engine import compute_signal_quality
    df = compute_signal_quality(df)

    # ── Step 12c: TREND continuation-edge scores [NEW] ────────────────────────
    log.info("Step 12c — TREND continuation-edge scores (pullback-reset, maturity, extension)")
    from trend_continuation_edge_engine import compute_trend_continuation_edge
    df = compute_trend_continuation_edge(df)

    # ── Step 13: V3 Backtest + diagnostics ───────────────────────────────────
    # [V3] portfolio.py removed — sizing is done inside run_backtest via
    # position_sizing.py, with stateful winner extension and trailing stops.
    log.info("Step 13 — V3 Backtest (winner extension, trailing stops, add-to-winner)")
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    result = run_backtest(df)

    run_diagnostics(
        portfolio_returns = result["portfolio_returns"],
        symbol_returns    = result["symbol_returns"],
        strategy_returns  = result["strategy_returns"],
        positions         = result["positions"],
        df_signals        = df,
        trade_log         = result["trade_log"],
        pair_trades       = result["pair_trades"],
        stats             = result["stats"],
        daily_records     = result.get("daily_records"),
    )

    result["df_signals"] = df   # expose full feature/signal frame for ablation forensics
    return result


def main():
    args = _parse_args()
    _setup_logging(args.verbose)
    run(
        fetch = args.fetch,
        start = args.start,
        end   = args.end,
    )


if __name__ == "__main__":
    main()
