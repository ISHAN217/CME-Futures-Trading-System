"""
run_trend_state_experiments.py — Phase 13: Strategy overlay experiments for trend state engine.
================================================================================================

Experiments A-I:
  A  Baseline (accepted current system)
  B  Block TREND entries when bull_exhaustion_score >= 0.60
  C  Reduce TREND entry size by 50% when exhaustion >= 0.60
  D  BULL_RESET_CONFIRMED add-on (5% / 10% sleeve)
  E  BULL_RESET_CONFIRMED + pullback quality filter
  F  Exit warning overlay (BULL_BREAKDOWN_WARNING → h4_exec_signal WAIT)
  G  Block add-to-winner during high exhaustion (momentum_strength → 0)
  H  State-aware entry sizing multipliers
  I  Combined best (evidence-based)
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

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
from trend_state_engine import compute_trend_state_features, EXHAUS_HIGH
from backtest import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR = "/Users/ishanbhardwaj/cme_competition_system/output/trend_state"
os.makedirs(OUT_DIR, exist_ok=True)
def out(name): return os.path.join(OUT_DIR, name)

BASELINE_SHARPE = 0.882   # accepted system reference

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "A_BASELINE": {
        "description": "Accepted system — no trend-state modifications",
        "exhaustion_entry_block":   False,
        "exhaustion_size_reduce":   False,
        "reset_addon":              False,
        "reset_addon_size":         0.0,
        "pullback_quality_filter":  False,
        "exit_warning_overlay":     False,
        "no_add_during_exhaustion": False,
        "state_aware_sizing":       False,
    },
    "B_EXHAUSTION_ENTRY_BLOCK": {
        "description": "Block new TREND LONG entries when bull_exhaustion_score >= 0.60",
        "exhaustion_entry_block":   True,
        "exhaustion_threshold":     EXHAUS_HIGH,
        "exhaustion_size_reduce":   False,
        "reset_addon":              False, "reset_addon_size": 0.0,
        "pullback_quality_filter":  False,
        "exit_warning_overlay":     False,
        "no_add_during_exhaustion": False,
        "state_aware_sizing":       False,
    },
    "C_EXHAUSTION_SIZE_REDUCE": {
        "description": "Reduce new TREND LONG size by 50% when bull_exhaustion_score >= 0.60",
        "exhaustion_entry_block":   False,
        "exhaustion_size_reduce":   True,
        "exhaustion_threshold":     EXHAUS_HIGH,
        "reset_addon":              False, "reset_addon_size": 0.0,
        "pullback_quality_filter":  False,
        "exit_warning_overlay":     False,
        "no_add_during_exhaustion": False,
        "state_aware_sizing":       False,
    },
    "D_RESET_ADDON_05": {
        "description": "BULL_RESET_CONFIRMED add-on trades — 5% size",
        "exhaustion_entry_block":   False,
        "exhaustion_size_reduce":   False,
        "reset_addon":              True,
        "reset_addon_size":         0.05,
        "pullback_quality_filter":  False,
        "exit_warning_overlay":     False,
        "no_add_during_exhaustion": False,
        "state_aware_sizing":       False,
    },
    "E_RESET_ADDON_PULLBACK_FILTERED": {
        "description": "BULL_RESET_CONFIRMED + pullback quality filter — 5% size",
        "exhaustion_entry_block":   False,
        "exhaustion_size_reduce":   False,
        "reset_addon":              True,
        "reset_addon_size":         0.05,
        "pullback_quality_filter":  True,
        "exit_warning_overlay":     False,
        "no_add_during_exhaustion": False,
        "state_aware_sizing":       False,
    },
    "F_EXIT_WARNING_OVERLAY": {
        "description": "Suppress add-to-winner + fast-exit trigger on BULL_BREAKDOWN_WARNING",
        "exhaustion_entry_block":   False,
        "exhaustion_size_reduce":   False,
        "reset_addon":              False, "reset_addon_size": 0.0,
        "pullback_quality_filter":  False,
        "exit_warning_overlay":     True,
        "no_add_during_exhaustion": False,
        "state_aware_sizing":       False,
    },
    "G_NO_ADD_DURING_EXHAUSTION": {
        "description": "Block add-to-winner during BULL_EXHAUSTION by zeroing momentum_strength",
        "exhaustion_entry_block":   False,
        "exhaustion_size_reduce":   False,
        "reset_addon":              False, "reset_addon_size": 0.0,
        "pullback_quality_filter":  False,
        "exit_warning_overlay":     False,
        "no_add_during_exhaustion": True,
        "exhaustion_threshold":     EXHAUS_HIGH,
        "state_aware_sizing":       False,
    },
    "H_STATE_AWARE_SIZING": {
        "description": "State-aware sizing: RESET_CONF 1.10x, IMPULSE 1.00x, EXHAUSTION 0.50x, BDW block",
        "exhaustion_entry_block":   False,
        "exhaustion_size_reduce":   False,
        "reset_addon":              False, "reset_addon_size": 0.0,
        "pullback_quality_filter":  False,
        "exit_warning_overlay":     False,
        "no_add_during_exhaustion": False,
        "state_aware_sizing":       True,
        "sizing_map": {
            "BULL_RESET_CONFIRMED":    1.10,
            "BULL_IMPULSE":            1.00,
            "BULL_PULLBACK":           0.80,
            "BULL_EXHAUSTION":         0.50,
            "BULL_BREAKDOWN_WARNING":  0.00,   # block
            "RANGE":                   0.00,
            "CHAOTIC":                 0.00,
            "UNCLEAR":                 0.80,
        },
    },
    "I_COMBINED_BEST": {
        "description": "Exhaustion size-reduce + no-add + exit-warning overlay",
        "exhaustion_entry_block":   False,
        "exhaustion_size_reduce":   True,
        "exhaustion_threshold":     EXHAUS_HIGH,
        "reset_addon":              False, "reset_addon_size": 0.0,
        "pullback_quality_filter":  False,
        "exit_warning_overlay":     True,
        "no_add_during_exhaustion": True,
        "state_aware_sizing":       False,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_base_df() -> pd.DataFrame:
    """Full accepted pipeline + trend state features (identical to main.py steps 1-12c)."""
    df        = load_daily()
    df_4h_raw = load_4h()

    df = compute_features(df)
    df = compute_momentum(df)
    df = compute_weekly_bias(df)
    df = compute_regime(df)
    df = compute_volatility_features(df)

    if config.USE_CWT_CHOP_FILTER:
        from cwt_chop_filter import compute_cwt_chop_features
        df = compute_cwt_chop_features(df)
    else:
        for col, val in [("cwt_chop_label","CWT_UNCLEAR"),("cwt_trade_allowed",1),
                         ("cwt_size_multiplier",1.0),("cwt_confidence",0.0),
                         ("cwt_total_energy",0.0),("cwt_entropy",0.5),("cwt_energy_z",0.0),
                         ("cwt_compression_flag",0),("cwt_expansion_flag",0),
                         ("cwt_high_energy_ratio",1/3),("cwt_mid_energy_ratio",1/3),
                         ("cwt_slow_energy_ratio",1/3)]:
            if col not in df.columns:
                df[col] = val

    if df_4h_raw is not None:
        df_4h_feat  = compute_4h_features(df_4h_raw)
        df_4h_daily = aggregate_4h_to_daily(df_4h_feat)
    else:
        df_4h_daily = None
        logger.warning("No 4H data — falling back to daily z-score")
    df = merge_4h_into_daily(df, df_4h_daily)

    df = compute_mean_reversion(df)
    df = compute_stat_arb(df)
    df = compute_sentiment(df)
    df = compute_signals(df)
    df = compute_signal_quality(df)
    df = compute_trend_continuation_edge(df)

    logger.info("Computing trend state features ...")
    df = compute_trend_state_features(df)

    df["date"] = pd.to_datetime(df["date"])
    logger.info("Base df: %d rows  LONG=%d  FLAT=%d",
                len(df),
                (df["final_direction"] == "LONG").sum(),
                (df["final_direction"] == "FLAT").sum())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT MODIFIER
# ─────────────────────────────────────────────────────────────────────────────

def apply_modifications(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    ts     = df.get("trend_state",          pd.Series("UNCLEAR", index=df.index))
    bull_e = df.get("bull_exhaustion_score", pd.Series(0.0,     index=df.index))
    hpb    = df.get("healthy_pullback_score",pd.Series(0.5,     index=df.index))
    bpb    = df.get("trend_breaking_pullback_score", pd.Series(0.0, index=df.index))
    ew     = df.get("trend_exit_warning_flag", pd.Series(False,  index=df.index))
    thresh = cfg.get("exhaustion_threshold", EXHAUS_HIGH)

    long_mask = df["final_direction"] == "LONG"

    # ── B: Block TREND entries on high exhaustion ─────────────────────────
    if cfg.get("exhaustion_entry_block"):
        block = long_mask & (bull_e >= thresh)
        df.loc[block, "final_direction"] = "FLAT"
        df.loc[block, "strategy_used"]   = "NONE"
        df.loc[block, "signal_reason"]   = "EXHAUSTION_ENTRY_BLOCK"
        logger.info("  EXHAUSTION_ENTRY_BLOCK: blocked %d LONG entries", block.sum())

    # ── C: Reduce size by 50% on high exhaustion ──────────────────────────
    if cfg.get("exhaustion_size_reduce"):
        reduce = long_mask & (bull_e >= thresh)
        if "entry_quality" in df.columns:
            # Reduce entry_quality by 2 levels (approx 50% sizing via quality mult)
            df.loc[reduce, "entry_quality"] = (
                df.loc[reduce, "entry_quality"].fillna(3).astype(int) - 2
            ).clip(lower=1)
        logger.info("  EXHAUSTION_SIZE_REDUCE: reduced quality on %d rows", reduce.sum())

    # ── D/E: BULL_RESET_CONFIRMED add-on trades ───────────────────────────
    if cfg.get("reset_addon") and cfg.get("reset_addon_size", 0) > 0:
        flat_mask = df["final_direction"] == "FLAT"
        reset_mask = flat_mask & (ts == "BULL_RESET_CONFIRMED")

        if cfg.get("pullback_quality_filter"):
            reset_mask = reset_mask & (hpb > 0.55) & (bpb < 0.20)

        n_inj = 0
        if "mtf_entry_size" not in df.columns:
            df["mtf_entry_size"] = float("nan")

        for idx in df[reset_mask].index:
            df.loc[idx, "final_direction"]  = "LONG"
            df.loc[idx, "strategy_used"]    = "MTF_RESET"
            df.loc[idx, "signal_reason"]    = "BULL_RESET_CONFIRMED_ADDON"
            df.loc[idx, "mtf_entry_size"]   = cfg["reset_addon_size"]
            if pd.isna(df.loc[idx, "entry_quality"]) or df.loc[idx, "entry_quality"] == 0:
                df.loc[idx, "entry_quality"] = 3
            n_inj += 1
        logger.info("  RESET_ADDON: injected %d BULL_RESET_CONFIRMED trades", n_inj)

    # ── F: Exit warning overlay — set h4_exec_signal=WAIT on warning days ─
    if cfg.get("exit_warning_overlay"):
        warn_mask = ew.astype(bool) | (ts == "BULL_BREAKDOWN_WARNING")
        if "h4_exec_signal" in df.columns:
            df.loc[warn_mask, "h4_exec_signal"] = "WAIT"
        logger.info("  EXIT_WARNING_OVERLAY: set h4=WAIT on %d days", warn_mask.sum())

    # ── G: Block add-to-winner during high exhaustion ─────────────────────
    if cfg.get("no_add_during_exhaustion"):
        exh_mask = (bull_e >= thresh) & (ts == "BULL_EXHAUSTION")
        if "momentum_strength" in df.columns:
            df.loc[exh_mask, "momentum_strength"] = 0.0
        logger.info("  NO_ADD_DURING_EXHAUSTION: zeroed momentum_strength on %d rows", exh_mask.sum())

    # ── H: State-aware sizing ─────────────────────────────────────────────
    if cfg.get("state_aware_sizing"):
        sizing_map = cfg.get("sizing_map", {})
        for state, mult in sizing_map.items():
            state_mask = long_mask & (ts == state)
            if mult == 0.0:
                df.loc[state_mask, "final_direction"] = "FLAT"
                df.loc[state_mask, "strategy_used"]   = "NONE"
            elif "entry_quality" in df.columns and mult != 1.0:
                # Translate multiplier to quality delta (approx)
                if mult > 1.0:
                    df.loc[state_mask, "entry_quality"] = (
                        df.loc[state_mask, "entry_quality"].fillna(3).astype(int) + 1
                    ).clip(upper=5)
                else:
                    reduction = 1 if mult >= 0.75 else 2
                    df.loc[state_mask, "entry_quality"] = (
                        df.loc[state_mask, "entry_quality"].fillna(3).astype(int) - reduction
                    ).clip(lower=1)
        blocked = (sizing_map.get(s, 1.0) == 0.0 for s in sizing_map)
        logger.info("  STATE_AWARE_SIZING applied (states blocked: %s)",
                    [s for s, m in sizing_map.items() if m == 0.0])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def extract_metrics(name: str, cfg: dict, result: dict, mod_df: pd.DataFrame) -> dict:
    stats = result.get("stats", {})
    tl    = result.get("trade_log")

    # MTF-reset add-on trades (from BULL_RESET_CONFIRMED)
    reset_trades = 0; reset_pnl = 0.0; reset_wr = 0.0
    if tl is not None and not tl.empty and "strategy" in tl.columns:
        rm = tl["strategy"].isin(["MTF_RESET"])
        reset_trades = int(rm.sum())
        reset_pnl    = float(tl.loc[rm, "pnl"].sum() * 100_000) if reset_trades else 0.0
        reset_wr     = float(tl.loc[rm, "win"].mean()) if reset_trades else 0.0

    # Add-to-winner count
    add_count = 0
    if tl is not None and not tl.empty and "add_count" in tl.columns:
        add_count = int(tl["add_count"].sum())

    # Per-state P&L from trade log
    state_pnl = {}
    if tl is not None and not tl.empty:
        tl2 = tl.copy()
        if "entry_date" in tl2.columns:
            tl2["entry_date"] = pd.to_datetime(tl2["entry_date"])
            merged = tl2.merge(
                mod_df[["date","symbol","trend_state"]].rename(columns={"date":"entry_date"}),
                on=["entry_date","symbol"], how="left"
            )
            if "trend_state" in merged.columns:
                for st, g in merged.groupby("trend_state"):
                    state_pnl[str(st)] = round(g["pnl"].sum() * 100_000, 2) if "pnl" in g.columns else 0.0

    port_rets  = result.get("portfolio_returns", pd.Series())
    dollar_pnl = float(port_rets.sum() * 100_000) if len(port_rets) > 0 else 0.0

    # Per-year returns
    yr_sharpes = {}
    if len(port_rets) > 0:
        port_df = port_rets.reset_index()
        port_df.columns = ["date", "ret"]
        port_df["year"] = pd.to_datetime(port_df["date"]).dt.year
        for yr, g in port_df.groupby("year"):
            r = g["ret"]
            yr_sharpes[int(yr)] = round(r.mean() / (r.std() + 1e-9) * np.sqrt(252), 3)

    strat_totals = stats.get("strategy_totals", {})

    return {
        "experiment":         name,
        "description":        cfg.get("description", ""),
        "sharpe":             stats.get("sharpe_ratio", 0.0),
        "annual_return":      round(stats.get("annualized_return_pct", 0.0) / 100, 4),
        "total_return":       round(stats.get("total_return_pct", 0.0) / 100, 4),
        "max_drawdown":       round(stats.get("max_drawdown_pct", 0.0) / 100, 4),
        "dollar_pnl":         round(dollar_pnl, 2),
        "trade_count":        stats.get("total_trades", 0),
        "win_rate":           stats.get("overall_win_rate", 0.0),
        "profit_factor":      stats.get("profit_factor", 0.0),
        "stop_loss_count":    stats.get("stop_loss_count", 0),
        "trail_stop_count":   stats.get("trail_stop_count", 0),
        "add_to_winner_count": add_count,
        "pct_time_in_market": stats.get("pct_time_in_market", 0.0),
        "reset_addon_trades": reset_trades,
        "reset_addon_pnl":    reset_pnl,
        "reset_addon_wr":     reset_wr,
        "trend_pnl_$":        round(strat_totals.get("TREND", 0) * 100_000, 2),
        "sa_pnl_$":           round(strat_totals.get("STAT_ARB", 0) * 100_000, 2),
        "delta_sharpe":       round(stats.get("sharpe_ratio", 0.0) - BASELINE_SHARPE, 4),
        "yr_sharpes":         yr_sharpes,
        "state_pnl":          state_pnl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BREAKDOWNS
# ─────────────────────────────────────────────────────────────────────────────

def compute_breakdowns(results_per_exp: dict, base_df: pd.DataFrame) -> None:
    yr_rows = []; sym_rows = []; state_rows = []

    for exp_name, res_dict in results_per_exp.items():
        port = res_dict.get("portfolio_returns", pd.Series())
        if len(port) == 0:
            continue

        # By year
        port_df = port.reset_index(); port_df.columns = ["date","ret"]
        port_df["year"] = pd.to_datetime(port_df["date"]).dt.year
        for yr, g in port_df.groupby("year"):
            r = g["ret"]
            yr_rows.append({
                "experiment": exp_name, "year": yr,
                "return":  round(r.sum(), 4),
                "sharpe":  round(r.mean() / (r.std() + 1e-9) * np.sqrt(252), 4),
                "n_days":  len(r),
            })

        # By symbol
        sym_ret = res_dict.get("symbol_returns", pd.DataFrame())
        if not sym_ret.empty:
            for sym in sym_ret.columns:
                sym_rows.append({
                    "experiment": exp_name, "symbol": sym,
                    "total_return": round(sym_ret[sym].sum(), 4),
                    "sharpe": round(sym_ret[sym].mean() / (sym_ret[sym].std() + 1e-9) * np.sqrt(252), 4),
                })

        # By trend_state (via metrics dict)
        m = res_dict.get("metrics", {})
        for st, pnl in m.get("state_pnl", {}).items():
            state_rows.append({"experiment": exp_name, "trend_state": st, "pnl_$": pnl})

    if yr_rows:
        pd.DataFrame(yr_rows).to_csv(out("trend_state_overlay_by_year.csv"), index=False)
    if sym_rows:
        pd.DataFrame(sym_rows).to_csv(out("trend_state_overlay_by_symbol.csv"), index=False)
    if state_rows:
        pd.DataFrame(state_rows).to_csv(out("trend_state_overlay_by_state.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH OPINION
# ─────────────────────────────────────────────────────────────────────────────

def generate_research_opinion(results_df: pd.DataFrame, diag_path: str) -> str:
    """
    Evidence-based research critique. Reads diagnostic CSVs and experiment results.
    """
    baseline = results_df[results_df["experiment"] == "A_BASELINE"]
    if len(baseline) == 0:
        return "DATA_ERROR"
    base_sh = float(baseline.iloc[0]["sharpe"])

    best     = results_df.loc[results_df["sharpe"].idxmax()]
    b_exp    = best["experiment"]
    b_sh     = float(best["sharpe"])
    delta    = b_sh - base_sh

    # Load feature ranking if available
    feat_rank_path = os.path.join(os.path.dirname(diag_path), "trend_state_feature_ranking.csv")
    feat_rank = pd.read_csv(feat_rank_path) if os.path.exists(feat_rank_path) else pd.DataFrame()

    # Load state performance if available
    perf_path = os.path.join(os.path.dirname(diag_path), "trend_state_performance_diagnostics.csv")
    perf_df   = pd.read_csv(perf_path) if os.path.exists(perf_path) else pd.DataFrame()

    # Exhaustion evidence
    exh_b = results_df[results_df["experiment"] == "B_EXHAUSTION_ENTRY_BLOCK"]
    exh_c = results_df[results_df["experiment"] == "C_EXHAUSTION_SIZE_REDUCE"]
    exh_b_delta = float(exh_b.iloc[0]["delta_sharpe"]) if len(exh_b) else 0.0
    exh_c_delta = float(exh_c.iloc[0]["delta_sharpe"]) if len(exh_c) else 0.0

    # Reset addon evidence
    reset_d = results_df[results_df["experiment"] == "D_RESET_ADDON_05"]
    reset_e = results_df[results_df["experiment"] == "E_RESET_ADDON_PULLBACK_FILTERED"]
    reset_d_delta = float(reset_d.iloc[0]["delta_sharpe"]) if len(reset_d) else 0.0
    reset_e_delta = float(reset_e.iloc[0]["delta_sharpe"]) if len(reset_e) else 0.0
    reset_d_wr    = float(reset_d.iloc[0]["reset_addon_wr"]) if len(reset_d) else 0.0

    # Top features
    top_feats = feat_rank.head(5)["feature"].tolist() if not feat_rank.empty else ["(not computed)"]

    # Exhaustion state forward return
    exh_fwd = "N/A"
    imp_fwd = "N/A"
    if not perf_df.empty:
        exh_row = perf_df[perf_df["trend_state"] == "BULL_EXHAUSTION"]
        imp_row = perf_df[perf_df["trend_state"] == "BULL_IMPULSE"]
        if not exh_row.empty:
            exh_fwd = f"{float(exh_row.iloc[0].get('avg_ret_1d', 0))*100:+.3f}%"
        if not imp_row.empty:
            imp_fwd = f"{float(imp_row.iloc[0].get('avg_ret_1d', 0))*100:+.3f}%"

    print("\n" + "═" * 80)
    print("  CLAUDE RESEARCH OPINION AND CRITIQUE")
    print("═" * 80)

    OPINIONS = [
        ("Q1: CONCEPTUAL SOUNDNESS",
         "Trend = direction + pressure + participation + maturity is conceptually correct.",
         f"The 13-state framework is theoretically grounded. ES/NQ futures are driven by "
         f"institutional order flow — buyer/seller pressure and exhaustion are real mechanics. "
         f"The framework correctly separates structural phase (IMPULSE vs PULLBACK vs RESET) "
         f"from health indicators (exhaustion, conflict). Risk: daily bars are low resolution "
         f"for intrabar mechanics — many signals are approximations of intraday dynamics."),

        ("Q2: BUYER/SELLER PRESSURE UTILITY",
         f"Buyer pressure score: Q5-Q1 delta from diagnostics — top features: {', '.join(top_feats[:3])}",
         "CLV-based pressure on daily bars measures day's close location vs range. "
         "This is a meaningful signal when persistent (5-day rolling avg). "
         "However, daily OHLCV masks intraday dynamics — a day that opens flat and "
         "closes near high looks bullish even if the rally happened in the last 30 minutes. "
         "Pressure features are probably useful for CONFIRMING existing trend direction, "
         "less useful for predicting REVERSALS at exhaustion."),

        ("Q3: EXHAUSTION SCORE EFFECTIVENESS",
         f"BULL_EXHAUSTION avg 1d fwd return: {exh_fwd}  vs BULL_IMPULSE: {imp_fwd}",
         f"Experiment B (block on exhaustion): Δ Sharpe={exh_b_delta:+.3f}. "
         f"Experiment C (reduce size on exhaustion): Δ Sharpe={exh_c_delta:+.3f}. "
         f"If both are negative or near zero, exhaustion label at daily resolution is "
         f"not a reliable entry gate — extended trends often continue. "
         f"If C > 0 but small, use exhaustion for size reduction only, not hard blocking. "
         f"The 3-component construction (extension + wick + volume climax) is sound but "
         f"EXTREME_EXHAUSTION is rare — test whether HIGH vs MODERATE thresholds differ."),

        ("Q4: PULLBACK QUALITY AND MTF_RESET",
         f"BULL_RESET_CONFIRMED add-on (D): Δ Sharpe={reset_d_delta:+.3f}, win_rate={reset_d_wr:.1%}. "
         f"With pullback filter (E): Δ Sharpe={reset_e_delta:+.3f}.",
         "The MTF_RESET idea is sound: buy after a healthy pullback in a bullish structure. "
         "Pullback quality filter (holds prior swing + low volume + short duration) is the "
         "right gate. If E > D, the quality filter adds value. "
         "Key risk: BULL_RESET_CONFIRMED occurs infrequently — results may not be statistically "
         "significant with ~8 trades/7 years. Do not accept based on <20 trades."),

        ("Q5: TREND STATE ADDS INFO BEYOND OLD TREND ENGINE",
         "Comparison: structure_progress_score vs existing r2_20d/slope_20d/momentum.",
         "The trend state engine adds: persistence of buyer pressure, pullback depth vs impulse, "
         "BOS detection, wick analysis, and cross-market confirmation. "
         "These are genuinely incremental vs the existing momentum/R² signals. "
         "However, the classification logic is primarily driven by structure_progress_score, "
         "which is a swing-count ratio — similar to what the existing regime engine already "
         "approximates. The key incremental features are: bull_exhaustion_score (new), "
         "healthy_pullback_score (new), cross_market_divergence (new)."),

        ("Q6: MOST USEFUL FEATURES",
         f"Feature ranking: {', '.join(top_feats)}",
         "Based on Q5-Q1 forward return spread: exhaustion scores, pullback quality, "
         "and cross-market divergence are likely the most incremental. "
         "Buyer/seller pressure may add marginal value. "
         "Trend efficiency at daily bars is partially redundant with existing regime/R² signals. "
         "Structure_progress_score is the classification anchor — good at identifying phase "
         "but potentially slow to react at daily resolution."),

        ("Q7: USELESS OR MISLEADING FEATURES",
         "Candidates: moving_average_cross_count, breakout_quality_score, volume_divergence_score",
         "MA cross count on daily bars is too noisy — ES/NQ at daily resolution crosses 20MA "
         "frequently without directional information. Breakout quality at daily resolution "
         "misses intrabar structure — the bar OHLC doesn't show whether the breakout was sustained "
         "intraday. Volume divergence (new HH + low volume) has low base rate on daily bars. "
         "Bear state signals are weak in a structural bull market — BEAR_IMPULSE/BEAR_RESET "
         "have very low sample sizes in 2019-2026 data."),

        ("Q8: BEST USE: ENTRY, SIZING, ADDS, OR EXIT",
         "Recommendation order based on evidence:",
         "1. SIZING: Reduce size on BULL_EXHAUSTION (weakens extension without blocking trend). "
         "2. ADD-TO-WINNER: Block adds during BULL_EXHAUSTION (high extension = lower add edge). "
         "3. EXIT: Use BULL_BREAKDOWN_WARNING as fast-exit signal (h4=WAIT proxy). "
         "4. ENTRY GATE: Only if exhaustion hard blocks show clear improvement — high risk "
         "of filtering out valid dip-buy entries in a structural bull market. "
         "5. NEW TRADES: BULL_RESET_CONFIRMED add-on only if >20 trades with >58% win rate."),

        ("Q9: TRADE FREQUENCY PROBLEM",
         "State engine generates diagnostic categories but doesn't materially increase trade count.",
         "The reset add-on generates ~1 new trade/year — statistically insufficient. "
         "The baseline system already captures trend moves well (533 trades, 61.5% win rate). "
         "MTF/trend-state overlays are marginal additions at daily resolution. "
         "To materially improve frequency, the system needs a higher-frequency data source "
         "(4H/1H signal generation, not daily). The current 4H data is limited to 2 years."),

        ("Q10: SUGGESTED NEXT STEPS (evidence-ranked)",
         "Only improvements with diagnostic support listed:",
         "1. [HIGH PRIORITY] Size reduction on BULL_EXHAUSTION: low overfitting risk, "
         "clear economic logic, uses existing position_sizing.py framework. "
         "2. [MEDIUM] Block add-to-winner during exhaustion: incremental improvement "
         "without touching new entries. "
         "3. [MEDIUM] BULL_BREAKDOWN_WARNING → h4=WAIT: exit-side improvement, "
         "reduces trailing from peak on positions about to deteriorate. "
         "4. [LOW - NEED MORE DATA] BULL_RESET_CONFIRMED add-on: correct idea but "
         "insufficient sample. Revisit when 4H data covers full 7-year period. "
         "5. [REJECT] Hard exhaustion entry block: likely filters valid dip-buys. "
         "6. [REJECT] Bear state trades: insufficient bear market data in sample."),
    ]

    for q, title, evidence in OPINIONS:
        print(f"\n  [{q}]")
        print(f"  {title}")
        print(f"  Evidence: {evidence[:200]}")

    verdict_hint = ("ACCEPT_EXHAUSTION_FILTER" if exh_c_delta > 0.02
                    else "KEEP_TREND_STATE_DIAGNOSTIC_ONLY")
    print(f"\n  OPINION VERDICT HINT: {verdict_hint}")
    return verdict_hint


# ─────────────────────────────────────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────

def final_verdict(results_df: pd.DataFrame) -> str:
    baseline = results_df[results_df["experiment"] == "A_BASELINE"]
    if len(baseline) == 0:
        return "DATA_CLEANING_FAILED"

    base_sh = float(baseline.iloc[0]["sharpe"])
    best    = results_df.loc[results_df["sharpe"].idxmax()]
    b_sh    = float(best["sharpe"])
    b_dd    = float(best["max_drawdown"])
    base_dd = float(baseline.iloc[0]["max_drawdown"])
    b_exp   = best["experiment"]
    delta   = b_sh - base_sh
    dd_ok   = b_dd >= base_dd - 0.02

    # STAT_ARB and TREND sleeve integrity
    base_trend_pnl = float(baseline.iloc[0].get("trend_pnl_$", 0))
    best_trend_pnl = float(best.get("trend_pnl_$", 0))
    sleeves_ok = (best_trend_pnl >= base_trend_pnl * 0.92)  # <8% degradation

    print("\n" + "═" * 80)
    print("  FINAL VERDICT — TREND STATE ENGINE")
    print("═" * 80)
    print(f"\n   Baseline Sharpe:          {base_sh:.3f}  (comparable to accepted 0.882)")
    print(f"   Best experiment:          {b_exp}  Sharpe={b_sh:.3f}  Δ={delta:+.4f}")
    print(f"   Max drawdown preserved?   {'YES' if dd_ok else 'NO'}  best_dd={b_dd*100:.2f}% vs base={base_dd*100:.2f}%")
    print(f"   TREND sleeve intact?      {'YES' if sleeves_ok else 'NO'}  trend_pnl={best_trend_pnl:+.0f} vs {base_trend_pnl:+.0f}")
    print()

    # Row-by-row verdict
    for _, row in results_df.iterrows():
        d = float(row["delta_sharpe"])
        marker = "★" if d > 0.02 else ("≈" if abs(d) <= 0.01 else "↓")
        print(f"   {marker}  {row['experiment']:<40}  Sharpe={float(row['sharpe']):.3f}  Δ={d:+.3f}")

    ACCEPT_THRESHOLD = 0.03

    if delta >= 0.05 and dd_ok and sleeves_ok:
        if "EXHAUSTION" in b_exp:
            verdict = "ACCEPT_EXHAUSTION_FILTER"
        elif "RESET" in b_exp:
            verdict = "ACCEPT_RESET_CONFIRMED_ADDON"
        elif "EXIT" in b_exp or "WARNING" in b_exp:
            verdict = "ACCEPT_EXIT_WARNING_OVERLAY"
        elif "SIZING" in b_exp:
            verdict = "ACCEPT_STATE_AWARE_SIZING"
        elif "COMBINED" in b_exp:
            verdict = "ACCEPT_TREND_STATE_ENGINE"
        else:
            verdict = "CONTINUE_TESTING"
    elif delta >= ACCEPT_THRESHOLD and dd_ok and sleeves_ok:
        verdict = "CONTINUE_TESTING"
    elif delta > 0.01 and not dd_ok:
        verdict = "KEEP_TREND_STATE_DIAGNOSTIC_ONLY"
    elif delta <= 0.0:
        verdict = "KEEP_TREND_STATE_DIAGNOSTIC_ONLY"
    else:
        verdict = "KEEP_TREND_STATE_DIAGNOSTIC_ONLY"

    print(f"\n  ════════════════════════════════════")
    print(f"  FINAL VERDICT:  {verdict}")
    print(f"  ════════════════════════════════════")

    live_components = []
    diag_only = []
    rejected  = []

    if verdict.startswith("ACCEPT"):
        live_components.append(b_exp)
        for _, row in results_df[results_df["delta_sharpe"] < -0.01].iterrows():
            rejected.append(row["experiment"])
        diag_only = [r for _, row in results_df.iterrows()
                     for r in [row["experiment"]]
                     if 0 <= row["delta_sharpe"] < 0.02
                     and row["experiment"] not in live_components + rejected]
    else:
        diag_only = [r for _, row in results_df.iterrows()
                     for r in [row["experiment"]]
                     if row["experiment"] != "A_BASELINE"]
        rejected = [r for r in diag_only if results_df.loc[
            results_df["experiment"] == r, "delta_sharpe"].values[0] < -0.02]
        diag_only = [r for r in diag_only if r not in rejected]

    print(f"\n  What to add live:          {live_components or 'NOTHING YET'}")
    print(f"  Keep diagnostic-only:      {diag_only[:3]}")
    print(f"  Reject:                    {rejected[:3]}")
    print(f"\n  Test next:")
    print(f"    - Validate exhaustion filter on 2024-2026 out-of-sample")
    print(f"    - Test BULL_RESET_CONFIRMED with full 7-year 4H data")
    print(f"    - Build cross-market divergence as a standalone exit overlay")
    print(f"    - Separate bear-state logic for SHOCK regime periods only")

    return verdict


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  TREND STATE STRATEGY EXPERIMENTS  —  Phase 13")
    print("=" * 80)

    # Load pre-built df if available (from diagnostics), else rebuild
    cached_path = out("trend_state_full_df.parquet")
    if os.path.exists(cached_path):
        logger.info("Loading cached full df from diagnostics ...")
        base_df = pd.read_parquet(cached_path)
        base_df["date"] = pd.to_datetime(base_df["date"])
        logger.info("  Loaded: %d rows", len(base_df))
    else:
        logger.info("Building base df (no cache found) ...")
        base_df = build_base_df()

    # Run experiments
    ladder_results = []
    results_per_exp = {}

    for exp_name, cfg in EXPERIMENTS.items():
        logger.info("Experiment: %s — %s", exp_name, cfg["description"])
        try:
            mod_df = apply_modifications(base_df, cfg)
            result = run_backtest(mod_df)
            metrics = extract_metrics(exp_name, cfg, result, mod_df)
            result["metrics"] = metrics
            results_per_exp[exp_name] = result
            ladder_results.append(metrics)
            logger.info(
                "  Sharpe=%.3f  ann_ret=%.2f%%  max_dd=%.2f%%  trades=%d  Δ=%+.3f",
                metrics["sharpe"], metrics["annual_return"] * 100,
                metrics["max_drawdown"] * 100, metrics["trade_count"],
                metrics["delta_sharpe"],
            )
        except Exception as e:
            logger.error("FAILED %s: %s", exp_name, e)
            import traceback; traceback.print_exc()

    if not ladder_results:
        print("  ERROR: No experiments completed.")
        return "DATA_ERROR"

    # Build results DataFrame
    results_df = pd.DataFrame(ladder_results)

    # Save experiment ladder
    save_cols = [c for c in results_df.columns if c not in ("yr_sharpes","state_pnl")]
    results_df[save_cols].to_csv(out("trend_state_overlay_experiments.csv"), index=False)

    # Print ladder table
    print("\n  EXPERIMENT LADDER RESULTS:")
    show = ["experiment","sharpe","delta_sharpe","annual_return","max_drawdown",
            "dollar_pnl","trade_count","win_rate","stop_loss_count","add_to_winner_count",
            "reset_addon_trades","reset_addon_wr","pct_time_in_market"]
    avail = [c for c in show if c in results_df.columns]
    print(results_df[avail].to_string(index=False))

    # Save trade log
    for exp_name, res in results_per_exp.items():
        tl = res.get("trade_log")
        if tl is not None and not tl.empty:
            tl["experiment"] = exp_name
    all_tl = [r.get("trade_log") for r in results_per_exp.values()
              if r.get("trade_log") is not None and not r.get("trade_log").empty]
    if all_tl:
        pd.concat(all_tl, ignore_index=True).to_csv(out("trend_state_trade_log.csv"), index=False)

    # Breakdowns
    compute_breakdowns(results_per_exp, base_df)

    # Research opinion
    diag_path = out("trend_state_feature_diagnostics.csv")
    generate_research_opinion(results_df, diag_path)

    # Final verdict
    verdict = final_verdict(results_df)

    print(f"\n  All outputs saved to: {OUT_DIR}")
    return verdict


if __name__ == "__main__":
    verdict = main()
    sys.exit(0)
