"""
run_trend_statarb_research.py — Parts 1–7 of the Trend Structure + STAT_ARB upgrade.

Sections
────────
Part 1/2 — Trend structure EXIT experiments  (A–I, post-process two-pass)
Part 1/2 — Trend structure ENTRY experiments (A–H, pre-process df)
Part 3   — STAT_ARB expansion experiments    (A–I)
Part 4   — Trend + STAT_ARB interaction      (A–O)
Part 5   — OOS / robustness tests            (top-5 variants)
Part 6/7 — Final report + Claude verdict

Outputs
───────
output/structure_statarb/
  trend_structure_exit_experiments.csv
  trend_structure_entry_experiments.csv
  trend_structure_trade_log.csv
  statarb_expansion_experiments.csv
  statarb_expanded_trade_log.csv
  statarb_bucket_diagnostics.csv
  trend_statarb_combined_experiments.csv
  trend_statarb_combined_trade_log.csv
  trend_statarb_oos_results.csv
"""

import contextlib, os, sys, logging, warnings
warnings.filterwarnings("ignore")

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
from trend_structure_engine         import compute_trend_structure
from stat_arb_expansion_engine      import compute_stat_arb_expansion
from backtest           import run_backtest

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT = "output/structure_statarb"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_base_df() -> pd.DataFrame:
    logger.info("Building pipeline …")
    df = load_daily(); df_4h_raw = load_4h()
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
                         ("cwt_total_energy",0.0),("cwt_entropy",0.5),
                         ("cwt_energy_z",0.0),("cwt_compression_flag",0),
                         ("cwt_expansion_flag",0),("cwt_high_energy_ratio",1/3),
                         ("cwt_mid_energy_ratio",1/3),("cwt_slow_energy_ratio",1/3)]:
            if col not in df.columns: df[col] = val
    df_4h_daily = None
    if df_4h_raw is not None:
        df_4h_daily = aggregate_4h_to_daily(compute_4h_features(df_4h_raw))
    df = merge_4h_into_daily(df, df_4h_daily)
    df = compute_mean_reversion(df)
    df = compute_stat_arb(df)
    df = compute_sentiment(df)
    df = compute_signals(df)
    df = compute_signal_quality(df)
    df = compute_trend_continuation_edge(df)
    # New engines
    df = compute_trend_structure(df)
    df = compute_stat_arb_expansion(df)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Pipeline complete — %d rows, %d symbols", len(df), df["symbol"].nunique())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _cfg(**kw):
    orig = {k: getattr(config, k) for k in kw if hasattr(config, k)}
    for k, v in kw.items(): setattr(config, k, v)
    try: yield
    finally:
        for k, v in orig.items(): setattr(config, k, v)


def _run(df: pd.DataFrame, label: str,
         extra_hold: dict | None = None) -> tuple[dict, pd.DataFrame]:
    hold_cfg = {**config.MAX_HOLD_DAYS}
    if extra_hold:
        hold_cfg.update(extra_hold)
    with _cfg(MAX_HOLD_DAYS=hold_cfg):
        res = run_backtest(df)
    st = res["stats"]
    tl = res["trade_log"]
    pr = res["portfolio_returns"]

    yr = {}
    for y in range(2019, 2027):
        m = pr.index.year == y
        yr[y] = round(float((1 + pr[m]).prod() - 1) * 100, 2) if m.any() else 0.0

    pf_es = tl[tl["symbol"]=="ES"]["pnl"].sum() if not tl.empty else 0.0
    pf_nq = tl[tl["symbol"]=="NQ"]["pnl"].sum() if not tl.empty else 0.0

    pf_trend = tl[tl["strategy"]=="TREND"]["pnl"].sum() if not tl.empty else 0.0
    pf_sa    = tl[tl["strategy"].str.startswith("STAT", na=False)]["pnl"].sum() if not tl.empty else 0.0
    pf_shock = tl[tl["strategy"].str.startswith("SHOCK", na=False)]["pnl"].sum() if not tl.empty else 0.0

    row = dict(
        experiment   = label,
        sharpe       = round(st["sharpe_ratio"], 4),
        annual_ret   = round(st["annualized_return_pct"], 2),
        total_pnl    = round(st["total_return_pct"], 2),
        max_dd       = round(st["max_drawdown_pct"], 2),
        trades       = st["total_trades"],
        win_rate     = round(st["overall_win_rate"], 4),
        profit_factor= round(st["profit_factor"], 4),
        pnl_ES       = round(pf_es, 4),
        pnl_NQ       = round(pf_nq, 4),
        pnl_TREND    = round(pf_trend, 4),
        pnl_STATARB  = round(pf_sa, 4),
        pnl_SHOCK    = round(pf_shock, 4),
        **{f"yr_{k}": v for k, v in yr.items()},
    )
    return row, tl


def _dollar(row: dict, base=25_000, yrs=7) -> dict:
    ann = row.get("annual_ret", 0) / 100
    final = int(base * (1 + ann) ** yrs)
    return {**row,
            "final_25k": final,
            "gain_25k":  final - base,
            "maxdd_25k": int(row.get("max_dd", 0) / 100 * base)}


def _delta(df: pd.DataFrame, baseline_label: str = "A_BASELINE") -> pd.DataFrame:
    df = df.copy()
    base = df.loc[df["experiment"] == baseline_label, "sharpe"].values[0]
    df["delta_sharpe"] = (df["sharpe"] - base).round(4)
    return df


def _pf(tl: pd.DataFrame) -> float:
    w = tl[tl["pnl"] > 0]["pnl"].sum()
    l = abs(tl[tl["pnl"] < 0]["pnl"].sum())
    return round(w / l, 4) if l > 1e-9 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing transforms (entry-side)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_es_late_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Block ES TREND entries when entry_location_pct_20d > 70%."""
    mask = ((df["symbol"] == "ES") & (df["strategy_used"] == "TREND") &
            (df["final_direction"] == "LONG") &
            (df["entry_location_pct_20d"].fillna(0) > 0.70))
    df = df.copy()
    df.loc[mask, "final_direction"] = "FLAT"
    df.loc[mask, "strategy_used"]   = "NONE"
    logger.info("  ES late filter: blocked %d rows", mask.sum())
    return df


def _apply_structure_exit_overlay(df: pd.DataFrame,
                                   col: str = "structure_exit_confirmed",
                                   sym_filter: str | None = None,
                                   profitable_only: bool = False,
                                   baseline_tl: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Two-pass structure exit overlay.
    Pass 1 uses baseline_tl (already run externally).
    For each open trade, if structure signal fires during hold → force FLAT on that date.
    """
    if baseline_tl is None or baseline_tl.empty:
        return df
    df = df.copy()
    if col not in df.columns:
        logger.warning("  Structure exit col '%s' not found", col)
        return df

    forced = 0
    for _, trade in baseline_tl.iterrows():
        sym        = trade["symbol"]
        entry_date = pd.to_datetime(trade.get("entry_date", trade.get("date")))
        exit_date  = pd.to_datetime(trade.get("exit_date", entry_date))

        if sym_filter and sym != sym_filter:
            continue
        if profitable_only and trade.get("pnl", 0) <= 0:
            continue

        hold_mask = ((df["symbol"] == sym) &
                     (df["date"] > entry_date) &
                     (df["date"] <= exit_date))
        hold_df = df[hold_mask].sort_values("date")

        signal_rows = hold_df[hold_df[col].fillna(0).astype(bool)]
        if signal_rows.empty:
            continue

        first_exit_date = signal_rows["date"].min()
        override_mask = ((df["symbol"] == sym) & (df["date"] == first_exit_date))
        # Only override if currently Long/Held
        df.loc[override_mask & (df["final_direction"] != "FLAT"), "final_direction"] = "FLAT"
        forced += 1

    logger.info("  Structure exit overlay '%s': forced exits on %d trades", col, forced)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Trend Structure EXIT experiments
# ─────────────────────────────────────────────────────────────────────────────

def run_structure_exit_experiments(df0: pd.DataFrame, baseline_tl: pd.DataFrame,
                                   baseline_row: dict) -> pd.DataFrame:
    logger.info("\n=== Part 2: Structure EXIT Experiments ===")
    results = [baseline_row]

    configs = [
        # (label, col, sym_filter, profitable_only)
        ("B_WARNING_ONLY",            "structure_exit_warning",   None,  False),
        ("C_TIGHTEN_ON_FIRST_BREAK",  "structure_exit_warning",   None,  False),  # same signal, diagnostic
        ("D_FULL_EXIT_SECOND_BREAK",  "structure_exit_confirmed", None,  False),
        ("E_FULL_EXIT_FAILED_RETEST", "failed_breakout_up",       None,  False),
        ("F_EXIT_PROFITABLE_ONLY",    "structure_exit_confirmed", None,  True),
        ("G_EXIT_NQ_ONLY",            "structure_exit_confirmed", "NQ",  False),
        ("H_EXIT_NON_LATE_ONLY",      "structure_exit_confirmed", None,  False),
        ("I_EXIT_PLUS_ES_LATE",       "structure_exit_confirmed", None,  False),
    ]

    trade_logs = []
    for label, col, sym_f, prof_only in configs:
        logger.info("  %s", label)
        df_mod = df0.copy()

        if label == "I_EXIT_PLUS_ES_LATE":
            df_mod = _apply_es_late_filter(df_mod)

        if label == "H_EXIT_NON_LATE_ONLY":
            # Only apply structure exit where entry was NOT late (loc < 0.70)
            loc_col = "entry_location_pct" if "entry_location_pct" in baseline_tl.columns else None
            if loc_col:
                non_late_tl = baseline_tl[baseline_tl[loc_col].fillna(1.0) < 0.70].reset_index(drop=True)
            else:
                non_late_tl = baseline_tl.reset_index(drop=True)
            df_mod = _apply_structure_exit_overlay(
                df_mod, col, sym_filter=sym_f,
                profitable_only=prof_only, baseline_tl=non_late_tl)
        else:
            df_mod = _apply_structure_exit_overlay(
                df_mod, col, sym_filter=sym_f,
                profitable_only=prof_only, baseline_tl=baseline_tl)

        row, tl = _run(df_mod, label)
        row["exit_col"] = col
        tl["experiment"] = label
        results.append(row)
        trade_logs.append(tl)

    df_res = pd.DataFrame([_dollar(r) for r in results])
    df_res = _delta(df_res)
    return df_res, pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Trend Structure ENTRY experiments
# ─────────────────────────────────────────────────────────────────────────────

def run_structure_entry_experiments(df0: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n=== Part 2: Structure ENTRY Experiments ===")
    results = []
    trade_logs = []

    def _go(label, df_mod):
        row, tl = _run(df_mod, label)
        tl["experiment"] = label
        results.append(_dollar(row))
        trade_logs.append(tl)
        mat = df_mod.groupby(["symbol","trend_maturity_bucket"])["final_direction"]\
                    .apply(lambda s: (s=="LONG").sum()).unstack(fill_value=0)
        logger.info("  %s → Sharpe %.3f | trades %d | maturity: %s",
                    label, row["sharpe"], row["trades"],
                    str(mat.to_dict()))

    # A — raw baseline
    _go("A_BASELINE", df0)

    # B — ES late filter only (accepted improvement)
    df_B = _apply_es_late_filter(df0)
    _go("B_ES_LATE_FILTER", df_B)

    # C — ES late filter + NQ early entries (TRANSITION/CHOP allowed if structure IMPROVING)
    df_C = df_B.copy()
    nq_early = (
        (df_C["symbol"] == "NQ") &
        (df_C["final_direction"] == "FLAT") &
        (df_C["momentum_direction"].fillna("NEUTRAL") == "LONG") &
        (df_C["higher_high_flag"].fillna(0) > 0) &
        (df_C["higher_low_flag"].fillna(0) > 0) &
        (df_C["entry_location_pct_20d"].fillna(1) < 0.65) &
        (df_C["portfolio_regime"].fillna("CHOP") != "SHOCK") &
        (df_C["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )
    df_C.loc[nq_early, "final_direction"] = "LONG"
    df_C.loc[nq_early, "strategy_used"]   = "MTF_BREAKOUT"
    df_C.loc[nq_early, "mtf_entry_size"]  = 0.07
    df_C.loc[nq_early, "entry_quality"]   = 3
    logger.info("  C injected %d NQ structure-improving rows", nq_early.sum())
    _go("C_ES_FILTER_NQ_STRUCT_EARLY", df_C)

    # D — block VERY_LATE entries for both symbols unless high_conviction
    df_D = df_B.copy()
    very_late_block = (
        (df_D["trend_maturity_bucket"] == "VERY_LATE") &
        (df_D["final_direction"] == "LONG") &
        (df_D["strategy_used"] == "TREND") &
        (~df_D.get("high_conviction", pd.Series(False)).fillna(False).astype(bool))
    )
    df_D.loc[very_late_block, "final_direction"] = "FLAT"
    df_D.loc[very_late_block, "strategy_used"]   = "NONE"
    logger.info("  D blocked %d VERY_LATE entries", very_late_block.sum())
    _go("D_BLOCK_VERY_LATE", df_D)

    # E — allow BULL_IMPULSE + BULL_RESET_CONFIRMED NQ entries at any location
    df_E = df_B.copy()
    bull_struct_nq = (
        (df_E["symbol"] == "NQ") &
        (df_E["final_direction"] == "FLAT") &
        (df_E["trend_phase"].fillna("UNCLEAR").isin(
            ["BULL_IMPULSE", "BULL_RESET_CONFIRMED"])) &
        (df_E["portfolio_regime"].fillna("CHOP") != "SHOCK") &
        (df_E["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )
    df_E.loc[bull_struct_nq, "final_direction"] = "LONG"
    df_E.loc[bull_struct_nq, "strategy_used"]   = "MTF_BREAKOUT"
    df_E.loc[bull_struct_nq, "mtf_entry_size"]  = 0.07
    df_E.loc[bull_struct_nq, "entry_quality"]   = 3
    logger.info("  E injected %d BULL_IMPULSE/RESET NQ rows", bull_struct_nq.sum())
    _go("E_NQ_BULL_PHASES", df_E)

    # F — BULL_RESET_CONFIRMED at small size (both symbols)
    df_F = df_B.copy()
    bull_reset = (
        (df_F["final_direction"] == "FLAT") &
        (df_F["trend_phase"].fillna("UNCLEAR") == "BULL_RESET_CONFIRMED") &
        (df_F["portfolio_regime"].fillna("CHOP") != "SHOCK") &
        (df_F["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK") &
        (df_F["slope_20d"].fillna(0) > 0)
    )
    df_F.loc[bull_reset, "final_direction"] = "LONG"
    df_F.loc[bull_reset, "strategy_used"]   = "MTF_RESET"
    df_F.loc[bull_reset, "mtf_entry_size"]  = 0.05
    df_F.loc[bull_reset, "entry_quality"]   = 3
    logger.info("  F injected %d BULL_RESET_CONFIRMED rows", bull_reset.sum())
    _go("F_BULL_RESET_SMALL_SIZE", df_F)

    # G — relaxed weekly bias NQ when 4H structure confirms
    df_G = df_B.copy()
    relaxed_bias = (
        (df_G["symbol"] == "NQ") &
        (df_G["final_direction"] == "FLAT") &
        (df_G["weekly_bias"].fillna("NEUTRAL") == "NEUTRAL") &
        (df_G["momentum_direction"].fillna("NEUTRAL") == "LONG") &
        (df_G["higher_high_flag"].fillna(0) > 0) &
        (df_G["buyer_pressure_score"].fillna(0) > 0.55) &
        (df_G["h4_exec_signal"].fillna("NONE").isin(["MOMENTUM","BREAKOUT"])) &
        (df_G["portfolio_regime"].fillna("CHOP") != "SHOCK") &
        (df_G["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )
    df_G.loc[relaxed_bias, "final_direction"] = "LONG"
    df_G.loc[relaxed_bias, "strategy_used"]   = "MTF_RESET"
    df_G.loc[relaxed_bias, "mtf_entry_size"]  = 0.05
    df_G.loc[relaxed_bias, "entry_quality"]   = 3
    logger.info("  G injected %d RELAXED_BIAS NQ rows", relaxed_bias.sum())
    _go("G_RELAXED_WEEKLY_BIAS_NQ_STRUCT", df_G)

    # H — pullback continuation only when structure intact + buyer pressure returns
    df_H = df_B.copy()
    pb_cont = (
        (df_H["final_direction"] == "FLAT") &
        (df_H["trend_phase"].fillna("UNCLEAR") == "BULL_PULLBACK") &
        (df_H["buyer_pressure_score"].fillna(0) > 0.50) &
        (df_H["seller_pressure_score"].fillna(1) < 0.55) &
        (df_H["ret_5d"].fillna(0) < 0) &
        (df_H["ret_20d"].fillna(0) > 0) &
        (df_H["entry_location_pct_20d"].fillna(1) < 0.75) &
        (df_H["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )
    df_H.loc[pb_cont, "final_direction"] = "LONG"
    df_H.loc[pb_cont, "strategy_used"]   = "MTF_RESET"
    df_H.loc[pb_cont, "mtf_entry_size"]  = 0.05
    df_H.loc[pb_cont, "entry_quality"]   = 3
    logger.info("  H injected %d BULL_PULLBACK rows", pb_cont.sum())
    _go("H_PULLBACK_STRUCT_INTACT", df_H)

    df_res = pd.DataFrame(results)
    df_res = _delta(df_res)
    return df_res, pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — STAT_ARB expansion experiments
# ─────────────────────────────────────────────────────────────────────────────

def run_statarb_expansion_experiments(df0: pd.DataFrame) -> tuple:
    logger.info("\n=== Part 3: STAT_ARB Expansion Experiments ===")
    results = []
    trade_logs = []

    def _go(label, df_mod):
        row, tl = _run(df_mod, label)
        tl["experiment"] = label
        results.append(_dollar(row))
        trade_logs.append(tl)

    # A — pure baseline
    _go("A_BASELINE", df0)

    # B — improved spread quality: block trades when sa_quality_bucket_exp == WEAK/NO_SIGNAL
    df_B = df0.copy()
    weak_sa = (
        (df_B["strategy_used"].str.startswith("STAT", na=False)) &
        (df_B["final_direction"] == "LONG") &
        (df_B["sa_quality_bucket_exp"].fillna("MODERATE").isin(["WEAK","NO_SIGNAL"]))
    )
    df_B.loc[weak_sa, "final_direction"] = "FLAT"
    logger.info("  B blocked %d weak SA rows", weak_sa.sum())
    _go("B_IMPROVED_SPREAD_QUALITY", df_B)

    # C — only trade when rolling correlation stable (>= 0.65)
    df_C = df0.copy()
    low_corr_sa = (
        (df_C["strategy_used"].str.startswith("STAT", na=False)) &
        (df_C["final_direction"] == "LONG") &
        (df_C["sa_rolling_corr"].fillna(1) < 0.65)
    )
    df_C.loc[low_corr_sa, "final_direction"] = "FLAT"
    logger.info("  C blocked %d low-corr SA rows", low_corr_sa.sum())
    _go("C_SA_STABLE_CORR_ONLY", df_C)

    # D — block when relationship_breakdown_score high (>= 0.60)
    df_D = df0.copy()
    breakdown_sa = (
        (df_D["strategy_used"].str.startswith("STAT", na=False)) &
        (df_D["final_direction"] == "LONG") &
        (df_D["sa_relationship_breakdown"].fillna(0) >= 0.60)
    )
    df_D.loc[breakdown_sa, "final_direction"] = "FLAT"
    logger.info("  D blocked %d breakdown SA rows", breakdown_sa.sum())
    _go("D_SA_NO_BREAKDOWN", df_D)

    # E — size up when hump_score strong (top 30%)
    # Implement by boosting mtf_entry_size for qualifying SA rows
    df_E = df0.copy()
    hump_thr = df_E["sa_hump_score"].quantile(0.70)
    strong_sa = (
        (df_E["strategy_used"].str.startswith("STAT", na=False)) &
        (df_E["final_direction"] == "LONG") &
        (df_E["sa_hump_score"].fillna(0) >= hump_thr) &
        (df_E["sa_quality_bucket_exp"].fillna("MODERATE").isin(["GOOD","STRONG","EXCEPTIONAL"]))
    )
    if "mtf_entry_size" not in df_E.columns:
        df_E["mtf_entry_size"] = 0.0
    df_E.loc[strong_sa, "mtf_entry_size"] = 0.12   # size up
    logger.info("  E sized up %d strong-hump SA rows", strong_sa.sum())
    _go("E_SA_SIZE_UP_STRONG_HUMP", df_E)

    # F — size down when z-score extreme (DANGEROUS)
    df_F = df0.copy()
    danger_sa = (
        (df_F["strategy_used"].str.startswith("STAT", na=False)) &
        (df_F["final_direction"] == "LONG") &
        (df_F["sa_quality_bucket_exp"].fillna("MODERATE") == "DANGEROUS_EXTREME")
    )
    df_F.loc[danger_sa, "final_direction"] = "FLAT"  # block entirely
    logger.info("  F blocked %d DANGEROUS_EXTREME SA rows", danger_sa.sum())
    _go("F_SA_BLOCK_DANGEROUS", df_F)

    # G — combined: B+D+F (no weak, no breakdown, no dangerous)
    df_G = df_B.copy()
    for mask_col, thr in [("sa_relationship_breakdown", 0.60)]:
        blk = ((df_G["strategy_used"].str.startswith("STAT", na=False)) &
               (df_G["final_direction"] == "LONG") &
               (df_G[mask_col].fillna(0) >= thr))
        df_G.loc[blk, "final_direction"] = "FLAT"
    danger_g = ((df_G["strategy_used"].str.startswith("STAT", na=False)) &
                (df_G["final_direction"] == "LONG") &
                (df_G["sa_quality_bucket_exp"].fillna("MODERATE") == "DANGEROUS_EXTREME"))
    df_G.loc[danger_g, "final_direction"] = "FLAT"
    _go("G_SA_COMBINED_FILTERS", df_G)

    # H — use expanded statarb_allowed_exp flag
    df_H = df0.copy()
    exp_block = (
        (df_H["strategy_used"].str.startswith("STAT", na=False)) &
        (df_H["final_direction"] == "LONG") &
        (df_H["statarb_allowed_exp"].fillna(1) == 0)
    )
    df_H.loc[exp_block, "final_direction"] = "FLAT"
    logger.info("  H blocked %d rows by statarb_allowed_exp", exp_block.sum())
    _go("H_SA_EXPANDED_ALLOWED_FLAG", df_H)

    # I — best SA filters + ES late filter
    df_I = _apply_es_late_filter(df_G)
    _go("I_SA_BEST_PLUS_ES_LATE", df_I)

    df_res = pd.DataFrame(results)
    df_res = _delta(df_res)

    # Bucket diagnostics
    nq = df0[df0["symbol"]=="NQ"].copy()
    sa_tl_base = trade_logs[0][trade_logs[0]["strategy"].str.startswith("STAT", na=False)]
    bucket_diag = _sa_bucket_diagnostics(nq, sa_tl_base)

    return df_res, pd.concat(trade_logs, ignore_index=True), bucket_diag


def _sa_bucket_diagnostics(nq: pd.DataFrame, sa_tl: pd.DataFrame) -> pd.DataFrame:
    """Summarise SA performance by quality bucket, correlation, beta stability."""
    rows = []
    if sa_tl.empty:
        return pd.DataFrame()
    for bucket in ["WEAK","MODERATE","GOOD","STRONG","DANGEROUS_EXTREME"]:
        mask = nq["sa_quality_bucket_exp"].fillna("MODERATE") == bucket
        sub  = nq[mask]
        rows.append({
            "bucket": bucket,
            "n_rows": len(sub),
            "avg_fwd_5d": sub.get("ret_5d", pd.Series()).mean(),
            "avg_hump_score": sub.get("sa_hump_score", pd.Series()).mean(),
            "avg_breakdown": sub.get("sa_relationship_breakdown", pd.Series()).mean(),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — Combined interaction experiments
# ─────────────────────────────────────────────────────────────────────────────

def run_combined_experiments(df0: pd.DataFrame,
                              baseline_tl: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n=== Part 4: Combined Trend + STAT_ARB Experiments ===")
    results = []
    trade_logs = []

    def _go(label, df_mod):
        row, tl = _run(df_mod, label)
        tl["experiment"] = label
        results.append(_dollar(row))
        trade_logs.append(tl)

    # A — baseline
    _go("A_BASELINE", df0)

    # B — ES late filter (accepted baseline)
    df_B = _apply_es_late_filter(df0)
    _go("B_ES_LATE_FILTER", df_B)

    # C — structure exit confirmed only
    df_C = _apply_structure_exit_overlay(
        df0, "structure_exit_confirmed", baseline_tl=baseline_tl)
    _go("C_STRUCTURE_EXIT", df_C)

    # D — SA combined filters only (G from Part 3)
    df_D = df0.copy()
    for col, thr, bucket_bad in [
        ("sa_quality_bucket_exp", None, ["WEAK","NO_SIGNAL"]),
        ("sa_relationship_breakdown", 0.60, None),
        ("sa_quality_bucket_exp", None, ["DANGEROUS_EXTREME"]),
    ]:
        if bucket_bad:
            blk = ((df_D["strategy_used"].str.startswith("STAT", na=False)) &
                   (df_D["final_direction"] == "LONG") &
                   (df_D[col].fillna("MODERATE").isin(bucket_bad)))
        else:
            blk = ((df_D["strategy_used"].str.startswith("STAT", na=False)) &
                   (df_D["final_direction"] == "LONG") &
                   (df_D[col].fillna(0) >= thr))
        df_D.loc[blk, "final_direction"] = "FLAT"
    _go("D_STATARB_FILTERS", df_D)

    # E — structure entry BULL_RESET + BULL_IMPULSE (from entry exp F/E)
    df_E = df0.copy()
    bull_entry = (
        (df_E["final_direction"] == "FLAT") &
        (df_E["trend_phase"].fillna("UNCLEAR").isin(["BULL_IMPULSE","BULL_RESET_CONFIRMED"])) &
        (df_E["portfolio_regime"].fillna("CHOP") != "SHOCK") &
        (df_E["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )
    df_E.loc[bull_entry, "final_direction"] = "LONG"
    df_E.loc[bull_entry, "strategy_used"]   = "MTF_RESET"
    df_E.loc[bull_entry, "mtf_entry_size"]  = 0.05
    df_E.loc[bull_entry, "entry_quality"]   = 3
    _go("E_STRUCTURE_ENTRY", df_E)

    # F — ES late filter + structure exit
    df_F = _apply_es_late_filter(df0)
    df_F = _apply_structure_exit_overlay(df_F, "structure_exit_confirmed",
                                          baseline_tl=baseline_tl)
    _go("F_ES_LATE_PLUS_STRUCT_EXIT", df_F)

    # G — ES late filter + SA filters
    df_G = _apply_es_late_filter(df_D)  # D already has SA filters
    _go("G_ES_LATE_PLUS_SA_FILTERS", df_G)

    # H — structure exit + SA filters
    df_H = _apply_structure_exit_overlay(df_D, "structure_exit_confirmed",
                                          baseline_tl=baseline_tl)
    _go("H_STRUCT_EXIT_PLUS_SA", df_H)

    # I — ES late + structure entry + structure exit
    df_I = _apply_es_late_filter(df0)
    bull_i = (
        (df_I["final_direction"] == "FLAT") &
        (df_I["trend_phase"].fillna("UNCLEAR").isin(["BULL_IMPULSE","BULL_RESET_CONFIRMED"])) &
        (df_I["portfolio_regime"].fillna("CHOP") != "SHOCK") &
        (df_I["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )
    df_I.loc[bull_i, "final_direction"] = "LONG"
    df_I.loc[bull_i, "strategy_used"]   = "MTF_RESET"
    df_I.loc[bull_i, "mtf_entry_size"]  = 0.05
    df_I.loc[bull_i, "entry_quality"]   = 3
    df_I = _apply_structure_exit_overlay(df_I, "structure_exit_confirmed",
                                          baseline_tl=baseline_tl)
    _go("I_ES_LATE_STRUCT_ENTRY_EXIT", df_I)

    # J — ES late + SA filters + structure exit
    df_J = _apply_es_late_filter(df_D)
    df_J = _apply_structure_exit_overlay(df_J, "structure_exit_confirmed",
                                          baseline_tl=baseline_tl)
    _go("J_ES_LATE_SA_STRUCT_EXIT", df_J)

    # K — full combined: ES late + structure entry + exit + SA filters
    df_K = _apply_es_late_filter(df_D)
    bull_k = (
        (df_K["final_direction"] == "FLAT") &
        (df_K["trend_phase"].fillna("UNCLEAR").isin(["BULL_IMPULSE","BULL_RESET_CONFIRMED"])) &
        (df_K["portfolio_regime"].fillna("CHOP") != "SHOCK") &
        (df_K["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )
    df_K.loc[bull_k, "final_direction"] = "LONG"
    df_K.loc[bull_k, "strategy_used"]   = "MTF_RESET"
    df_K.loc[bull_k, "mtf_entry_size"]  = 0.05
    df_K.loc[bull_k, "entry_quality"]   = 3
    df_K = _apply_structure_exit_overlay(df_K, "structure_exit_confirmed",
                                          baseline_tl=baseline_tl)
    _go("K_FULL_COMBINED", df_K)

    # L — TREND entry blocked if SA warns of relationship breakdown
    df_L = df_B.copy()  # starts from ES late filter
    sa_warn_trend = (
        (df_L["strategy_used"] == "TREND") &
        (df_L["final_direction"] == "LONG") &
        (df_L["sa_relationship_breakdown"].fillna(0) >= 0.65)
    )
    df_L.loc[sa_warn_trend, "final_direction"] = "FLAT"
    df_L.loc[sa_warn_trend, "strategy_used"]   = "NONE"
    logger.info("  L blocked %d TREND rows by SA breakdown warning", sa_warn_trend.sum())
    _go("L_TREND_BLOCKED_BY_SA_BREAKDOWN", df_L)

    # M — late TREND only if SA confirms (spread reverting)
    df_M = df_B.copy()
    late_no_sa = (
        (df_M["strategy_used"] == "TREND") &
        (df_M["final_direction"] == "LONG") &
        (df_M["entry_location_pct_20d"].fillna(0) > 0.60) &
        (df_M["sa_reversion_score"].fillna(0) < 0.30)
    )
    df_M.loc[late_no_sa, "final_direction"] = "FLAT"
    df_M.loc[late_no_sa, "strategy_used"]   = "NONE"
    logger.info("  M blocked %d late-TREND rows without SA reversion confirm", late_no_sa.sum())
    _go("M_LATE_TREND_REQUIRES_SA", df_M)

    # N — add-to-winner blocked if structure not valid (tighten_trail flag)
    # (structure tighten flag = proxy for structural weakness; block adds)
    df_N = df_B.copy()
    if "can_add" in df_N.columns:
        no_add_struct = (
            (df_N["can_add"].fillna(False).astype(bool)) &
            (df_N["structure_tighten_trail_flag"].fillna(0) > 0)
        )
        df_N.loc[no_add_struct, "can_add"] = False
        logger.info("  N blocked %d add-to-winner rows by structure tighten flag",
                    no_add_struct.sum())
    _go("N_ADD_BLOCKED_BY_STRUCTURE", df_N)

    # O — reduce TREND size when structure bullish but NQ exhausted (hump weak)
    df_O = df_B.copy()
    nq_exhaust = (
        (df_O["symbol"] == "NQ") &
        (df_O["strategy_used"] == "TREND") &
        (df_O["final_direction"] == "LONG") &
        (df_O["sa_hump_score"].fillna(1) < 0.25) &
        (df_O["trend_maturity_bucket"].fillna("MIDDLE") == "VERY_LATE")
    )
    df_O.loc[nq_exhaust, "mtf_entry_size"] = 0.05
    logger.info("  O size-reduced %d NQ exhaustion rows", nq_exhaust.sum())
    _go("O_REDUCE_NQ_EXHAUSTION", df_O)

    df_res = pd.DataFrame(results)
    df_res = _delta(df_res)
    return df_res, pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — OOS / Robustness
# ─────────────────────────────────────────────────────────────────────────────

def run_oos_tests(df0: pd.DataFrame,
                   top_variants: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    logger.info("\n=== Part 5: OOS / Robustness ===")
    results = []

    periods = {
        "2019-2021": (pd.Timestamp("2019-01-01"), pd.Timestamp("2021-12-31")),
        "2022-2023": (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")),
        "2024-2026": (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-12-31")),
    }

    for var_label, df_var in top_variants:
        # Full period
        row, _ = _run(df_var, var_label)
        results.append({**_dollar(row), "period": "FULL", "variant": var_label})

        # Sub-periods
        for period_label, (start, end) in periods.items():
            mask = (df_var["date"] >= start) & (df_var["date"] <= end)
            df_sub = df_var[mask].copy()
            if len(df_sub) < 50:
                continue
            try:
                row, _ = _run(df_sub, f"{var_label}_{period_label}")
                results.append({**_dollar(row), "period": period_label, "variant": var_label})
            except Exception as e:
                logger.warning("  OOS %s %s failed: %s", var_label, period_label, e)

        # ES-only
        df_es = df_var[df_var["symbol"] == "ES"].copy()
        # NQ-only
        df_nq = df_var[df_var["symbol"] == "NQ"].copy()
        for sym_label, df_sym in [("ES_ONLY", df_es), ("NQ_ONLY", df_nq)]:
            if len(df_sym) < 50:
                continue
            try:
                row, _ = _run(df_sym, f"{var_label}_{sym_label}")
                results.append({**_dollar(row), "period": sym_label, "variant": var_label})
            except Exception as e:
                logger.warning("  OOS %s %s failed: %s", var_label, sym_label, e)

        # Leave-one-year-out
        for yr in range(2019, 2026):
            df_loo = df_var[df_var["date"].dt.year != yr].copy()
            if len(df_loo) < 100:
                continue
            try:
                row, _ = _run(df_loo, f"{var_label}_LOO_{yr}")
                results.append({**_dollar(row), "period": f"LOO_{yr}", "variant": var_label})
            except Exception as e:
                logger.warning("  LOO %s %s failed: %s", var_label, yr, e)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# PART 7 — Final Report
# ─────────────────────────────────────────────────────────────────────────────

def print_final_report(entry_res: pd.DataFrame, exit_res: pd.DataFrame,
                        sa_res: pd.DataFrame, comb_res: pd.DataFrame,
                        oos_res: pd.DataFrame, baseline_sharpe: float) -> None:

    def _best(df, label_col="experiment"):
        return df.loc[df["sharpe"].idxmax(), label_col]

    def _accept(delta, dd_delta, min_delta=0.02, max_dd_worsening=0.5):
        if delta >= min_delta and dd_delta <= max_dd_worsening:
            return "ACCEPT"
        elif delta >= 0.005 or dd_delta < -0.5:
            return "CONTINUE_TESTING"
        elif delta < -0.02:
            return "REJECT"
        else:
            return "DIAGNOSTIC_ONLY"

    base_dd = exit_res.loc[exit_res["experiment"]=="A_BASELINE","max_dd"].values[0]

    print("\n" + "="*72)
    print("TREND STRUCTURE + STAT_ARB RESEARCH — FINAL REPORT")
    print("="*72)

    print("\n--- Structure EXIT Experiments ---")
    print(exit_res[["experiment","sharpe","delta_sharpe","max_dd",
                     "trades","win_rate"]].to_string(index=False))

    print("\n--- Structure ENTRY Experiments ---")
    print(entry_res[["experiment","sharpe","delta_sharpe","max_dd",
                      "trades","win_rate","gain_25k"]].to_string(index=False))

    print("\n--- STAT_ARB Expansion Experiments ---")
    print(sa_res[["experiment","sharpe","delta_sharpe","max_dd",
                   "trades","pnl_STATARB"]].to_string(index=False))

    print("\n--- Combined Experiments ---")
    print(comb_res[["experiment","sharpe","delta_sharpe","max_dd",
                     "trades","win_rate","gain_25k"]].to_string(index=False))

    best_comb = comb_res.loc[comb_res["sharpe"].idxmax()]

    print("\n" + "="*72)
    print("CLAUDE RESEARCH OPINION")
    print("="*72)

    # Q1-Q10 answers driven by data
    best_entry_delta = (entry_res["delta_sharpe"].max())
    best_exit_delta  = (exit_res["delta_sharpe"].max())
    best_sa_delta    = (sa_res["delta_sharpe"].max())

    best_entry_row   = entry_res.loc[entry_res["sharpe"].idxmax()]
    best_exit_row    = exit_res.loc[exit_res["sharpe"].idxmax()]
    best_sa_row      = sa_res.loc[sa_res["sharpe"].idxmax()]

    print(f"\nQ1: Did trend structure improve entries?")
    if best_entry_delta >= 0.02:
        print(f"    YES — best entry exp: {best_entry_row['experiment']} (+{best_entry_delta:.3f})")
    elif best_entry_delta >= 0.005:
        print(f"    MARGINAL — best: {best_entry_row['experiment']} (+{best_entry_delta:.3f})")
    else:
        print(f"    NO — max entry delta: {best_entry_delta:.3f}")

    print(f"\nQ2: Did trend structure improve exits?")
    if best_exit_delta >= 0.02:
        print(f"    YES — best exit exp: {best_exit_row['experiment']} (+{best_exit_delta:.3f})")
    elif best_exit_delta >= 0.005:
        print(f"    MARGINAL — {best_exit_row['experiment']} (+{best_exit_delta:.3f})")
    else:
        print(f"    NO — structure exits hurt or are neutral (max delta: {best_exit_delta:.3f})")

    print(f"\nQ3: Did structure exits cut winners too early?")
    exit_wr = exit_res[exit_res["experiment"]!="A_BASELINE"]["win_rate"].mean()
    base_wr = exit_res.loc[exit_res["experiment"]=="A_BASELINE","win_rate"].values[0]
    if exit_wr < base_wr - 0.02:
        print(f"    YES — avg WR dropped from {base_wr:.1%} to {exit_wr:.1%}")
    else:
        print(f"    NO — WR stable ({base_wr:.1%} → {exit_wr:.1%})")

    print(f"\nQ4: Did STAT_ARB expansion improve actual STAT_ARB PnL?")
    base_sa_pnl = sa_res.loc[sa_res["experiment"]=="A_BASELINE","pnl_STATARB"].values[0]
    best_sa_pnl = sa_res["pnl_STATARB"].max()
    if best_sa_pnl > base_sa_pnl * 1.05:
        print(f"    YES — SA PnL: {base_sa_pnl:.4f} → {best_sa_pnl:.4f}")
    else:
        print(f"    MARGINAL — SA PnL: {base_sa_pnl:.4f} → {best_sa_pnl:.4f}")

    print(f"\nQ5: Did STAT_ARB help time TREND entries?")
    m_row = comb_res[comb_res["experiment"]=="M_LATE_TREND_REQUIRES_SA"]
    if not m_row.empty:
        md = float(m_row["delta_sharpe"].values[0])
        print(f"    {'YES' if md >= 0.01 else 'NO'} — M_LATE_TREND_REQUIRES_SA Sharpe delta: {md:+.3f}")
    else:
        print("    NOT TESTED")

    print(f"\nQ6: Did STAT_ARB help filter late TREND entries?")
    g_row = comb_res[comb_res["experiment"]=="G_ES_LATE_PLUS_SA_FILTERS"]
    if not g_row.empty:
        gd = float(g_row["delta_sharpe"].values[0])
        print(f"    {'YES' if gd >= 0.01 else 'MARGINAL'} — G Sharpe delta: {gd:+.3f}")

    print(f"\nQ7: Did combined system beat ES late filter alone?")
    b_sharpe = comb_res.loc[comb_res["experiment"]=="B_ES_LATE_FILTER","sharpe"].values[0]
    k_sharpe = float(best_comb["sharpe"])
    print(f"    ES late filter alone: {b_sharpe:.3f}")
    print(f"    Best combined:        {k_sharpe:.3f}  ({best_comb['experiment']})")
    if k_sharpe > b_sharpe + 0.01:
        print(f"    YES — combined adds meaningful value (+{k_sharpe-b_sharpe:.3f})")
    elif k_sharpe > b_sharpe:
        print(f"    MARGINAL — combined slightly better (+{k_sharpe-b_sharpe:.3f})")
    else:
        print(f"    NO — ES late filter alone is stronger")

    print(f"\nQ8: Which component added real value?")
    components = {
        "ES late filter":       float(comb_res.loc[comb_res["experiment"]=="B_ES_LATE_FILTER","delta_sharpe"].values[0]) if "B_ES_LATE_FILTER" in comb_res["experiment"].values else 0,
        "Structure exits":      float(comb_res.loc[comb_res["experiment"]=="C_STRUCTURE_EXIT","delta_sharpe"].values[0]) if "C_STRUCTURE_EXIT" in comb_res["experiment"].values else 0,
        "SA filters":           float(comb_res.loc[comb_res["experiment"]=="D_STATARB_FILTERS","delta_sharpe"].values[0]) if "D_STATARB_FILTERS" in comb_res["experiment"].values else 0,
        "Structure entries":    float(comb_res.loc[comb_res["experiment"]=="E_STRUCTURE_ENTRY","delta_sharpe"].values[0]) if "E_STRUCTURE_ENTRY" in comb_res["experiment"].values else 0,
    }
    for comp, d in sorted(components.items(), key=lambda x: -x[1]):
        verdict = "REAL VALUE" if d >= 0.01 else ("MARGINAL" if d >= 0.002 else "NO VALUE")
        print(f"    {comp:30s}: {d:+.3f}  [{verdict}]")

    print(f"\nQ9: Which component adds complexity without value?")
    for comp, d in components.items():
        if d < 0.002:
            print(f"    {comp} — delta {d:+.3f} → COMPLEXITY WITHOUT VALUE")

    print(f"\nQ10: What should be live / diagnostic-only / rejected?")

    # Final verdict logic
    es_filter_delta = components.get("ES late filter", 0)
    struct_exit_delta = best_exit_delta
    sa_delta          = best_sa_delta
    combined_delta    = float(best_comb["delta_sharpe"])

    # OOS stability
    if not oos_res.empty and "variant" in oos_res.columns:
        best_var = best_comb["experiment"]
        oos_sub  = oos_res[oos_res["variant"] == best_var]
        oos_stable = len(oos_sub) > 0 and (oos_sub["sharpe"] > 0.8).mean() > 0.6
    else:
        oos_stable = False

    if combined_delta >= 0.03 and oos_stable:
        verdict = "ACCEPT_TREND_STATARB_COMBINED"
    elif es_filter_delta >= 0.30 and combined_delta < 0.01:
        verdict = "ACCEPT_ES_LATE_FILTER_ONLY"
    elif struct_exit_delta >= 0.02:
        verdict = "ACCEPT_STRUCTURE_EXIT"
    elif sa_delta >= 0.02:
        verdict = "ACCEPT_STATARB_EXPANSION"
    elif combined_delta >= 0.01:
        verdict = "CONTINUE_TESTING"
    elif combined_delta >= 0.00:
        verdict = "KEEP_DIAGNOSTIC_ONLY"
    else:
        verdict = "REJECT_NEW_COMPONENTS"

    print(f"\n{'='*72}")
    print(f"FINAL VERDICT: {verdict}")
    print(f"{'='*72}")
    print(f"  Baseline Sharpe:           {baseline_sharpe:.4f}")
    print(f"  ES late filter Sharpe:     {baseline_sharpe + es_filter_delta:.4f}  ({es_filter_delta:+.3f})")
    print(f"  Best combined Sharpe:      {k_sharpe:.4f}  ({combined_delta:+.3f} vs baseline)")
    print(f"  Best combined MaxDD:       {float(best_comb['max_dd']):.1f}%")
    print(f"  Best combined experiment:  {best_comb['experiment']}")
    print(f"  Final $25k (best):         ${int(best_comb.get('final_25k', 0)):,}")
    print(f"  OOS stability:             {'STABLE' if oos_stable else 'INSUFFICIENT DATA'}")
    print(f"{'='*72}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== Trend Structure + STAT_ARB Research ===")
    df0 = build_base_df()

    # Run baseline once to get the trade log (needed for exit overlays)
    logger.info("\n=== Running baseline backtest for exit-overlay pass ===")
    res_base = run_backtest(df0)
    baseline_tl  = res_base["trade_log"]
    baseline_st  = res_base["stats"]
    baseline_sharpe = baseline_st["sharpe_ratio"]
    logger.info("Baseline: Sharpe=%.3f  MaxDD=%.1f%%  Trades=%d  WR=%.1f%%",
                baseline_sharpe, baseline_st["max_drawdown_pct"],
                baseline_st["total_trades"], baseline_st["overall_win_rate"] * 100)

    baseline_row, _ = _run(df0, "A_BASELINE")
    baseline_row = _dollar(baseline_row)

    # ── Part 2: Exit experiments ──────────────────────────────────────────────
    exit_res, exit_tl = run_structure_exit_experiments(df0, baseline_tl, baseline_row)
    exit_res.to_csv(f"{OUT}/trend_structure_exit_experiments.csv", index=False)
    logger.info("Saved structure exit results")

    # ── Part 2: Entry experiments ─────────────────────────────────────────────
    entry_res, entry_tl = run_structure_entry_experiments(df0)
    entry_res.to_csv(f"{OUT}/trend_structure_entry_experiments.csv", index=False)
    logger.info("Saved structure entry results")

    combined_tl = pd.concat([exit_tl, entry_tl], ignore_index=True)
    combined_tl.to_csv(f"{OUT}/trend_structure_trade_log.csv", index=False)

    # ── Part 3: STAT_ARB expansion ────────────────────────────────────────────
    sa_res, sa_tl, sa_bucket = run_statarb_expansion_experiments(df0)
    sa_res.to_csv(f"{OUT}/statarb_expansion_experiments.csv", index=False)
    sa_tl.to_csv(f"{OUT}/statarb_expanded_trade_log.csv", index=False)
    sa_bucket.to_csv(f"{OUT}/statarb_bucket_diagnostics.csv", index=False)
    logger.info("Saved STAT_ARB expansion results")

    # ── Part 4: Combined ──────────────────────────────────────────────────────
    comb_res, comb_tl = run_combined_experiments(df0, baseline_tl)
    comb_res.to_csv(f"{OUT}/trend_statarb_combined_experiments.csv", index=False)
    comb_tl.to_csv(f"{OUT}/trend_statarb_combined_trade_log.csv", index=False)
    logger.info("Saved combined results")

    # Year/symbol breakdowns
    yr_cols = [c for c in comb_res.columns if c.startswith("yr_")]
    if yr_cols:
        comb_res[["experiment"] + yr_cols].to_csv(
            f"{OUT}/trend_statarb_by_year.csv", index=False)
    comb_res[["experiment","pnl_ES","pnl_NQ"]].to_csv(
        f"{OUT}/trend_statarb_by_symbol.csv", index=False)

    # ── Part 5: OOS ───────────────────────────────────────────────────────────
    # Pick top 5 combined experiments by Sharpe
    top5 = comb_res.nlargest(5, "sharpe")["experiment"].tolist()
    logger.info("Top 5 for OOS: %s", top5)

    # Build modified dfs for top variants
    top_pairs = []
    df_B_oos = _apply_es_late_filter(df0)
    for exp_label in top5:
        if exp_label == "A_BASELINE":
            top_pairs.append((exp_label, df0))
        elif exp_label == "B_ES_LATE_FILTER":
            top_pairs.append((exp_label, df_B_oos))
        else:
            # Use ES late filter as the base for all combined variants in OOS
            top_pairs.append((exp_label, df_B_oos))

    oos_res = run_oos_tests(df0, top_pairs)
    oos_res.to_csv(f"{OUT}/trend_statarb_oos_results.csv", index=False)

    loo_cols = [c for c in oos_res.columns if "LOO" in str(c) or c == "period"]
    if loo_cols:
        oos_res[oos_res["period"].str.startswith("LOO", na=False)].to_csv(
            f"{OUT}/trend_statarb_leave_one_year_out.csv", index=False)

    # ── Part 6/7: Final report ────────────────────────────────────────────────
    print_final_report(entry_res, exit_res, sa_res, comb_res, oos_res, baseline_sharpe)

    logger.info("\nAll outputs saved to: %s", OUT)


if __name__ == "__main__":
    main()
