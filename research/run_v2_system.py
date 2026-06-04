"""
run_v2_system.py — V2 Architecture Experiment Runner.

13 experiments (A through M):
  Long-only experiments:
    A_BASELINE_LONG_ONLY      — raw baseline (no modifications)
    B_BULL_RESET_ONLY         — add BULL_RESET_CONFIRMED long entries
    C_CONFIRMATION_SIZING     — confirmation-scaled sizing (schedules A-E)
    D_WINNER_EXTENSION        — extend winners by +1/+2/+3 days
    E_REGIME_ROUTER_ONLY      — regime activation flags (gating only, no new signals)
    F_LONG_V2_CORE            — stack B+C+D+E (full long-sleeve V2)

  Short-sleeve experiments (each on top of A_BASELINE):
    G_BEAR_TREND_SHORT        — BEAR_TREND_SHORT only
    H_FAILED_RALLY_SHORT      — FAILED_RALLY_SHORT only
    I_SHOCK_SHORT             — SHOCK_CONT short only
    J_BEAR_RESET_SHORT        — BEAR_RESET_CONFIRMED short only
    K_ALL_SHORTS_SMALL        — all 4 short types, small sizes

  Combined experiments:
    L_LONG_PLUS_BEST_SHORT    — F_LONG_V2_CORE + best individual short
    M_FULL_V2_SYSTEM          — F_LONG_V2_CORE + all confirmed shorts

Full metrics per experiment:
  Sharpe, Sortino, annual%, final $25k, MaxDD, avg DD,
  worst week, worst month, trades, WR, PF, avg trade PnL,
  PnL by year, PnL by symbol, PnL by strategy,
  short-specific: short trades, short WR, short PF, short avg PnL

Robustness splits:
  2019-2021, 2022-2023, 2024-2026, LOO (leave-one-year-out),
  ES-only, NQ-only, high-vol, low-vol

Final verdict: one of 7 outcomes based on objective criteria.
"""

import contextlib
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import config
from data_loader                    import load_daily, load_4h
from features                       import compute_features
from momentum_engine                import compute_momentum
from weekly_bias                    import compute_weekly_bias
from regime_engine                  import compute_regime
from volatility_engine              import compute_volatility_features
from intraday_4h_engine             import (compute_4h_features,
                                            aggregate_4h_to_daily,
                                            merge_4h_into_daily)
from mean_reversion                 import compute_mean_reversion
from stat_arb                       import compute_stat_arb
from sentiment_layer                import compute_sentiment
from signal_engine                  import compute_signals
from signal_quality_engine          import compute_signal_quality
from trend_continuation_edge_engine import compute_trend_continuation_edge
from regime_router                  import compute_regime_router
from short_sleeve_engine            import compute_short_signals
from backtest                       import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR = "output/v2_system"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Size constants (V2 long sleeve) ─────────────────────────────────────────
BULL_RESET_SIZE   = 0.08   # BULL_RESET_CONFIRMED entries
WINNER_EXT_SIZE   = 0.04   # add-to-winner extension size

# Confirmation-scaled sizing: initial TREND entry reduced, winner add increased
_SCHED_A = dict(VOL_SCALE_MAX=0.28, ADD_WINNER_SIZE_FRAC=0.50)   # baseline-like
_SCHED_B = dict(VOL_SCALE_MAX=0.24, ADD_WINNER_SIZE_FRAC=0.60)   # mild shift
_SCHED_C = dict(VOL_SCALE_MAX=0.20, ADD_WINNER_SIZE_FRAC=0.75)   # moderate
_SCHED_D = dict(VOL_SCALE_MAX=0.18, ADD_WINNER_SIZE_FRAC=0.90)   # aggressive add
_SCHED_E = dict(VOL_SCALE_MAX=0.15, ADD_WINNER_SIZE_FRAC=1.00)   # max aggressive

_HOLD_CFG = {**config.MAX_HOLD_DAYS, "MTF_BREAKOUT": 2, "MTF_RESET": 3,
             "BEAR_TREND": 4, "FAILED_RALLY": 3, "SHOCK_CONT": 2,
             "BEAR_RESET": 3}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_base_df() -> pd.DataFrame:
    logger.info("Building pipeline…")
    df = load_daily()
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
        for col, val in [
            ("cwt_chop_label", "CWT_UNCLEAR"), ("cwt_trade_allowed", 1),
            ("cwt_size_multiplier", 1.0),      ("cwt_confidence", 0.0),
            ("cwt_total_energy", 0.0),         ("cwt_entropy", 0.5),
            ("cwt_energy_z", 0.0),             ("cwt_compression_flag", 0),
            ("cwt_expansion_flag", 0),         ("cwt_high_energy_ratio", 1/3),
            ("cwt_mid_energy_ratio", 1/3),     ("cwt_slow_energy_ratio", 1/3),
        ]:
            if col not in df.columns:
                df[col] = val

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

    # Entry location features (needed by multiple engines)
    df = _add_entry_location(df)

    # Regime router (adds activation flags + router_size_scale)
    df = compute_regime_router(df)

    df["date"] = pd.to_datetime(df["date"])
    logger.info("Pipeline done: %d rows, %d cols", len(df), len(df.columns))
    return df


def _add_entry_location(df: pd.DataFrame) -> pd.DataFrame:
    if "entry_location_pct_20d" in df.columns:
        return df
    df = df.copy()
    df["entry_location_pct_20d"] = np.nan
    for sym, g in df.groupby("symbol", sort=False):
        idx = g.index
        s   = pd.Series(g["close"].values, index=idx)
        mn  = s.rolling(20, min_periods=5).min()
        mx  = s.rolling(20, min_periods=5).max()
        loc = (s - mn) / (mx - mn).replace(0, np.nan)
        df.loc[idx, "entry_location_pct_20d"] = loc.clip(0, 1).values
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Long-sleeve modifications
# ─────────────────────────────────────────────────────────────────────────────

def apply_es_late_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Block ES TREND LONG entries at late location (> 70%)."""
    loc  = df.get("entry_location_pct_20d", pd.Series(0.5, index=df.index)).fillna(0.5)
    mask = (
        (df["symbol"]          == "ES") &
        (df["strategy_used"]   == "TREND") &
        (df["final_direction"] == "LONG") &
        (loc > 0.70)
    )
    df = df.copy()
    df.loc[mask, "final_direction"] = "FLAT"
    df.loc[mask, "strategy_used"]   = "NONE"
    logger.info("  ES late filter blocked %d rows", int(mask.sum()))
    return df


def apply_bull_reset(df: pd.DataFrame) -> pd.DataFrame:
    """Inject BULL_RESET_CONFIRMED long entries on flat rows."""
    slope  = df["slope_20d"].fillna(0.0)
    ret20  = df["ret_20d"].fillna(0.0)
    ret5   = df["ret_5d"].fillna(0.0)
    bk_cnt = df.get("structure_break_count", pd.Series(0, index=df.index)).fillna(0)
    loc20  = df.get("entry_location_pct_20d", pd.Series(0.5, index=df.index)).fillna(0.5)
    regime = df["portfolio_regime"].fillna("CHOP")
    sent   = df["portfolio_sentiment_flag"].fillna("NORMAL")

    mask = (
        (df["final_direction"] == "FLAT") &
        (regime.isin(["TREND", "TRANSITION"])) &
        (slope > 0) &
        (ret20 > 0) &
        (ret5 < 0) &
        (bk_cnt == 0) &
        (loc20 < 0.80) &
        (sent != "HIGH_RISK")
    )
    df = df.copy()
    df.loc[mask, "final_direction"] = "LONG"
    df.loc[mask, "strategy_used"]   = "BULL_RESET_CONFIRMED"
    df.loc[mask, "mtf_entry_size"]  = BULL_RESET_SIZE
    df.loc[mask, "entry_quality"]   = 4
    logger.info("  BULL_RESET injected %d rows", int(mask.sum()))
    return df


def apply_winner_extension(df: pd.DataFrame, extra_days: int) -> pd.DataFrame:
    """
    Extend max hold for TREND/BULL_RESET winners.
    This patches config rather than modifying df rows.
    Returns the df unchanged — the config patch is applied in the run() call.
    """
    return df   # config patched at run time via override_config


# ─────────────────────────────────────────────────────────────────────────────
# Config context manager
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def override_config(**kwargs):
    orig = {k: getattr(config, k) for k in kwargs if hasattr(config, k)}
    for k, v in kwargs.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in orig.items():
            setattr(config, k, v)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sortino(returns: pd.Series) -> float:
    if len(returns) < 20:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) < 5:
        return float("nan")
    ds_std = float(downside.std()) * np.sqrt(252)
    if ds_std < 1e-9:
        return float("nan")
    return float(returns.mean()) * 252 / ds_std


def _profit_factor(tl: pd.DataFrame) -> float:
    if tl.empty or "pnl" not in tl.columns:
        return float("nan")
    w = tl[tl["pnl"] > 0]["pnl"].sum()
    l = abs(tl[tl["pnl"] < 0]["pnl"].sum())
    return w / l if l > 1e-9 else float("nan")


def _worst_week(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    weekly = returns.resample("W").sum()
    return float(weekly.min()) * 100 if not weekly.empty else 0.0


def _worst_month(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    monthly = returns.resample("ME").sum()
    return float(monthly.min()) * 100 if not monthly.empty else 0.0


def _avg_drawdown(returns: pd.Series) -> float:
    """Average of all drawdown periods (not just max)."""
    if returns.empty:
        return 0.0
    cumret = (1 + returns).cumprod()
    roll_max = cumret.cummax()
    dd = (cumret - roll_max) / roll_max
    in_dd = dd < 0
    if not in_dd.any():
        return 0.0
    return float(dd[in_dd].mean()) * 100


def _annual_returns_by_year(returns: pd.Series) -> dict:
    yr = {}
    for y in range(2019, 2027):
        m = returns.index.year == y
        if m.any():
            yr[y] = round(float((1 + returns[m]).prod() - 1) * 100, 2)
    return yr


def _pnl_by_symbol(tl: pd.DataFrame) -> dict:
    if tl.empty:
        return {"ES": 0.0, "NQ": 0.0}
    return {sym: round(float(g["pnl"].sum()), 4)
            for sym, g in tl.groupby("symbol")}


def _pnl_by_strategy(tl: pd.DataFrame) -> dict:
    if tl.empty or "strategy" not in tl.columns:
        return {}
    return {s: round(float(g["pnl"].sum()), 4)
            for s, g in tl.groupby("strategy")}


def _short_metrics(tl: pd.DataFrame) -> dict:
    short_strats = {"BEAR_TREND", "FAILED_RALLY", "SHOCK_CONT", "BEAR_RESET"}
    if tl.empty or "strategy" not in tl.columns:
        return dict(short_trades=0, short_wr=float("nan"),
                    short_pf=float("nan"), short_avg_pnl=float("nan"),
                    short_pnl_total=0.0)
    sh = tl[tl["strategy"].isin(short_strats)]
    if sh.empty:
        return dict(short_trades=0, short_wr=float("nan"),
                    short_pf=float("nan"), short_avg_pnl=float("nan"),
                    short_pnl_total=0.0)
    return dict(
        short_trades    = len(sh),
        short_wr        = round(float(sh["win"].mean()), 4),
        short_pf        = round(_profit_factor(sh), 4),
        short_avg_pnl   = round(float(sh["pnl"].mean()), 4),
        short_pnl_total = round(float(sh["pnl"].sum()), 4),
    )


def _final_equity(annual_ret_pct: float, years: float = 7.0) -> int:
    return int(25000 * (1 + annual_ret_pct / 100) ** years)


# ─────────────────────────────────────────────────────────────────────────────
# Single backtest runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(label: str, df: pd.DataFrame,
                   config_overrides: dict | None = None) -> tuple[dict, pd.DataFrame]:
    cfg = {**{"MAX_HOLD_DAYS": _HOLD_CFG}, **(config_overrides or {})}
    with override_config(**cfg):
        res = run_backtest(df)

    st  = res["stats"]
    tl  = res["trade_log"]
    pr  = res["portfolio_returns"]

    yr  = _annual_returns_by_year(pr)
    sym_pnl  = _pnl_by_symbol(tl)
    strat_pnl = _pnl_by_strategy(tl)
    short_m  = _short_metrics(tl)

    ann = st["annualized_return_pct"]
    row = dict(
        experiment      = label,
        sharpe          = round(st["sharpe_ratio"], 4),
        sortino         = round(_sortino(pr), 4),
        annual_ret_pct  = round(ann, 2),
        final_25k       = _final_equity(ann),
        max_dd_pct      = round(st["max_drawdown_pct"], 2),
        avg_dd_pct      = round(_avg_drawdown(pr), 2),
        worst_week_pct  = round(_worst_week(pr), 2),
        worst_month_pct = round(_worst_month(pr), 2),
        trades          = st["total_trades"],
        win_rate        = round(st.get("overall_win_rate", 0), 4),
        profit_factor   = round(st.get("profit_factor", 0), 4),
        avg_trade_pnl   = round(float(tl["pnl"].mean()), 4) if not tl.empty else 0.0,
        pnl_ES          = sym_pnl.get("ES", 0.0),
        pnl_NQ          = sym_pnl.get("NQ", 0.0),
        **short_m,
        **{f"yr_{k}": v for k, v in yr.items()},
        **{f"strat_{k}": v for k, v in strat_pnl.items()},
    )
    logger.info(
        "%-30s  Sharpe=%.3f  Sortino=%.3f  Ann=%.1f%%  "
        "MaxDD=%.1f%%  Trades=%d  WR=%.1f%%",
        label, row["sharpe"], row["sortino"], ann,
        row["max_dd_pct"], row["trades"], row["win_rate"] * 100,
    )
    return row, tl


# ─────────────────────────────────────────────────────────────────────────────
# Robustness helper
# ─────────────────────────────────────────────────────────────────────────────

def _sub_period_sharpe(df: pd.DataFrame, start_yr: int, end_yr: int,
                       label: str, cfg: dict | None = None) -> float:
    mask = (df["date"].dt.year >= start_yr) & (df["date"].dt.year <= end_yr)
    sub  = df[mask].copy()
    if len(sub) < 50:
        return float("nan")
    with override_config(**{"MAX_HOLD_DAYS": _HOLD_CFG, **(cfg or {})}):
        res = run_backtest(sub)
    sh = res["stats"]["sharpe_ratio"]
    logger.info("  Robustness [%d-%d] %-25s  Sharpe=%.3f", start_yr, end_yr, label, sh)
    return round(sh, 4)


def run_robustness(label: str, df: pd.DataFrame,
                   cfg: dict | None = None) -> dict:
    """Run sub-period, LOO, and symbol-split robustness checks."""
    logger.info("Robustness: %s", label)
    out = {"experiment": label}

    # Sub-period splits
    out["sharpe_2019_2021"] = _sub_period_sharpe(df, 2019, 2021, label, cfg)
    out["sharpe_2022_2023"] = _sub_period_sharpe(df, 2022, 2023, label, cfg)
    out["sharpe_2024_2026"] = _sub_period_sharpe(df, 2024, 2026, label, cfg)

    # Leave-one-year-out
    loo = {}
    for yr in range(2019, 2027):
        mask = df["date"].dt.year != yr
        sub  = df[mask].copy()
        if len(sub) < 50:
            continue
        with override_config(**{"MAX_HOLD_DAYS": _HOLD_CFG, **(cfg or {})}):
            res = run_backtest(sub)
        loo[yr] = round(res["stats"]["sharpe_ratio"], 4)
    out["loo_sharpes"] = loo
    out["loo_min"]     = round(min(loo.values()), 4) if loo else float("nan")
    out["loo_mean"]    = round(np.mean(list(loo.values())), 4) if loo else float("nan")

    # Symbol-only splits
    for sym in ["ES", "NQ"]:
        sub = df[df["symbol"] == sym].copy()
        if len(sub) < 30:
            out[f"sharpe_{sym}_only"] = float("nan")
            continue
        with override_config(**{"MAX_HOLD_DAYS": _HOLD_CFG, **(cfg or {})}):
            res = run_backtest(sub)
        out[f"sharpe_{sym}_only"] = round(res["stats"]["sharpe_ratio"], 4)
        logger.info("  Robustness [%s-only] %-22s  Sharpe=%.3f",
                    sym, label, out[f"sharpe_{sym}_only"])

    # High-vol vs low-vol splits (using vol_regime if available)
    if "vol_regime" in df.columns:
        for vol_tag, levels in [("high_vol", ["HIGH", "ELEVATED"]),
                                 ("low_vol",  ["NORMAL"])]:
            sub = df[df["vol_regime"].isin(levels)].copy()
            if len(sub) < 30:
                out[f"sharpe_{vol_tag}"] = float("nan")
                continue
            with override_config(**{"MAX_HOLD_DAYS": _HOLD_CFG, **(cfg or {})}):
                res = run_backtest(sub)
            out[f"sharpe_{vol_tag}"] = round(res["stats"]["sharpe_ratio"], 4)
            logger.info("  Robustness [%s] %-25s  Sharpe=%.3f",
                        vol_tag, label, out[f"sharpe_{vol_tag}"])

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Final verdict
# ─────────────────────────────────────────────────────────────────────────────

_VERDICT_CODES = {
    "ACCEPT_V2_LONG_ARCHITECTURE":   "V2 long sleeve accepted (Sharpe ≥ 1.20, robust)",
    "ACCEPT_SHORT_SLEEVE":           "Short sleeve adds value (short Sharpe contribution ≥ 0.05)",
    "ACCEPT_FULL_V2":                "Full V2 system accepted (long + short both pass)",
    "REJECT_SHORT_SLEEVE":           "Short sleeve rejected (hurt long Sharpe or short WR < 45%)",
    "REJECT_V2_LONG":                "V2 long sleeve rejected (Sharpe < baseline or MaxDD worse)",
    "CONTINUE_TESTING_BEST_LONG":    "Best long modification warrants further testing",
    "CONTINUE_TESTING_BEST_SHORT":   "Best individual short warrants further testing",
}


def _determine_verdict(results: list[dict], base_sharpe: float,
                        rob: dict) -> tuple[str, list[str]]:
    findings = []

    # Find key experiments
    def _get(exp):
        rows = [r for r in results if r["experiment"] == exp]
        return rows[0] if rows else None

    base_row  = _get("A_BASELINE_LONG_ONLY")
    long_row  = _get("F_LONG_V2_CORE")
    full_row  = _get("M_FULL_V2_SYSTEM")
    short_rows = [_get(e) for e in
                  ["G_BEAR_TREND_SHORT", "H_FAILED_RALLY_SHORT",
                   "I_SHOCK_SHORT", "J_BEAR_RESET_SHORT"]]
    short_rows = [r for r in short_rows if r is not None]

    # ── Long sleeve verdict ───────────────────────────────────────────────────
    long_ok = False
    if long_row:
        delta_sh = long_row["sharpe"] - base_sharpe
        dd_ok    = long_row["max_dd_pct"] >= base_row["max_dd_pct"] - 1.0
        loo_ok   = (rob.get("F_LONG_V2_CORE", {}).get("loo_min", 0) or 0) > 0.80
        long_ok  = (long_row["sharpe"] >= 1.20 and delta_sh >= 0.0 and
                    dd_ok and loo_ok)
        findings.append(
            f"F_LONG_V2_CORE: Sharpe={long_row['sharpe']:.3f} "
            f"(Δ={delta_sh:+.3f}), MaxDD={long_row['max_dd_pct']:.1f}%, "
            f"LOO_min={'n/a' if not loo_ok else rob.get('F_LONG_V2_CORE',{}).get('loo_min','?')}"
        )

    # ── Short sleeve verdict ──────────────────────────────────────────────────
    best_short = None
    best_short_delta = -999
    short_sleeve_ok = False

    for sr in short_rows:
        if sr and base_row:
            delta = sr["sharpe"] - base_sharpe
            short_wr = sr.get("short_wr", 0) or 0
            if delta > best_short_delta:
                best_short_delta = delta
                best_short = sr
            if (delta >= 0.05 and short_wr >= 0.45
                    and sr.get("short_trades", 0) >= 10):
                short_sleeve_ok = True
                findings.append(
                    f"SHORT accepted: {sr['experiment']} "
                    f"Sharpe Δ={delta:+.3f}, WR={short_wr:.1%}, "
                    f"trades={sr.get('short_trades',0)}"
                )

    if full_row:
        full_delta = full_row["sharpe"] - base_sharpe
        short_m_ok = (full_row.get("short_wr", 0) or 0) >= 0.45
        findings.append(
            f"M_FULL_V2: Sharpe={full_row['sharpe']:.3f} "
            f"(Δ={full_delta:+.3f}), short_WR={full_row.get('short_wr','?')}"
        )

    # ── Decision tree ─────────────────────────────────────────────────────────
    if long_ok and short_sleeve_ok and full_row and full_row["sharpe"] >= 1.20:
        verdict = "ACCEPT_FULL_V2"
    elif long_ok and short_sleeve_ok:
        verdict = "ACCEPT_SHORT_SLEEVE"
    elif long_ok:
        verdict = "ACCEPT_V2_LONG_ARCHITECTURE"
    elif not long_ok and long_row and long_row["sharpe"] < base_sharpe:
        verdict = "REJECT_V2_LONG"
    elif short_sleeve_ok and not long_ok:
        verdict = "CONTINUE_TESTING_BEST_SHORT"
    elif best_short and best_short_delta > 0.02:
        verdict = "CONTINUE_TESTING_BEST_SHORT"
        if best_short:
            findings.append(
                f"Best short: {best_short['experiment']} Δ={best_short_delta:+.3f}"
            )
    else:
        verdict = "REJECT_SHORT_SLEEVE"

    return verdict, findings


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df0 = build_base_df()

    results   = []
    rob_results = {}

    # ── A: Raw baseline ───────────────────────────────────────────────────────
    logger.info("=== A_BASELINE_LONG_ONLY ===")
    row, tl_A = run_experiment("A_BASELINE_LONG_ONLY", df0)
    results.append(row)
    base_sharpe = row["sharpe"]

    # Apply ES late filter to all long experiments (accepted in prior session)
    df_filtered = apply_es_late_filter(df0)

    # ── B: BULL_RESET entries ─────────────────────────────────────────────────
    logger.info("=== B_BULL_RESET_ONLY ===")
    df_B = apply_bull_reset(df_filtered)
    row, _ = run_experiment("B_BULL_RESET_ONLY", df_B)
    results.append(row)

    # ── C: Confirmation-scaled sizing (best schedule found by grid) ───────────
    logger.info("=== C_CONFIRMATION_SIZING ===")
    best_c_sharpe = -999.0
    best_c_sched  = "A"
    best_c_row    = None
    for sname, sched in [("A", _SCHED_A), ("B", _SCHED_B), ("C", _SCHED_C),
                          ("D", _SCHED_D), ("E", _SCHED_E)]:
        r, _ = run_experiment(f"C_SCHED_{sname}", df_filtered, config_overrides=sched)
        results.append(r)
        if r["sharpe"] > best_c_sharpe:
            best_c_sharpe = r["sharpe"]
            best_c_sched  = sname
            best_c_row    = r
    row = {**best_c_row, "experiment": "C_CONFIRMATION_SIZING_BEST"}
    results.append(row)
    best_c_cfg = {"A": _SCHED_A, "B": _SCHED_B, "C": _SCHED_C,
                  "D": _SCHED_D, "E": _SCHED_E}[best_c_sched]
    logger.info("  Best sizing schedule: %s (Sharpe=%.4f)", best_c_sched, best_c_sharpe)

    # ── D: Winner extension ───────────────────────────────────────────────────
    logger.info("=== D_WINNER_EXTENSION ===")
    best_ext_sharpe = -999.0
    best_ext_days   = 0
    best_ext_row    = None
    for extra in [1, 2, 3]:
        ext_hold = {k: v + extra for k, v in _HOLD_CFG.items()
                    if k in ("TREND", "BULL_RESET_CONFIRMED")}
        hold_cfg = {**_HOLD_CFG, **ext_hold}
        r, _ = run_experiment(f"D_EXT_PLUS{extra}", df_filtered,
                              config_overrides={"MAX_HOLD_DAYS": hold_cfg})
        results.append(r)
        if r["sharpe"] > best_ext_sharpe:
            best_ext_sharpe = r["sharpe"]
            best_ext_days   = extra
            best_ext_row    = r
    row = {**best_ext_row, "experiment": "D_WINNER_EXTENSION_BEST"}
    results.append(row)
    best_ext_hold = {**_HOLD_CFG}
    if best_ext_days > 0:
        for k in ("TREND", "BULL_RESET_CONFIRMED"):
            if k in best_ext_hold:
                best_ext_hold[k] += best_ext_days
    logger.info("  Best extension: +%d days (Sharpe=%.4f)", best_ext_days, best_ext_sharpe)

    # ── E: Regime router (gating only, no new long signals) ──────────────────
    logger.info("=== E_REGIME_ROUTER_ONLY ===")
    # The router flags are already on df — use regime activation to gate entries
    df_E = _apply_regime_gating(df_filtered)
    row, _ = run_experiment("E_REGIME_ROUTER_ONLY", df_E)
    results.append(row)

    # ── F: Long V2 Core (B + best C + best D + E) ────────────────────────────
    logger.info("=== F_LONG_V2_CORE ===")
    df_F = apply_bull_reset(_apply_regime_gating(df_filtered))
    row, tl_F = run_experiment("F_LONG_V2_CORE", df_F,
                               config_overrides={**best_c_cfg,
                                                 "MAX_HOLD_DAYS": best_ext_hold})
    results.append(row)

    # Robustness on F
    logger.info("Robustness for F_LONG_V2_CORE…")
    rob_results["F_LONG_V2_CORE"] = run_robustness(
        "F_LONG_V2_CORE", df_F,
        cfg={**best_c_cfg, "MAX_HOLD_DAYS": best_ext_hold}
    )

    # ── G-J: Individual short sleeves (on baseline with ES filter) ───────────
    short_type_map = {
        "G_BEAR_TREND_SHORT":   "BEAR_TREND",
        "H_FAILED_RALLY_SHORT": "FAILED_RALLY",
        "I_SHOCK_SHORT":        "SHOCK_CONT",
        "J_BEAR_RESET_SHORT":   "BEAR_RESET",
    }
    for exp_name, short_type in short_type_map.items():
        logger.info("=== %s ===", exp_name)
        df_short = _inject_single_short(df_filtered, short_type)
        row, _ = run_experiment(exp_name, df_short)
        results.append(row)

    # ── K: All shorts small (on baseline with ES filter) ─────────────────────
    logger.info("=== K_ALL_SHORTS_SMALL ===")
    df_K = compute_short_signals(df_filtered)
    row, _ = run_experiment("K_ALL_SHORTS_SMALL", df_K)
    results.append(row)

    # ── Find best individual short by Sharpe delta ────────────────────────────
    short_exp_names = list(short_type_map.keys())
    short_rows_so_far = [r for r in results if r["experiment"] in short_exp_names]
    best_short_exp = max(short_rows_so_far,
                         key=lambda r: r["sharpe"],
                         default=None)
    best_short_type = (short_type_map.get(best_short_exp["experiment"])
                       if best_short_exp else None)
    logger.info("  Best individual short: %s",
                best_short_exp["experiment"] if best_short_exp else "none")

    # ── L: Long V2 Core + best short ─────────────────────────────────────────
    logger.info("=== L_LONG_PLUS_BEST_SHORT ===")
    if best_short_type:
        df_L = _inject_single_short(df_F, best_short_type)
    else:
        df_L = df_F
    row, _ = run_experiment("L_LONG_PLUS_BEST_SHORT", df_L,
                            config_overrides={**best_c_cfg,
                                             "MAX_HOLD_DAYS": best_ext_hold})
    results.append(row)

    # ── M: Full V2 (Long V2 Core + all shorts) ───────────────────────────────
    logger.info("=== M_FULL_V2_SYSTEM ===")
    df_M = compute_short_signals(df_F)
    row, tl_M = run_experiment("M_FULL_V2_SYSTEM", df_M,
                               config_overrides={**best_c_cfg,
                                                "MAX_HOLD_DAYS": best_ext_hold})
    results.append(row)

    # Robustness on M
    logger.info("Robustness for M_FULL_V2_SYSTEM…")
    rob_results["M_FULL_V2_SYSTEM"] = run_robustness(
        "M_FULL_V2_SYSTEM", df_M,
        cfg={**best_c_cfg, "MAX_HOLD_DAYS": best_ext_hold}
    )

    # ── Compile results ───────────────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    res_df["delta_sharpe"] = (res_df["sharpe"] - base_sharpe).round(4)
    res_df.to_csv(f"{OUT_DIR}/v2_results.csv", index=False)

    rob_df = pd.DataFrame(list(rob_results.values()))
    rob_df.to_csv(f"{OUT_DIR}/v2_robustness.csv", index=False)

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict, findings = _determine_verdict(results, base_sharpe, rob_results)

    # ── Print report ──────────────────────────────────────────────────────────
    _print_report(res_df, rob_results, base_sharpe, verdict, findings,
                  best_c_sched, best_ext_days)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for signal injection
# ─────────────────────────────────────────────────────────────────────────────

def _apply_regime_gating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gate existing long entries using router flags.
    Blocks entries where the relevant allow_* flag is 0.
    """
    df = df.copy()

    if "allow_trend_long" not in df.columns:
        return df   # regime router not run

    # Block TREND longs when regime says disallow
    trend_long_blocked = (
        (df["strategy_used"] == "TREND") &
        (df["final_direction"] == "LONG") &
        (df["allow_trend_long"] == 0)
    )
    df.loc[trend_long_blocked, "final_direction"] = "FLAT"
    df.loc[trend_long_blocked, "strategy_used"]   = "NONE"

    # Block STAT_ARB when regime says disallow
    sa_blocked = (
        (df["strategy_used"] == "STAT_ARB") &
        (df["final_direction"] == "LONG") &
        (df.get("allow_stat_arb", pd.Series(1, index=df.index)) == 0)
    )
    df.loc[sa_blocked, "final_direction"] = "FLAT"
    df.loc[sa_blocked, "strategy_used"]   = "NONE"

    n_blocked = int(trend_long_blocked.sum()) + int(sa_blocked.sum())
    logger.info("  Regime gating blocked %d entries", n_blocked)
    return df


def _inject_single_short(df: pd.DataFrame, short_type: str) -> pd.DataFrame:
    """
    Run compute_short_signals then blank out all short types except short_type.
    """
    df = compute_short_signals(df)
    # Zero out everything that isn't the requested type
    wrong_type = (df["short_signal_type"] != short_type) & (df["short_signal_type"] != "NONE")
    df.loc[wrong_type, "final_direction"]    = "FLAT"
    df.loc[wrong_type, "strategy_used"]      = "NONE"
    df.loc[wrong_type, "short_signal_type"]  = "NONE"
    df.loc[wrong_type, "short_signal_size"]  = 0.0
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(res: pd.DataFrame, rob: dict, base_sharpe: float,
                  verdict: str, findings: list[str],
                  best_sched: str, best_ext: int) -> None:
    w = "=" * 80

    print(f"\n{w}")
    print("V2 SYSTEM — FULL RESULTS")
    print(w)

    core_exps = ["A_BASELINE_LONG_ONLY", "B_BULL_RESET_ONLY",
                 "C_CONFIRMATION_SIZING_BEST", "D_WINNER_EXTENSION_BEST",
                 "E_REGIME_ROUTER_ONLY", "F_LONG_V2_CORE",
                 "G_BEAR_TREND_SHORT", "H_FAILED_RALLY_SHORT",
                 "I_SHOCK_SHORT", "J_BEAR_RESET_SHORT", "K_ALL_SHORTS_SMALL",
                 "L_LONG_PLUS_BEST_SHORT", "M_FULL_V2_SYSTEM"]
    sub = res[res["experiment"].isin(core_exps)].copy()

    cols = ["experiment", "sharpe", "delta_sharpe", "sortino",
            "annual_ret_pct", "final_25k", "max_dd_pct", "avg_dd_pct",
            "worst_week_pct", "worst_month_pct",
            "trades", "win_rate", "profit_factor", "avg_trade_pnl"]
    print(sub[[c for c in cols if c in sub.columns]].to_string(index=False))

    print(f"\n{w}")
    print("SHORT-SLEEVE METRICS")
    print(w)
    short_cols = ["experiment", "short_trades", "short_wr",
                  "short_pf", "short_avg_pnl", "short_pnl_total"]
    print(sub[[c for c in short_cols if c in sub.columns]].to_string(index=False))

    print(f"\n{w}")
    print("PnL BY SYMBOL")
    print(w)
    sym_cols = ["experiment", "pnl_ES", "pnl_NQ"]
    print(sub[[c for c in sym_cols if c in sub.columns]].to_string(index=False))

    yr_cols = sorted([c for c in sub.columns if c.startswith("yr_")])
    if yr_cols:
        print(f"\n{w}")
        print("ANNUAL RETURNS % BY YEAR")
        print(w)
        print(sub[["experiment"] + yr_cols].to_string(index=False))

    strat_cols = [c for c in sub.columns if c.startswith("strat_")]
    if strat_cols:
        print(f"\n{w}")
        print("PnL BY STRATEGY")
        print(w)
        print(sub[["experiment"] + strat_cols].to_string(index=False))

    # Sizing schedule sub-table
    sched_exps = [c for c in res["experiment"] if c.startswith("C_SCHED_")]
    if sched_exps:
        print(f"\n{w}")
        print(f"CONFIRMATION SIZING SCHEDULES (best: {best_sched})")
        print(w)
        sched_sub = res[res["experiment"].isin(sched_exps)]
        print(sched_sub[["experiment","sharpe","delta_sharpe","annual_ret_pct",
                          "max_dd_pct","trades","win_rate"]].to_string(index=False))

    # Extension sub-table
    ext_exps = [c for c in res["experiment"] if c.startswith("D_EXT_")]
    if ext_exps:
        print(f"\n{w}")
        print(f"WINNER EXTENSION (best: +{best_ext} days)")
        print(w)
        ext_sub = res[res["experiment"].isin(ext_exps)]
        print(ext_sub[["experiment","sharpe","delta_sharpe","annual_ret_pct",
                        "max_dd_pct","trades","win_rate"]].to_string(index=False))

    # Robustness
    if rob:
        print(f"\n{w}")
        print("ROBUSTNESS SUMMARY")
        print(w)
        for exp_name, rdata in rob.items():
            print(f"\n  {exp_name}:")
            print(f"    2019-2021 Sharpe: {rdata.get('sharpe_2019_2021', 'n/a')}")
            print(f"    2022-2023 Sharpe: {rdata.get('sharpe_2022_2023', 'n/a')}")
            print(f"    2024-2026 Sharpe: {rdata.get('sharpe_2024_2026', 'n/a')}")
            print(f"    LOO min Sharpe:   {rdata.get('loo_min', 'n/a')}")
            print(f"    LOO mean Sharpe:  {rdata.get('loo_mean', 'n/a')}")
            print(f"    ES-only Sharpe:   {rdata.get('sharpe_ES_only', 'n/a')}")
            print(f"    NQ-only Sharpe:   {rdata.get('sharpe_NQ_only', 'n/a')}")
            print(f"    High-vol Sharpe:  {rdata.get('sharpe_high_vol', 'n/a')}")
            print(f"    Low-vol Sharpe:   {rdata.get('sharpe_low_vol', 'n/a')}")
            if "loo_sharpes" in rdata:
                print(f"    LOO by year:      {rdata['loo_sharpes']}")

    # Final verdict
    print(f"\n{w}")
    print("FINAL VERDICT")
    print(w)
    print(f"  VERDICT: {verdict}")
    print(f"  MEANING: {_VERDICT_CODES.get(verdict, 'Unknown')}")
    print(f"\n  Key findings:")
    for f in findings:
        print(f"    • {f}")
    print(f"\n  Baseline Sharpe:      {base_sharpe:.4f}")
    best_long = res[res["experiment"] == "F_LONG_V2_CORE"]
    if not best_long.empty:
        fl = best_long.iloc[0]
        print(f"  F_LONG_V2_CORE:       Sharpe={fl['sharpe']:.4f}  "
              f"MaxDD={fl['max_dd_pct']:.1f}%  "
              f"Final_$25k=${fl['final_25k']:,}")
    best_full = res[res["experiment"] == "M_FULL_V2_SYSTEM"]
    if not best_full.empty:
        fm = best_full.iloc[0]
        print(f"  M_FULL_V2_SYSTEM:     Sharpe={fm['sharpe']:.4f}  "
              f"MaxDD={fm['max_dd_pct']:.1f}%  "
              f"Short_WR={fm.get('short_wr','?')}")
    print(w)


if __name__ == "__main__":
    main()
