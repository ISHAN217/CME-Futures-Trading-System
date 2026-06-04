"""
run_trend_capture_experiments.py — Trend Capture Improvement Experiments
=========================================================================
AUDIT FINDINGS (from run_trade_timing_analysis.py):
  - 86% avg entry point through 20-day swing (momentum chaser)
  - 56% of TREND trades are LATE → 48% WR, -0.28% avg PnL
  - ADDED trades: 43, 88% WR, +10% capture — this is the alpha engine
  - NO_ADD trades: 490, 59% WR, near-zero net edge
  - ES enters at 101% of swing, NQ at 72%
  - 35% of exits are TOO_EARLY; 51% of swing left on table post-exit

STRATEGY:
  1. Filter LATE entries (block when price is too extended in 20d range)
  2. Expand add-to-winner (earlier trigger, larger size, second add)
  3. Hold confirmed winners longer (target-specific extension, not blanket)
  4. Apply ES-specific stricter treatment
  5. Combine accepted components

All experiments via DataFrame modification or config monkey-patching.
No backtest.py edits. No lookahead in feature computation.

OUTPUTS (all in output/timing/):
  late_entry_filter_sweep.csv
  late_entry_blocked_trade_attribution.csv
  add_to_winner_expansion_sweep.csv
  add_to_winner_trade_audit.csv
  winner_extension_sweep.csv
  winner_extension_trade_audit.csv
  es_specific_trend_treatment.csv
  trend_capture_improvement_combined.csv
"""

import contextlib
import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
from backtest           import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR = "/Users/ishanbhardwaj/cme_competition_system/output/timing"
os.makedirs(OUT_DIR, exist_ok=True)

# Lookback for late-entry feature computation
LATE_LOOKBACK_20 = 20
LATE_LOOKBACK_40 = 40


# ─────────────────────────────────────────────────────────────────────────────
# 1. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_base_df() -> pd.DataFrame:
    """Full accepted pipeline — identical to main.py steps 1-12c."""
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
        for col, val in [
            ("cwt_chop_label",       "CWT_UNCLEAR"),
            ("cwt_trade_allowed",    1),
            ("cwt_size_multiplier",  1.0),
            ("cwt_confidence",       0.0),
            ("cwt_total_energy",     0.0),
            ("cwt_entropy",          0.5),
            ("cwt_energy_z",         0.0),
            ("cwt_compression_flag", 0),
            ("cwt_expansion_flag",   0),
            ("cwt_high_energy_ratio",1/3),
            ("cwt_mid_energy_ratio", 1/3),
            ("cwt_slow_energy_ratio",1/3),
        ]:
            if col not in df.columns:
                df[col] = val

    if df_4h_raw is not None:
        df_4h_feat  = compute_4h_features(df_4h_raw)
        df_4h_daily = aggregate_4h_to_daily(df_4h_feat)
    else:
        df_4h_daily = None
    df = merge_4h_into_daily(df, df_4h_daily)
    df = compute_mean_reversion(df)
    df = compute_stat_arb(df)
    df = compute_sentiment(df)
    df = compute_signals(df)
    df = compute_signal_quality(df)
    df = compute_trend_continuation_edge(df)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. LATE ENTRY FEATURES (backward-looking only, no lookahead)
# ─────────────────────────────────────────────────────────────────────────────

def compute_late_entry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add entry location features per row — all backward-looking.
    Features:
      entry_location_pct_20d  — where close sits in the rolling 20d range [0=low, 1=high]
      entry_location_pct_40d  — same for 40d window
      pre_entry_missed_move_20d_pct — (close - min_20d) / close * 100 [% moved from 20d low]
      swing_capture_expected   — (1 - entry_location_pct_20d) * 100 [% of range remaining]
      late_entry_flag_70/75/80/85/90  — bool flags
      add_eligible_flag        — not late_80 OR high_conviction
    """
    df = df.copy()
    # Initialise with NaN
    for col in ["entry_location_pct_20d", "entry_location_pct_40d",
                "pre_entry_missed_move_20d_pct", "swing_capture_expected",
                "late_entry_flag_70", "late_entry_flag_75", "late_entry_flag_80",
                "late_entry_flag_85", "late_entry_flag_90", "add_eligible_flag"]:
        df[col] = np.nan

    for sym, g in df.groupby("symbol", sort=False):
        idx  = g.index
        cl   = g["close"].values
        s    = pd.Series(cl, index=idx)

        # Rolling min/max (strictly backward-looking using .shift(0) — current day included)
        min_20 = s.rolling(LATE_LOOKBACK_20, min_periods=5).min()
        max_20 = s.rolling(LATE_LOOKBACK_20, min_periods=5).max()
        min_40 = s.rolling(LATE_LOOKBACK_40, min_periods=10).min()
        max_40 = s.rolling(LATE_LOOKBACK_40, min_periods=10).max()

        rng_20 = (max_20 - min_20).replace(0, np.nan)
        rng_40 = (max_40 - min_40).replace(0, np.nan)

        loc_20 = (s - min_20) / rng_20
        loc_40 = (s - min_40) / rng_40

        df.loc[idx, "entry_location_pct_20d"]      = loc_20.values
        df.loc[idx, "entry_location_pct_40d"]      = loc_40.values
        df.loc[idx, "pre_entry_missed_move_20d_pct"] = ((s - min_20) / s * 100).values
        df.loc[idx, "swing_capture_expected"]      = ((1 - loc_20) * 100).values

        for thr, col in [(0.70, "late_entry_flag_70"), (0.75, "late_entry_flag_75"),
                         (0.80, "late_entry_flag_80"), (0.85, "late_entry_flag_85"),
                         (0.90, "late_entry_flag_90")]:
            df.loc[idx, col] = (loc_20 > thr).astype(float).values

    # add_eligible: not late_80 OR high_conviction
    hc = df.get("high_conviction", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    df["add_eligible_flag"] = ((df["late_entry_flag_80"] < 1) | hc).astype(float)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def override_config(**kwargs):
    """Temporarily monkey-patch config values; restore on exit."""
    originals = {k: getattr(config, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in originals.items():
            setattr(config, k, v)


def _run(name: str, df: pd.DataFrame, cfg_overrides: dict = None) -> dict:
    """Run backtest with optional config patches. Returns full result dict."""
    overrides = cfg_overrides or {}
    with override_config(**overrides):
        result = run_backtest(df)
    return result


def _year_pnl(tl: pd.DataFrame) -> dict:
    """PnL sum by entry year from trade log."""
    if tl is None or tl.empty:
        return {}
    tl = tl.copy()
    tl["year"] = pd.to_datetime(tl["entry_date"]).dt.year
    return tl.groupby("year")["pnl"].sum().round(4).to_dict()


def _sym_pnl(tl: pd.DataFrame) -> dict:
    """PnL sum by symbol from trade log."""
    if tl is None or tl.empty:
        return {}
    return tl.groupby("symbol")["pnl"].sum().round(4).to_dict()


def flatten(name: str, result: dict, baseline_tl: pd.DataFrame = None) -> dict:
    """
    Flatten a backtest result into a single metrics dict for CSV export.
    Includes: perf metrics, trade stats, sym/year PnL, add-to-winner info.
    Optional baseline_tl: used to compute blocked_trade attribution.
    """
    st  = result["stats"]
    tl  = result["trade_log"]
    dr  = result.get("daily_records", pd.DataFrame())
    sym = result.get("symbol_returns", pd.DataFrame())

    # Core performance
    row = {
        "experiment":          name,
        "sharpe":              st["sharpe_ratio"],
        "annual_ret_pct":      st["annualized_return_pct"],
        "total_pnl_pct":       st["total_return_pct"],
        "max_dd_pct":          st["max_drawdown_pct"],
        "win_rate":            round(st.get("overall_win_rate", 0.0), 4),
        "profit_factor":       st.get("profit_factor", 0.0),
        "trade_count":         st.get("total_trades", 0),
        "avg_hold_days":       st.get("avg_holding_days", 0.0),
        "stop_loss_count":     st.get("stop_loss_count", 0),
        "trail_stop_count":    st.get("trail_stop_count", 0),
        "fast_exit_count":     st.get("fast_exit_count", 0),
        "extension_count":     st.get("extension_count", 0),
        "add_to_winner_count": st.get("add_to_winner_count", 0),
        "extended_win_rate":   st.get("extended_win_rate", 0.0),
    }

    # Symbol PnL from symbol_returns
    if not sym.empty:
        for s in sym.columns:
            row[f"pnl_{s}"] = round(float(sym[s].sum()), 4)

    # Year PnL
    if tl is not None and not tl.empty:
        for yr, v in _year_pnl(tl).items():
            row[f"pnl_{yr}"] = v

        # TREND-specific
        trend = tl[tl["strategy"] == "TREND"]
        row["trend_trade_count"] = len(trend)
        row["trend_win_rate"]    = round(trend["win"].mean(), 4) if len(trend) else 0.0
        row["trend_avg_pnl"]     = round(trend["pnl"].mean(), 4) if len(trend) else 0.0

        # Add-to-winner trades
        added = tl[(tl["strategy"] == "TREND") & (tl.get("add_count", pd.Series(0)) > 0)] \
                if "add_count" in tl.columns else pd.DataFrame()
        # Re-check
        if "add_count" in tl.columns:
            added = tl[(tl["strategy"] == "TREND") & (tl["add_count"] > 0)]
        else:
            added = pd.DataFrame()
        row["added_trade_count"] = len(added)
        row["added_win_rate"]    = round(added["win"].mean(), 4) if len(added) else 0.0
        row["added_avg_pnl"]     = round(added["pnl"].mean(), 4) if len(added) else 0.0
        row["added_pnl_sum"]     = round(added["pnl"].sum(), 4) if len(added) else 0.0

    # Blocked trade attribution vs baseline
    if baseline_tl is not None and tl is not None and not tl.empty and not baseline_tl.empty:
        attr = _blocked_attribution(baseline_tl, tl)
        row.update(attr)

    return row


def _blocked_attribution(baseline_tl: pd.DataFrame, exp_tl: pd.DataFrame) -> dict:
    """
    Find TREND trades in baseline that are absent in experiment (blocked).
    Returns count, PnL, avoided losers, missed winners.
    """
    if baseline_tl.empty or exp_tl.empty:
        return {}

    b = baseline_tl[baseline_tl["strategy"] == "TREND"].copy()
    e = exp_tl[exp_tl["strategy"] == "TREND"].copy()

    b["_key"] = b["symbol"] + "_" + b["entry_date"].astype(str)
    e["_key"] = e["symbol"] + "_" + e["entry_date"].astype(str)

    blocked = b[~b["_key"].isin(e["_key"])]

    n_blocked        = len(blocked)
    avoided_losers   = int((blocked["pnl"] <= 0).sum())
    missed_winners   = int((blocked["pnl"] > 0).sum())
    blocked_pnl_sum  = round(blocked["pnl"].sum(), 4)
    blocked_avg_pnl  = round(blocked["pnl"].mean(), 4) if n_blocked > 0 else 0.0

    return {
        "blocked_trades":     n_blocked,
        "avoided_losers":     avoided_losers,
        "missed_winners":     missed_winners,
        "blocked_pnl_sum":    blocked_pnl_sum,
        "blocked_avg_pnl":    blocked_avg_pnl,
        "block_loser_rate":   round(avoided_losers / max(n_blocked, 1), 3),
    }


def _add_winner_audit(result: dict) -> pd.DataFrame:
    """
    From daily_records, identify days when an add-to-winner fired
    (add_count incremented). Return per-add audit rows.
    """
    dr = result.get("daily_records", pd.DataFrame())
    tl = result.get("trade_log",     pd.DataFrame())
    if dr.empty or tl.empty:
        return pd.DataFrame()

    # Sort so we can detect add_count transitions
    dr = dr.sort_values(["symbol", "date"]).copy()
    dr["prev_add_count"] = dr.groupby("symbol")["add_count"].shift(1).fillna(0)
    add_days = dr[(dr["add_count"] > dr["prev_add_count"]) &
                  (dr["strategy"] == "TREND")].copy()

    rows = []
    for _, arow in add_days.iterrows():
        sym   = arow["symbol"]
        adate = arow["date"]
        cum_at_add = arow["cum_ret"]

        # Find corresponding trade
        t = tl[(tl["symbol"] == sym) &
                (pd.to_datetime(tl["entry_date"]) <= pd.to_datetime(adate)) &
                (pd.to_datetime(tl["exit_date"])  >= pd.to_datetime(adate))]
        if t.empty:
            continue
        t = t.iloc[0]

        post_add_pnl = t["pnl"] - cum_at_add  # rough post-add return
        rows.append({
            "symbol":        sym,
            "add_date":      adate,
            "entry_date":    t["entry_date"],
            "exit_date":     t["exit_date"],
            "exit_reason":   t["exit_reason"],
            "cum_ret_at_add":round(cum_at_add, 4),
            "final_pnl":     round(t["pnl"], 4),
            "post_add_pnl":  round(post_add_pnl, 4),
            "post_add_win":  post_add_pnl > 0,
            "entry_quality": t.get("entry_quality", 0),
            "add_count":     t.get("add_count", 1),
        })
    return pd.DataFrame(rows)


def _print_sweep(rows: list, section: str) -> None:
    """Print a sweep result table."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    key_cols = ["experiment", "sharpe", "annual_ret_pct", "max_dd_pct",
                "trade_count", "win_rate", "profit_factor",
                "trend_win_rate", "add_to_winner_count", "added_win_rate",
                "extension_count"]
    cols = [c for c in key_cols if c in df.columns]
    print(f"\n{'─'*100}")
    print(f"  {section}")
    print(f"{'─'*100}")
    print(df[cols].to_string(index=False))


def _delta(rows: list, baseline_sharpe: float) -> None:
    """Show Sharpe delta vs baseline for each experiment."""
    for r in rows:
        delta = r.get("sharpe", 0) - baseline_sharpe
        sign  = "+" if delta >= 0 else ""
        logger.info("  %-45s  Sharpe=%.3f  Δ=%s%.3f  MaxDD=%.1f%%  Trades=%d",
                    r["experiment"], r.get("sharpe", 0),
                    sign, delta,
                    r.get("max_dd_pct", 0), r.get("trade_count", 0))


# ─────────────────────────────────────────────────────────────────────────────
# 4. SECTION 1 — LATE ENTRY FILTER
# ─────────────────────────────────────────────────────────────────────────────

def _block_late(df: pd.DataFrame, flag_col: str, sym_filter: str = None) -> pd.DataFrame:
    """
    Set final_direction='FLAT' for TREND LONG rows where flag_col is True.
    Optional sym_filter: apply only to that symbol.
    """
    d    = df.copy()
    mask = (d["strategy_used"] == "TREND") & \
           (d["final_direction"] == "LONG") & \
           (d[flag_col] > 0.5)
    if sym_filter:
        mask &= (d["symbol"] == sym_filter)
    d.loc[mask, "final_direction"] = "FLAT"
    return d


def _block_late_cond(df: pd.DataFrame, flag_col: str,
                     unless_hc: bool = False, unless_add_eligible: bool = False,
                     unless_nq: bool = False, sym_filter: str = None) -> pd.DataFrame:
    """Block late entries with conditional pass-throughs."""
    d    = df.copy()
    mask = (d["strategy_used"] == "TREND") & \
           (d["final_direction"] == "LONG") & \
           (d[flag_col] > 0.5)
    if sym_filter:
        mask &= (d["symbol"] == sym_filter)
    # Exemptions
    if unless_hc:
        hc   = d.get("high_conviction", pd.Series(False, index=d.index)).fillna(False).astype(bool)
        mask &= ~hc
    if unless_add_eligible:
        ae   = (d.get("add_eligible_flag", pd.Series(0.0, index=d.index)).fillna(0) > 0.5)
        mask &= ~ae
    if unless_nq:
        mask &= (d["symbol"] != "NQ")
    d.loc[mask, "final_direction"] = "FLAT"
    return d


def run_late_filter_experiments(df_base: pd.DataFrame, baseline_result: dict) -> pd.DataFrame:
    """
    Section 1: A_BASELINE through L_BLOCK_ES_LATE_80_AND_NQ_LATE_90.
    Returns sweep DataFrame.
    """
    logger.info("═" * 60)
    logger.info("SECTION 1 — LATE ENTRY FILTER (12 experiments)")
    logger.info("═" * 60)

    btl = baseline_result["trade_log"]
    rows = []

    def _run_and_record(name, df_mod, cfg=None):
        r = _run(name, df_mod, cfg)
        row = flatten(name, r, btl)
        rows.append(row)
        delta = row["sharpe"] - btl_sharpe
        logger.info("  %-45s  Sharpe=%.3f (Δ%+.3f)  MaxDD=%.1f%%  Trades=%d  Blocked=%d",
                    name, row["sharpe"], delta,
                    row["max_dd_pct"], row["trade_count"],
                    row.get("blocked_trades", 0))
        return r

    btl_sharpe = baseline_result["stats"]["sharpe_ratio"]

    # A — Baseline (already run, just record)
    row_a = flatten("A_BASELINE", baseline_result)
    rows.append(row_a)
    logger.info("  %-45s  Sharpe=%.3f (baseline)  MaxDD=%.1f%%  Trades=%d",
                "A_BASELINE", btl_sharpe, row_a["max_dd_pct"], row_a["trade_count"])

    # B-F: Threshold sweeps (symmetric block, both ES and NQ)
    for name, col in [
        ("B_BLOCK_LATE_70", "late_entry_flag_70"),
        ("C_BLOCK_LATE_75", "late_entry_flag_75"),
        ("D_BLOCK_LATE_80", "late_entry_flag_80"),
        ("E_BLOCK_LATE_85", "late_entry_flag_85"),
        ("F_BLOCK_LATE_90", "late_entry_flag_90"),
    ]:
        _run_and_record(name, _block_late(df_base, col))

    # G-L: Conditional variants
    _run_and_record("G_BLOCK_LATE_80_UNLESS_HIGH_CONVICTION",
                    _block_late_cond(df_base, "late_entry_flag_80", unless_hc=True))

    _run_and_record("H_BLOCK_LATE_80_UNLESS_ADD_ELIGIBLE",
                    _block_late_cond(df_base, "late_entry_flag_80", unless_add_eligible=True))

    _run_and_record("I_BLOCK_LATE_80_UNLESS_HC_OR_NQ",
                    _block_late_cond(df_base, "late_entry_flag_80", unless_hc=True, unless_nq=True))

    _run_and_record("J_BLOCK_ES_LATE_70_ONLY",
                    _block_late(df_base, "late_entry_flag_70", sym_filter="ES"))

    _run_and_record("K_BLOCK_ES_LATE_80_ONLY",
                    _block_late(df_base, "late_entry_flag_80", sym_filter="ES"))

    _run_and_record("L_BLOCK_ES_LATE_80_AND_NQ_LATE_90",
                    _block_late(
                        _block_late(df_base, "late_entry_flag_80", sym_filter="ES"),
                        "late_entry_flag_90",
                        sym_filter="NQ",
                    ))

    df_sweep = pd.DataFrame(rows)

    # ── Blocked trade attribution ──
    attr_rows = []
    trend_base = btl[btl["strategy"] == "TREND"].copy() if btl is not None else pd.DataFrame()
    trend_base["_key"] = trend_base["symbol"] + "_" + trend_base["entry_date"].astype(str)
    for exp_name, flag_col, sym_filter in [
        ("B_BLOCK_LATE_70",  "late_entry_flag_70", None),
        ("C_BLOCK_LATE_75",  "late_entry_flag_75", None),
        ("D_BLOCK_LATE_80",  "late_entry_flag_80", None),
        ("E_BLOCK_LATE_85",  "late_entry_flag_85", None),
        ("F_BLOCK_LATE_90",  "late_entry_flag_90", None),
        ("K_BLOCK_ES_LATE_80_ONLY", "late_entry_flag_80", "ES"),
    ]:
        # Identify blocked rows from df_base
        mask = (df_base["strategy_used"] == "TREND") & \
               (df_base["final_direction"] == "LONG") & \
               (df_base[flag_col] > 0.5)
        if sym_filter:
            mask &= (df_base["symbol"] == sym_filter)
        blocked_dates = set(
            df_base.loc[mask, "symbol"] + "_" + df_base.loc[mask, "date"].astype(str)
        )
        blocked_in_base = trend_base[trend_base["_key"].isin(blocked_dates)]
        for _, bt in blocked_in_base.iterrows():
            attr_rows.append({
                "experiment":    exp_name,
                "symbol":        bt["symbol"],
                "entry_date":    bt["entry_date"],
                "exit_reason":   bt.get("exit_reason", ""),
                "pnl":           round(bt["pnl"], 4),
                "win":           bt["pnl"] > 0,
                "holding_days":  bt.get("holding_days", 0),
                "high_conv":     bt.get("high_conviction", False),
                "add_count":     bt.get("add_count", 0),
            })
    df_attr = pd.DataFrame(attr_rows)

    return df_sweep, df_attr


# ─────────────────────────────────────────────────────────────────────────────
# 5. SECTION 2 — ADD-TO-WINNER EXPANSION
# ─────────────────────────────────────────────────────────────────────────────

def run_add_to_winner_experiments(df_base: pd.DataFrame, baseline_result: dict) -> pd.DataFrame:
    """
    Section 2: A_BASELINE through J_ADD_AND_EXTEND_HOLD.
    Experiments use config monkey-patching and/or DataFrame momentum_strength zeroing.
    """
    logger.info("═" * 60)
    logger.info("SECTION 2 — ADD-TO-WINNER EXPANSION (10 experiments)")
    logger.info("═" * 60)

    btl_sharpe = baseline_result["stats"]["sharpe_ratio"]
    btl        = baseline_result["trade_log"]
    rows       = []

    def _go(name, df_mod=None, cfg=None):
        df  = df_mod if df_mod is not None else df_base
        r   = _run(name, df, cfg)
        row = flatten(name, r)
        rows.append(row)
        delta = row["sharpe"] - btl_sharpe
        logger.info("  %-45s  Sharpe=%.3f (Δ%+.3f)  Adds=%d  AddWR=%.0f%%  MaxDD=%.1f%%",
                    name, row["sharpe"], delta,
                    row.get("add_to_winner_count", 0),
                    row.get("added_win_rate", 0) * 100,
                    row.get("max_dd_pct", 0))
        return r

    # A — Baseline
    row_a = flatten("A_BASELINE", baseline_result)
    rows.append(row_a)
    logger.info("  %-45s  Sharpe=%.3f (baseline)  Adds=%d  AddWR=%.0f%%",
                "A_BASELINE", btl_sharpe,
                row_a.get("add_to_winner_count", 0),
                row_a.get("added_win_rate", 0) * 100)

    # B — Earlier trigger (0.25%), small size (25% of entry)
    _go("B_EARLIER_ADD_TRIGGER_SMALL",
        cfg=dict(ADD_WINNER_MIN_CUM_RET=0.0025, ADD_WINNER_SIZE_FRAC=0.25))

    # C — Earlier trigger (0.25%), current size (50%)
    _go("C_EARLIER_ADD_TRIGGER_CURRENT_SIZE",
        cfg=dict(ADD_WINNER_MIN_CUM_RET=0.0025))

    # D — Current trigger (0.50%), larger size (75%)
    _go("D_CURRENT_TRIGGER_LARGER_ADD",
        cfg=dict(ADD_WINNER_SIZE_FRAC=0.75))

    # E — Allow second add, small (25% per add)
    _go("E_ALLOW_SECOND_ADD_SMALL",
        cfg=dict(ADD_WINNER_MAX_ADDS=2, ADD_WINNER_SIZE_FRAC=0.25))

    # F — Allow second add, current size — approximates "if first profitable"
    #     (second add requires cum_ret >= threshold again after first add; this
    #      naturally screens for ongoing profitability via the cumulative ret check)
    _go("F_ALLOW_SECOND_ADD_CURRENT_SIZE",
        cfg=dict(ADD_WINNER_MAX_ADDS=2))

    # G — Block adds on LATE entries (zero momentum on rows that were late)
    #     Proxy: set momentum_strength = 0 on late_80 rows so winner_conditions_hold() fails
    df_g = df_base.copy()
    late_mask = df_g["late_entry_flag_80"] > 0.5
    df_g.loc[late_mask, "momentum_strength"] = 0.0
    _go("G_ADD_ONLY_IF_ENTRY_NOT_LATE", df_mod=df_g)

    # H — NQ-only adds (block ES adds by zeroing momentum_strength for ES)
    df_h = df_base.copy()
    df_h.loc[df_h["symbol"] == "ES", "momentum_strength"] = 0.0
    _go("H_ADD_ONLY_IF_NQ", df_mod=df_h)

    # I — Earlier add (0.25%) + second add allowed (MFE naturally screened)
    _go("I_EARLIER_ADD_PLUS_SECOND_ADD",
        cfg=dict(ADD_WINNER_MIN_CUM_RET=0.0025, ADD_WINNER_MAX_ADDS=2))

    # J — Add expansion + extended hold (earlier trigger + second add + +2d max hold)
    _go("J_ADD_AND_EXTEND_HOLD",
        cfg=dict(ADD_WINNER_MIN_CUM_RET=0.0025, ADD_WINNER_MAX_ADDS=2,
                 MAX_EXTENDED_HOLD_DAYS=9))

    # Audit: inspect baseline add-to-winner mechanics
    audit = _add_winner_audit(baseline_result)

    return pd.DataFrame(rows), audit


# ─────────────────────────────────────────────────────────────────────────────
# 6. SECTION 3 — WINNER EXTENSION
# ─────────────────────────────────────────────────────────────────────────────

def _force_can_extend_for_dates(df: pd.DataFrame, dates_by_sym: dict) -> pd.DataFrame:
    """
    Force can_extend = True for specific (sym, date) pairs.
    dates_by_sym: {sym: set_of_dates}
    """
    d = df.copy()
    for sym, dset in dates_by_sym.items():
        mask = (d["symbol"] == sym) & (d["date"].isin(dset))
        d.loc[mask, "can_extend"] = True
    return d


def _get_added_trade_date_ranges(result: dict, extra_days: int = 7) -> dict:
    """
    Return {sym: set_of_dates} covering days 3-N of trades that had adds.
    Used for two-pass extension experiments.
    """
    tl = result["trade_log"]
    if tl is None or tl.empty or "add_count" not in tl.columns:
        return {}
    added = tl[(tl["strategy"] == "TREND") & (tl["add_count"] > 0)]
    dates_by_sym = {}
    for _, row in added.iterrows():
        sym = row["symbol"]
        ed  = pd.Timestamp(row["entry_date"])
        xd  = pd.Timestamp(row["exit_date"]) + pd.Timedelta(days=extra_days)
        if sym not in dates_by_sym:
            dates_by_sym[sym] = set()
        # Add all dates from day 3 of the trade onward
        dates_by_sym[sym].update(
            pd.date_range(ed + pd.Timedelta(days=2), xd, freq="B")
        )
    return dates_by_sym


def _get_profitable_at_day3_dates(result: dict, extra_days: int = 7) -> dict:
    """
    Return {sym: set_of_dates} for days 3+ of TREND trades that had
    positive cum_ret at days_held==3 in the baseline daily_records.
    """
    dr = result.get("daily_records", pd.DataFrame())
    if dr.empty:
        return {}

    # days at exactly day 3
    d3 = dr[(dr["strategy"] == "TREND") &
            (dr["days_held"] == 3) &
            (dr["cum_ret"]   > 0.0)]

    tl  = result["trade_log"]
    dates_by_sym = {}
    for _, row in d3.iterrows():
        sym   = row["symbol"]
        d3_dt = pd.Timestamp(row["date"])

        # Find this trade's exit
        if tl is not None and not tl.empty:
            match = tl[(tl["symbol"] == sym) &
                       (pd.to_datetime(tl["entry_date"]) <= d3_dt) &
                       (pd.to_datetime(tl["exit_date"])  >= d3_dt)]
            if not match.empty:
                xd = pd.Timestamp(match.iloc[0]["exit_date"]) + pd.Timedelta(days=extra_days)
            else:
                xd = d3_dt + pd.Timedelta(days=extra_days)
        else:
            xd = d3_dt + pd.Timedelta(days=extra_days)

        if sym not in dates_by_sym:
            dates_by_sym[sym] = set()
        dates_by_sym[sym].update(pd.date_range(d3_dt, xd, freq="B"))
    return dates_by_sym


def run_winner_extension_experiments(df_base: pd.DataFrame, baseline_result: dict) -> pd.DataFrame:
    """
    Section 3: A_BASELINE through J_EXTEND_ADDED_PLUS_HIGH_CONVICTION_2D.
    Two-pass for add-based / profit-based extensions; one-pass for precomputable filters.
    """
    logger.info("═" * 60)
    logger.info("SECTION 3 — WINNER EXTENSION (10 experiments)")
    logger.info("═" * 60)

    btl_sharpe = baseline_result["stats"]["sharpe_ratio"]
    rows       = []
    audit_rows = []

    def _go(name, df_mod, cfg=None):
        r   = _run(name, df_mod, cfg)
        row = flatten(name, r)
        rows.append(row)
        delta = row["sharpe"] - btl_sharpe
        logger.info("  %-45s  Sharpe=%.3f (Δ%+.3f)  Ext=%d  ExtWR=%.0f%%  AvgHold=%.1fd  MaxDD=%.1f%%",
                    name, row["sharpe"], delta,
                    row.get("extension_count", 0),
                    row.get("extended_win_rate", 0) * 100,
                    row.get("avg_hold_days", 0),
                    row.get("max_dd_pct", 0))
        return r

    # A — Baseline
    row_a = flatten("A_BASELINE", baseline_result)
    rows.append(row_a)
    logger.info("  %-45s  Sharpe=%.3f (baseline)  Ext=%d",
                "A_BASELINE", btl_sharpe, row_a.get("extension_count", 0))

    # B/C/D — Two-pass: extend added trades by +1d / +2d / +3d max hold
    for extra_days, hard_cap, name in [
        (1, config.MAX_EXTENDED_HOLD_DAYS + 1, "B_EXTEND_ADDED_TRADES_1D"),
        (2, config.MAX_EXTENDED_HOLD_DAYS + 2, "C_EXTEND_ADDED_TRADES_2D"),
        (3, config.MAX_EXTENDED_HOLD_DAYS + 3, "D_EXTEND_ADDED_TRADES_3D"),
    ]:
        dates_by_sym = _get_added_trade_date_ranges(baseline_result, extra_days=extra_days + 4)
        df_mod       = _force_can_extend_for_dates(df_base, dates_by_sym)
        _go(name, df_mod, cfg=dict(MAX_EXTENDED_HOLD_DAYS=hard_cap))

    # E/F — Two-pass: extend trades that were profitable at day 3
    for extra_days, hard_cap, name in [
        (3, config.MAX_EXTENDED_HOLD_DAYS + 1, "E_EXTEND_PROFITABLE_TREND_1D"),
        (5, config.MAX_EXTENDED_HOLD_DAYS + 2, "F_EXTEND_PROFITABLE_TREND_2D"),
    ]:
        dates_by_sym = _get_profitable_at_day3_dates(baseline_result, extra_days=extra_days)
        df_mod       = _force_can_extend_for_dates(df_base, dates_by_sym)
        _go(name, df_mod, cfg=dict(MAX_EXTENDED_HOLD_DAYS=hard_cap))

    # G — Extend EARLY/MIDDLE entries only (+2d)
    #     Pre-computable: use entry_location_pct_20d < 0.60 (not LATE)
    df_g = df_base.copy()
    not_late = df_g["entry_location_pct_20d"] < 0.60
    # Force can_extend=True for rows where we're holding AND the ENTRY day was not-late
    # Proxy: mark can_extend on all not-late days for TREND signal rows
    trend_long = (df_g["strategy_used"] == "TREND") & (df_g["final_direction"] == "LONG") & not_late
    df_g.loc[trend_long, "can_extend"] = True
    _go("G_EXTEND_EARLY_MIDDLE_ONLY_2D", df_g, cfg=dict(MAX_EXTENDED_HOLD_DAYS=config.MAX_EXTENDED_HOLD_DAYS + 2))

    # H — NQ winners only (+2d)
    df_h = df_base.copy()
    nq_trend = (df_h["symbol"] == "NQ") & (df_h["strategy_used"] == "TREND")
    df_h.loc[nq_trend, "can_extend"] = True
    _go("H_EXTEND_NQ_WINNERS_2D", df_h, cfg=dict(MAX_EXTENDED_HOLD_DAYS=config.MAX_EXTENDED_HOLD_DAYS + 2))

    # I — High conviction winners only (+2d)
    df_i = df_base.copy()
    hc_trend = (df_i.get("high_conviction", pd.Series(False, index=df_i.index)).fillna(False)) & \
               (df_i["strategy_used"] == "TREND")
    df_i.loc[hc_trend, "can_extend"] = True
    _go("I_EXTEND_HIGH_CONVICTION_WINNERS_2D", df_i, cfg=dict(MAX_EXTENDED_HOLD_DAYS=config.MAX_EXTENDED_HOLD_DAYS + 2))

    # J — Added trades + high conviction (+2d)
    dates_by_sym = _get_added_trade_date_ranges(baseline_result, extra_days=4)
    df_j = _force_can_extend_for_dates(df_base, dates_by_sym)
    df_j.loc[hc_trend, "can_extend"] = True
    _go("J_EXTEND_ADDED_PLUS_HIGH_CONVICTION_2D", df_j,
        cfg=dict(MAX_EXTENDED_HOLD_DAYS=config.MAX_EXTENDED_HOLD_DAYS + 2))

    # Extension trade audit from baseline
    tl_b = baseline_result["trade_log"]
    if tl_b is not None and not tl_b.empty and "extension_used" in tl_b.columns:
        ext_trades = tl_b[tl_b["extension_used"] == True]
        for _, et in ext_trades.iterrows():
            audit_rows.append({
                "symbol":        et["symbol"],
                "entry_date":    et["entry_date"],
                "exit_date":     et["exit_date"],
                "exit_reason":   et["exit_reason"],
                "holding_days":  et["holding_days"],
                "pnl":           round(et["pnl"], 4),
                "win":           et["pnl"] > 0,
                "add_count":     et.get("add_count", 0),
                "high_conviction": et.get("high_conviction", False),
            })

    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SECTION 4 — ES-SPECIFIC TREND TREATMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_es_specific_experiments(df_base: pd.DataFrame, baseline_result: dict) -> pd.DataFrame:
    """
    Section 4: A_BASELINE through I_ES_ALLOWED_ONLY_IF_NQ_CONFIRMS.
    ES-specific sizing, filtering, and disabling experiments.
    """
    logger.info("═" * 60)
    logger.info("SECTION 4 — ES-SPECIFIC TREND TREATMENT (9 experiments)")
    logger.info("═" * 60)

    btl_sharpe = baseline_result["stats"]["sharpe_ratio"]
    btl        = baseline_result["trade_log"]
    rows       = []

    def _go(name, df_mod, cfg=None):
        r   = _run(name, df_mod, cfg)
        row = flatten(name, r, btl)
        rows.append(row)
        delta = row["sharpe"] - btl_sharpe

        es_pnl = row.get("pnl_ES", "?")
        nq_pnl = row.get("pnl_NQ", "?")
        logger.info("  %-45s  Sharpe=%.3f (Δ%+.3f)  ES_PnL=%.4f  NQ_PnL=%.4f  MaxDD=%.1f%%",
                    name, row["sharpe"], delta,
                    es_pnl if isinstance(es_pnl, float) else 0,
                    nq_pnl if isinstance(nq_pnl, float) else 0,
                    row.get("max_dd_pct", 0))
        return r

    # A — Baseline
    row_a = flatten("A_BASELINE", baseline_result, btl)
    rows.append(row_a)
    logger.info("  %-45s  Sharpe=%.3f (baseline)  ES=%.4f  NQ=%.4f",
                "A_BASELINE", btl_sharpe,
                row_a.get("pnl_ES", 0), row_a.get("pnl_NQ", 0))

    # B — ES TREND size 75% (reduce entry_quality by 1 for ES TREND)
    df_b = df_base.copy()
    es_trend = (df_b["symbol"] == "ES") & (df_b["strategy_used"] == "TREND") & (df_b["final_direction"] == "LONG")
    df_b.loc[es_trend, "entry_quality"] = (df_b.loc[es_trend, "entry_quality"].fillna(3).astype(int) - 1).clip(lower=1)
    _go("B_ES_TREND_SIZE_75", df_b)

    # C — ES TREND size 50% (reduce entry_quality by 2)
    df_c = df_base.copy()
    df_c.loc[es_trend, "entry_quality"] = (df_c.loc[es_trend, "entry_quality"].fillna(3).astype(int) - 2).clip(lower=1)
    _go("C_ES_TREND_SIZE_50", df_c)

    # D — ES TREND only if high conviction
    df_d = df_base.copy()
    es_no_hc = es_trend & ~(df_d.get("high_conviction", pd.Series(False, index=df_d.index)).fillna(False))
    df_d.loc[es_no_hc, "final_direction"] = "FLAT"
    _go("D_ES_TREND_ONLY_HIGH_CONVICTION", df_d)

    # E — Block ES late 70
    _go("E_ES_BLOCK_LATE_70", _block_late(df_base, "late_entry_flag_70", sym_filter="ES"))

    # F — Block ES late 80
    _go("F_ES_BLOCK_LATE_80", _block_late(df_base, "late_entry_flag_80", sym_filter="ES"))

    # G — Turn off ES TREND entirely
    df_g = df_base.copy()
    df_g.loc[es_trend, "final_direction"] = "FLAT"
    _go("G_ES_TREND_OFF_KEEP_STAT_ARB", df_g)

    # H — NQ TREND only + STAT_ARB (turn off ES TREND)
    #     (identical to G for TREND; STAT_ARB uses NQ-only by construction)
    _go("H_NQ_TREND_ONLY_PLUS_STAT_ARB", df_g)   # same df as G

    # I — ES TREND only if NQ also has TREND signal on same date
    #     Build a set of dates where NQ has a TREND LONG signal
    nq_trend_dates = set(
        df_base.loc[(df_base["symbol"] == "NQ") &
                    (df_base["strategy_used"] == "TREND") &
                    (df_base["final_direction"] == "LONG"), "date"]
    )
    df_i = df_base.copy()
    es_no_nq_confirm = es_trend & ~(df_i["date"].isin(nq_trend_dates))
    df_i.loc[es_no_nq_confirm, "final_direction"] = "FLAT"
    _go("I_ES_ALLOWED_ONLY_IF_NQ_CONFIRMS", df_i)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 8. SECTION 5 — COMBINED BEST TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_combined_experiments(
    df_base: pd.DataFrame,
    baseline_result: dict,
    best_late_filter: str = "K_BLOCK_ES_LATE_80_ONLY",
    best_late_flag:   str = "late_entry_flag_80",
    best_late_sym:    str = "ES",
    best_add_cfg:     dict = None,
    best_ext_cfg:     dict = None,
    best_es_cfg:      dict = None,
) -> pd.DataFrame:
    """
    Section 5: A_BASELINE through I_ALL_ACCEPTED_COMPONENTS.
    Fills in accepted best variants from Sections 1-4.
    """
    logger.info("═" * 60)
    logger.info("SECTION 5 — COMBINED BEST TEST (9 experiments)")
    logger.info("═" * 60)

    btl_sharpe = baseline_result["stats"]["sharpe_ratio"]
    btl        = baseline_result["trade_log"]
    rows       = []

    # Default "best" configs (will be overridden by caller if needed)
    add_cfg = best_add_cfg or {}
    ext_cfg = best_ext_cfg or {}

    def _go(name, df_mod, cfg=None):
        r   = _run(name, df_mod, cfg)
        row = flatten(name, r, btl)
        rows.append(row)
        delta = row["sharpe"] - btl_sharpe
        logger.info("  %-45s  Sharpe=%.3f (Δ%+.3f)  MaxDD=%.1f%%  Trades=%d  Adds=%d",
                    name, row["sharpe"], delta,
                    row.get("max_dd_pct", 0), row.get("trade_count", 0),
                    row.get("add_to_winner_count", 0))
        return r

    # A — Baseline
    row_a = flatten("A_BASELINE", baseline_result, btl)
    rows.append(row_a)
    logger.info("  %-45s  Sharpe=%.3f (baseline)", "A_BASELINE", btl_sharpe)

    # B — Best late filter only
    df_b = _block_late(df_base, best_late_flag, sym_filter=best_late_sym)
    _go("B_BEST_LATE_FILTER", df_b)

    # C — Best add expansion only
    _go("C_BEST_ADD_EXPANSION", df_base, cfg=add_cfg if add_cfg else None)

    # D — Best winner extension only
    dates_ext = _get_added_trade_date_ranges(baseline_result, extra_days=4)
    df_d = _force_can_extend_for_dates(df_base, dates_ext)
    hard_cap_d = config.MAX_EXTENDED_HOLD_DAYS + 2
    _go("D_BEST_WINNER_EXTENSION", df_d,
        cfg=dict(MAX_EXTENDED_HOLD_DAYS=hard_cap_d, **(ext_cfg or {})))

    # E — Best ES treatment only
    es_trend = (df_base["symbol"] == "ES") & (df_base["strategy_used"] == "TREND") & \
               (df_base["final_direction"] == "LONG")
    df_e = _block_late(df_base, "late_entry_flag_80", sym_filter="ES")
    _go("E_BEST_ES_TREATMENT", df_e)

    # F — Late filter + add expansion
    df_f = _block_late(df_base, best_late_flag, sym_filter=best_late_sym)
    _go("F_LATE_FILTER_PLUS_ADD", df_f, cfg=add_cfg if add_cfg else None)

    # G — Add expansion + winner extension
    dates_ext_g = _get_added_trade_date_ranges(baseline_result, extra_days=4)
    df_g = _force_can_extend_for_dates(df_base, dates_ext_g)
    _go("G_ADD_PLUS_EXTENSION", df_g,
        cfg=dict(MAX_EXTENDED_HOLD_DAYS=config.MAX_EXTENDED_HOLD_DAYS + 2,
                 **(add_cfg or {})))

    # H — Late filter + ES treatment
    df_h = _block_late(
        _block_late(df_base, best_late_flag, sym_filter=best_late_sym),
        "late_entry_flag_70",
        sym_filter="ES",
    )
    _go("H_LATE_FILTER_PLUS_ES_TREATMENT", df_h)

    # I — All accepted components
    df_i = _block_late(df_base, best_late_flag, sym_filter=best_late_sym)
    dates_ext_i = _get_added_trade_date_ranges(baseline_result, extra_days=4)
    df_i = _force_can_extend_for_dates(df_i, dates_ext_i)
    _go("I_ALL_ACCEPTED_COMPONENTS", df_i,
        cfg=dict(MAX_EXTENDED_HOLD_DAYS=config.MAX_EXTENDED_HOLD_DAYS + 2,
                 **(add_cfg or {})))

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 9. FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_final_report(sweeps: dict, baseline_sharpe: float) -> None:
    """Answer the 10 final questions based on sweep results."""

    print("\n" + "═" * 80)
    print("  TREND CAPTURE IMPROVEMENT — FINAL REPORT")
    print("═" * 80)

    def _best(df, metric="sharpe", higher_is_better=True):
        """Return the best row in the sweep (excluding baseline)."""
        non_base = df[~df["experiment"].str.startswith("A_")]
        if non_base.empty:
            return None
        if higher_is_better:
            return non_base.loc[non_base[metric].idxmax()]
        else:
            return non_base.loc[non_base[metric].idxmin()]

    def _verdict(sharpe, max_dd_abs, baseline_sharpe, baseline_dd_abs):
        # max_dd_abs and baseline_dd_abs are positive (e.g., 3.7, 5.7)
        # delta_dd < 0 = drawdown IMPROVED; delta_dd > 0 = drawdown WORSENED
        delta_s  = sharpe       - baseline_sharpe
        delta_dd = max_dd_abs   - baseline_dd_abs   # positive = worse, negative = better
        if delta_s > 0.02 and delta_dd <= 0.3:       # improved Sharpe, DD not materially worse
            return "ACCEPT"
        elif delta_s > 0.005 or (delta_dd < -0.3 and delta_s > -0.01):  # marginal Sharpe or clear DD gain
            return "CONTINUE_TESTING"
        elif delta_s < -0.02:
            return "REJECT"
        else:
            return "KEEP_DIAGNOSTIC_ONLY"

    s1 = sweeps.get("late_filter",   pd.DataFrame())
    s2 = sweeps.get("add_expansion", pd.DataFrame())
    s3 = sweeps.get("extension",     pd.DataFrame())
    s4 = sweeps.get("es_specific",   pd.DataFrame())
    s5 = sweeps.get("combined",      pd.DataFrame())

    base_dd = sweeps.get("baseline_dd", -5.7)

    print("\n  Q1: Does blocking late entries improve performance?")
    if not s1.empty:
        best_s1 = _best(s1)
        if best_s1 is not None:
            delta = best_s1["sharpe"] - baseline_sharpe
            print(f"     Best filter: {best_s1['experiment']}")
            print(f"     Sharpe: {best_s1['sharpe']:.3f} (Δ{delta:+.3f} vs {baseline_sharpe:.3f} baseline)")
            print(f"     MaxDD: {best_s1['max_dd_pct']:.1f}%  Trades: {best_s1['trade_count']}")
            ans = "YES" if delta > 0.005 else ("MARGINAL" if delta > -0.005 else "NO")
            print(f"     Answer: {ans}")

    print("\n  Q2: Which late threshold works best?")
    if not s1.empty:
        non_base = s1[~s1["experiment"].str.startswith("A_")].copy()
        if not non_base.empty:
            best_thr = non_base.loc[non_base["sharpe"].idxmax()]
            print(f"     Best threshold experiment: {best_thr['experiment']}")
            print(f"     Sharpe={best_thr['sharpe']:.3f}  MaxDD={best_thr['max_dd_pct']:.1f}%")
            # Print all threshold rows
            thr_rows = non_base[non_base["experiment"].str.contains("BLOCK_LATE_|BLOCK_ES_LATE")]
            for _, r in thr_rows.iterrows():
                delta = r["sharpe"] - baseline_sharpe
                print(f"       {r['experiment']:<40}  Sharpe={r['sharpe']:.3f}  Δ={delta:+.3f}  Trades={r['trade_count']}")

    print("\n  Q3: Does ES need stricter treatment?")
    if not s4.empty:
        es_rows = s4[s4["experiment"].str.contains("ES")].copy()
        if not es_rows.empty:
            best_es = es_rows.loc[es_rows["sharpe"].idxmax()]
            delta   = best_es["sharpe"] - baseline_sharpe
            ans     = "YES — ACCEPT" if delta > 0.01 else ("MARGINAL" if delta > 0 else "NO — REJECT")
            print(f"     Best ES experiment: {best_es['experiment']}  Sharpe={best_es['sharpe']:.3f}  Δ={delta:+.3f}")
            print(f"     Answer: {ans}")

    print("\n  Q4: Can add-to-winner be expanded?")
    if not s2.empty:
        best_s2 = _best(s2)
        if best_s2 is not None:
            delta = best_s2["sharpe"] - baseline_sharpe
            ans   = "YES" if delta > 0.005 else ("MARGINAL" if delta > -0.005 else "NO")
            print(f"     Best add experiment: {best_s2['experiment']}")
            print(f"     Sharpe={best_s2['sharpe']:.3f}  Δ={delta:+.3f}  Adds={best_s2.get('add_to_winner_count', '?')}")
            print(f"     Answer: {ans}")

    print("\n  Q5: Can winners be held longer?")
    if not s3.empty:
        best_s3 = _best(s3)
        if best_s3 is not None:
            delta = best_s3["sharpe"] - baseline_sharpe
            ans   = "YES" if delta > 0.005 else ("MARGINAL" if delta > -0.005 else "NO")
            print(f"     Best extension experiment: {best_s3['experiment']}")
            print(f"     Sharpe={best_s3['sharpe']:.3f}  Δ={delta:+.3f}  AvgHold={best_s3.get('avg_hold_days', '?')}")
            print(f"     Answer: {ans}")

    # Q6-9: Which change improves each metric most?
    all_sweeps = [s for s in [s1, s2, s3, s4] if not s.empty]
    if all_sweeps:
        all_df = pd.concat(all_sweeps, ignore_index=True)
        all_df = all_df[~all_df["experiment"].str.startswith("A_")]

        print("\n  Q6: Which change improves trend capture most?")
        # Proxy: highest add_to_winner_count (more adds = better capture)
        if "add_to_winner_count" in all_df.columns:
            best_cap = all_df.loc[all_df["add_to_winner_count"].idxmax()]
            print(f"     {best_cap['experiment']}  Adds={best_cap['add_to_winner_count']}")

        print("\n  Q7: Which change improves Sharpe most?")
        if "sharpe" in all_df.columns:
            best_sh = all_df.loc[all_df["sharpe"].idxmax()]
            delta   = best_sh["sharpe"] - baseline_sharpe
            print(f"     {best_sh['experiment']}  Sharpe={best_sh['sharpe']:.3f}  Δ={delta:+.3f}")

        print("\n  Q8: Which change improves PnL most?")
        if "total_pnl_pct" in all_df.columns:
            best_pnl = all_df.loc[all_df["total_pnl_pct"].idxmax()]
            print(f"     {best_pnl['experiment']}  PnL={best_pnl['total_pnl_pct']:.2f}%")

        print("\n  Q9: Which change reduces drawdown most?")
        if "max_dd_pct" in all_df.columns:
            best_dd = all_df.loc[all_df["max_dd_pct"].idxmax()]   # least negative = max
            print(f"     {best_dd['experiment']}  MaxDD={best_dd['max_dd_pct']:.1f}%")

    print("\n  Q10: FINAL VERDICTS")
    base_dd_abs = abs(base_dd)

    verdicts = {}

    for label, sweep in [("LATE_ENTRY_FILTER", s1), ("ADD_TO_WINNER_EXPANSION", s2),
                         ("WINNER_EXTENSION", s3), ("ES_SPECIFIC_TREND_TREATMENT", s4),
                         ("COMBINED_TREND_CAPTURE_UPGRADE", s5)]:
        if sweep.empty:
            verdicts[label] = "INSUFFICIENT_DATA"
            continue
        non_base = sweep[~sweep["experiment"].str.startswith("A_")]
        if non_base.empty:
            verdicts[label] = "INSUFFICIENT_DATA"
            continue
        best = non_base.loc[non_base["sharpe"].idxmax()]
        v    = _verdict(best["sharpe"], abs(best["max_dd_pct"]), baseline_sharpe, base_dd_abs)
        verdicts[label] = v

    for label, v in verdicts.items():
        print(f"     {'ACCEPT' if v=='ACCEPT' else v:<30}  {label}")

    print("\n" + "═" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  TREND CAPTURE IMPROVEMENT EXPERIMENTS")
    print("=" * 80)

    # ── Step 1: Build and cache base DataFrame ─────────────────────────────
    logger.info("Building full pipeline ...")
    df_base = build_base_df()
    logger.info("Computing late entry features ...")
    df_base = compute_late_entry_features(df_base)

    # Quick sanity: how many TREND LONG rows at each late threshold?
    trend_long = (df_base["strategy_used"] == "TREND") & (df_base["final_direction"] == "LONG")
    tl_count   = trend_long.sum()
    for thr, col in [(70, "late_entry_flag_70"), (80, "late_entry_flag_80"), (90, "late_entry_flag_90")]:
        n_late = (trend_long & (df_base[col] > 0.5)).sum()
        logger.info("  TREND LONG signals: %d total | late>%d%%: %d (%.0f%%)",
                    tl_count, thr, n_late, 100 * n_late / max(tl_count, 1))

    # ── Step 2: Baseline backtest ──────────────────────────────────────────
    logger.info("Running baseline backtest ...")
    baseline_result = _run("A_BASELINE", df_base)
    baseline_sharpe = baseline_result["stats"]["sharpe_ratio"]
    baseline_dd     = baseline_result["stats"]["max_drawdown_pct"]
    baseline_tl     = baseline_result["trade_log"]
    logger.info("BASELINE: Sharpe=%.3f  MaxDD=%.1f%%  Trades=%d",
                baseline_sharpe, baseline_dd,
                baseline_result["stats"].get("total_trades", 0))

    # ── Step 3: Section 1 — Late Entry Filter ─────────────────────────────
    s1_sweep, s1_attr = run_late_filter_experiments(df_base, baseline_result)
    s1_sweep.to_csv(os.path.join(OUT_DIR, "late_entry_filter_sweep.csv"), index=False)
    s1_attr.to_csv(os.path.join(OUT_DIR, "late_entry_blocked_trade_attribution.csv"), index=False)
    logger.info("Saved late_entry_filter_sweep.csv + blocked_trade_attribution.csv")
    _print_sweep(s1_sweep.to_dict("records"), "SECTION 1: LATE ENTRY FILTER")

    # ── Step 4: Section 2 — Add-to-winner ─────────────────────────────────
    s2_sweep, s2_audit = run_add_to_winner_experiments(df_base, baseline_result)
    s2_sweep.to_csv(os.path.join(OUT_DIR, "add_to_winner_expansion_sweep.csv"), index=False)
    s2_audit.to_csv(os.path.join(OUT_DIR, "add_to_winner_trade_audit.csv"), index=False)
    logger.info("Saved add_to_winner_expansion_sweep.csv + trade_audit.csv")
    _print_sweep(s2_sweep.to_dict("records"), "SECTION 2: ADD-TO-WINNER EXPANSION")

    # ── Step 5: Section 3 — Winner Extension ──────────────────────────────
    s3_sweep, s3_audit = run_winner_extension_experiments(df_base, baseline_result)
    s3_sweep.to_csv(os.path.join(OUT_DIR, "winner_extension_sweep.csv"), index=False)
    s3_audit.to_csv(os.path.join(OUT_DIR, "winner_extension_trade_audit.csv"), index=False)
    logger.info("Saved winner_extension_sweep.csv + trade_audit.csv")
    _print_sweep(s3_sweep.to_dict("records"), "SECTION 3: WINNER EXTENSION")

    # ── Step 6: Section 4 — ES-Specific ───────────────────────────────────
    s4_sweep = run_es_specific_experiments(df_base, baseline_result)
    s4_sweep.to_csv(os.path.join(OUT_DIR, "es_specific_trend_treatment.csv"), index=False)
    logger.info("Saved es_specific_trend_treatment.csv")
    _print_sweep(s4_sweep.to_dict("records"), "SECTION 4: ES-SPECIFIC TREATMENT")

    # ── Step 7: Identify best components for Section 5 ────────────────────
    def _best_row(df):
        non = df[~df["experiment"].str.startswith("A_")]
        return non.loc[non["sharpe"].idxmax()] if not non.empty else None

    best_s1 = _best_row(s1_sweep)
    best_s2 = _best_row(s2_sweep)
    best_s3 = _best_row(s3_sweep)

    # Determine best late filter name/flag/sym from S1
    best_late_flag = "late_entry_flag_80"
    best_late_sym  = "ES"
    if best_s1 is not None:
        exp_name = best_s1["experiment"]
        if "90" in exp_name: best_late_flag = "late_entry_flag_90"
        elif "85" in exp_name: best_late_flag = "late_entry_flag_85"
        elif "75" in exp_name: best_late_flag = "late_entry_flag_75"
        elif "70" in exp_name: best_late_flag = "late_entry_flag_70"
        if "NQ" in exp_name and "ES" not in exp_name: best_late_sym = "NQ"
        elif "ES" not in exp_name: best_late_sym = None

    # Determine best add config from S2
    _add_configs = {
        "B_EARLIER_ADD_TRIGGER_SMALL":     dict(ADD_WINNER_MIN_CUM_RET=0.0025, ADD_WINNER_SIZE_FRAC=0.25),
        "C_EARLIER_ADD_TRIGGER_CURRENT_SIZE": dict(ADD_WINNER_MIN_CUM_RET=0.0025),
        "D_CURRENT_TRIGGER_LARGER_ADD":    dict(ADD_WINNER_SIZE_FRAC=0.75),
        "E_ALLOW_SECOND_ADD_SMALL":        dict(ADD_WINNER_MAX_ADDS=2, ADD_WINNER_SIZE_FRAC=0.25),
        "F_ALLOW_SECOND_ADD_CURRENT_SIZE": dict(ADD_WINNER_MAX_ADDS=2),
        "I_EARLIER_ADD_PLUS_SECOND_ADD":   dict(ADD_WINNER_MIN_CUM_RET=0.0025, ADD_WINNER_MAX_ADDS=2),
    }
    best_add_cfg = _add_configs.get(best_s2["experiment"] if best_s2 is not None else "", {})

    logger.info("Best S1: %s  |  Best S2: %s  |  Best S3: %s",
                best_s1["experiment"] if best_s1 is not None else "None",
                best_s2["experiment"] if best_s2 is not None else "None",
                best_s3["experiment"] if best_s3 is not None else "None")

    # ── Step 8: Section 5 — Combined ──────────────────────────────────────
    s5_sweep = run_combined_experiments(
        df_base, baseline_result,
        best_late_flag=best_late_flag,
        best_late_sym=best_late_sym,
        best_add_cfg=best_add_cfg,
    )
    s5_sweep.to_csv(os.path.join(OUT_DIR, "trend_capture_improvement_combined.csv"), index=False)
    logger.info("Saved trend_capture_improvement_combined.csv")
    _print_sweep(s5_sweep.to_dict("records"), "SECTION 5: COMBINED BEST TEST")

    # ── Step 9: Final Report ───────────────────────────────────────────────
    write_final_report(
        sweeps=dict(
            late_filter=s1_sweep,
            add_expansion=s2_sweep,
            extension=s3_sweep,
            es_specific=s4_sweep,
            combined=s5_sweep,
            baseline_dd=baseline_dd,
        ),
        baseline_sharpe=baseline_sharpe,
    )

    # ── Step 10: Summary CSV ───────────────────────────────────────────────
    all_sweeps = pd.concat([s1_sweep, s2_sweep, s3_sweep, s4_sweep, s5_sweep],
                           ignore_index=True)
    all_sweeps.to_csv(os.path.join(OUT_DIR, "all_experiments_summary.csv"), index=False)
    logger.info("Saved all_experiments_summary.csv (%d experiments)", len(all_sweeps))
    logger.info("All outputs written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
