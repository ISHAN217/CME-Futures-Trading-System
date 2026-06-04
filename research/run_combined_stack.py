"""
run_combined_stack.py — Full stacked system combining all accepted improvements.

Stack:
  Layer 0 — Raw baseline                         (Sharpe ~0.882)
  Layer 1 — ES late-entry filter (loc > 70%)     (from run_trend_capture_experiments J)
  Layer 2 — EARLY_TREND_NQ injection             (from run_early_pullback_experiments B)
  Layer 3 — RELAXED_WEEKLY_BIAS_NQ injection     (from run_early_pullback_experiments D)

Experiments:
  A_BASELINE           — raw, no changes
  B_ES_LATE_FILTER     — Layer 1 only
  C_ES_FILTER_EARLY    — Layers 1+2
  D_ES_FILTER_RELAXED  — Layers 1+3
  E_FULL_STACK         — Layers 1+2+3  (the target)
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
from backtest           import run_backtest

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT_DIR = "output/combined_stack"
os.makedirs(OUT_DIR, exist_ok=True)

EARLY_TREND_SIZE  = 0.07
RELAXED_BIAS_SIZE = 0.05
_HOLD_CFG = {**config.MAX_HOLD_DAYS, "MTF_BREAKOUT": 2, "MTF_RESET": 3}


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_base_df():
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
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Entry-location features ───────────────────────────────────────────────────

def add_entry_features(df):
    df = df.copy()
    for col in ["entry_location_pct_20d", "late_entry_flag_70"]:
        df[col] = np.nan
    for sym, g in df.groupby("symbol", sort=False):
        idx = g.index
        s   = pd.Series(g["close"].values, index=idx)
        mn  = s.rolling(20, min_periods=5).min()
        mx  = s.rolling(20, min_periods=5).max()
        loc = (s - mn) / (mx - mn).replace(0, np.nan)
        df.loc[idx, "entry_location_pct_20d"] = loc.clip(0, 1).values
        df.loc[idx, "late_entry_flag_70"]      = (loc > 0.70).astype(float).values
    return df


# ── Layer 1: ES late-entry filter ─────────────────────────────────────────────

def apply_es_late_filter(df):
    """Block ES TREND entries when entry_location_pct_20d > 70%."""
    mask = (
        (df["symbol"]          == "ES") &
        (df["strategy_used"]   == "TREND") &
        (df["final_direction"] == "LONG") &
        (df["late_entry_flag_70"] > 0.5)
    )
    df = df.copy()
    df.loc[mask, "final_direction"] = "FLAT"
    df.loc[mask, "strategy_used"]   = "NONE"
    blocked = int(mask.sum())
    logger.info("  ES late filter blocked %d rows", blocked)
    return df, blocked


# ── Layer 2: EARLY_TREND_NQ ───────────────────────────────────────────────────

def apply_early_trend_nq(df):
    """Inject NQ early-trend signals on flat rows."""
    loc = df["entry_location_pct_20d"].fillna(1.0)
    r2  = df["r2_20d"].fillna(0.0)
    mask = (
        (df["symbol"]          == "NQ") &
        (df["final_direction"] == "FLAT") &
        (df["momentum_direction"].fillna("NEUTRAL") == "LONG") &
        (df["slope_20d"].fillna(0.0) > 0) &
        (r2 < 0.65) &
        (df["weekly_bias"].fillna("NEUTRAL") != "SHORT") &
        (loc < 0.65) &
        (df["portfolio_regime"].fillna("CHOP") != "SHOCK") &
        (df["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )
    df = df.copy()
    df.loc[mask, "final_direction"] = "LONG"
    df.loc[mask, "strategy_used"]   = "MTF_BREAKOUT"
    df.loc[mask, "mtf_entry_size"]  = EARLY_TREND_SIZE
    df.loc[mask, "entry_quality"]   = 3
    n = int(mask.sum())
    logger.info("  EARLY_TREND_NQ injected %d rows", n)
    return df, n


# ── Layer 3: RELAXED_WEEKLY_BIAS_NQ ──────────────────────────────────────────

def apply_relaxed_weekly_bias(df):
    """Inject NQ entries when weekly_bias==NEUTRAL but quality/h4 confirms."""
    h4_ok   = df["h4_exec_signal"].fillna("NONE").isin(["MOMENTUM","BREAKOUT"])
    qual_ok = df["trend_quality_bucket"].fillna("MODERATE").isin(
                  ["GOOD","STRONG","EXCEPTIONAL"])
    hc_ok   = df.get("high_conviction", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    mask = (
        (df["symbol"]          == "NQ") &
        (df["final_direction"] == "FLAT") &
        (df["weekly_bias"].fillna("NEUTRAL") == "NEUTRAL") &
        (df["momentum_direction"].fillna("NEUTRAL") == "LONG") &
        (df["portfolio_regime"].fillna("CHOP").isin(["TREND","TRANSITION"])) &
        (h4_ok | qual_ok | hc_ok) &
        (df["portfolio_sentiment_flag"].fillna("NORMAL") != "HIGH_RISK")
    )
    df = df.copy()
    df.loc[mask, "final_direction"] = "LONG"
    df.loc[mask, "strategy_used"]   = "MTF_RESET"
    df.loc[mask, "mtf_entry_size"]  = RELAXED_BIAS_SIZE
    df.loc[mask, "entry_quality"]   = 3
    n = int(mask.sum())
    logger.info("  RELAXED_WEEKLY_BIAS_NQ injected %d rows", n)
    return df, n


# ── Run one backtest and extract stats ────────────────────────────────────────

@contextlib.contextmanager
def override_config(**kwargs):
    orig = {k: getattr(config, k) for k in kwargs if hasattr(config, k)}
    for k, v in kwargs.items(): setattr(config, k, v)
    try: yield
    finally:
        for k, v in orig.items(): setattr(config, k, v)


def run(label, df):
    with override_config(MAX_HOLD_DAYS=_HOLD_CFG):
        res = run_backtest(df)
    st  = res["stats"]
    tl  = res["trade_log"]
    pr  = res["portfolio_returns"]

    pnl_es = tl[tl["symbol"]=="ES"]["pnl"].sum() if not tl.empty else 0.0
    pnl_nq = tl[tl["symbol"]=="NQ"]["pnl"].sum() if not tl.empty else 0.0

    new_strats = {"MTF_RESET","MTF_BREAKOUT"}
    new_tl = tl[tl["strategy"].isin(new_strats)] if not tl.empty else pd.DataFrame()
    n_new  = len(new_tl)
    new_wr = new_tl["win"].mean() if n_new > 0 else float("nan")
    new_pf = _pf(new_tl)

    yr = {}
    for y in range(2019, 2027):
        m = pr.index.year == y
        if m.any():
            yr[y] = round(float((1 + pr[m]).prod() - 1) * 100, 2)

    row = dict(
        experiment        = label,
        sharpe            = round(st["sharpe_ratio"], 4),
        annual_ret_pct    = round(st["annualized_return_pct"], 2),
        total_pnl_pct     = round(st["total_return_pct"], 2),
        max_dd_pct        = round(st["max_drawdown_pct"], 2),
        trade_count       = st["total_trades"],
        win_rate          = round(st["overall_win_rate"], 4),
        profit_factor     = round(st["profit_factor"], 4),
        new_trade_count   = n_new,
        new_trade_wr      = round(new_wr, 4) if not np.isnan(new_wr) else float("nan"),
        new_trade_pf      = round(new_pf, 4) if not np.isnan(new_pf) else float("nan"),
        pnl_ES            = round(pnl_es, 4),
        pnl_NQ            = round(pnl_nq, 4),
        **{f"yr_{k}": v for k, v in yr.items()},
    )
    return row, tl


def _pf(tl):
    if tl.empty or "pnl" not in tl.columns: return float("nan")
    w = tl[tl["pnl"] > 0]["pnl"].sum()
    l = abs(tl[tl["pnl"] < 0]["pnl"].sum())
    return w / l if l > 1e-9 else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df0 = add_entry_features(build_base_df())

    results = []
    trade_logs = {}

    # A — raw baseline
    logger.info("=== A_BASELINE ===")
    row, tl = run("A_BASELINE", df0)
    results.append(row); trade_logs["A"] = tl

    # B — ES late filter only
    logger.info("=== B_ES_LATE_FILTER ===")
    df1, _ = apply_es_late_filter(df0)
    row, tl = run("B_ES_LATE_FILTER", df1)
    results.append(row); trade_logs["B"] = tl

    # C — ES filter + EARLY_TREND_NQ
    logger.info("=== C_ES_FILTER_PLUS_EARLY ===")
    df2, _ = apply_early_trend_nq(df1)
    row, tl = run("C_ES_FILTER_PLUS_EARLY", df2)
    results.append(row); trade_logs["C"] = tl

    # D — ES filter + RELAXED_WEEKLY_BIAS
    logger.info("=== D_ES_FILTER_PLUS_RELAXED ===")
    df3, _ = apply_relaxed_weekly_bias(df1)
    row, tl = run("D_ES_FILTER_PLUS_RELAXED", df3)
    results.append(row); trade_logs["D"] = tl

    # E — FULL STACK: ES filter + EARLY + RELAXED
    logger.info("=== E_FULL_STACK ===")
    df4, _ = apply_early_trend_nq(df1)       # layer 2 on top of layer 1
    df4, _ = apply_relaxed_weekly_bias(df4)  # layer 3 on top
    row, tl = run("E_FULL_STACK", df4)
    results.append(row); trade_logs["E"] = tl

    res = pd.DataFrame(results)
    base_sharpe = res.loc[res["experiment"]=="A_BASELINE","sharpe"].values[0]
    res["delta_sharpe"] = (res["sharpe"] - base_sharpe).round(4)
    res["final_$25k"]   = (25000 * (1 + res["annual_ret_pct"]/100)**7).round(0).astype(int)
    res["gain_$25k"]    = res["final_$25k"] - 25000
    res["maxdd_$25k"]   = (res["max_dd_pct"]/100 * 25000).round(0).astype(int)

    res.to_csv(f"{OUT_DIR}/combined_stack_results.csv", index=False)

    # ── Print report ─────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("COMBINED STACK — RESULTS")
    print("="*72)

    cols = ["experiment","sharpe","delta_sharpe","annual_ret_pct",
            "max_dd_pct","trade_count","win_rate","profit_factor",
            "new_trade_count","new_trade_wr","new_trade_pf",
            "final_$25k","gain_$25k","maxdd_$25k"]
    print(res[[c for c in cols if c in res.columns]].to_string(index=False))

    yr_cols = [c for c in res.columns if c.startswith("yr_")]
    if yr_cols:
        print("\n--- Annual Return % by Year ---")
        print(res[["experiment"] + yr_cols].to_string(index=False))

    print("\n--- PnL by Symbol ---")
    print(res[["experiment","pnl_ES","pnl_NQ"]].to_string(index=False))

    print("\n" + "="*72)
    print("STACK BREAKDOWN (incremental Sharpe from each layer)")
    print("="*72)
    rows_map = {r["experiment"]: r for r in results}
    sharpe_A = rows_map["A_BASELINE"]["sharpe"]
    sharpe_B = rows_map["B_ES_LATE_FILTER"]["sharpe"]
    sharpe_C = rows_map["C_ES_FILTER_PLUS_EARLY"]["sharpe"]
    sharpe_E = rows_map["E_FULL_STACK"]["sharpe"]
    print(f"  Raw baseline:                        {sharpe_A:.4f}")
    print(f"  + ES late-entry filter (loc>70%):   {sharpe_B:.4f}  ({sharpe_B-sharpe_A:+.4f})")
    print(f"  + EARLY_TREND_NQ:                   {sharpe_C:.4f}  ({sharpe_C-sharpe_B:+.4f})")
    print(f"  + RELAXED_WEEKLY_BIAS_NQ:           {sharpe_E:.4f}  ({sharpe_E-sharpe_C:+.4f})")
    print(f"\n  FULL STACK vs Baseline:             {sharpe_E-sharpe_A:+.4f}")
    print(f"  FULL STACK Sharpe:                  {sharpe_E:.4f}")
    print(f"  MaxDD (full stack):                 {rows_map['E_FULL_STACK']['max_dd_pct']:.1f}%")
    e_row = res[res["experiment"] == "E_FULL_STACK"].iloc[0]
    print(f"  Final $25k → ${e_row['final_$25k']:,}  "
          f"(+${e_row['gain_$25k']:,})")
    print("="*72)


if __name__ == "__main__":
    main()
