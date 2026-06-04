"""
run_sizing.py — Position sizing parameter sweep.

Baseline: Config D (MR+PB disabled, SL=-3.0%, trail 1.5%/1.0%).
Sweeps TARGET_VOL_DAILY × VOL_SCALE_MAX to find the right balance between
position size and risk.

WHY this matters:
  Current average position = $4,652 (18.6% of $25k). Root cause:
    vol_scalar = TARGET_VOL_DAILY / ewma_vol ≈ 0.004 / 0.012 ≈ 0.33
  Raising TARGET_VOL_DAILY directly raises vol_scalar and thus position size.
  Raising VOL_SCALE_MAX raises the TREND base weight ceiling.

Grid:
  TARGET_VOL_DAILY : [0.004, 0.006, 0.008, 0.010, 0.012, 0.015]
  VOL_SCALE_MAX    : [0.30, 0.40, 0.50]

For each combo reports:
  Sharpe, Sortino, Ann Return %, Max DD %, Win Rate, Profit Factor,
  Avg Entry Weight, Avg Dollar Position ($25k base), Max Dollar Position,
  Utilisation (avg_weight / MAX_COMBINED_EXPOSURE), Dollar PnL.

Usage:
    python run_sizing.py
"""

import importlib
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# ── Ensure project root on path ───────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import main as _main_mod

# ── Baseline config to restore between runs ───────────────────────────────────
_ORIG_TARGET_VOL   = config.TARGET_VOL_DAILY
_ORIG_VOL_MAX      = config.VOL_SCALE_MAX

# Competition account size
CAPITAL = 25_000.0

# ── Sweep grid ────────────────────────────────────────────────────────────────
TARGET_VOL_GRID = [0.004, 0.006, 0.008, 0.010, 0.012, 0.015]
VOL_MAX_GRID    = [0.30, 0.40, 0.50]


def _set_config(**kwargs):
    for k, v in kwargs.items():
        setattr(config, k, v)


def _restore_config():
    _set_config(
        TARGET_VOL_DAILY = _ORIG_TARGET_VOL,
        VOL_SCALE_MAX    = _ORIG_VOL_MAX,
    )


def _extract(result: dict, tvd: float, vsm: float) -> dict:
    """Pull metrics from a run_backtest result."""
    stats     = result.get("stats", {})
    trade_log = result.get("trade_log", pd.DataFrame())
    port_ret  = result.get("portfolio_returns", pd.Series(dtype=float))

    sharpe    = float(stats.get("sharpe_ratio", np.nan))
    ann_ret   = float(stats.get("annualized_return_pct", np.nan))
    tot_ret   = float(stats.get("total_return_pct", np.nan))
    max_dd    = float(stats.get("max_drawdown_pct", np.nan))
    pf        = float(stats.get("profit_factor", np.nan))

    # Sortino — compute from portfolio returns
    downside = port_ret[port_ret < 0]
    sortino  = (port_ret.mean() / downside.std() * np.sqrt(252)) if len(downside) > 5 else np.nan

    # Win rate
    win_rate  = float(trade_log["win"].mean()) if "win" in trade_log.columns and not trade_log.empty else np.nan

    # Dollar PnL
    cum_ret   = (1 + port_ret).prod() - 1
    dollar_pnl = cum_ret * CAPITAL

    # ── Position size metrics ─────────────────────────────────────────────────
    if not trade_log.empty and "entry_weight" in trade_log.columns:
        ew = trade_log["entry_weight"].dropna()
        # Only look at actual entries (weight > 0)
        ew_pos = ew[ew > 0]
        avg_w   = ew_pos.mean()  if len(ew_pos) > 0 else np.nan
        med_w   = ew_pos.median() if len(ew_pos) > 0 else np.nan
        max_w   = ew_pos.max()   if len(ew_pos) > 0 else np.nan
        p25_w   = ew_pos.quantile(0.25) if len(ew_pos) > 0 else np.nan
        p75_w   = ew_pos.quantile(0.75) if len(ew_pos) > 0 else np.nan
        n_trades = len(ew_pos)
    else:
        avg_w = med_w = max_w = p25_w = p75_w = np.nan
        n_trades = 0

    avg_dollar = avg_w * CAPITAL if not np.isnan(avg_w) else np.nan
    max_dollar = max_w * CAPITAL if not np.isnan(max_w) else np.nan

    # Utilisation = avg combined exposure vs cap
    util = avg_w / config.MAX_COMBINED_EXPOSURE if not np.isnan(avg_w) else np.nan

    return {
        "tvd_label":    f"{tvd*100:.1f}%",
        "vol_max":      vsm,
        "target_vol":   tvd,
        "sharpe":       round(sharpe,  3),
        "sortino":      round(sortino, 3),
        "ann_ret_pct":  round(ann_ret, 2),
        "total_ret_pct":round(tot_ret, 2),
        "max_dd_pct":   round(max_dd,  2),
        "profit_factor":round(pf,      3),
        "win_rate":     round(win_rate,3),
        "dollar_pnl":   round(dollar_pnl, 0),
        "n_trades":     n_trades,
        "avg_weight":   round(avg_w,   4) if not np.isnan(avg_w) else np.nan,
        "med_weight":   round(med_w,   4) if not np.isnan(med_w) else np.nan,
        "max_weight":   round(max_w,   4) if not np.isnan(max_w) else np.nan,
        "p25_weight":   round(p25_w,   4) if not np.isnan(p25_w) else np.nan,
        "p75_weight":   round(p75_w,   4) if not np.isnan(p75_w) else np.nan,
        "avg_dollar":   round(avg_dollar, 0) if not np.isnan(avg_dollar) else np.nan,
        "max_dollar":   round(max_dollar, 0) if not np.isnan(max_dollar) else np.nan,
        "utilisation":  round(util,    3) if not np.isnan(util) else np.nan,
    }


def run_sweep():
    import logging
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.WARNING,   # suppress pipeline noise
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    total = len(TARGET_VOL_GRID) * len(VOL_MAX_GRID)
    print(f"\n{'='*70}")
    print(f"  SIZING SWEEP — {total} combos  (baseline: Config D, SL=-3%, trail 1.5/1.0)")
    print(f"  Capital: ${CAPITAL:,.0f}   MAX_COMBINED_EXPOSURE: {config.MAX_COMBINED_EXPOSURE}")
    print(f"{'='*70}\n")

    results = []
    idx = 0

    for vsm in VOL_MAX_GRID:
        for tvd in TARGET_VOL_GRID:
            idx += 1
            label = f"TVD={tvd*100:.1f}% / VMAX={vsm:.2f}"
            print(f"[{idx:2d}/{total}] Running {label} ...", flush=True)

            _set_config(TARGET_VOL_DAILY=tvd, VOL_SCALE_MAX=vsm)

            try:
                # Re-import modules that cache config at import time
                importlib.reload(config)
                _set_config(TARGET_VOL_DAILY=tvd, VOL_SCALE_MAX=vsm)

                result = _main_mod.run(fetch=False)
                row    = _extract(result, tvd, vsm)
                results.append(row)

                print(
                    f"         Sharpe={row['sharpe']:.3f}  "
                    f"AnnRet={row['ann_ret_pct']:.1f}%  "
                    f"MaxDD={row['max_dd_pct']:.2f}%  "
                    f"AvgPos=${row['avg_dollar']:,.0f}  "
                    f"MaxPos=${row['max_dollar']:,.0f}  "
                    f"DollarPnL=${row['dollar_pnl']:,.0f}"
                )

            except Exception as e:
                print(f"         ERROR: {e}")
                results.append({"tvd_label": f"{tvd*100:.1f}%", "vol_max": vsm,
                                  "target_vol": tvd, "error": str(e)})
            finally:
                _restore_config()

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out_path = Path("output/sizing_sweep_results.csv")
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    # ── Pretty print table ─────────────────────────────────────────────────────
    _print_table(df)
    _print_recommendation(df)

    return df


def _print_table(df: pd.DataFrame):
    cols = ["tvd_label", "vol_max", "sharpe", "sortino", "ann_ret_pct",
            "max_dd_pct", "profit_factor", "win_rate", "avg_dollar", "max_dollar",
            "utilisation", "dollar_pnl", "n_trades"]
    view = df[[c for c in cols if c in df.columns]].copy()

    # Highlight best Sharpe per VOL_MAX group
    print(f"\n{'='*110}")
    print(f"  {'TVD':>6}  {'VMAX':>5}  {'Sharpe':>7}  {'Sortino':>8}  "
          f"{'AnnRet%':>8}  {'MaxDD%':>7}  {'PF':>5}  {'WR':>5}  "
          f"{'AvgPos$':>9}  {'MaxPos$':>9}  {'Util%':>6}  {'$PnL':>9}  {'N':>4}")
    print(f"  {'-'*104}")

    for _, r in view.iterrows():
        def _f(v, fmt=""):
            try:
                return format(v, fmt)
            except Exception:
                return str(v)

        print(
            f"  {_f(r.get('tvd_label',''),'>6')}  "
            f"  {_f(r.get('vol_max', ''),'>4')}  "
            f"  {_f(r.get('sharpe', np.nan),'>6.3f')}  "
            f"  {_f(r.get('sortino', np.nan),'>7.3f')}  "
            f"  {_f(r.get('ann_ret_pct', np.nan),'>7.1f')}  "
            f"  {_f(r.get('max_dd_pct', np.nan),'>6.2f')}  "
            f"  {_f(r.get('profit_factor', np.nan),'>4.2f')}  "
            f"  {_f(r.get('win_rate', np.nan),'>4.3f')}  "
            f"  {_f(r.get('avg_dollar', np.nan),'>9,.0f')}  "
            f"  {_f(r.get('max_dollar', np.nan),'>9,.0f')}  "
            f"  {_f(r.get('utilisation', np.nan)*100 if pd.notna(r.get('utilisation')) else np.nan,'>5.1f')}%  "
            f"  {_f(r.get('dollar_pnl', np.nan),'>9,.0f')}  "
            f"  {_f(r.get('n_trades', 0),'>4.0f')}"
        )

    print(f"  {'='*104}")


def _print_recommendation(df: pd.DataFrame):
    if "sharpe" not in df.columns or df["sharpe"].isna().all():
        return

    best = df.loc[df["sharpe"].idxmax()]

    print(f"\n{'='*70}")
    print(f"  RECOMMENDATION")
    print(f"{'='*70}")
    print(f"  Best Sharpe: {best['sharpe']:.3f}  →  "
          f"TARGET_VOL_DAILY={best['target_vol']:.3f}  "
          f"VOL_SCALE_MAX={best['vol_max']:.2f}")
    print(f"  Average position size at best config: ${best['avg_dollar']:,.0f}  "
          f"({best['avg_dollar']/25000*100:.1f}% of capital)")
    print(f"  Max position size: ${best['max_dollar']:,.0f}  "
          f"({best['max_dollar']/25000*100:.1f}% of capital)")
    print(f"  Ann Return: {best['ann_ret_pct']:.1f}%  "
          f"Max DD: {best['max_dd_pct']:.2f}%  "
          f"Dollar PnL: ${best['dollar_pnl']:,.0f}")

    # Practical advice
    current_best_dd = best["max_dd_pct"]
    print(f"\n  APPLY TO config.py:")
    print(f"    TARGET_VOL_DAILY = {best['target_vol']:.3f}")
    print(f"    VOL_SCALE_MAX    = {best['vol_max']:.2f}")

    # Show Sharpe cost of different sizes
    print(f"\n  SHARPE vs POSITION SIZE trade-off:")
    size_df = df[["tvd_label", "vol_max", "sharpe", "avg_dollar", "max_dd_pct"]].sort_values("sharpe", ascending=False)
    for _, r in size_df.head(6).iterrows():
        print(f"    TVD={r['tvd_label']} VMAX={r['vol_max']:.2f} → "
              f"Sharpe={r['sharpe']:.3f}  "
              f"AvgPos=${r['avg_dollar']:,.0f}  "
              f"MaxDD={r['max_dd_pct']:.2f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_sweep()
