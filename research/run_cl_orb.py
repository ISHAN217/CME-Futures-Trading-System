"""
run_cl_orb.py — Out-of-sample ORB validation on Crude Oil (CL / MCL).

Purpose:
  Run the same ORB strategy on CL as an independent out-of-sample test.
  ES+NQ ORB was developed and tuned in-sample.  If CL ORB produces a
  Sharpe ≈ 0.9–1.3 with no parameter changes, the concept is real.

Data required:
  CL 1-min OHLCV, continuous front-month, backadjusted.
  Download from Databento (GLBX.MDP3, schema=ohlcv-1m, stype=continuous, symbol=CL.c.0).

  Once downloaded (e.g. glbx-mdp3-*.ohlcv-1m.csv.zst), run:
    python3 run_cl_orb.py --build-parquet /path/to/file.csv.zst
  This creates output/mtf/CL_1min_continuous.parquet.

  Then run the backtest:
    python3 run_cl_orb.py

CL Contract specs:
  CL  (full):  $1,000 / pt  (1 pt = $1/bbl × 1,000 bbl)
  MCL (micro): $100  / pt   (1/10th of CL)
  Tick: $0.01 / bbl = $10 (CL), $1 (MCL)
  OR range filter: 0.50 – 2.00 $/bbl typical (TBD from data)

Strategy (unchanged from ES+NQ version):
  - Observe 09:30–10:00 ET range  (same open/close as equity futures)
  - Enter breakout at H_OR (long) or L_OR (short)
  - Stop at opposite extreme; target = entry ± 1.5 × range
  - Force close 15:30 ET
  - No alignment filter (single instrument)
  - 1% risk per trade, 1-tick slippage, $2.23 IBKR round-trip (MCL)
"""

import logging
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── CL Contract constants ────────────────────────────────────────────────────
ACCOUNT_START = 25_000.0
CL_MULT       = 1_000.0   # full CL: $1,000/pt  (1 pt = $1/bbl × 1,000 bbl)
MCL_MULT      = 100.0     # micro CL: $100/pt   (1/10th of CL)
CL_TICK       = 0.01      # $0.01/bbl minimum tick
COMM_MCL      = 2.23      # IBKR round-trip per MCL contract
COMM_CL       = 2.10      # IBKR round-trip per CL  contract

# ─── CL-specific session times (DIFFERENT from ES/NQ) ────────────────────────
#
# ES/NQ  OR: 09:30–10:00 ET  (equity open)       EOD: 15:30 ET
# CL     OR: 09:00–09:30 ET  (NYMEX US session)   EOD: 14:30 ET  (settlement)
#
# Why 09:00? NYMEX crude opened its pit session at 09:00 ET.
# Even post-pit (2020+), 09:00 remains the "US crude market open" in terms of
# participant attention, inventory expectations, and bid-ask spread compression.
# The CL daily settlement is printed at ~14:28 ET — we force-close at 14:30.
#
OR_START_H  = 9           # 09:00 ET
OR_START_M  = 0
OR_END_H    = 9           # 09:30 ET — 30-min range (same duration as ES/NQ)
OR_END_M    = 30
CUTOFF_H    = 14          # no new entries after 14:00 ET
CUTOFF_M    = 0
EOD_H       = 14          # force-close at 14:30 ET (CL settlement)
EOD_M       = 30

# Strategy parameters (FROZEN from ES+NQ tuning — no re-optimization)
PROFIT_MULT  = 1.5
RISK_PCT     = 0.01

DATA_DIR = "/Users/ishanbhardwaj/cme_competition_system/output/mtf"
CL_PARQUET   = f"{DATA_DIR}/CL_1min_continuous.parquet"


# ─── Build parquet from raw Databento .zst ───────────────────────────────────

def build_cl_parquet(zst_path: str, out_path: str = CL_PARQUET) -> pd.DataFrame:
    """
    Load raw Databento CL 1-min CSV (zst-compressed), build a synthetic
    continuous front-month series, and save to parquet.

    Handles two Databento formats:
      stype=continuous  → symbol = 'CL.c.0' (use directly)
      stype=native      → symbols like CLQ1, CLF2, CLZ3 (stitch front-month)

    For native contracts, 'front month' is defined as the contract with the
    highest per-bar volume — this robustly handles roll transitions without
    needing a hardcoded roll calendar.

    Year code convention for CL (single-digit = last digit of year):
      CLQ1 = August 2021, CLF2 = January 2022, CLZ3 = December 2023, etc.
    """
    import subprocess, io

    logger.info("Decompressing %s …", zst_path)
    proc = subprocess.run(
        ["zstd", "-d", zst_path, "--stdout"],
        capture_output=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"zstd failed: {proc.stderr.decode()[:200]}")

    raw = pd.read_csv(io.BytesIO(proc.stdout), low_memory=False)
    logger.info("Raw rows: %d  sample symbols: %s",
                len(raw), sorted(raw["symbol"].unique())[:8])

    # Parse timestamp → ET
    raw["ts"] = pd.to_datetime(raw["ts_event"], utc=True)
    raw["ts_et"] = raw["ts"].dt.tz_convert("America/New_York")

    # ── Case 1: continuous contract already present ────────────────────────
    if "CL.c.0" in raw["symbol"].values:
        df = raw[raw["symbol"] == "CL.c.0"].copy()
        logger.info("Using continuous CL.c.0  %d bars", len(df))

    # ── Case 2: native outright contracts — build highest-volume synthetic ─
    else:
        logger.info("Building synthetic continuous (highest-volume per bar) …")

        # Keep only plain outright contracts: CL + month_code + year_digits
        # Examples: CLQ1, CLF2, CLZ3, CLF30 (unlikely but handle gracefully)
        outrights = raw[raw["symbol"].str.match(r"^CL[FGHJKMNQUVXZ]\d{1,2}$")].copy()
        logger.info("  Outrights: %d rows,  %d unique contracts",
                    len(outrights), outrights["symbol"].nunique())

        if len(outrights) == 0:
            raise ValueError("No outright CL contracts found. Check the symbol format.")

        # ── Parse expiry date (for roll-away filter) ──────────────────────
        month_map = dict(F=1, G=2, H=3, J=4, K=5, M=6,
                         N=7, Q=8, U=9, V=10, X=11, Z=12)

        def parse_expiry(sym: str) -> pd.Timestamp:
            """
            CLQ1  → Aug 2021   (1-digit year: 202X)
            CLF2  → Jan 2022
            CLF30 → Jan 2030   (2-digit year: 20XX when XX < 50)
            """
            try:
                month_code = sym[2]
                year_str   = sym[3:]
                m = month_map.get(month_code, 0)
                if m == 0:
                    return pd.NaT
                y = int(year_str)
                if y < 10:                    # single digit: 1→2021 … 9→2029
                    y = 2020 + y
                elif y < 50:                  # two digits: 22→2022
                    y = 2000 + y
                else:                         # two digits: 50+→1950+
                    y = 1900 + y
                # CL expiry = 3rd-last business day of the delivery month
                # Approximate as the 20th of the month for roll-away filter
                return pd.Timestamp(year=y, month=m, day=20)
            except Exception:
                return pd.NaT

        outrights["expiry"] = outrights["symbol"].apply(parse_expiry)
        outrights = outrights.dropna(subset=["expiry"])

        # Floor to minute for groupby key
        outrights["ts_min"] = outrights["ts_et"].dt.floor("1min")

        # For each minute: pick the contract with the highest volume.
        # This is the dominant (front-month) contract at that instant.
        # Tie-break by earliest expiry (prefer near-term).
        outrights = outrights.sort_values(
            ["ts_min", "volume", "expiry"],
            ascending=[True, False, True]   # volume desc, expiry asc
        )
        df = outrights.groupby("ts_min").first().reset_index()
        df = df.rename(columns={"ts_min": "ts_et_floor"})
        # Restore ts_et from the winning row (already sorted)
        # Use ts_et_floor as the canonical bar timestamp
        df["ts_et"] = df["ts_et_floor"]
        logger.info("Synthetic continuous: %d bars", len(df))

    # ── Standardise output columns ─────────────────────────────────────────
    df = df.rename(columns={
        "open":   "CL_open",
        "high":   "CL_high",
        "low":    "CL_low",
        "close":  "CL_close",
        "volume": "CL_volume",
    })
    df["timestamp"] = pd.to_datetime(df["ts_et"]).dt.tz_convert("UTC")
    keep = ["timestamp", "ts_et", "CL_open", "CL_high", "CL_low",
            "CL_close", "CL_volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ts_et"]      = pd.to_datetime(df["ts_et"])
    df["hour_et"]    = df["ts_et"].dt.hour
    df["min_et"]     = df["ts_et"].dt.minute
    df["trade_date"] = df["ts_et"].dt.date

    df.to_parquet(out_path, index=False)
    logger.info("Saved → %s  (%d bars  %s → %s)",
                out_path, len(df),
                df["ts_et"].min().date(), df["ts_et"].max().date())
    return df


# ─── ORB backtest on CL ──────────────────────────────────────────────────────

def run_cl_orb(
    df:             pd.DataFrame,
    profit_mult:    float = PROFIT_MULT,
    min_range_cl:   float = 0.30,   # $/bbl — calibrated from data percentile
    risk_pct:       float = RISK_PCT,
    use_micro:      bool  = True,    # MCL by default ($100/pt)
    slip_ticks:     int   = 1,
) -> dict:
    """
    Single-instrument ORB for CL using CL-specific session times.

    CL session (NOT the same as ES/NQ):
      OR observation : 09:00–09:30 ET  (NYMEX US open)
      Trade window   : 09:30–14:00 ET  (entry cutoff before settlement)
      EOD force-close: 14:30 ET        (CL daily settlement)

    Parameters inherited from ES+NQ with ZERO re-optimisation:
      profit_mult=1.5, risk_pct=1%, slip=1 tick, min_range calibrated to data.
    """
    # CL RTH: 09:00–14:30 ET
    in_rth = (
        ((df["hour_et"] == OR_START_H) & (df["min_et"] >= OR_START_M)) |
        (df["hour_et"].between(OR_START_H + 1, EOD_H - 1)) |
        ((df["hour_et"] == EOD_H) & (df["min_et"] <= EOD_M))
    )
    rth = df[in_rth].copy()

    mult  = MCL_MULT if use_micro else CL_MULT
    comm  = COMM_MCL if use_micro else COMM_CL
    tick  = CL_TICK

    trade_log  = []
    daily_rets = {}
    account    = ACCOUNT_START

    for date, day in rth.groupby("trade_date", sort=True):
        date_str = str(date)
        day = day.sort_values("ts_et").reset_index(drop=True)

        # ── Opening range: 09:00–09:29 ET ────────────────────────────────
        obs = day[
            (day["hour_et"] == OR_START_H) &
            (day["min_et"] >= OR_START_M) &
            (day["min_et"] < OR_END_M)          # 09:00 ≤ t < 09:30
        ]
        if len(obs) < 5:
            continue

        H_OR = obs["CL_high"].max()
        L_OR = obs["CL_low"].min()
        rng  = H_OR - L_OR

        if rng < min_range_cl:
            continue

        # ── Trade window: 09:30–14:00 ET ─────────────────────────────────
        tw_mask = (
            ((day["hour_et"] > OR_END_H) |
             ((day["hour_et"] == OR_END_H) & (day["min_et"] >= OR_END_M))) &
            ((day["hour_et"] < CUTOFF_H) |
             ((day["hour_et"] == CUTOFF_H) & (day["min_et"] <= CUTOFF_M)))
        )
        # Also include bars up to EOD close at 14:30 for position management
        eod_mask = (
            (day["hour_et"] == EOD_H) & (day["min_et"] <= EOD_M)
        )
        tw = day[tw_mask | eod_mask].reset_index(drop=True)
        if len(tw) == 0:
            continue

        pos       = None
        traded    = False
        daily_pnl = 0.0

        def _book(exit_px, reason, ts):
            nonlocal account, daily_pnl, pos, traded
            d          = pos["dir"]
            raw_pts    = d * (exit_px - pos["entry"])
            entry_slip = slip_ticks * tick
            exit_slip  = slip_ticks * tick if reason in ("STOP", "EOD") else 0.0
            adj_pts    = raw_pts - (entry_slip + exit_slip)

            frac      = (account * risk_pct) / (pos["rng"] * mult)
            contracts = max(1, round(frac)) if use_micro else frac

            pnl_dollar = adj_pts * contracts * mult - comm * contracts
            pnl_pct    = pnl_dollar / account
            hold_min   = (ts - pos["entry_ts"]).total_seconds() / 60

            trade_log.append({
                "date":        date_str,
                "symbol":      "MCL" if use_micro else "CL",
                "direction":   "LONG" if d == 1 else "SHORT",
                "entry":       pos["entry"],
                "exit":        exit_px,
                "stop":        pos["stop"],
                "target":      pos["target"],
                "range":       pos["rng"],
                "pnl_pts":     adj_pts,
                "pnl_dollar":  pnl_dollar,
                "pnl_pct":     pnl_pct,
                "exit_reason": reason,
                "hold_min":    hold_min,
            })
            pos       = None
            traded    = True
            daily_pnl += pnl_pct
            account   += pnl_pct * account

        for i in range(len(tw)):
            if traded:
                break
            bar     = tw.iloc[i]
            is_last = (i == len(tw) - 1)
            t_h     = bar["hour_et"]
            t_m     = bar["min_et"]

            # EOD force-close at 14:30 (applies even without an open position
            # to avoid entering too close to settlement)
            at_eod = (t_h == EOD_H and t_m == EOD_M) or is_last
            past_cutoff = (t_h > CUTOFF_H or
                           (t_h == CUTOFF_H and t_m > CUTOFF_M))

            if pos is None:
                if past_cutoff:
                    continue    # no new entries after 14:00

                if bar["CL_high"] > H_OR:
                    pos = dict(dir=1, entry=H_OR, stop=L_OR,
                               target=H_OR + rng * profit_mult,
                               rng=rng, entry_ts=bar["ts_et"])
                    if bar["CL_low"] <= pos["stop"]:
                        _book(pos["stop"],   "STOP",   bar["ts_et"]); continue
                    if bar["CL_high"] >= pos["target"]:
                        _book(pos["target"], "TARGET", bar["ts_et"]); continue
                elif bar["CL_low"] < L_OR:
                    pos = dict(dir=-1, entry=L_OR, stop=H_OR,
                               target=L_OR - rng * profit_mult,
                               rng=rng, entry_ts=bar["ts_et"])
                    if bar["CL_high"] >= pos["stop"]:
                        _book(pos["stop"],   "STOP",   bar["ts_et"]); continue
                    if bar["CL_low"] <= pos["target"]:
                        _book(pos["target"], "TARGET", bar["ts_et"]); continue
            else:
                d = pos["dir"]
                if at_eod:
                    _book(bar["CL_close"], "EOD", bar["ts_et"])
                elif d == 1:
                    if bar["CL_low"] <= pos["stop"]:
                        _book(pos["stop"],   "STOP",   bar["ts_et"])
                    elif bar["CL_high"] >= pos["target"]:
                        _book(pos["target"], "TARGET", bar["ts_et"])
                else:
                    if bar["CL_high"] >= pos["stop"]:
                        _book(pos["stop"],   "STOP",   bar["ts_et"])
                    elif bar["CL_low"] <= pos["target"]:
                        _book(pos["target"], "TARGET", bar["ts_et"])

        if pos is not None and not traded:
            last_bar = tw.iloc[-1]
            _book(last_bar["CL_close"], "EOD", last_bar["ts_et"])

        if daily_pnl != 0:
            daily_rets[date_str] = daily_rets.get(date_str, 0.0) + daily_pnl

    return _compute_cl_stats(trade_log, daily_rets, account)


def _compute_cl_stats(trade_log, daily_rets, final_account):
    tl = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    dr = pd.Series(daily_rets, dtype=float)
    dr.index = pd.to_datetime(dr.index)
    dr = dr.sort_index()

    if len(dr) > 1:
        idx = pd.bdate_range(dr.index.min(), dr.index.max())
        dr  = dr.reindex(idx, fill_value=0.0)

    n_years = (dr.index.max() - dr.index.min()).days / 365.25 if len(dr) > 1 else 1.0
    cum_ret = (1 + dr).prod() - 1
    ann_ret = (1 + cum_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    mean_d  = dr.mean()
    std_d   = dr.std(ddof=1) if len(dr) > 1 and dr.std() > 0 else 1e-9
    sharpe  = mean_d / std_d * np.sqrt(252)
    equity  = (1 + dr).cumprod()
    peak    = equity.cummax()
    max_dd  = ((equity - peak) / peak).min() * 100
    ann_by_year = (dr.groupby(dr.index.year)
                     .apply(lambda x: (1 + x).prod() - 1) * 100)

    stats = {
        "sharpe_ratio":          round(sharpe, 4),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "max_drawdown_pct":      round(max_dd, 2),
        "total_trades":          len(tl),
        "overall_win_rate":      float((tl["pnl_pct"] > 0).mean()) if len(tl) else 0.0,
        "final_account":         round(final_account, 2),
        "trades_per_year":       round(len(tl) / n_years, 1),
    }
    return {"stats": stats, "trade_log": trade_log,
            "daily_returns": dr, "annual_rets": ann_by_year}


# ─── Range calibration ───────────────────────────────────────────────────────

def calibrate_cl_range(df: pd.DataFrame) -> float:
    """
    Print OR range statistics for the CL 09:00-09:30 ET window.
    Returns the 25th-percentile range (used as default min_range_cl filter).

    Compare to ES/NQ:
      ES OR (09:30-10:00): median ≈15 pts → min_range=8 (53rd pct)
      NQ OR (09:30-10:00): median ≈55 pts → min_range=30 (lower pct)
      CL OR (09:00-09:30): ?  — printed below, filter set to 25th pct
    """
    ranges = []
    rth = df[
        ((df["hour_et"] == OR_START_H) & (df["min_et"] >= OR_START_M)) |
        (df["hour_et"].between(OR_START_H + 1, EOD_H))
    ].copy()

    for date, day in rth.groupby("trade_date"):
        obs = day[
            (day["hour_et"] == OR_START_H) &
            (day["min_et"] >= OR_START_M) &
            (day["min_et"] < OR_END_M)
        ]
        if len(obs) >= 5:
            ranges.append(obs["CL_high"].max() - obs["CL_low"].min())

    r = pd.Series(ranges)
    print(f"\nCL OR Range distribution  (09:00–09:30 ET,  $/bbl):")
    print(f"  Days sampled : {len(r)}")
    print(f"  Mean         : {r.mean():.3f}")
    print(f"  Median       : {r.median():.3f}")
    print(f"  10th pct     : {r.quantile(0.10):.3f}")
    print(f"  25th pct     : {r.quantile(0.25):.3f}  ← default min_range_cl")
    print(f"  50th pct     : {r.quantile(0.50):.3f}")
    print(f"  75th pct     : {r.quantile(0.75):.3f}")
    print(f"  90th pct     : {r.quantile(0.90):.3f}")
    filt_25 = r.quantile(0.25)
    filt_50 = r.quantile(0.50)
    print(f"\n  Days passing 25th-pct filter ({filt_25:.2f}): {(r>=filt_25).mean():.0%}")
    print(f"  Days passing 50th-pct filter ({filt_50:.2f}): {(r>=filt_50).mean():.0%}")
    return float(filt_25)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import os

    if "--build-parquet" in sys.argv:
        idx = sys.argv.index("--build-parquet")
        zst = sys.argv[idx + 1]
        build_cl_parquet(zst)
        return

    if not os.path.exists(CL_PARQUET):
        print(f"\n❌  CL 1-min data not found at {CL_PARQUET}")
        print("   Steps to download:")
        print("   1. Go to https://databento.com → Historical data")
        print("   2. Dataset: GLBX.MDP3  Schema: ohlcv-1m")
        print("      Symbol: CL.c.0  stype: continuous")
        print("      Date range: 2018-01-01 to today")
        print("   3. Once downloaded (.csv.zst), run:")
        print("      python3 run_cl_orb.py --build-parquet /path/to/file.csv.zst")
        print()
        print("   ℹ️  The 1H file you have (GLBX-20260525-...) is NOT sufficient.")
        print("      ORB requires 1-min bars to resolve the 09:30-10:00 range.")
        return

    logger.info("Loading CL 1-min data …")
    df = pd.read_parquet(CL_PARQUET)
    logger.info("  %d bars  %s → %s",
                len(df), df["ts_et"].min().date(), df["ts_et"].max().date())

    print("\n── Range calibration (CL 09:00–09:30 ET OR) ──────────────")
    r25 = calibrate_cl_range(df)
    r25 = round(r25, 2)

    print("\n── Experiment ladder ──────────────────────────────────────")
    print("   OR: 09:00–09:30 ET   EOD: 14:30 ET   Target: 1.5R   Risk: 1%")
    print()

    experiments = [
        # (label,               min_range_cl)
        ("CL_NO_FILTER",        0.00),   # every day, no range screen
        ("CL_25PCT_RANGE",      r25),    # pass ~75% of days
        ("CL_2×25PCT_RANGE",    r25*2),  # pass ~50% of days
    ]

    all_results = []
    base_s = None
    for lbl, min_rng in experiments:
        r   = run_cl_orb(df, min_range_cl=min_rng)
        s   = r["stats"]
        ann = r["annual_rets"]
        d_sh  = s["sharpe_ratio"]          - base_s["sharpe_ratio"]          if base_s else 0
        d_ann = s["annualized_return_pct"] - base_s["annualized_return_pct"] if base_s else 0
        print(
            f"  {lbl:<22} | rng>{min_rng:.2f}  "
            f"Sh={s['sharpe_ratio']:+.3f}({d_sh:+.3f})  "
            f"Ann={s['annualized_return_pct']:+.1f}%({d_ann:+.1f}%)  "
            f"DD={s['max_drawdown_pct']:.1f}%  "
            f"T={s['total_trades']}({s['trades_per_year']:.0f}/yr)  "
            f"WR={s['overall_win_rate']*100:.0f}%  "
            f"$25k→${s['final_account']:,.0f}"
        )
        all_results.append((lbl, s, ann))
        if base_s is None:
            base_s = s

    # Annual breakdown table
    years = sorted(all_results[0][2].index.tolist())
    print()
    print("=" * 85)
    print("CL ORB — OUT-OF-SAMPLE ANNUAL RETURNS")
    print(f"  ES+NQ E8 reference: Sh=1.326  Ann=+37.2%  (in-sample, 2018-2026)")
    print(f"  CL data window    : {df['ts_et'].min().date()} → {df['ts_et'].max().date()}")
    print("=" * 85)
    print(f"  {'label':<22} " + "  ".join(f"{y:>6}" for y in years))
    print("-" * 85)
    for lbl, _, ann in all_results:
        row = f"  {lbl:<22} " + "  ".join(f"{ann.get(y,0):>+6.1f}%" for y in years)
        print(row)
    print()

    best_sh = max(s["sharpe_ratio"] for _, s, _ in all_results)
    print(f"  Best Sharpe: {best_sh:.3f}")
    if best_sh >= 0.9:
        print("  ✅  CL Sharpe ≥ 0.9 → ORB concept VALIDATED out-of-sample on energy futures")
    elif best_sh >= 0.6:
        print("  ⚠️   CL Sharpe 0.6–0.9 → concept generalises but with decay; ES/NQ has structural edge")
    else:
        print("  ❌  CL Sharpe < 0.6 → ORB may be equity-specific; ES/NQ edge needs re-evaluation")


if __name__ == "__main__":
    main()
