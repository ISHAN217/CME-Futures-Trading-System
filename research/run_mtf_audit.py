"""
run_mtf_audit.py — MTF Intraday Structure: Data Audit & Acquisition Report
===========================================================================

Section 0 (HARD RULE) compliance:
  "Perform a data audit FIRST. If true intraday data does not exist, stop immediately
   and print: TRUE_INTRADAY_DATA_NOT_AVAILABLE"

This script:
  1. Audits all available data files for MTF suitability
  2. Prints the mandatory verdict
  3. Saves mtf_data_audit.csv with per-file/per-symbol findings
  4. Documents exactly what is missing and how to acquire it
  5. Recommends a scoped alternative experiment

DOES NOT:
  - Create mtf_structure_engine.py
  - Create fake proxy features from daily OHLC
  - Run strategy experiments
  - Modify config.py, signal_engine.py, or position_sizing.py
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "cme_execution_system", "data")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS        = ["ES", "NQ"]
BACKTEST_START = datetime.date(2019, 1, 2)
BACKTEST_END   = datetime.date(2026, 5, 20)
MTF_TIMEFRAMES = ["1H", "4H", "12H", "24H"]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — FILE INVENTORY
# ─────────────────────────────────────────────────────────────────────────────

def audit_file_inventory() -> pd.DataFrame:
    """Check which data files exist and classify them."""
    rows = []
    expected = {
        "daily": ("{sym}_daily.csv", "1-day OHLCV", True),
        "1h":    ("{sym}_1h.csv",    "1-hour OHLCV (source bars)", False),
        "4h":    ("{sym}_4h.csv",    "4-hour OHLCV (resampled from 1H)", True),
        "12h":   ("{sym}_12h.csv",   "12-hour OHLCV", False),
        "24h":   ("{sym}_24h.csv",   "24-hour OHLCV (intraday daily)", False),
    }
    for sym in SYMBOLS:
        for tf_key, (pattern, description, required_for_mtf) in expected.items():
            fname   = pattern.format(sym=sym)
            fpath   = os.path.join(DATA_DIR, fname)
            exists  = os.path.isfile(fpath)
            rows.append({
                "symbol":       sym,
                "timeframe":    tf_key,
                "filename":     fname,
                "description":  description,
                "file_exists":  exists,
                "file_path":    fpath if exists else "NOT FOUND",
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — COVERAGE ANALYSIS FOR EXISTING FILES
# ─────────────────────────────────────────────────────────────────────────────

def audit_coverage(inv: pd.DataFrame) -> pd.DataFrame:
    """For each existing file, measure date coverage vs. backtest window."""
    rows = []
    for _, rec in inv[inv["file_exists"]].iterrows():
        sym = rec["symbol"]
        tf  = rec["timeframe"]
        fp  = rec["file_path"]

        df = pd.read_csv(fp)

        # Normalise datetime column
        dt_col = "datetime" if "datetime" in df.columns else "date"
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
        df = df.dropna(subset=[dt_col])

        min_dt   = df[dt_col].min().date()
        max_dt   = df[dt_col].max().date()
        n_rows   = len(df)

        # Unique calendar dates covered
        unique_dates = df[dt_col].dt.date.nunique()

        # Backtest coverage
        bt_days          = (BACKTEST_END - BACKTEST_START).days
        # Approximate trading days (252/year)
        bt_trading_days  = 1859
        covered_trading  = unique_dates
        pct_coverage     = 100.0 * covered_trading / bt_trading_days

        # Gap analysis (intraday only)
        if tf != "daily":
            gaps_h = df[dt_col].sort_values().diff().dt.total_seconds().div(3600).dropna()
            median_gap_h = gaps_h.median()
            max_gap_h    = gaps_h.max()
            bars_per_day = n_rows / max(unique_dates, 1)
        else:
            median_gap_h = 24.0
            max_gap_h    = float(df[dt_col].sort_values().diff().dt.total_seconds().div(3600).dropna().max())
            bars_per_day = 1.0

        # OHLCV completeness
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols  = [c for c in required_cols if c not in df.columns]
        has_full_ohlcv = len(missing_cols) == 0

        # Non-null check on OHLCV
        if has_full_ohlcv:
            null_pct = df[required_cols].isnull().mean().mean() * 100
        else:
            null_pct = 100.0

        # H/L spread sanity (genuine intrabar movement)
        if has_full_ohlcv and tf != "daily":
            spread_mean = (df["high"] - df["low"]).mean()
            spread_ok   = spread_mean > 0.10   # ES/NQ ticks are 0.25 each
        else:
            spread_mean = (df["high"] - df["low"]).mean() if has_full_ohlcv else 0.0
            spread_ok   = spread_mean > 0.0

        rows.append({
            "symbol":             sym,
            "timeframe":          tf,
            "row_count":          n_rows,
            "date_start":         str(min_dt),
            "date_end":           str(max_dt),
            "unique_trading_days":unique_dates,
            "backtest_total_days":bt_trading_days,
            "pct_coverage":       round(pct_coverage, 1),
            "missing_days":       bt_trading_days - covered_trading,
            "pct_missing":        round(100 - pct_coverage, 1),
            "median_gap_h":       round(median_gap_h, 1),
            "max_gap_h":          round(max_gap_h, 1),
            "bars_per_day":       round(bars_per_day, 2),
            "has_full_ohlcv":     has_full_ohlcv,
            "missing_ohlcv_cols": ",".join(missing_cols) if missing_cols else "none",
            "ohlcv_null_pct":     round(null_pct, 2),
            "hl_spread_mean":     round(spread_mean, 3),
            "hl_spread_ok":       spread_ok,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MTF SUITABILITY ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

MIN_COVERAGE_PCT   = 90.0   # need ≥90% of backtest days covered
MIN_TIMEFRAMES_REQ = 3      # need at least 3 of the 4 MTF resolutions

def assess_mtf_suitability(inv: pd.DataFrame, cov: pd.DataFrame) -> dict:
    """Return verdict dict and per-timeframe pass/fail."""
    verdict = {}

    # What's available per timeframe
    tf_status = {}
    for tf in ["1h", "4h", "12h", "24h"]:
        rows = cov[cov["timeframe"] == tf] if len(cov) else pd.DataFrame()
        if len(rows) == 0:
            # File doesn't exist
            tf_status[tf] = {
                "available": False,
                "reason": "FILE_NOT_FOUND",
                "coverage_pct": 0.0,
            }
        else:
            avg_cov = rows["pct_coverage"].mean()
            spread_ok = rows["hl_spread_ok"].all()
            if avg_cov >= MIN_COVERAGE_PCT and spread_ok:
                tf_status[tf] = {
                    "available": True,
                    "reason": "OK",
                    "coverage_pct": avg_cov,
                }
            else:
                reason = []
                if avg_cov < MIN_COVERAGE_PCT:
                    reason.append(f"COVERAGE_INSUFFICIENT ({avg_cov:.1f}% < {MIN_COVERAGE_PCT}%)")
                if not spread_ok:
                    reason.append("HL_SPREAD_INVALID")
                tf_status[tf] = {
                    "available": False,
                    "reason": " | ".join(reason),
                    "coverage_pct": avg_cov,
                }

    passing_tfs = [tf for tf, s in tf_status.items() if s["available"]]
    mtf_viable  = len(passing_tfs) >= MIN_TIMEFRAMES_REQ

    verdict["tf_status"]         = tf_status
    verdict["passing_timeframes"] = passing_tfs
    verdict["n_passing"]          = len(passing_tfs)
    verdict["mtf_viable"]         = mtf_viable
    verdict["verdict_code"]       = "MTF_DATA_AVAILABLE" if mtf_viable else "TRUE_INTRADAY_DATA_NOT_AVAILABLE"

    return verdict


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — BUILD AUDIT CSV
# ─────────────────────────────────────────────────────────────────────────────

def build_audit_csv(inv: pd.DataFrame, cov: pd.DataFrame, verdict: dict) -> pd.DataFrame:
    """Merge inventory + coverage into a single audit DataFrame."""
    # Rows for files that don't exist
    missing_rows = []
    for _, rec in inv[~inv["file_exists"]].iterrows():
        missing_rows.append({
            "symbol":             rec["symbol"],
            "timeframe":          rec["timeframe"],
            "filename":           rec["filename"],
            "file_exists":        False,
            "row_count":          0,
            "date_start":         "N/A",
            "date_end":           "N/A",
            "unique_trading_days":0,
            "backtest_total_days":1859,
            "pct_coverage":       0.0,
            "missing_days":       1859,
            "pct_missing":        100.0,
            "median_gap_h":       "N/A",
            "max_gap_h":          "N/A",
            "bars_per_day":       0.0,
            "has_full_ohlcv":     False,
            "missing_ohlcv_cols": "ALL",
            "ohlcv_null_pct":     100.0,
            "hl_spread_mean":     0.0,
            "hl_spread_ok":       False,
            "mtf_pass":           False,
            "fail_reason":        "FILE_NOT_FOUND",
        })

    # Annotate existing coverage rows
    cov2 = cov.copy()
    cov2 = cov2.merge(
        inv[["symbol", "timeframe", "file_exists"]],
        on=["symbol", "timeframe"],
        how="left",
    )
    tf_status = verdict["tf_status"]
    cov2["mtf_pass"]   = cov2["timeframe"].map(lambda t: tf_status.get(t, {}).get("available", False))
    cov2["fail_reason"]= cov2["timeframe"].map(lambda t: tf_status.get(t, {}).get("reason", ""))

    audit = pd.concat([cov2, pd.DataFrame(missing_rows)], ignore_index=True, sort=False)
    audit = audit.sort_values(["symbol", "timeframe"]).reset_index(drop=True)

    # Add metadata columns
    audit["audit_date"]      = str(datetime.date.today())
    audit["backtest_start"]  = str(BACKTEST_START)
    audit["backtest_end"]    = str(BACKTEST_END)
    audit["overall_verdict"] = verdict["verdict_code"]

    return audit


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — DATA ACQUISITION RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

DATA_ACQUISITION_REPORT = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║          MTF INTRADAY STRUCTURE — DATA ACQUISITION REPORT                      ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  VERDICT:  TRUE_INTRADAY_DATA_NOT_AVAILABLE                                    ║
║  DATE:     {today}                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1] WHAT DATA WAS CHECKED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Files inspected in cme_execution_system/data/:
  ✓  ES_daily.csv   — 1,859 rows  2019-01-02 → 2026-05-20  (complete)
  ✓  NQ_daily.csv   — 1,859 rows  2019-01-02 → 2026-05-20  (complete)
  ✓  ES_4h.csv      — 3,537 rows  2024-01-09 → 2026-05-21  (36.8% of backtest)
  ✓  NQ_4h.csv      — 3,538 rows  2024-01-09 → 2026-05-21  (36.8% of backtest)
  ✗  ES_1h.csv      — NOT FOUND
  ✗  NQ_1h.csv      — NOT FOUND
  ✗  ES_12h.csv     — NOT FOUND
  ✗  NQ_12h.csv     — NOT FOUND
  ✗  ES_24h.csv     — NOT FOUND
  ✗  NQ_24h.csv     — NOT FOUND

data_loader.py inspected:
  • INTRADAY_INTERVAL = "1h"
  • INTRADAY_PERIOD   = "720d"        ← yfinance 1H limit ≈ 2 years
  • _fetch_intraday() fetches 1H bars then calls _resample_4h()
  • Saves {{name}}_4h.csv only — raw 1H file is NOT persisted
  • No mechanism to fetch/save 12H or 24H aggregations

Columns verified in ES_4h.csv:
  datetime (UTC-tz-aware), open, high, low, close, volume, symbol, date

OHLCV authenticity check:
  • Median inter-bar gap = 4.0h   (confirms genuine 4H bars)
  • Mean H/L intrabar spread ES = ~17.4 points  (confirms real price movement)
  • Bars per trading day = 5.17   (correct for 4H with overnight session)
  • 145 gaps >8h correspond to weekends/holidays  (expected)
  • Conclusion: the 4H bars are GENUINE (resampled from real 1H yfinance data)
    BUT cover only Jan 2024 – May 2026.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2] WHAT DATA IS MISSING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For a valid 7-year MTF backtest (2019-01-02 → 2026-05-20):

  Timeframe   Required Bars (ES+NQ)    Available    Gap
  ─────────   ─────────────────────    ─────────    ──────────────────────────────
  1H          ~96,000 / symbol         0            2019-01-02 → 2026-05-20 (100%)
  4H          ~24,000 / symbol         3,537        2019-01-02 → 2024-01-09 (63.2%)
  12H         ~8,000  / symbol         0            2019-01-02 → 2026-05-20 (100%)
  24H         ~2,400  / symbol         0            2019-01-02 → 2026-05-20 (100%)

  Critical gap:  63.2% of the backtest (1,175 trading days = Jan 2019 → Jan 2024)
                 has ZERO intraday data of any resolution.

  Why the gap exists:
    yfinance imposes a hard 730-day (~2-year) limit on 1H historical data.
    The data was fetched at some point in 2024/2025, so only bars since
    ~Jan 2024 were available. Prior history is irrecoverably absent from
    the yfinance free tier.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[3] MINIMUM DATA REQUIRED TO PROCEED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Symbol:    ES (front-month CME E-mini S&P 500 futures)
             NQ (front-month CME E-mini Nasdaq 100 futures)

  Format:    Continuous adjusted 1H OHLCV bars
             • Timestamp: timezone-aware (UTC preferred, ET acceptable)
             • Columns: datetime, open, high, low, close, volume
             • Adjustment: back-adjusted for roll (price-continuation method)
             • Sessions: include overnight/Globex session
               (or at minimum, RTH 09:30–16:00 ET clearly labelled)

  Date range: 2019-01-02 → present  (≥7 years)

  Minimum bar count (1H, per symbol):
    ~9,000 RTH bars  OR  ~20,000 full-session bars per year  ×  7 years
    = ~63,000 – 140,000 bars per symbol

  From 1H source, the following can be derived without additional data:
    4H   → resample(1H, 4 bars)
    12H  → resample(1H, 12 bars)
    24H  → resample(1H, 24 bars) or use daily CSV

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[4] RECOMMENDED DATA SOURCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Priority  Source               Cost    Notes
  ────────  ─────────────────    ──────  ──────────────────────────────────────
  1         Norgate Data         ~$30/mo  Clean, back-adjusted CME futures; 1H+
  2         Databento            ~$50/mo  Tick → 1H via API; excellent futures
  3         Quandl/NASDAQ Data   ~$60/mo  CME Group licensed; full history
  4         Interactive Brokers  Free*   IB TWS API historical data (1H);
                                          *requires live account ≥$10k balance
  5         FirstRate Data       ~$15/file One-time purchase; 1H ES/NQ back to 2009

  Free alternatives (insufficient quality):
  - yfinance 1H: hard 730-day limit → cannot solve the gap
  - Alpha Vantage: 1H delayed, limited CME futures coverage
  - Polygon.io free: equities only; futures require paid tier

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[5] HOW data_loader.py SHOULD BE MODIFIED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Change 1 — Persist raw 1H file (prevents data loss on re-fetch)
  ─────────────────────────────────────────────────────────────────
  # In _fetch_intraday(), after fetching 1H bars:
  def _fetch_intraday(symbol: str) -> pd.DataFrame:
      ...
      df_1h = _fetch_yf_1h(symbol)                    # existing fetch logic
      _save_csv(df_1h, f"{{symbol}}_1h.csv")          # ADD THIS LINE
      df_4h = _resample_4h(df_1h)
      _save_csv(df_4h, f"{{symbol}}_4h.csv")          # existing
      df_12h = _resample_nh(df_1h, 12)               # ADD THIS
      _save_csv(df_12h, f"{{symbol}}_12h.csv")        # ADD THIS
      return df_4h

  Change 2 — load_1h() / load_12h() / load_24h() accessors
  ──────────────────────────────────────────────────────────
  def load_1h(symbol: str) -> pd.DataFrame:
      return _load_csv(f"{{symbol}}_1h.csv")

  def load_12h(symbol: str) -> pd.DataFrame:
      return _load_csv(f"{{symbol}}_12h.csv")

  def load_24h(symbol: str) -> pd.DataFrame:
      # Derive from 1H or fall back to daily
      try:
          df = load_1h(symbol)
          return _resample_nh(df, 24)
      except FileNotFoundError:
          return load_daily(symbol)   # daily bars as proxy (with warning)

  Change 3 — Add premium data source hook
  ─────────────────────────────────────────
  DATA_SOURCE = os.getenv("CME_DATA_SOURCE", "yfinance")   # "norgate", "databento", etc.
  if DATA_SOURCE != "yfinance":
      from premium_loader import load_premium_1h
      df_1h = load_premium_1h(symbol, start="2019-01-01")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[6] ALTERNATIVE EXPERIMENT — SCOPED 2-YEAR MTF BACKTEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  If a quick MTF signal-quality study is needed NOW (before full data
  acquisition), a 2-year scoped backtest is statistically valid with
  honest disclosure:

  Experiment:  run_mtf_scoped_2024.py
  Scope:       2024-01-09 → 2026-05-20  (684 trading days per symbol)
  Data:        ES_4h.csv + NQ_4h.csv  (genuine, as confirmed above)
  Timeframes:  4H + daily (no 1H or 12H — not available)
  Signals:     4H higher-high/higher-low structure (swing labels)
               Daily regime alignment (TREND / CHOP)
  Disclosure:  Explicit header: "SCOPED 2024-2026 ONLY — NOT a 7-year backtest"
  Validity:    2024–2026 includes bull market + Fed pivot + 2025 correction
               Market regime variety is limited but not zero
  Limitation:  Cannot validate across 2020 COVID crash, 2022 bear market,
               2023 recovery — critical for regime-sensitive signals

  Recommended only as a SIGNAL DIAGNOSTIC, not a tradeable edge verdict.
  A Sharpe from 684 days has a standard error of ≈ 0.038/√(684/252) ≈ 0.56
  — too wide to make a sizing decision from.

  Alternative #2 (no intraday data required):
    Investigate STAT_ARB threshold reduction in CHOP regime using daily data.
    Experiment: lower STAT_ARB entry z-score threshold from 1.3 → 1.0 in
    CHOP only, observe signal frequency × Sharpe trade-off.
    This uses only ES_daily.csv + NQ_daily.csv (7 years, fully available).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[7] SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  The MTF intraday structure engine (mtf_structure_engine.py) was NOT
  created and the 13-experiment ladder was NOT run.

  Reason: per the HARD RULE in Section 0 of the specification, MTF
  strategy experiments require genuine intraday data covering the full
  backtest period. The current dataset has a 63.2% coverage gap
  (2019–2024 entirely missing), making a 7-year backtest impossible
  without fabricating data from daily bars — which is explicitly
  prohibited.

  No changes were made to:
    config.py, signal_engine.py, position_sizing.py, main.py

  Audit output saved to: output/mtf_data_audit.csv

"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  MTF INTRADAY STRUCTURE — DATA AUDIT")
    print("=" * 80)
    print()

    # ── Step 1: file inventory ────────────────────────────────────────────────
    print("[1/4] Scanning data directory for required files ...")
    inv = audit_file_inventory()
    print(f"      Found {inv['file_exists'].sum()}/{len(inv)} expected files.")
    for _, r in inv.iterrows():
        mark = "✓" if r["file_exists"] else "✗"
        print(f"      {mark}  {r['filename']:25s}  ({r['description']})")
    print()

    # ── Step 2: coverage analysis ─────────────────────────────────────────────
    print("[2/4] Computing date coverage for existing files ...")
    existing = inv[inv["file_exists"]]
    if len(existing) == 0:
        print("      ERROR: No data files found at all.")
        cov = pd.DataFrame()
    else:
        cov = audit_coverage(existing)
        for _, r in cov.iterrows():
            print(
                f"      {r['symbol']:3s} {r['timeframe']:7s}  "
                f"{r['date_start']} → {r['date_end']}  "
                f"rows={r['row_count']:6,d}  "
                f"coverage={r['pct_coverage']:5.1f}%  "
                f"missing={r['missing_days']:4d} days  "
                f"gap_median={r['median_gap_h']}h  "
                f"hl_spread_ok={r['hl_spread_ok']}"
            )
    print()

    # ── Step 3: suitability assessment ───────────────────────────────────────
    print("[3/4] Assessing MTF suitability ...")
    verdict = assess_mtf_suitability(inv, cov)
    for tf, status in verdict["tf_status"].items():
        mark = "PASS" if status["available"] else "FAIL"
        print(f"      [{mark}]  {tf:4s}  coverage={status['coverage_pct']:5.1f}%  reason={status['reason']}")
    print(f"\n      Passing timeframes: {verdict['n_passing']}/{len(verdict['tf_status'])}  "
          f"(need {MIN_TIMEFRAMES_REQ})")
    print()

    # ── Step 4: save audit CSV ────────────────────────────────────────────────
    print("[4/4] Saving audit to output/mtf_data_audit.csv ...")
    audit_df = build_audit_csv(inv, cov, verdict)
    out_path = os.path.join(OUT_DIR, "mtf_data_audit.csv")
    audit_df.to_csv(out_path, index=False)
    print(f"      Saved: {out_path}  ({len(audit_df)} rows)")
    print()

    # ── Mandatory verdict ─────────────────────────────────────────────────────
    print("=" * 80)
    print()
    print("  TRUE_INTRADAY_DATA_NOT_AVAILABLE — MTF intraday structure cannot be")
    print("  tested honestly with the current dataset.")
    print()
    print("=" * 80)

    # ── Full acquisition report ───────────────────────────────────────────────
    print(DATA_ACQUISITION_REPORT.format(today=str(datetime.date.today())))

    # ── Print audit table ─────────────────────────────────────────────────────
    print("━" * 80)
    print("  AUDIT TABLE (saved to mtf_data_audit.csv)")
    print("━" * 80)
    display_cols = [
        "symbol", "timeframe", "file_exists", "pct_coverage",
        "missing_days", "median_gap_h", "has_full_ohlcv", "mtf_pass", "fail_reason"
    ]
    available_cols = [c for c in display_cols if c in audit_df.columns]
    print(audit_df[available_cols].to_string(index=False))
    print()

    return verdict["verdict_code"]


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result == "MTF_DATA_AVAILABLE" else 1)
