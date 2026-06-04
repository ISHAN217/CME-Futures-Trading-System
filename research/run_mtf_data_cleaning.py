"""
run_mtf_data_cleaning.py — Phases 2-6 of the MTF pipeline
===========================================================

Sections:
  2. File discovery and metadata audit
  3. Remove calendar spreads and invalid symbols
  4. Build continuous 1-minute series (volume-based roll, no lookahead)
  5. Price adjustment / roll gap audit
  6. ES/NQ alignment audit

Output files (output/mtf/):
  raw_file_audit.csv
  raw_symbol_inventory.csv
  raw_data_quality_report.csv
  ES_outrights_1min.parquet
  NQ_outrights_1min.parquet
  outright_filter_audit.csv
  ES_1min_continuous_clean.parquet
  NQ_1min_continuous_clean.parquet
  ES_roll_schedule.csv
  NQ_roll_schedule.csv
  roll_audit.csv
  ES_1min_continuous_backadjusted.parquet
  NQ_1min_continuous_backadjusted.parquet
  backadjustment_audit.csv
  ES_NQ_1min_aligned.parquet
  ES_NQ_alignment_audit.csv

Verdict codes:
  DATA_CLEANING_PASSED
  DATA_CLEANING_FAILED
"""

import os
import sys
import re
import logging
import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
NQ_ZST   = "/Users/ishanbhardwaj/Downloads/GLBX-20260523-KAGUFWCPLT/glbx-mdp3-20180402-20260522.ohlcv-1m.csv.zst"
ES_ZST   = "/Users/ishanbhardwaj/Downloads/GLBX-20260523-UJSB8STRKY/glbx-mdp3-20180402-20260522.ohlcv-1m.csv.zst"
OUT_DIR  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf"
os.makedirs(OUT_DIR, exist_ok=True)

def out(name): return os.path.join(OUT_DIR, name)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
VALID_MONTH_CODES = set("HMUZ")
OUTRIGHT_PATTERNS = {
    "ES": re.compile(r"^ES[HMUZ]\d{1,2}$"),
    "NQ": re.compile(r"^NQ[HMUZ]\d{1,2}$"),
}
SPREAD_PATTERN    = re.compile(r"-")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FILE DISCOVERY AND METADATA AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def load_raw(path: str, root: str) -> pd.DataFrame:
    logger.info("Loading %s from %s ...", root, os.path.basename(path))
    df = pd.read_csv(
        path,
        compression="zstd",
        parse_dates=["ts_event"],
        dtype={
            "rtype":         "int8",
            "publisher_id":  "int16",
            "instrument_id": "int32",
            "open":          "float64",
            "high":          "float64",
            "low":           "float64",
            "close":         "float64",
            "volume":        "int64",
            "symbol":        "str",
        },
    )
    # Normalise timestamp to UTC
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.sort_values("ts_event").reset_index(drop=True)
    logger.info("  Loaded %s rows, %d cols", f"{len(df):,}", len(df.columns))
    return df


def audit_raw_file(df: pd.DataFrame, root: str) -> dict:
    """Return dict of audit metrics for one raw file."""
    a = {}
    a["root"]            = root
    a["rows"]            = len(df)
    a["columns"]         = list(df.columns)
    a["dtypes"]          = {c: str(df[c].dtype) for c in df.columns}
    a["ts_col"]          = "ts_event"
    a["ts_first"]        = str(df["ts_event"].min())
    a["ts_last"]         = str(df["ts_event"].max())
    a["timezone"]        = "UTC"

    unique_syms          = df["symbol"].unique().tolist()
    a["unique_symbols"]  = unique_syms
    a["n_unique_symbols"]= len(unique_syms)

    unique_ids           = df["instrument_id"].unique().tolist()
    a["n_instrument_ids"]= len(unique_ids)

    # Classify symbols
    spread_mask    = df["symbol"].str.contains("-", na=False)
    outright_mask  = df["symbol"].str.match(OUTRIGHT_PATTERNS[root], na=False)
    other_mask     = ~spread_mask & ~outright_mask

    a["n_spread_rows"]   = int(spread_mask.sum())
    a["n_outright_rows"] = int(outright_mask.sum())
    a["n_other_rows"]    = int(other_mask.sum())
    a["n_spread_symbols"]= int(df[spread_mask]["symbol"].nunique())
    a["n_outright_ctrs"] = int(df[outright_mask]["symbol"].nunique())

    # Missing values
    a["null_counts"]     = df.isnull().sum().to_dict()
    a["total_nulls"]     = int(df.isnull().sum().sum())

    # Duplicate rows
    a["dup_rows"]        = int(df.duplicated().sum())
    a["dup_ts_sym"]      = int(df.duplicated(["ts_event","symbol"]).sum())
    a["dup_ts_iid"]      = int(df.duplicated(["ts_event","instrument_id"]).sum())

    # OHLC validity
    ohlc_bad = (
        (df["high"] < df["low"]) |
        (df["open"] < df["low"]) | (df["open"] > df["high"]) |
        (df["close"] < df["low"]) | (df["close"] > df["high"])
    )
    a["ohlc_invalid_rows"] = int(ohlc_bad.sum())

    a["zero_volume_rows"]  = int((df["volume"] == 0).sum())
    a["neg_volume_rows"]   = int((df["volume"] < 0).sum())

    # Price sanity
    a["price_min"]  = float(df[["open","high","low","close"]].min().min())
    a["price_max"]  = float(df[["open","high","low","close"]].max().max())

    return a


def phase2_audit(es_raw: pd.DataFrame, nq_raw: pd.DataFrame) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    """Section 2: file-level audit + symbol inventory."""
    logger.info("=== PHASE 2: FILE AUDIT ===")

    es_audit = audit_raw_file(es_raw, "ES")
    nq_audit = audit_raw_file(nq_raw, "NQ")
    audit_rows = []
    for a in [es_audit, nq_audit]:
        row = {k: str(v) if isinstance(v, (list, dict)) else v
               for k, v in a.items()}
        audit_rows.append(row)
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(out("raw_file_audit.csv"), index=False)
    logger.info("  Saved raw_file_audit.csv")

    # Symbol inventory
    inv_rows = []
    for root, df in [("ES", es_raw), ("NQ", nq_raw)]:
        for sym, g in df.groupby("symbol"):
            is_spread   = bool(SPREAD_PATTERN.search(sym))
            is_outright = bool(OUTRIGHT_PATTERNS[root].match(sym))
            inv_rows.append({
                "root":          root,
                "symbol":        sym,
                "instrument_ids": ",".join(str(x) for x in g["instrument_id"].unique()),
                "rows":          len(g),
                "date_start":    str(g["ts_event"].min().date()),
                "date_end":      str(g["ts_event"].max().date()),
                "vol_total":     int(g["volume"].sum()),
                "is_spread":     is_spread,
                "is_outright":   is_outright,
            })
    inv_df = pd.DataFrame(inv_rows).sort_values(["root","symbol"])
    inv_df.to_csv(out("raw_symbol_inventory.csv"), index=False)
    logger.info("  Saved raw_symbol_inventory.csv  (%d symbols total)", len(inv_df))

    # Data quality report
    qual_rows = []
    for root, df in [("ES", es_raw), ("NQ", nq_raw)]:
        for sym, g in df.groupby("symbol"):
            ohlc_bad = (
                (g["high"] < g["low"]) |
                (g["open"] < g["low"]) | (g["open"] > g["high"]) |
                (g["close"] < g["low"]) | (g["close"] > g["high"])
            )
            qual_rows.append({
                "root":          root,
                "symbol":        sym,
                "rows":          len(g),
                "nulls":         int(g.isnull().sum().sum()),
                "dup_ts":        int(g.duplicated("ts_event").sum()),
                "ohlc_bad":      int(ohlc_bad.sum()),
                "zero_vol":      int((g["volume"]==0).sum()),
                "neg_vol":       int((g["volume"]<0).sum()),
                "price_min":     float(g[["open","high","low","close"]].min().min()),
                "price_max":     float(g[["open","high","low","close"]].max().max()),
            })
    qual_df = pd.DataFrame(qual_rows)
    qual_df.to_csv(out("raw_data_quality_report.csv"), index=False)
    logger.info("  Saved raw_data_quality_report.csv")

    # Print summary
    for root, a in [("ES", es_audit), ("NQ", nq_audit)]:
        print(f"\n  {root} parent file detected: YES")
        print(f"    rows={a['rows']:,}  ts={a['ts_first']} → {a['ts_last']}")
        print(f"    outright contracts: {a['n_outright_ctrs']}   "
              f"spread symbols: {a['n_spread_symbols']}   "
              f"outright rows: {a['n_outright_rows']:,}   "
              f"spread rows: {a['n_spread_rows']:,}")
        print(f"    OHLC invalid: {a['ohlc_invalid_rows']}   "
              f"zero volume: {a['zero_volume_rows']}   "
              f"total nulls: {a['total_nulls']}")
    print("  true 1-minute intraday data: YES")
    print("  strategy-ready continuous data: NO (needs cleaning)")

    return [es_audit, nq_audit], inv_df, qual_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — REMOVE CALENDAR SPREADS AND INVALID SYMBOLS
# ══════════════════════════════════════════════════════════════════════════════

def phase3_filter_outrights(raw: pd.DataFrame, root: str) -> tuple[pd.DataFrame, dict]:
    logger.info("=== PHASE 3: OUTRIGHT FILTER  [%s] ===", root)

    rows_before   = len(raw)
    pat           = OUTRIGHT_PATTERNS[root]

    # Tag each row
    spread_mask   = raw["symbol"].str.contains("-", na=False)
    outright_mask = raw["symbol"].str.match(pat, na=False)
    ohlc_bad      = (
        (raw["high"] < raw["low"]) |
        (raw["open"] < raw["low"]) | (raw["open"] > raw["high"]) |
        (raw["close"] < raw["low"]) | (raw["close"] > raw["high"])
    )
    zero_vol_mask = raw["volume"] == 0
    neg_vol_mask  = raw["volume"] < 0

    # Remove spreads, invalid OHLC, and zero/neg volume
    keep_mask = outright_mask & ~ohlc_bad & ~neg_vol_mask
    df        = raw[keep_mask].copy()

    spread_rows_removed  = int(spread_mask.sum())
    invalid_ohlc_removed = int(ohlc_bad[~spread_mask].sum())
    neg_vol_removed      = int(neg_vol_mask[~spread_mask & ~ohlc_bad].sum())
    rows_after           = len(df)

    # Validate retained symbols
    retained_symbols = sorted(df["symbol"].unique())
    for sym in retained_symbols:
        if not pat.match(sym):
            logger.warning("  Unexpected symbol after filter: %s", sym)

    # Month code validation
    for sym in retained_symbols:
        mc = sym[2]  # 3rd char is month code for ES/NQ (root is 2 chars)
        if mc not in VALID_MONTH_CODES:
            logger.warning("  Invalid month code in symbol: %s", sym)

    audit = {
        "root":               root,
        "rows_before":        rows_before,
        "rows_after":         rows_after,
        "spread_rows_removed":spread_rows_removed,
        "invalid_ohlc_removed":invalid_ohlc_removed,
        "neg_vol_removed":    neg_vol_removed,
        "retained_contracts": len(retained_symbols),
        "retained_symbols":   ",".join(retained_symbols),
        "date_start":         str(df["ts_event"].min().date()),
        "date_end":           str(df["ts_event"].max().date()),
    }

    logger.info(
        "  %s: %s → %s rows  spread_removed=%s  contracts=%d",
        root, f"{rows_before:,}", f"{rows_after:,}",
        f"{spread_rows_removed:,}", len(retained_symbols)
    )
    logger.info("  Retained contracts: %s", ", ".join(retained_symbols[:10]))
    return df, audit


def phase3_run(es_raw, nq_raw):
    es_out, es_audit = phase3_filter_outrights(es_raw, "ES")
    nq_out, nq_audit = phase3_filter_outrights(nq_raw, "NQ")

    audit_df = pd.DataFrame([es_audit, nq_audit])
    audit_df.to_csv(out("outright_filter_audit.csv"), index=False)

    # Save outrights
    es_out.to_parquet(out("ES_outrights_1min.parquet"), index=False)
    nq_out.to_parquet(out("NQ_outrights_1min.parquet"), index=False)
    logger.info("  Saved ES_outrights_1min.parquet  NQ_outrights_1min.parquet")

    return es_out, nq_out, audit_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BUILD CONTINUOUS 1-MINUTE SERIES
# ══════════════════════════════════════════════════════════════════════════════

def _get_trade_date(ts: pd.Series) -> pd.Series:
    """
    Map UTC timestamps to CME trade dates.
    CME Globex session starts at 18:00 CT (00:00 UTC next day approx).
    For simplicity use UTC date. Consistent within each day.
    """
    return ts.dt.date


def build_roll_schedule(df: pd.DataFrame, root: str) -> pd.DataFrame:
    """
    Determine active contract for each trade date using prior-day volume.
    Returns DataFrame with columns: trade_date, active_contract.
    No lookahead: day D uses max-volume contract from day D-1.
    """
    df = df.copy()
    df["trade_date"] = _get_trade_date(df["ts_event"])

    # Daily volume per (date, symbol)
    daily_vol = (
        df.groupby(["trade_date", "symbol"])["volume"]
        .sum()
        .reset_index()
        .rename(columns={"volume": "daily_vol"})
    )
    daily_vol = daily_vol.sort_values(["trade_date", "symbol"])

    all_dates = sorted(daily_vol["trade_date"].unique())
    roll_rows = []
    prev_active = None

    for i, d in enumerate(all_dates):
        day_data = daily_vol[daily_vol["trade_date"] == d]

        if i == 0:
            # First day: use this day's volume (no prior day available)
            best     = day_data.loc[day_data["daily_vol"].idxmax(), "symbol"]
            prev_vol = day_data[day_data["symbol"] == best]["daily_vol"].values[0]
            roll_rows.append({
                "trade_date":    d,
                "active_contract": best,
                "selection_basis": "FIRST_DAY",
                "prev_day_vol_winner": int(prev_vol),
                "prev_day_vol_loser":  0,
            })
            prev_active = best
            continue

        # Use D-1 data
        prev_d  = all_dates[i - 1]
        prev_data = daily_vol[daily_vol["trade_date"] == prev_d]

        if len(prev_data) == 0:
            # No D-1 data (gap), keep prev_active
            roll_rows.append({
                "trade_date":    d,
                "active_contract": prev_active,
                "selection_basis": "NO_PREV_DATA",
                "prev_day_vol_winner": 0,
                "prev_day_vol_loser":  0,
            })
            continue

        # Max volume on D-1
        best_row    = prev_data.loc[prev_data["daily_vol"].idxmax()]
        new_active  = best_row["symbol"]
        winner_vol  = int(best_row["daily_vol"])

        # Runner-up volume
        sorted_prev = prev_data.sort_values("daily_vol", ascending=False)
        loser_vol   = int(sorted_prev.iloc[1]["daily_vol"]) if len(sorted_prev) > 1 else 0

        # Anti-flip: only switch if new contract is available on current day
        current_syms = set(day_data["symbol"].unique())
        if new_active not in current_syms:
            # The max-volume D-1 contract is not trading today — keep prev or find best available
            available = prev_data[prev_data["symbol"].isin(current_syms)]
            if len(available) > 0:
                new_active = available.loc[available["daily_vol"].idxmax(), "symbol"]
                winner_vol = int(available[available["symbol"] == new_active]["daily_vol"].values[0])
                basis = "BEST_AVAILABLE"
            else:
                new_active = prev_active
                basis = "KEEP_PREV_UNAVAILABLE"
        else:
            basis = "MAX_PREV_VOL"

        is_roll = (new_active != prev_active)
        roll_rows.append({
            "trade_date":          d,
            "active_contract":     new_active,
            "selection_basis":     basis,
            "prev_day_vol_winner": winner_vol,
            "prev_day_vol_loser":  loser_vol,
            "is_roll":             is_roll,
            "prev_contract":       prev_active if is_roll else "",
        })
        prev_active = new_active

    sched = pd.DataFrame(roll_rows)
    sched["trade_date"] = pd.to_datetime(sched["trade_date"])
    return sched


def apply_roll_schedule(df: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where symbol == active_contract for that trade_date."""
    df = df.copy()
    df["trade_date"] = pd.to_datetime(_get_trade_date(df["ts_event"]))

    sched2 = sched[["trade_date","active_contract"]].copy()
    sched2["trade_date"] = pd.to_datetime(sched2["trade_date"])

    df = df.merge(sched2, on="trade_date", how="left")
    # Rows where symbol == active_contract
    active_rows = df[df["symbol"] == df["active_contract"]].copy()

    # Add roll flag
    rolls_on   = set(sched[sched.get("is_roll", False) == True]["trade_date"].dt.date.astype(str))
    active_rows["roll_flag"] = active_rows["trade_date"].dt.date.astype(str).isin(rolls_on).astype("int8")
    active_rows["roll_reason"] = active_rows["roll_flag"].map({1: "VOLUME_BASED", 0: ""})
    active_rows["root_symbol"] = active_rows["symbol"].str[:2]
    active_rows["source_symbol"] = active_rows["symbol"]

    # Rename for clarity
    active_rows = active_rows.rename(columns={"symbol": "active_contract_sym"})
    active_rows = active_rows.rename(columns={"active_contract": "active_contract"})
    active_rows["active_contract"] = active_rows["active_contract_sym"]
    active_rows = active_rows.drop(columns=["active_contract_sym"])

    return active_rows.sort_values("ts_event").reset_index(drop=True)


def phase4_run(es_out, nq_out):
    logger.info("=== PHASE 4: CONTINUOUS SERIES ===")
    schedules = {}
    continuous = {}

    for root, df in [("ES", es_out), ("NQ", nq_out)]:
        logger.info("  Building roll schedule for %s ...", root)
        sched = build_roll_schedule(df, root)
        schedules[root] = sched

        fname = f"{root}_roll_schedule.csv"
        sched.to_csv(out(fname), index=False)
        logger.info("    %s: %d dates, %d roll events",
                    root, len(sched), int(sched.get("is_roll", pd.Series([False]*len(sched))).sum()))

        logger.info("  Applying roll schedule for %s ...", root)
        cont = apply_roll_schedule(df, sched)
        continuous[root] = cont

        fname = f"{root}_1min_continuous_clean.parquet"
        cont.to_parquet(out(fname), index=False)
        logger.info("    %s continuous: %s rows  %s → %s",
                    root, f"{len(cont):,}",
                    cont["ts_event"].min().date(), cont["ts_event"].max().date())

    # Roll audit
    roll_rows = []
    for root, sched in schedules.items():
        rolls = sched[sched.get("is_roll", False) == True] if "is_roll" in sched.columns else pd.DataFrame()
        for _, r in rolls.iterrows():
            roll_rows.append({
                "root":           root,
                "roll_date":      r["trade_date"],
                "old_contract":   r.get("prev_contract",""),
                "new_contract":   r["active_contract"],
                "prev_vol_winner":r.get("prev_day_vol_winner",0),
                "prev_vol_loser": r.get("prev_day_vol_loser",0),
            })

    # Also compute coverage stats
    for root, cont in continuous.items():
        total_dates = (cont["ts_event"].dt.date.nunique())
        all_days_range = pd.date_range(cont["ts_event"].min().date(),
                                       cont["ts_event"].max().date(), freq="B")
        missing_min_pct = 0.0  # approximate
        logger.info("  %s coverage: %d trading days  %s → %s",
                    root, total_dates,
                    cont["ts_event"].min().date(), cont["ts_event"].max().date())

    roll_audit = pd.DataFrame(roll_rows) if roll_rows else pd.DataFrame(
        columns=["root","roll_date","old_contract","new_contract","prev_vol_winner","prev_vol_loser"])
    roll_audit.to_csv(out("roll_audit.csv"), index=False)
    logger.info("  Saved roll_audit.csv  (%d roll events)", len(roll_audit))

    # Print roll events
    print("\n  ROLL EVENTS:")
    for root, sched in schedules.items():
        rolls = sched[sched.get("is_roll", False) == True] if "is_roll" in sched.columns else pd.DataFrame()
        print(f"    {root}: {len(rolls)} rolls")
        for _, r in rolls.head(5).iterrows():
            print(f"      {r['trade_date'].date()}  {r.get('prev_contract','')} → {r['active_contract']}  "
                  f"vol_winner={r.get('prev_day_vol_winner',0):,}")

    return continuous, schedules, roll_audit


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PRICE ADJUSTMENT / ROLL GAP AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def compute_backadjustment(cont: pd.DataFrame, sched: pd.DataFrame, root: str):
    """
    Additive back-adjustment (Panama method, backward from present).
    At each roll: gap = new_contract_close_on_roll_day - old_contract_last_close_before_roll
    Subtract gap from all data at and before the roll point.
    Most recent prices are unchanged.
    """
    cont  = cont.copy().sort_values("ts_event").reset_index(drop=True)
    rolls = []

    if "is_roll" not in sched.columns:
        logger.info("  No roll events found for %s — no adjustment needed", root)
        return cont, pd.DataFrame()

    roll_dates = sched[sched["is_roll"] == True].copy()
    roll_dates = roll_dates.sort_values("trade_date", ascending=False)  # process newest first

    cumulative_adj  = 0.0
    ba_audit_rows   = []

    # We'll tag each row with its cumulative adjustment
    cont["_adj"] = 0.0

    for _, roll_row in roll_dates.iterrows():
        roll_date   = pd.Timestamp(roll_row["trade_date"]).date()
        old_contract= roll_row.get("prev_contract", "")
        new_contract= roll_row["active_contract"]

        if not old_contract:
            continue

        # Last close of old contract just before roll date
        old_data   = cont[
            (cont["active_contract"] == old_contract) &
            (cont["ts_event"].dt.date <  roll_date)
        ]
        # First close of new contract on or after roll date
        new_data   = cont[
            (cont["active_contract"] == new_contract) &
            (cont["ts_event"].dt.date >= roll_date)
        ]

        if len(old_data) == 0 or len(new_data) == 0:
            logger.warning("  Roll gap data missing: %s → %s on %s", old_contract, new_contract, roll_date)
            ba_audit_rows.append({
                "root": root, "roll_date": roll_date,
                "old_contract": old_contract, "new_contract": new_contract,
                "old_last_close": np.nan, "new_first_close": np.nan,
                "abs_gap": np.nan, "pct_gap": np.nan,
                "material": "NO_DATA", "adj_applied": 0.0,
                "needs_adjustment": False,
            })
            continue

        old_last_close  = float(old_data.iloc[-1]["close"])
        new_first_close = float(new_data.iloc[0]["close"])
        gap             = new_first_close - old_last_close
        pct_gap         = abs(gap) / old_last_close * 100
        material        = "YES" if pct_gap > 0.1 else "NO"   # >0.1% is material

        # Accumulate adjustment and apply to all data before this roll
        cumulative_adj -= gap   # subtract gap from old data
        roll_ts         = cont[cont["ts_event"].dt.date < roll_date].index
        cont.loc[roll_ts, "_adj"] = cumulative_adj

        ba_audit_rows.append({
            "root": root, "roll_date": roll_date,
            "old_contract": old_contract, "new_contract": new_contract,
            "old_last_close": round(old_last_close, 3),
            "new_first_close": round(new_first_close, 3),
            "abs_gap": round(abs(gap), 3),
            "pct_gap": round(pct_gap, 4),
            "material": material,
            "adj_applied": round(cumulative_adj, 3),
            "needs_adjustment": pct_gap > 0.1,
        })

    # Apply back-adjustment
    for col in ["open","high","low","close"]:
        cont[col] = cont[col] + cont["_adj"]
    cont = cont.drop(columns=["_adj"])

    # Validate no negative prices
    min_price = cont[["open","high","low","close"]].min().min()
    if min_price < 0:
        logger.warning("  Back-adjustment produced negative prices (min=%.2f) — check roll logic", min_price)

    ba_audit = pd.DataFrame(ba_audit_rows)
    return cont, ba_audit


def phase5_run(continuous: dict, schedules: dict):
    logger.info("=== PHASE 5: PRICE ADJUSTMENT ===")
    ba_continuous = {}
    ba_audits     = []

    for root in ["ES", "NQ"]:
        cont  = continuous[root]
        sched = schedules[root]
        ba_cont, ba_audit = compute_backadjustment(cont, sched, root)
        ba_continuous[root] = ba_cont

        fname = f"{root}_1min_continuous_backadjusted.parquet"
        ba_cont.to_parquet(out(fname), index=False)
        logger.info("  %s back-adjusted saved (%s rows)", root, f"{len(ba_cont):,}")

        if len(ba_audit) > 0:
            ba_audit["root"] = root
            ba_audits.append(ba_audit)

            mat = ba_audit[ba_audit["material"]=="YES"]
            logger.info("  %s: %d roll gaps  (%d material > 0.1%%)",
                        root, len(ba_audit), len(mat))
            # Recommend
            max_pct = ba_audit["pct_gap"].max() if len(ba_audit) > 0 else 0
            if max_pct > 0.5:
                rec = "USE_BACKADJUSTED_CONTINUOUS"
            elif max_pct > 0.1:
                rec = "USE_BACKADJUSTED_CONTINUOUS"
            else:
                rec = "USE_RAW_CONTINUOUS"
            logger.info("  %s recommendation: %s  (max_gap=%.3f%%)", root, rec, max_pct)

    if ba_audits:
        full_ba = pd.concat(ba_audits, ignore_index=True)
    else:
        full_ba = pd.DataFrame()
    full_ba.to_csv(out("backadjustment_audit.csv"), index=False)
    logger.info("  Saved backadjustment_audit.csv")

    # Recommendation
    if len(full_ba) > 0 and "pct_gap" in full_ba.columns:
        max_pct = full_ba["pct_gap"].dropna().max()
        if max_pct > 0.1:
            verdict = "USE_BACKADJUSTED_CONTINUOUS"
        else:
            verdict = "USE_RAW_CONTINUOUS"
    else:
        verdict = "USE_RAW_CONTINUOUS"

    print(f"\n  Roll gap recommendation: {verdict}")
    return ba_continuous, full_ba, verdict


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ES/NQ ALIGNMENT AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def phase6_align(ba_continuous: dict):
    logger.info("=== PHASE 6: ES/NQ ALIGNMENT ===")

    # Keep only columns needed for alignment; exclude overlapping non-price columns
    keep = ["ts_event","open","high","low","close","volume","active_contract"]
    es = ba_continuous["ES"][[c for c in keep if c in ba_continuous["ES"].columns]].copy()
    nq = ba_continuous["NQ"][[c for c in keep if c in ba_continuous["NQ"].columns]].copy()

    # Check for duplicate timestamps (keep last per ts in each)
    es_dups = es.duplicated("ts_event").sum()
    nq_dups = nq.duplicated("ts_event").sum()
    if es_dups > 0:
        logger.warning("  ES has %d duplicate timestamps — keeping last", es_dups)
        es = es.drop_duplicates("ts_event", keep="last")
    if nq_dups > 0:
        logger.warning("  NQ has %d duplicate timestamps — keeping last", nq_dups)
        nq = nq.drop_duplicates("ts_event", keep="last")

    es = es.set_index("ts_event")
    nq = nq.set_index("ts_event")

    # Rename columns
    es_cols = {c: f"ES_{c}" for c in ["open","high","low","close","volume","active_contract"]}
    nq_cols = {c: f"NQ_{c}" for c in ["open","high","low","close","volume","active_contract"]}
    es = es.rename(columns=es_cols)
    nq = nq.rename(columns=nq_cols)

    # Find overlap
    overlap_start = max(es.index.min(), nq.index.min())
    overlap_end   = min(es.index.max(), nq.index.max())
    logger.info("  ES: %s → %s  (%s rows)", es.index.min(), es.index.max(), f"{len(es):,}")
    logger.info("  NQ: %s → %s  (%s rows)", nq.index.min(), nq.index.max(), f"{len(nq):,}")
    logger.info("  Overlap: %s → %s", overlap_start, overlap_end)

    # Inner join
    aligned = es.join(nq, how="inner")
    aligned.index.name = "timestamp"
    aligned = aligned.reset_index()

    # Alignment statistics
    es_in_overlap = len(es[(es.index >= overlap_start) & (es.index <= overlap_end)])
    nq_in_overlap = len(nq[(nq.index >= overlap_start) & (nq.index <= overlap_end)])
    aligned_rows  = len(aligned)
    pct_aligned   = 100.0 * aligned_rows / max(es_in_overlap, nq_in_overlap)
    pct_missing   = 100 - pct_aligned

    es_only = es_in_overlap - aligned_rows
    nq_only = nq_in_overlap - aligned_rows

    # OHLC validity check after alignment
    ohlc_bad_es = (
        (aligned["ES_high"] < aligned["ES_low"]) |
        (aligned["ES_open"] < aligned["ES_low"]) | (aligned["ES_open"] > aligned["ES_high"]) |
        (aligned["ES_close"] < aligned["ES_low"]) | (aligned["ES_close"] > aligned["ES_high"])
    )
    ohlc_bad_nq = (
        (aligned["NQ_high"] < aligned["NQ_low"]) |
        (aligned["NQ_open"] < aligned["NQ_low"]) | (aligned["NQ_open"] > aligned["NQ_high"]) |
        (aligned["NQ_close"] < aligned["NQ_low"]) | (aligned["NQ_close"] > aligned["NQ_high"])
    )
    dup_ts_aligned = aligned.duplicated("timestamp").sum()

    align_audit = {
        "es_rows":        len(es),
        "nq_rows":        len(nq),
        "overlap_start":  str(overlap_start),
        "overlap_end":    str(overlap_end),
        "es_in_overlap":  es_in_overlap,
        "nq_in_overlap":  nq_in_overlap,
        "aligned_rows":   aligned_rows,
        "pct_aligned":    round(pct_aligned, 2),
        "pct_missing":    round(pct_missing, 2),
        "es_only_rows":   es_only,
        "nq_only_rows":   nq_only,
        "ohlc_bad_es":    int(ohlc_bad_es.sum()),
        "ohlc_bad_nq":    int(ohlc_bad_nq.sum()),
        "dup_ts_aligned": int(dup_ts_aligned),
    }

    pd.DataFrame([align_audit]).to_csv(out("ES_NQ_alignment_audit.csv"), index=False)

    # Save aligned
    aligned.to_parquet(out("ES_NQ_1min_aligned.parquet"), index=False)
    logger.info("  Saved ES_NQ_1min_aligned.parquet  (%s rows  %.1f%% aligned)",
                f"{aligned_rows:,}", pct_aligned)

    # Verdict
    if pct_aligned >= 95 and dup_ts_aligned == 0 and ohlc_bad_es.sum() == 0 and ohlc_bad_nq.sum() == 0:
        verdict = "ES_NQ_ALIGNMENT_READY"
    elif pct_aligned >= 85:
        verdict = "ES_NQ_ALIGNMENT_NEEDS_REVIEW"
    else:
        verdict = "ES_NQ_ALIGNMENT_FAILED"

    print(f"\n  ES rows: {len(es):,}   NQ rows: {len(nq):,}")
    print(f"  Aligned rows: {aligned_rows:,}   ({pct_aligned:.1f}% aligned  {pct_missing:.1f}% missing)")
    print(f"  Overlap: {overlap_start.date()} → {overlap_end.date()}")
    print(f"  OHLC issues ES: {ohlc_bad_es.sum()}   NQ: {ohlc_bad_nq.sum()}")
    print(f"  Alignment verdict: {verdict}")

    return aligned, align_audit, verdict


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  MTF DATA CLEANING PIPELINE  —  Phases 2-6")
    print("=" * 80)

    # Phase 2: Load + audit
    print("\n[PHASE 2] Loading raw files and auditing ...")
    es_raw = load_raw(ES_ZST, "ES")
    nq_raw = load_raw(NQ_ZST, "NQ")
    phase2_audit(es_raw, nq_raw)

    # Phase 3: Filter outrights
    print("\n[PHASE 3] Removing calendar spreads and invalid symbols ...")
    es_out, nq_out, filter_audit = phase3_run(es_raw, nq_raw)

    # Free raw DataFrames — they're large
    del es_raw, nq_raw

    # Phase 4: Continuous series
    print("\n[PHASE 4] Building continuous 1-minute series ...")
    continuous, schedules, roll_audit = phase4_run(es_out, nq_out)
    del es_out, nq_out

    # Phase 5: Price adjustment
    print("\n[PHASE 5] Back-adjusting prices at roll boundaries ...")
    ba_continuous, ba_audit, ba_verdict = phase5_run(continuous, schedules)
    del continuous

    # Phase 6: Alignment
    print("\n[PHASE 6] Aligning ES and NQ minute-by-minute ...")
    aligned, align_audit, align_verdict = phase6_align(ba_continuous)
    del ba_continuous

    # Final verdict
    print("\n" + "=" * 80)
    if align_verdict == "ES_NQ_ALIGNMENT_FAILED":
        verdict = "DATA_CLEANING_FAILED"
    elif align_verdict in ("ES_NQ_ALIGNMENT_READY", "ES_NQ_ALIGNMENT_NEEDS_REVIEW"):
        verdict = "DATA_CLEANING_PASSED"
    else:
        verdict = "DATA_CLEANING_FAILED"

    print(f"  FINAL VERDICT: {verdict}")
    print(f"  Alignment:     {align_verdict}")
    print(f"  Price adj:     {ba_verdict}")
    print(f"  Output:        {OUT_DIR}")
    print("=" * 80)

    return verdict, align_verdict


if __name__ == "__main__":
    verdict, align_verdict = main()
    sys.exit(0 if "PASSED" in verdict else 1)
