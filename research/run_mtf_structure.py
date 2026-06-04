"""
run_mtf_structure.py — Phase 8: Compute MTF structure features from resampled bars
====================================================================================

Input:  ES_1H/4H/12H/24H.parquet  NQ_1H/4H/12H/24H.parquet  (from Phase 7)
        ES_daily.csv  NQ_daily.csv  (existing daily data)

Output:
  ES_MTF_features.parquet
  NQ_MTF_features.parquet
  ES_NQ_MTF_features_aligned.parquet
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_daily
from mtf_structure_engine import compute_mtf_structure, load_tf_bars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf"
SYMBOLS  = ["ES", "NQ"]


def main():
    print("=" * 80)
    print("  MTF STRUCTURE ENGINE  —  Phase 8")
    print("=" * 80)

    # Check that Phase 7 outputs exist
    required = [f"{sym}_{tf}.parquet" for sym in SYMBOLS for tf in ["1H","4H","12H","24H"]]
    missing  = [f for f in required if not os.path.exists(os.path.join(OUT_DIR, f))]
    if missing:
        print(f"  ERROR: Missing Phase 7 outputs: {missing}")
        print("  Run run_mtf_resample.py first.")
        sys.exit(1)

    # Load daily data for trading date index
    logger.info("Loading daily trading dates ...")
    daily_df = load_daily()   # returns all symbols combined
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    logger.info("  %s daily rows for %d symbols", f"{len(daily_df):,}", daily_df["symbol"].nunique())

    # Load TF bars
    logger.info("Loading resampled TF bars ...")
    bars_by_tf = load_tf_bars(OUT_DIR, SYMBOLS)
    for tf, df in bars_by_tf.items():
        if df is not None:
            logger.info("  %s: %s rows", tf, f"{len(df):,}")
        else:
            logger.warning("  %s: NOT LOADED", tf)

    # Compute MTF structure — this is the heavy step
    logger.info("Computing MTF structure features (this may take several minutes) ...")
    mtf_df = compute_mtf_structure(daily_df, bars_by_tf, symbols=SYMBOLS)
    logger.info("  MTF features computed: %s rows", f"{len(mtf_df):,}")

    if len(mtf_df) == 0:
        print("  ERROR: No MTF features computed. Check TF data availability.")
        sys.exit(1)

    # Save per-symbol
    for sym in SYMBOLS:
        sym_df = mtf_df[mtf_df["symbol"] == sym]
        fp     = os.path.join(OUT_DIR, f"{sym}_MTF_features.parquet")
        sym_df.to_parquet(fp, index=False)
        logger.info("  Saved %s_MTF_features.parquet  (%s rows)", sym, f"{len(sym_df):,}")

    # Save aligned (both symbols)
    mtf_df.to_parquet(os.path.join(OUT_DIR, "ES_NQ_MTF_features_aligned.parquet"), index=False)
    logger.info("  Saved ES_NQ_MTF_features_aligned.parquet")

    # Quick summary
    print("\n  MTF FEATURE SUMMARY:")
    for col in ["mtf_final_bias","structure_1h_label","structure_4h_label",
                "structure_12h_label","structure_24h_label"]:
        if col in mtf_df.columns:
            vc = mtf_df[col].value_counts()
            print(f"\n  {col}:")
            for v, c in vc.items():
                print(f"    {v:30s}  {c:6d}  ({100*c/len(mtf_df):.1f}%)")

    print("\n  PHASE 8 COMPLETE — MTF features saved.")


if __name__ == "__main__":
    main()
