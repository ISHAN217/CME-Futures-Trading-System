"""
run_mtf_pipeline.py — Full MTF pipeline orchestrator
=====================================================

Runs all phases in order, stopping on failure:

  Phase 2-6:  run_mtf_data_cleaning.py   → DATA_CLEANING_PASSED / FAILED
  Phase 7:    run_mtf_resample.py         → RESAMPLE_COMPLETE
  Phase 8:    run_mtf_structure.py        → MTF features saved
  Phase 9-10: run_mtf_diagnostics.py     → signal verdict
  Phase 11-17:run_mtf_experiments.py     → final strategy verdict

Usage:
  python3 run_mtf_pipeline.py              # run all phases
  python3 run_mtf_pipeline.py --from 7    # restart from phase 7
"""

import os
import sys
import subprocess
import argparse
import time
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "output", "mtf")

PHASES = [
    (2,  "run_mtf_data_cleaning.py",  "DATA CLEANING (Phases 2-6)"),
    (7,  "run_mtf_resample.py",        "RESAMPLE 1H/4H/12H/24H (Phase 7)"),
    (8,  "run_mtf_structure.py",       "MTF STRUCTURE ENGINE (Phase 8)"),
    (9,  "run_mtf_diagnostics.py",     "DIAGNOSTICS + INVERSE/RANDOM (Phases 9-10)"),
    (11, "run_mtf_experiments.py",     "STRATEGY LADDER (Phases 11-17)"),
]

# Files that must exist before the next phase can run
PHASE_GATES = {
    7:  [os.path.join(OUT_DIR, "ES_NQ_1min_aligned.parquet")],
    8:  [os.path.join(OUT_DIR, "ES_1H.parquet"),
         os.path.join(OUT_DIR, "NQ_24H.parquet")],
    9:  [os.path.join(OUT_DIR, "ES_MTF_features.parquet"),
         os.path.join(OUT_DIR, "NQ_MTF_features.parquet")],
    11: [],   # diagnostics always runs even if signal is weak
}


def run_phase(script: str, phase_num: int, description: str) -> bool:
    print(f"\n{'═'*80}")
    print(f"  PHASE {phase_num}:  {description}")
    print(f"{'═'*80}")

    script_path = os.path.join(SCRIPT_DIR, script)
    if not os.path.exists(script_path):
        logger.error("Script not found: %s", script_path)
        return False

    t0  = time.time()
    ret = subprocess.run([sys.executable, script_path], cwd=SCRIPT_DIR)
    elapsed = time.time() - t0

    if ret.returncode != 0:
        # Data cleaning returns 1 for DATA_CLEANING_FAILED (intentional)
        # Check gate files to determine if we truly passed
        gates = PHASE_GATES.get(phase_num + 1, [])
        if gates and all(os.path.exists(g) for g in gates):
            logger.info("  Phase %d completed in %.1fs (exit code %d but gate files exist)",
                        phase_num, elapsed, ret.returncode)
            return True
        logger.error("  Phase %d FAILED (exit code %d) in %.1fs",
                     phase_num, ret.returncode, elapsed)
        return False

    logger.info("  Phase %d completed in %.1fs", phase_num, elapsed)
    return True


def check_gate(phase_num: int) -> bool:
    """Check if required output files exist before running next phase."""
    gates = PHASE_GATES.get(phase_num, [])
    missing = [g for g in gates if not os.path.exists(g)]
    if missing:
        logger.error("  Gate check failed for phase %d — missing files:", phase_num)
        for m in missing:
            logger.error("    %s", m)
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_phase", type=int, default=2,
                        help="Start from this phase number (2=full run, 7=skip cleaning, etc.)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 80)
    print("  MTF STRATEGY PIPELINE — FULL RUN")
    print("=" * 80)
    print(f"  Starting from phase: {args.from_phase}")
    print(f"  Output directory:    {OUT_DIR}")

    t_start = time.time()

    for phase_num, script, description in PHASES:
        if phase_num < args.from_phase:
            logger.info("  Skipping phase %d (%s)", phase_num, description)
            continue

        # Check gate files from previous phase
        if not check_gate(phase_num):
            logger.error("  STOPPING: Phase %d gate check failed", phase_num)
            print(f"\n  DATA_CLEANING_FAILED — cannot proceed to phase {phase_num}")
            sys.exit(1)

        success = run_phase(script, phase_num, description)

        if not success:
            logger.error("  STOPPING: Phase %d failed", phase_num)
            print(f"\n  PIPELINE_FAILED at phase {phase_num}")
            sys.exit(1)

    elapsed = time.time() - t_start
    print(f"\n{'═'*80}")
    print(f"  PIPELINE COMPLETE  ({elapsed/60:.1f} minutes)")
    print(f"  All outputs in: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
