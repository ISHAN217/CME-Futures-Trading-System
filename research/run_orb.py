"""
run_orb.py — Opening Range Breakout (ORB) system for ES + NQ.

True intraday system:
  - Observe first 30 min of RTH (09:30–10:00 ET) → lock H_OR / L_OR
  - Trade breakout of that range: enter at the level, stop at opposite extreme
  - Target = entry ± Range × PROFIT_MULT
  - Force-close at 15:30 ET if not filled
  - Zero overnight exposure

Improvements (v2):
  Rec 1 — Daily loss circuit breaker  (stop after -1.5%/day)
  Rec 2 — ES+NQ alignment filter      (both must break same direction)
  Rec 3 — Skip news days              (FOMC + NFP + CPI)

Data: ES/NQ 1-min continuous back-adjusted (Databento, Databento)
      ES_NQ_1min_aligned.parquet — both symbols pre-aligned by timestamp
      2018-04-02 → 2026-05-22  (~8.1 years)

Experiments:
  E1  ORB_STRICT_BASE   — strict range filter only (best from v1)
  E2  +CIRCUIT          — add daily loss circuit breaker
  E3  +ALIGN            — add ES+NQ alignment filter
  E4  +NEWS_SKIP        — add news-day skip
  E5  +ALL_THREE        — all improvements combined

Benchmark: DAILY_BASELINE Sh≈1.40, Ann≈4.8%, MaxDD≈8%
"""

import logging
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

# ─── Constants ────────────────────────────────────────────────────────────────
ACCOUNT_START = 25_000.0
ES_MULT       = 50.0
NQ_MULT       = 20.0
RISK_PCT      = 0.01

OR_MINS       = 30
PROFIT_MULT   = 1.5
CUTOFF_H      = 15
CUTOFF_M      = 30
MIN_RANGE_ES  = 8.0     # strict filter from v1 best experiment
MIN_RANGE_NQ  = 30.0

DATA_DIR = "/Users/ishanbhardwaj/cme_competition_system/output/mtf"


# ─── News dates ───────────────────────────────────────────────────────────────

def _build_news_dates() -> set:
    """
    Returns a set of date strings (YYYY-MM-DD) to skip:
      - FOMC decision days (known dates 2018-2026)
      - NFP release days  (first Friday of each month, programmatic)
      - CPI release days  (BLS schedule, hardcoded)
    """
    fomc = {
        # 2018
        '2018-01-31','2018-03-21','2018-05-02','2018-06-13',
        '2018-08-01','2018-09-26','2018-11-08','2018-12-19',
        # 2019
        '2019-01-30','2019-03-20','2019-05-01','2019-06-19',
        '2019-07-31','2019-09-18','2019-10-30','2019-12-11',
        # 2020
        '2020-01-29','2020-03-03','2020-03-15','2020-04-29',
        '2020-06-10','2020-07-29','2020-09-16','2020-11-05','2020-12-16',
        # 2021
        '2021-01-27','2021-03-17','2021-04-28','2021-06-16',
        '2021-07-28','2021-09-22','2021-11-03','2021-12-15',
        # 2022
        '2022-01-26','2022-03-16','2022-05-04','2022-06-15',
        '2022-07-27','2022-09-21','2022-11-02','2022-12-14',
        # 2023
        '2023-02-01','2023-03-22','2023-05-03','2023-06-14',
        '2023-07-26','2023-09-20','2023-11-01','2023-12-13',
        # 2024
        '2024-01-31','2024-03-20','2024-05-01','2024-06-12',
        '2024-07-31','2024-09-18','2024-11-07','2024-12-18',
        # 2025
        '2025-01-29','2025-03-19','2025-05-07','2025-06-18',
        '2025-07-30','2025-09-17','2025-10-29','2025-12-10',
        # 2026
        '2026-01-28','2026-03-18','2026-05-06',
    }

    # NFP: first Friday of each month
    nfp = set()
    for year in range(2018, 2027):
        for month in range(1, 13):
            first = pd.Timestamp(year=year, month=month, day=1)
            days_to_fri = (4 - first.weekday()) % 7
            nfp.add((first + pd.Timedelta(days=days_to_fri)).strftime('%Y-%m-%d'))

    # CPI: BLS schedule (approx 2nd week of month, source: BLS release calendar)
    cpi = {
        '2018-01-12','2018-02-14','2018-03-13','2018-04-11','2018-05-10','2018-06-12',
        '2018-07-12','2018-08-10','2018-09-13','2018-10-11','2018-11-14','2018-12-12',
        '2019-01-11','2019-02-13','2019-03-12','2019-04-10','2019-05-10','2019-06-12',
        '2019-07-11','2019-08-13','2019-09-12','2019-10-10','2019-11-13','2019-12-11',
        '2020-01-14','2020-02-13','2020-03-11','2020-04-10','2020-05-12','2020-06-10',
        '2020-07-14','2020-08-12','2020-09-11','2020-10-13','2020-11-12','2020-12-10',
        '2021-01-13','2021-02-10','2021-03-10','2021-04-13','2021-05-12','2021-06-10',
        '2021-07-13','2021-08-11','2021-09-14','2021-10-13','2021-11-10','2021-12-10',
        '2022-01-12','2022-02-10','2022-03-10','2022-04-12','2022-05-11','2022-06-10',
        '2022-07-13','2022-08-10','2022-09-13','2022-10-13','2022-11-10','2022-12-13',
        '2023-01-12','2023-02-14','2023-03-14','2023-04-12','2023-05-10','2023-06-13',
        '2023-07-12','2023-08-10','2023-09-13','2023-10-12','2023-11-14','2023-12-12',
        '2024-01-11','2024-02-13','2024-03-12','2024-04-10','2024-05-15','2024-06-12',
        '2024-07-11','2024-08-14','2024-09-11','2024-10-10','2024-11-13','2024-12-11',
        '2025-01-15','2025-02-12','2025-03-12','2025-04-10','2025-05-13','2025-06-11',
        '2025-07-15','2025-08-12','2025-09-10','2025-10-14','2025-11-12','2025-12-10',
        '2026-01-14','2026-02-11','2026-03-11','2026-04-10','2026-05-13',
    }

    all_dates = fomc | nfp | cpi
    logger.info("News calendar: %d skip dates (FOMC=%d  NFP=%d  CPI=%d)",
                len(all_dates), len(fomc), len(nfp), len(cpi))
    return all_dates


# ─── Data load ────────────────────────────────────────────────────────────────

def load_aligned_data() -> pd.DataFrame:
    """
    Load pre-aligned ES+NQ 1-min parquet.
    Columns: timestamp, ES_open/high/low/close/volume, NQ_open/high/low/close/volume
    Add ET time columns for session filtering.
    """
    logger.info("Loading aligned 1-min data …")
    df = pd.read_parquet(f"{DATA_DIR}/ES_NQ_1min_aligned.parquet")
    df["ts_et"]     = df["timestamp"].dt.tz_convert("America/New_York")
    df["hour_et"]   = df["ts_et"].dt.hour
    df["min_et"]    = df["ts_et"].dt.minute
    df["trade_date"] = df["ts_et"].dt.date
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        "Loaded %d bars  %s → %s",
        len(df),
        df["ts_et"].min().date(),
        df["ts_et"].max().date(),
    )
    return df


# ─── Pre-compute alignment ────────────────────────────────────────────────────

def _precompute_alignment(
    df_rth: pd.DataFrame,
    or_end_h: int,
    or_end_m: int,
    min_range_es: float,
    min_range_nq: float,
) -> dict:
    """
    First pass (no positions, no P&L):
    For each trading day, determine which direction each symbol first broke
    out of its opening range during the trade window.

    Returns: dict  date_str → {'ES': int, 'NQ': int}
             where int in {+1, -1, 0}  (0 = never broke, or range too small)
    """
    alignment = {}
    for date, day in df_rth.groupby("trade_date", sort=True):
        day = day.sort_values("ts_et").reset_index(drop=True)
        date_str = str(date)

        # Observation window
        obs = day[(day["hour_et"] == 9) & (day["min_et"] >= 30)]
        if or_end_h == 9:
            obs = obs[obs["min_et"] < or_end_m]
        if len(obs) < 5:
            continue

        H_ES = obs["ES_high"].max(); L_ES = obs["ES_low"].min()
        H_NQ = obs["NQ_high"].max(); L_NQ = obs["NQ_low"].min()
        rng_es = H_ES - L_ES;  rng_nq = H_NQ - L_NQ

        # Trade window
        tw_mask = (
            ((day["hour_et"] > or_end_h) |
             ((day["hour_et"] == or_end_h) & (day["min_et"] >= or_end_m))) &
            ~((day["hour_et"] > CUTOFF_H) |
              ((day["hour_et"] == CUTOFF_H) & (day["min_et"] > CUTOFF_M)))
        )
        tw = day[tw_mask]

        es_dir = nq_dir = 0
        for _, bar in tw.iterrows():
            if es_dir == 0 and rng_es >= min_range_es:
                if bar["ES_high"] > H_ES:
                    es_dir = 1
                elif bar["ES_low"] < L_ES:
                    es_dir = -1
            if nq_dir == 0 and rng_nq >= min_range_nq:
                if bar["NQ_high"] > H_NQ:
                    nq_dir = 1
                elif bar["NQ_low"] < L_NQ:
                    nq_dir = -1
            if es_dir != 0 and nq_dir != 0:
                break

        alignment[date_str] = {"ES": es_dir, "NQ": nq_dir}
    return alignment


def _precompute_breakouts(
    df_rth: pd.DataFrame,
    or_end_h: int,
    or_end_m: int,
    min_range_es: float,
    min_range_nq: float,
) -> dict:
    """
    For each trading day, record the FIRST bar index where each symbol
    breaks its opening range, and in which direction.

    Returns: {date_str: {
        'ES': {'dir': int, 'bar_idx': int} | None,
        'NQ': {'dir': int, 'bar_idx': int} | None,
        'H_ES': float, 'L_ES': float, 'rng_ES': float,
        'H_NQ': float, 'L_NQ': float, 'rng_NQ': float,
    }}
    bar_idx is a 0-based index into the trade-window (tw) DataFrame.
    """
    result = {}
    for date, day in df_rth.groupby("trade_date", sort=True):
        day      = day.sort_values("ts_et").reset_index(drop=True)
        date_str = str(date)

        obs = day[(day["hour_et"] == 9) & (day["min_et"] >= 30)]
        if or_end_h == 9:
            obs = obs[obs["min_et"] < or_end_m]
        if len(obs) < 5:
            continue

        H_ES = obs["ES_high"].max();  L_ES = obs["ES_low"].min()
        H_NQ = obs["NQ_high"].max();  L_NQ = obs["NQ_low"].min()
        rng_es = H_ES - L_ES;  rng_nq = H_NQ - L_NQ

        tw_mask = (
            ((day["hour_et"] > or_end_h) |
             ((day["hour_et"] == or_end_h) & (day["min_et"] >= or_end_m))) &
            ~((day["hour_et"] > CUTOFF_H) |
              ((day["hour_et"] == CUTOFF_H) & (day["min_et"] > CUTOFF_M)))
        )
        tw = day[tw_mask].reset_index(drop=True)

        es_bk = nq_bk = None
        for idx in range(len(tw)):
            bar = tw.iloc[idx]
            if es_bk is None and rng_es >= min_range_es:
                if bar["ES_high"] > H_ES:
                    es_bk = {"dir":  1, "bar_idx": idx}
                elif bar["ES_low"] < L_ES:
                    es_bk = {"dir": -1, "bar_idx": idx}
            if nq_bk is None and rng_nq >= min_range_nq:
                if bar["NQ_high"] > H_NQ:
                    nq_bk = {"dir":  1, "bar_idx": idx}
                elif bar["NQ_low"] < L_NQ:
                    nq_bk = {"dir": -1, "bar_idx": idx}
            if es_bk and nq_bk:
                break

        result[date_str] = {
            "ES": es_bk, "NQ": nq_bk,
            "H_ES": H_ES, "L_ES": L_ES, "rng_ES": rng_es,
            "H_NQ": H_NQ, "L_NQ": L_NQ, "rng_NQ": rng_nq,
        }
    return result


# ─── Core backtest (simultaneous ES+NQ per day) ───────────────────────────────

def run_orb(
    df: pd.DataFrame,
    or_mins:          int   = OR_MINS,
    profit_mult:      float = PROFIT_MULT,
    min_range_es:     float = MIN_RANGE_ES,
    min_range_nq:     float = MIN_RANGE_NQ,
    risk_pct:         float = RISK_PCT,
    # Improvements
    daily_loss_limit: float | None = None,   # Rec 1: e.g. 0.015 → trip after -1.5%/day
    align_filter:     bool  = False,          # Rec 2: both symbols same direction
    news_dates:       set   | None = None,    # Rec 3: set of date strings to skip
) -> dict:
    """
    ORB backtest — processes ES and NQ simultaneously within each day.
    This allows the daily loss circuit breaker to share state across symbols.

    Entry: fill-at-level when bar HIGH > H_OR (long) or LOW < L_OR (short).
    Exit:  stop at opposite OR extreme, target at entry ± Range×profit_mult,
           or force-close at 15:30 ET.
    """
    # Observation window end
    obs_end_abs = 9 * 60 + 30 + or_mins
    or_end_h    = obs_end_abs // 60
    or_end_m    = obs_end_abs % 60

    # RTH filter
    in_rth = (
        ((df["hour_et"] == 9)  & (df["min_et"] >= 30)) |
        (df["hour_et"].between(10, 14)) |
        ((df["hour_et"] == 15) & (df["min_et"] <= 30))
    )
    rth = df[in_rth].copy()

    # Pre-compute alignment if needed
    alignment = {}
    if align_filter:
        logger.info("  Pre-computing ES/NQ alignment …")
        alignment = _precompute_alignment(rth, or_end_h, or_end_m, min_range_es, min_range_nq)

    skip_dates = news_dates or set()

    trade_log  = []
    daily_rets = {}
    account    = ACCOUNT_START

    for date, day in rth.groupby("trade_date", sort=True):
        date_str = str(date)

        # ── Rec 3: Skip news days ────────────────────────────────────────
        if date_str in skip_dates:
            continue

        day = day.sort_values("ts_et").reset_index(drop=True)

        # Observation window
        obs = day[(day["hour_et"] == 9) & (day["min_et"] >= 30)]
        if or_end_h == 9:
            obs = obs[obs["min_et"] < or_end_m]
        if len(obs) < 5:
            continue

        H_ES = obs["ES_high"].max();  L_ES = obs["ES_low"].min()
        H_NQ = obs["NQ_high"].max();  L_NQ = obs["NQ_low"].min()
        rng_es = H_ES - L_ES;  rng_nq = H_NQ - L_NQ

        # ── Rec 2: Alignment filter ──────────────────────────────────────
        trade_es = rng_es >= min_range_es
        trade_nq = rng_nq >= min_range_nq

        if align_filter:
            day_align = alignment.get(date_str, {"ES": 0, "NQ": 0})
            es_dir_precomp = day_align["ES"]
            nq_dir_precomp = day_align["NQ"]
            # Only trade if both broke in the same non-zero direction
            both_agree = (
                es_dir_precomp != 0 and
                nq_dir_precomp != 0 and
                es_dir_precomp == nq_dir_precomp
            )
            if not both_agree:
                continue   # misaligned day — skip entirely

        # Trade window
        tw_mask = (
            ((day["hour_et"] > or_end_h) |
             ((day["hour_et"] == or_end_h) & (day["min_et"] >= or_end_m))) &
            ~((day["hour_et"] > CUTOFF_H) |
              ((day["hour_et"] == CUTOFF_H) & (day["min_et"] > CUTOFF_M)))
        )
        tw = day[tw_mask].reset_index(drop=True)
        if len(tw) == 0:
            continue

        # Per-symbol position state
        pos     = {"ES": None, "NQ": None}
        traded  = set()     # symbols that have completed their ONE trade today
        daily_pnl = 0.0     # shared P&L across both symbols this day

        syms = []
        if trade_es:
            syms.append(("ES", H_ES, L_ES, rng_es, ES_MULT))
        if trade_nq:
            syms.append(("NQ", H_NQ, L_NQ, rng_nq, NQ_MULT))

        def _book(sym, exit_px, reason, ts):
            """Settle position, update shared account/daily_pnl."""
            nonlocal account, daily_pnl
            _settle(pos, sym, exit_px, reason, ts, trade_log, date_str, account)
            pnl_d = _last_pnl(trade_log)
            daily_pnl += pnl_d
            account   += pnl_d * account   # compound
            traded.add(sym)

        # ── Minute-by-minute scan ────────────────────────────────────────
        for i in range(len(tw)):
            bar     = tw.iloc[i]
            is_last = (i == len(tw) - 1)

            for sym, H_OR, L_OR, rng, mult in syms:
                if sym in traded:
                    continue   # already took one trade this symbol today

                p = pos[sym]

                if p is None:
                    # ── Rec 1: Circuit breaker blocks NEW entries only ─
                    if daily_loss_limit and daily_pnl <= -daily_loss_limit:
                        continue

                    # Long breakout
                    if bar[f"{sym}_high"] > H_OR:
                        pos[sym] = dict(dir=1, entry=H_OR, stop=L_OR,
                                        target=H_OR + rng * profit_mult,
                                        rng=rng, mult=mult, entry_ts=bar["ts_et"])
                        p = pos[sym]
                        # Same-bar resolution check
                        if bar[f"{sym}_low"] <= p["stop"]:
                            _book(sym, p["stop"],   "STOP",   bar["ts_et"]); continue
                        if bar[f"{sym}_high"] >= p["target"]:
                            _book(sym, p["target"], "TARGET", bar["ts_et"]); continue

                    # Short breakout
                    elif bar[f"{sym}_low"] < L_OR:
                        pos[sym] = dict(dir=-1, entry=L_OR, stop=H_OR,
                                        target=L_OR - rng * profit_mult,
                                        rng=rng, mult=mult, entry_ts=bar["ts_et"])
                        p = pos[sym]
                        # Same-bar resolution check
                        if bar[f"{sym}_high"] >= p["stop"]:
                            _book(sym, p["stop"],   "STOP",   bar["ts_et"]); continue
                        if bar[f"{sym}_low"] <= p["target"]:
                            _book(sym, p["target"], "TARGET", bar["ts_et"]); continue

                else:
                    # Manage open position — circuit breaker does NOT block exits
                    d = p["dir"]
                    if is_last:
                        _book(sym, bar[f"{sym}_close"], "EOD", bar["ts_et"])
                    elif d == 1:
                        if bar[f"{sym}_low"] <= p["stop"]:
                            _book(sym, p["stop"],   "STOP",   bar["ts_et"])
                        elif bar[f"{sym}_high"] >= p["target"]:
                            _book(sym, p["target"], "TARGET", bar["ts_et"])
                    else:
                        if bar[f"{sym}_high"] >= p["stop"]:
                            _book(sym, p["stop"],   "STOP",   bar["ts_et"])
                        elif bar[f"{sym}_low"] <= p["target"]:
                            _book(sym, p["target"], "TARGET", bar["ts_et"])

        # Force-close any position that opened on the very last bar
        last_bar = tw.iloc[-1]
        for sym, H_OR, L_OR, rng, mult in syms:
            if pos[sym] is not None and sym not in traded:
                _book(sym, last_bar[f"{sym}_close"], "EOD", last_bar["ts_et"])

        if daily_pnl != 0:
            daily_rets[date_str] = daily_rets.get(date_str, 0.0) + daily_pnl

    return _compute_stats(trade_log, daily_rets, account)


# ─── Wait-for-confirmation backtest ─────────────────────────────────────────

MES_MULT = 5.0    # Micro ES  $/pt  (1/10th of ES)
MNQ_MULT = 2.0    # Micro NQ  $/pt  (1/10th of NQ)
TICK     = 0.25   # Minimum tick size for ES, NQ, MES, MNQ (pts)
COMM_MICRO = 0.74  # IBKR round-trip per MES/MNQ contract
COMM_FULL  = 4.10  # IBKR round-trip per ES/NQ  contract


def run_orb_wait_confirm(
    df: pd.DataFrame,
    or_mins:          int   = OR_MINS,
    profit_mult:      float = PROFIT_MULT,
    min_range_es:     float = MIN_RANGE_ES,
    min_range_nq:     float = MIN_RANGE_NQ,
    risk_pct:         float = RISK_PCT,
    news_dates:       set   | None = None,
    # ── Realistic execution parameters ──────────────────────────────────
    use_micro:        bool  = False,   # True → MES/MNQ whole contracts
    slip_ticks:       int   = 0,       # ticks of slippage per side (entry+stop/EOD)
    commission_rt:    float | None = None,  # $ per contract round-trip; None=auto
) -> dict:
    """
    ORB alignment with ZERO lookahead — realistic live-trading version.

    Logic per day:
      1. Observe opening range (09:30–10:00 ET) for both symbols.
      2. Scan trade window bar by bar.
      3. When FIRST symbol breaks its OR → start waiting (no entry yet).
      4. When SECOND symbol breaks its OR in the SAME direction → enter BOTH:
           - Second sym: entry at its OR level (limit fill, as expected)
           - First  sym: entry at THIS bar's open (market fill — we enter
             at current price, not the OR level we saw earlier)
           - If opposite direction → misaligned, cancel, no trade.
      5. Manage both positions normally until stop / target / EOD.

    The key difference vs ORB_+ALIGN:
      ORB_+ALIGN pre-computes alignment then enters first sym at H_OR_ES
        (the level it broke hours ago) — optimistic.
      This version enters first sym at CURRENT MARKET PRICE when second
        confirms — realistic, potentially much worse fill.

    Records 'lag_bars' per trade: number of bars between first and second
    breakout.  Large lag = large lookahead effect in ORB_+ALIGN.
    """
    obs_end_abs = 9 * 60 + 30 + or_mins
    or_end_h    = obs_end_abs // 60
    or_end_m    = obs_end_abs % 60

    in_rth = (
        ((df["hour_et"] == 9)  & (df["min_et"] >= 30)) |
        (df["hour_et"].between(10, 14)) |
        ((df["hour_et"] == 15) & (df["min_et"] <= 30))
    )
    rth = df[in_rth].copy()

    logger.info("  Pre-computing breakout bars …")
    breakouts = _precompute_breakouts(rth, or_end_h, or_end_m, min_range_es, min_range_nq)

    skip_dates = news_dates or set()
    trade_log  = []
    daily_rets = {}
    account    = ACCOUNT_START

    for date, day in rth.groupby("trade_date", sort=True):
        date_str = str(date)
        if date_str in skip_dates:
            continue

        bk = breakouts.get(date_str)
        if bk is None:
            continue

        es_bk = bk["ES"];  nq_bk = bk["NQ"]

        # Both must have broken, in the same direction
        if es_bk is None or nq_bk is None:
            continue
        if es_bk["dir"] != nq_bk["dir"]:
            continue   # misaligned day

        direction   = es_bk["dir"]
        H_ES        = bk["H_ES"];  L_ES = bk["L_ES"];  rng_ES = bk["rng_ES"]
        H_NQ        = bk["H_NQ"];  L_NQ = bk["L_NQ"];  rng_NQ = bk["rng_NQ"]
        es_idx      = es_bk["bar_idx"]
        nq_idx      = nq_bk["bar_idx"]
        confirm_idx = max(es_idx, nq_idx)
        lag_bars    = abs(es_idx - nq_idx)

        # Label first vs second
        if es_idx <= nq_idx:
            first_sym,  second_sym  = "ES", "NQ"
            first_H,  first_L,  first_rng,  first_mult  = H_ES, L_ES, rng_ES, ES_MULT
            second_H, second_L, second_rng, second_mult = H_NQ, L_NQ, rng_NQ, NQ_MULT
        else:
            first_sym,  second_sym  = "NQ", "ES"
            first_H,  first_L,  first_rng,  first_mult  = H_NQ, L_NQ, rng_NQ, NQ_MULT
            second_H, second_L, second_rng, second_mult = H_ES, L_ES, rng_ES, ES_MULT

        day = day.sort_values("ts_et").reset_index(drop=True)
        tw_mask = (
            ((day["hour_et"] > or_end_h) |
             ((day["hour_et"] == or_end_h) & (day["min_et"] >= or_end_m))) &
            ~((day["hour_et"] > CUTOFF_H) |
              ((day["hour_et"] == CUTOFF_H) & (day["min_et"] > CUTOFF_M)))
        )
        tw = day[tw_mask].reset_index(drop=True)
        if confirm_idx >= len(tw):
            continue

        conf_bar = tw.iloc[confirm_idx]

        # ── Entry prices ─────────────────────────────────────────────────
        if direction == 1:
            entry_second = second_H
            stop_second  = second_L
            tgt_second   = second_H + second_rng * profit_mult

            if lag_bars == 0:
                # Simultaneous: first sym also fills at its OR level
                entry_first       = first_H
                eff_rng_first     = first_rng
            else:
                # First sym enters at market (conf bar open — current price)
                entry_first   = float(conf_bar[f"{first_sym}_open"])
                eff_rng_first = entry_first - first_L   # stop is still L_OR

            stop_first = first_L
            tgt_first  = entry_first + eff_rng_first * profit_mult

        else:  # short
            entry_second = second_L
            stop_second  = second_H
            tgt_second   = second_L - second_rng * profit_mult

            if lag_bars == 0:
                entry_first   = first_L
                eff_rng_first = first_rng
            else:
                entry_first   = float(conf_bar[f"{first_sym}_open"])
                eff_rng_first = first_H - entry_first

            stop_first = first_H
            tgt_first  = entry_first - eff_rng_first * profit_mult

        # Validate: first sym's stop must still be valid
        if eff_rng_first <= 0:
            continue   # price already moved past stop — skip day

        # ── Open both positions ──────────────────────────────────────────
        pos = {
            second_sym: dict(dir=direction, entry=entry_second, stop=stop_second,
                             target=tgt_second, rng=second_rng, mult=second_mult,
                             entry_ts=conf_bar["ts_et"]),
            first_sym:  dict(dir=direction, entry=entry_first,  stop=stop_first,
                             target=tgt_first,  rng=eff_rng_first, mult=first_mult,
                             entry_ts=conf_bar["ts_et"]),
        }
        traded    = set()
        daily_pnl = 0.0

        # ── Resolve execution parameters for this day ────────────────────
        _comm = (commission_rt if commission_rt is not None
                 else (COMM_MICRO if use_micro else COMM_FULL))

        def _eff_mult(sym):
            if use_micro:
                return MES_MULT if sym == "ES" else MNQ_MULT
            return ES_MULT if sym == "ES" else NQ_MULT

        def _bk(sym, exit_px, reason, ts):
            nonlocal account, daily_pnl
            p        = pos[sym]
            d        = p["dir"]
            raw_pts  = d * (exit_px - p["entry"])  # gross pts before slippage

            # Slippage: entry always costs 1 side; stop/EOD exit costs 1 side
            # TARGET is a resting limit order → fills at exact level (0 slippage)
            entry_slip = slip_ticks * TICK
            exit_slip  = slip_ticks * TICK if reason in ("STOP", "EOD") else 0.0
            adj_pts    = raw_pts - (entry_slip + exit_slip)   # always a cost

            # Contracts: integer (micro or full) vs fractional
            em = _eff_mult(sym)
            frac = (account * risk_pct) / (p["rng"] * em)
            if use_micro:
                contracts = max(1, round(frac))   # nearest integer, ≥1 micro
            else:
                contracts = frac                  # fractional (original behaviour)

            pnl_dollar = adj_pts * contracts * em - _comm * contracts
            pnl_pct    = pnl_dollar / account
            hold_min   = (ts - p["entry_ts"]).total_seconds() / 60

            trade_log.append({
                "date":        date_str,
                "symbol":      sym,
                "direction":   "LONG" if d == 1 else "SHORT",
                "entry":       p["entry"],
                "exit":        exit_px,
                "stop":        p["stop"],
                "target":      p["target"],
                "range":       p["rng"],
                "pnl_pts":     adj_pts,
                "pnl_dollar":  pnl_dollar,
                "pnl_pct":     pnl_pct,
                "exit_reason": reason,
                "hold_min":    hold_min,
                "lag_bars":    lag_bars,
                "contracts":   contracts,
            })
            pos[sym]   = None
            daily_pnl += pnl_pct
            account   += pnl_pct * account
            traded.add(sym)

        # Same-bar resolution at confirm bar
        for sym in [second_sym, first_sym]:
            p = pos[sym]
            if p is None:
                continue
            d = p["dir"]
            if d == 1:
                if conf_bar[f"{sym}_low"]  <= p["stop"]:
                    _bk(sym, p["stop"],   "STOP",   conf_bar["ts_et"])
                elif conf_bar[f"{sym}_high"] >= p["target"]:
                    _bk(sym, p["target"], "TARGET", conf_bar["ts_et"])
            else:
                if conf_bar[f"{sym}_high"] >= p["stop"]:
                    _bk(sym, p["stop"],   "STOP",   conf_bar["ts_et"])
                elif conf_bar[f"{sym}_low"]  <= p["target"]:
                    _bk(sym, p["target"], "TARGET", conf_bar["ts_et"])

        # Scan from confirm_idx + 1 onward
        for i in range(confirm_idx + 1, len(tw)):
            bar     = tw.iloc[i]
            is_last = (i == len(tw) - 1)
            for sym in [second_sym, first_sym]:
                if sym in traded:
                    continue
                p = pos[sym]
                if p is None:
                    continue
                d = p["dir"]
                if is_last:
                    _bk(sym, bar[f"{sym}_close"], "EOD", bar["ts_et"])
                elif d == 1:
                    if bar[f"{sym}_low"]  <= p["stop"]:
                        _bk(sym, p["stop"],   "STOP",   bar["ts_et"])
                    elif bar[f"{sym}_high"] >= p["target"]:
                        _bk(sym, p["target"], "TARGET", bar["ts_et"])
                else:
                    if bar[f"{sym}_high"] >= p["stop"]:
                        _bk(sym, p["stop"],   "STOP",   bar["ts_et"])
                    elif bar[f"{sym}_low"]  <= p["target"]:
                        _bk(sym, p["target"], "TARGET", bar["ts_et"])

        # Force-close any position that opened on the confirm bar itself
        last_bar = tw.iloc[-1]
        for sym in [second_sym, first_sym]:
            if pos[sym] is not None and sym not in traded:
                _bk(sym, last_bar[f"{sym}_close"], "EOD", last_bar["ts_et"])

        if daily_pnl != 0:
            daily_rets[date_str] = daily_rets.get(date_str, 0.0) + daily_pnl

    return _compute_stats(trade_log, daily_rets, account)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _settle(pos, sym, exit_px, reason, exit_ts, trade_log, date_str, account):
    """Close position, append to trade_log."""
    p = pos[sym]
    d = p["dir"]
    rng_pts = d * (exit_px - p["entry"])
    contracts = (account * RISK_PCT) / (p["rng"] * p["mult"])
    pnl_dollar = rng_pts * contracts * p["mult"]
    pnl_pct = pnl_dollar / account
    hold_min = (exit_ts - p["entry_ts"]).total_seconds() / 60

    trade_log.append({
        "date":       date_str,
        "symbol":     sym,
        "direction":  "LONG" if d == 1 else "SHORT",
        "entry":      p["entry"],
        "exit":       exit_px,
        "stop":       p["stop"],
        "target":     p["target"],
        "range":      p["rng"],
        "pnl_pts":    rng_pts,
        "pnl_dollar": pnl_dollar,
        "pnl_pct":    pnl_pct,
        "exit_reason": reason,
        "hold_min":   hold_min,
    })
    pos[sym] = None


def _last_pnl(trade_log):
    return trade_log[-1]["pnl_pct"] if trade_log else 0.0


# ─── Stats ────────────────────────────────────────────────────────────────────

def _compute_stats(trade_log, daily_rets, final_account):
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

    mean_d = dr.mean()
    std_d  = dr.std(ddof=1) if len(dr) > 1 and dr.std() > 0 else 1e-9
    sharpe = mean_d / std_d * np.sqrt(252)

    equity = (1 + dr).cumprod()
    peak   = equity.cummax()
    max_dd = ((equity - peak) / peak).min() * 100

    ann_by_year = dr.groupby(dr.index.year).apply(lambda x: (1 + x).prod() - 1) * 100

    stats = {
        "sharpe_ratio":          round(sharpe, 4),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "total_return_pct":      round(cum_ret * 100, 2),
        "max_drawdown_pct":      round(max_dd, 2),
        "total_trades":          len(tl),
        "overall_win_rate":      float((tl["pnl_pct"] > 0).mean()) if len(tl) else 0.0,
        "final_account":         round(final_account, 2),
        "trades_per_year":       round(len(tl) / n_years, 1),
    }
    return {"stats": stats, "trade_log": trade_log, "daily_returns": dr, "annual_rets": ann_by_year}


# ─── Reporting ────────────────────────────────────────────────────────────────

def _report(label, result, base_stats=None):
    s  = result["stats"]
    tl = pd.DataFrame(result["trade_log"]) if result["trade_log"] else pd.DataFrame()
    d_sh  = s["sharpe_ratio"]          - base_stats["sharpe_ratio"]          if base_stats else 0
    d_ann = s["annualized_return_pct"] - base_stats["annualized_return_pct"] if base_stats else 0

    logger.info(
        "%-30s | Sh=%+.3f(%+.3f)  Ann=%+.2f%%(%+.2f%%)  DD=%.2f%%  "
        "T=%d(%.1f/yr)  WR=%.1f%%  $25k→$%s",
        label,
        s["sharpe_ratio"], d_sh,
        s["annualized_return_pct"], d_ann,
        s["max_drawdown_pct"],
        s["total_trades"], s["trades_per_year"],
        s["overall_win_rate"] * 100,
        f"{s['final_account']:,.0f}",
    )
    if not tl.empty:
        by_sym = tl.groupby("symbol").agg(
            n=("pnl_pct","count"),
            wr=("pnl_pct", lambda x: (x>0).mean()),
            avg=("pnl_pct","mean"),
        )
        for sym, row in by_sym.iterrows():
            logger.info("  %-4s n=%d  WR=%.1f%%  avg=+%.4f%%", sym,
                        int(row["n"]), row["wr"]*100, row["avg"]*100)
        ec = tl["exit_reason"].value_counts()
        logger.info("  Exits: %s", "  ".join(f"{k}={v}" for k, v in ec.items()))
        if "circuit_tripped" in tl.columns:
            logger.info("  Circuit trips: %d days", tl["circuit_tripped"].sum())
    return s, result["annual_rets"]


def _print_table(results):
    print()
    print("=" * 105)
    print("ORB v2 — WITH IMPROVEMENTS  |  DAILY BASELINE ref: Sh=1.40  Ann=+4.8%  DD=-8%")
    print("=" * 105)
    print(
        f"{'label':<30} {'sharpe':>7} {'Δsh':>7} {'annual':>8} {'Δann':>7} "
        f"{'maxdd':>7} {'trades':>7} {'wr%':>6}  {'$25k→':>12}"
    )
    print("-" * 105)
    base_sh  = results[0][1]["sharpe_ratio"]
    base_ann = results[0][1]["annualized_return_pct"]
    for lbl, s, _ in results:
        d_sh  = s["sharpe_ratio"]          - base_sh
        d_ann = s["annualized_return_pct"] - base_ann
        print(
            f"{lbl:<30} {s['sharpe_ratio']:>7.3f} {d_sh:>+7.3f} "
            f"{s['annualized_return_pct']:>+7.2f}% {d_ann:>+6.2f}% "
            f"{s['max_drawdown_pct']:>6.2f}% {s['total_trades']:>7d} "
            f"{s['overall_win_rate']*100:>5.1f}%  ${s['final_account']:>11,.0f}"
        )

    years = sorted(results[0][2].index.tolist())
    print()
    print("=" * 105)
    print("ANNUAL RETURNS BY YEAR")
    print("=" * 105)
    hdr = f"{'label':<30} " + "  ".join(f"{y}" for y in years)
    print(hdr)
    print("-" * 105)
    baseline = {2019:4.0,2020:6.3,2021:8.7,2022:-0.9,2023:6.0,2024:2.8,2025:6.0,2026:2.7}
    print(f"{'DAILY_BASELINE (ref)':<30} " +
          "  ".join(f"{baseline.get(y,0):>+6.1f}%" for y in years))
    for lbl, _, ann in results:
        print(f"{lbl:<30} " + "  ".join(f"{ann.get(y,0):>+6.1f}%" for y in years))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df         = load_aligned_data()
    news       = _build_news_dates()
    results    = []

    logger.info("=" * 60)
    logger.info("ORB v2 EXPERIMENTS — ES+NQ  2018–2026  $25k account")
    logger.info("=" * 60)

    # ── E1: Strict range base (best from v1, now on aligned data) ────────
    logger.info("\n--- E1: STRICT_BASE (30min OR, 1.5R, ES>8 NQ>30 range filter) ---")
    r1 = run_orb(df, min_range_es=8.0, min_range_nq=30.0)
    s1, a1 = _report("ORB_STRICT_BASE", r1)
    results.append(("ORB_STRICT_BASE", s1, a1))

    # ── E2: + Circuit breaker (-1.5%/day) ────────────────────────────────
    logger.info("\n--- E2: +CIRCUIT BREAKER (trip at -1.5%/day) ---")
    r2 = run_orb(df, min_range_es=8.0, min_range_nq=30.0, daily_loss_limit=0.015)
    s2, a2 = _report("ORB_+CIRCUIT", r2, s1)
    results.append(("ORB_+CIRCUIT", s2, a2))

    # ── E3: + Alignment filter (ES+NQ must break same direction) ─────────
    logger.info("\n--- E3: +ALIGN FILTER (both symbols same direction) ---")
    r3 = run_orb(df, min_range_es=8.0, min_range_nq=30.0, align_filter=True)
    s3, a3 = _report("ORB_+ALIGN", r3, s1)
    results.append(("ORB_+ALIGN", s3, a3))

    # ── E4: + News day skip (FOMC + NFP + CPI) ───────────────────────────
    logger.info("\n--- E4: +NEWS SKIP (FOMC + NFP + CPI dates) ---")
    r4 = run_orb(df, min_range_es=8.0, min_range_nq=30.0, news_dates=news)
    s4, a4 = _report("ORB_+NEWS_SKIP", r4, s1)
    results.append(("ORB_+NEWS_SKIP", s4, a4))

    # ── E5: All three combined ────────────────────────────────────────────
    logger.info("\n--- E5: ALL THREE COMBINED ---")
    r5 = run_orb(df, min_range_es=8.0, min_range_nq=30.0,
                 daily_loss_limit=0.015,
                 align_filter=True,
                 news_dates=news)
    s5, a5 = _report("ORB_+ALL_THREE", r5, s1)
    results.append(("ORB_+ALL_THREE", s5, a5))

    # ── E6: Wait-for-confirmation (zero lookahead, fractional, no slip) ──
    logger.info("\n--- E6: ALIGN_WAIT_CONFIRM (realistic — no lookahead) ---")
    r6 = run_orb_wait_confirm(df, min_range_es=8.0, min_range_nq=30.0)
    s6, a6 = _report("ORB_ALIGN_WAIT", r6, s1)
    results.append(("ORB_ALIGN_WAIT", s6, a6))

    # ── E7: + MES/MNQ whole contracts (no slippage yet) ──────────────────
    logger.info("\n--- E7: +MES/MNQ whole contracts (integer sizing, no slip) ---")
    r7 = run_orb_wait_confirm(df, min_range_es=8.0, min_range_nq=30.0,
                              use_micro=True, slip_ticks=0)
    s7, a7 = _report("ORB_WAIT+MICRO", r7, s6)
    results.append(("ORB_WAIT+MICRO", s7, a7))

    # ── E8: + 1-tick slippage both sides + IBKR commission ───────────────
    logger.info("\n--- E8: +MICRO +1-tick slippage +commission (fully realistic) ---")
    r8 = run_orb_wait_confirm(df, min_range_es=8.0, min_range_nq=30.0,
                              use_micro=True, slip_ticks=1)
    s8, a8 = _report("ORB_WAIT+MICRO+SLIP", r8, s6)
    results.append(("ORB_WAIT+MICRO+SLIP", s8, a8))

    # ── E7: Wait-for-confirmation diagnostic — lag stats ─────────────────
    logger.info("\n--- E7: LAG ANALYSIS (how many bars between first and second breakout?) ---")
    tl6 = pd.DataFrame(r6["trade_log"])
    if not tl6.empty and "lag_bars" in tl6.columns:
        lag = tl6.groupby("lag_bars").size()
        logger.info("  Lag distribution (bars between 1st and 2nd breakout):")
        for lag_val, cnt in lag.items():
            pct = cnt / len(tl6) * 100
            logger.info("    lag=%3d bars  %4d trades  %.1f%%", lag_val, cnt, pct)
        logger.info("  Median lag: %.1f bars  (%.0f min)",
                    tl6["lag_bars"].median(), tl6["lag_bars"].median())
        logger.info("  Mean   lag: %.1f bars  (%.0f min)",
                    tl6["lag_bars"].mean(),   tl6["lag_bars"].mean())
        logger.info("  Pct same-bar (lag=0): %.1f%%",
                    (tl6["lag_bars"] == 0).mean() * 100)

    _print_table(results)

    # ── Final comparison: lookahead impact ───────────────────────────────
    print()
    print("=" * 70)
    print("LOOKAHEAD BIAS ASSESSMENT")
    print("=" * 70)
    print(f"  ORB_+ALIGN  (pre-computed, WITH lookahead): "
          f"Sh={s3['sharpe_ratio']:.3f}  Ann={s3['annualized_return_pct']:+.1f}%")
    print(f"  ORB_ALIGN_WAIT (wait-for-confirm, NO lookahead): "
          f"Sh={s6['sharpe_ratio']:.3f}  Ann={s6['annualized_return_pct']:+.1f}%")
    delta_sh  = s3["sharpe_ratio"]          - s6["sharpe_ratio"]
    delta_ann = s3["annualized_return_pct"] - s6["annualized_return_pct"]
    print(f"  Lookahead inflation: ΔSh={delta_sh:+.3f}  ΔAnn={delta_ann:+.1f}%")
    if delta_sh < 0.2:
        print("  ✅ Lookahead effect is SMALL — alignment filter result is credible")
    elif delta_sh < 0.5:
        print("  ⚠️  Lookahead effect is MODERATE — some optimism in the backtest")
    else:
        print("  ❌ Lookahead effect is LARGE — wait-for-confirm version is the real system")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
