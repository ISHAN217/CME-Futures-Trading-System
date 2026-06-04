"""
config.py — Competition System Configuration.

This is a competition-optimised version of the multi-strategy system,
designed specifically for 5–7 day CME trading windows.

Every parameter has a justification. Changes from the base system are
marked [CHANGED] with a reason. New parameters are marked [NEW].

Core philosophy for competition mode:
  - Realize PnL in 1–3 days, not 5+
  - Take more signals from a proven high-quality set
  - Size by actual volatility, not fixed weights
  - Block losing patterns proven in backtest
"""

# ── Symbols ───────────────────────────────────────────────────────────────────
SYMBOLS      = ["ES=F", "NQ=F"]
SYMBOL_NAMES = {"ES=F": "ES", "NQ=F": "NQ"}

# ── Data ──────────────────────────────────────────────────────────────────────
START_DATE = "2019-01-01"
END_DATE   = None
DATA_DIR   = "../cme_execution_system/data"  # reuse cached data by default
OUTPUT_DIR = "output"

INTRADAY_INTERVAL = "1h"
INTRADAY_PERIOD   = "720d"

# ── Momentum engine [NEW] ──────────────────────────────────────────────────────
# Multi-timeframe momentum replaces single weekly bias.
# WHY: Blending 5d/20d/60d gives a more stable, less whipsaw-prone signal
#      than pure weekly returns. The 20d window dominates because it matches
#      the competition-relevant time horizon.
MOM_WINDOWS  = [5, 20, 60]   # lookback periods
MOM_WEIGHTS  = [0.25, 0.50, 0.25]  # 20d is primary signal
MOM_LONG_THRESH = 0.10   # score above this → LONG bias
MOM_NEUT_BAND   = 0.05   # dead zone near zero → NEUTRAL

# ── Weekly bias ───────────────────────────────────────────────────────────────
# [CHANGED] Only LONG/NEUTRAL — no SHORT bias.
# WHY: Backtest confirmed weekly SHORT bias is correct only 35-40% of the time
#      for ES/NQ. Structural bull market means SHORT bias is noise, not signal.
WEEKLY_REG_WINDOW  = 40   # days for slope/R² computation
WEEKLY_LONG_THRESH = 0.20
# No SHORT threshold — system never trades short in trend mode.

# ── Daily regime ──────────────────────────────────────────────────────────────
# [CHANGED] Added TRANSITION regime between TREND and CHOP.
# WHY: The 2-day regime buffer in the base system caused position carry-over
#      losses at regime boundaries. Making TRANSITION an explicit regime with
#      its own rules (half-size, no new trend entries) is cleaner and more
#      controllable.
TREND_R2_MIN        = 0.25   # R² above this → TREND
TRANSITION_R2_MIN   = 0.10   # R² between here and TREND_R2_MIN → TRANSITION [NEW]
SHOCK_VOL_RATIO     = 2.0    # 5d/20d vol spike → SHOCK
SHOCK_RETURN_SIGMA  = 2.5    # |ret| / vol_20d → SHOCK

# ── Volatility engine [NEW] ───────────────────────────────────────────────────
# WHY: Fixed weights (0.25, 0.20, 0.15) ignore actual market volatility.
#      During high-vol regimes, fixed weights take on too much risk.
#      Inverse-EWMA sizing keeps dollar risk roughly constant.
EWMA_LAMBDA      = 0.94    # RiskMetrics standard; fast response to vol changes
ATR_PERIOD       = 14      # ATR lookback for stop reference
TARGET_VOL_DAILY = 0.010   # [OPTIMISED] 100bps target daily vol — sizing sweep showed best Sharpe/DD trade-off
                            # 0.004→0.010: AvgPos $4,652→$6,803, Sharpe 0.768→0.846, MaxDD -3.74%→-5.56%
VOL_SCALE_MIN    = 0.05    # minimum weight even in high-vol environment
VOL_SCALE_MAX    = 0.30    # maximum weight

# ── 4H execution engine ───────────────────────────────────────────────────────
H4_ZSCORE_WINDOW        = 10
H4_PULLBACK_ZSCORE      = -1.0   # zscore below this → pullback signal
H4_BREAKOUT_LOOKBACK    = 8      # bars for high/low breakout [NEW]
H4_OVEREXTENSION_ZSCORE = 2.0    # block new entries
H4_MOM_WINDOWS          = [1, 3, 6]
# Entry quality score (1-5) thresholds
H4_MOMENTUM_THRESHOLD = 0.0  # h4_momentum_score above this → MOMENTUM signal
H4_QUALITY_HIGH = 4   # ≥4 → high conviction, allow full size [NEW]
H4_QUALITY_LOW  = 2   # <2 → skip entry

# ── Mean reversion ────────────────────────────────────────────────────────────
MR_ZSCORE_WINDOW_4H  = 10
MR_ZSCORE_WINDOW_DAY = 5
MR_ENTRY_THRESHOLD   = 1.2
MR_EXIT_THRESHOLD    = 0.30

# [CHANGED] MR pullback in TREND regime — new fast-alpha mode.
# WHY: In a trending market, 4H pullbacks to z < -1.0 often revert within
#      1-2 days. This generates fast PnL bursts within the competition window
#      without changing the overall trend position.
TREND_PULLBACK_ENTRY_Z  = -1.0   # enter LONG when 4H z drops below this in TREND
TREND_PULLBACK_MOM_MIN  = 0.05   # momentum_score must still be positive
TREND_PULLBACK_WEIGHT   = 0.15   # smaller weight for pullback trades

# ── Stat arb (NQ dominance model) ────────────────────────────────────────────
# [CHANGED] Only LONG NQ direction. No ES hedge.
# WHY: NQ LONG on spread divergence had 70% win rate in backtest.
#      ES SHORT hedge lost at 35% (ES also goes up, just less than NQ).
#      Running NQ-only captures the full edge without the drag.
SA_BETA_WINDOW      = 60
SA_ZSCORE_WINDOW    = 20
SA_ENTRY_THRESHOLD  = 1.5
SA_EXIT_THRESHOLD   = 0.40   # [CHANGED] faster exit (was 0.50)
SA_NQ_ONLY          = True
SA_NO_ES_HEDGE      = True

# ── Sentiment / event layer ───────────────────────────────────────────────────
SENTIMENT_VOL_RATIO  = 2.0
SENTIMENT_RET_SIGMA  = 2.5
SENTIMENT_VOL_Z      = 2.0
SENTIMENT_GAP_THRESH = 0.015
SENTIMENT_MULTIPLIERS = {
    "NORMAL":       1.00,
    "EVENT_ACTIVE": 0.50,
    "HIGH_RISK":    0.00,
}

# ── Signal engine — competition mode ─────────────────────────────────────────
# [CHANGED] All max holds reduced for 5–7 day window.
# WHY: A 5-day hold in a 7-day competition window consumes the entire window
#      if the trade goes wrong. Faster exits keep options open for better trades.
ALLOW_EQUITY_SHORTS = False   # never

MAX_HOLD_DAYS = {
    "TREND":       3,  # base timer — backtest can extend to MAX_EXTENDED_HOLD_DAYS [v3]
    "PULLBACK":    5,  # [CHANGED v3 was 2] — allow pullbacks to develop fully
    "MEAN_REVERT": 3,  # [CHANGED v3 was 2] — extra day for MR to complete reversion
    "STAT_ARB":    5,  # [CHANGED v3 was 3] — spreads can take longer to converge
}

TRANSITION_SIZE_MULT = 0.50  # half size in TRANSITION regime [NEW]

# ── V3: Winner extension & dynamic risk management [NEW] ─────────────────────
# WHY: Previous system exited trades on fixed timers, cutting winners short.
#      V3 lets TREND winners run up to 7 days when conditions remain valid,
#      while protecting profits via trailing stops and cutting losers fast.

# TREND hard cap — even with extension, never hold past this many days
MAX_EXTENDED_HOLD_DAYS = 7

# Winner extension: allow h4_exec_signal=NONE to still qualify for extension.
# When False (default): requires active 4H signal to extend.
# When True: TREND+weekly=LONG+mom>0 is sufficient (4H silence = steady trend).
EXTEND_ALLOW_H4_NONE = False

# Stop loss: per-trade cumulative return threshold (not portfolio-level)
# WHY: -1.5% on a position means the trade thesis is wrong. Cut it.
STOP_LOSS_THRESHOLD = -0.030   # [OPTIMISED] sweep in run_stoploss.py — -3.0% best Sharpe 0.755

# Trailing profit stop: arms once profitable enough, fires on drawdown
# WHY: Lets winners run without giving back all gains.
TRAIL_ARM_THRESHOLD  = 0.015   # [OPTIMISED] arm when cum_ret >= +1.5% (was 1.0%)
TRAIL_STOP_DRAWDOWN  = 0.010   # [OPTIMISED] fire when drawdown from peak >= 1.0% (was 0.7%)

# Fast exit: protect profits when conviction fades
# WHY: If h4 signal flips to WAIT or confidence drops, book the gain.
FAST_EXIT_PROFIT_FLOOR    = 0.008  # only trigger when cum_ret >= +0.8%
FAST_EXIT_CONFIDENCE_DROP = 0.30   # exit if confidence_score below this

# Add-to-winner: increase TREND position size when trade is working
# WHY: The best trades deserve more capital; add once on clear profitability.
ADD_WINNER_MIN_CUM_RET = 0.005  # must be +0.5% profitable before adding
ADD_WINNER_MAX_ADDS    = 1      # max 1 add per trade (prevents pyramiding risk)
ADD_WINNER_SIZE_FRAC   = 0.50   # add 50% of original entry weight

# High-conviction upscaling [NEW]
# WHY: When momentum is strong AND 4H quality is high, the edge is larger.
#      Sizing up in these cases improves Sharpe without proportionally
#      increasing max drawdown.
HIGH_CONV_MOM_THRESH  = 0.65   # momentum_strength above this
HIGH_CONV_QUALITY_MIN = 4      # 4H quality score ≥ this
HIGH_CONV_MULTIPLIER  = 1.20   # +20% size (max still capped)

# ── Portfolio ─────────────────────────────────────────────────────────────────
MAX_WEIGHT_PER_SYMBOL  = 0.30
MAX_COMBINED_EXPOSURE  = 0.50
MEAN_REVERT_WEIGHT     = 0.18  # slightly lower than base (0.20)
STAT_ARB_NQ_WEIGHT     = 0.20  # NQ-only SA gets slightly higher weight

# ── Backtest ──────────────────────────────────────────────────────────────────
TRANSACTION_COST_BPS  = 1.0
TRADING_DAYS_PER_YEAR = 252

# ══════════════════════════════════════════════════════════════
# CWT CHOP FILTER  [NEW]
# ══════════════════════════════════════════════════════════════
USE_CWT_CHOP_FILTER = False   # [REJECTED] ablation: hard disable MR/PB beats CWT by 0.049 Sharpe

CWT_WINDOW     = 64     # rolling window for CWT computation
CWT_MIN_PERIODS = 48    # minimum data points before CWT is active

CWT_SCALES = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]

CWT_HIGH_SCALE_MAX = 6   # scales <= this are "high freq / noise"
CWT_MID_SCALE_MAX  = 16  # scales <= this (and > HIGH) are "mid freq"
                          # scales > MID are "slow / trending"

CWT_ENTROPY_HIGH = 0.80  # above = chaotic (energy spread across all scales)
CWT_ENTROPY_LOW  = 0.45  # below = very coherent (dominated by one scale)

CWT_ENERGY_COMPRESSION_Z = -0.75  # below = energy compression
CWT_ENERGY_EXPANSION_Z   =  1.00  # above = energy expansion/breakout

CWT_RANGE_SIZE_MULT       = 0.75  # tradable range chop
CWT_CHAOTIC_SIZE_MULT     = 0.00  # chaotic noise — block
CWT_COMPRESSION_SIZE_MULT = 0.00  # compression — block
CWT_UNCLEAR_SIZE_MULT     = 0.50  # unclear — half size

CWT_APPLY_TO_MEAN_REVERT = True
CWT_APPLY_TO_PULLBACK    = True
CWT_APPLY_TO_STAT_ARB    = False  # v1: tag only, no blocking
CWT_APPLY_TO_TREND       = False  # never apply CWT to TREND

# ── Ablation flags [NEW] ──────────────────────────────────────────────────────
# Used by run_ablation.py to disable strategies without CWT.
# Normal operation: both False. Set via monkey-patch in experiment runner.
DISABLE_MEAN_REVERT = True    # [OPTIMISED] ablation Config D — MR/PB disabled; Sharpe 0.724 vs 0.675 CWT
DISABLE_PULLBACK    = True    # [OPTIMISED] ablation Config D

# ══════════════════════════════════════════════════════════════
# SIGNAL QUALITY SIZING  [NEW]
# ══════════════════════════════════════════════════════════════
# Overlay that scales position size by signal quality bucket.
# Does NOT replace vol scaling — applied multiplicatively on top.
# Normal operation: USE_SIGNAL_QUALITY_SIZING = False.
# Set True + configure multipliers to activate.

# [OPTIMISED] STAT_ARB quality overlay accepted — Sharpe 0.882 vs 0.846 baseline
# Aggressive multiplier on SA only. TREND quality rejected (non-monotonic).
USE_SIGNAL_QUALITY_SIZING      = True
APPLY_QUALITY_SIZING_TO_TREND  = False   # rejected — EXCEPTIONAL bucket WORST performer
APPLY_QUALITY_SIZING_TO_STAT_ARB = True
STAT_ARB_QUALITY_MODE          = "hump"  # hump outperforms linear; penalises extreme-z
QUALITY_SIZING_MODE            = "multiplier"

# TREND: unused (APPLY_QUALITY_SIZING_TO_TREND=False)
TREND_QUALITY_MULTIPLIERS = {
    "WEAK":        0.50,
    "MODERATE":    0.75,
    "GOOD":        1.00,
    "STRONG":      1.10,
    "EXCEPTIONAL": 1.25,
}
# STAT_ARB: Aggressive — SA quality IS monotonically predictive (STRONG > GOOD)
# Inverse SA scored 0.824 vs Aggressive 0.882 → quality adds genuine +0.058 Sharpe
STATARB_QUALITY_MULTIPLIERS = {
    "WEAK":        0.25,
    "MODERATE":    0.75,
    "GOOD":        1.00,
    "STRONG":      1.50,
    "EXCEPTIONAL": 2.00,
}

# ══════════════════════════════════════════════════════════════
# TREND CONTINUATION EDGE SIZING  [NEW]
# ══════════════════════════════════════════════════════════════
# Sizes TREND positions by continuation-edge bucket, NOT old trend_quality.
# Does NOT affect STAT_ARB (SA uses its own hump-quality multiplier).
# Normal operation: both False. Activated by run_trend_edge.py experiments.

USE_TREND_EDGE_MULTIPLIER    = False   # apply multiplier schedule to TREND
USE_TREND_EDGE_DIRECT_WEIGHT = False   # assign direct portfolio-% weight to TREND

# Multiplier schedule (used when USE_TREND_EDGE_MULTIPLIER = True)
TREND_EDGE_MULTIPLIERS = {
    "NO_EDGE":     0.00,
    "WEAK":        0.50,
    "MODERATE":    0.75,
    "STRONG":      1.25,
    "EXCEPTIONAL": 1.50,
}

# Direct weight schedules (used when USE_TREND_EDGE_DIRECT_WEIGHT = True)
# Active schedule is set via monkey-patch in run_trend_edge.py
TREND_EDGE_DIRECT_WEIGHT_SCHEDULE = {
    "NO_EDGE":     0.00,
    "WEAK":        0.10,
    "MODERATE":    0.15,
    "STRONG":      0.20,
    "EXCEPTIONAL": 0.25,
}

TREND_EDGE_DIRECT_WEIGHT_SCHEDULE_AGGRESSIVE = {
    "NO_EDGE":     0.00,
    "WEAK":        0.10,
    "MODERATE":    0.15,
    "STRONG":      0.25,
    "EXCEPTIONAL": 0.30,
}

TREND_EDGE_DIRECT_WEIGHT_SCHEDULE_CONSERVATIVE = {
    "NO_EDGE":     0.00,
    "WEAK":        0.05,
    "MODERATE":    0.10,
    "STRONG":      0.15,
    "EXCEPTIONAL": 0.20,
}

# ══════════════════════════════════════════════════════════════
# TACTICAL SHORT SLEEVE  [NEW]
# ══════════════════════════════════════════════════════════════
# Optional short-signal overlay. Operates independently of TREND/STAT_ARB.
# Does NOT modify existing long-side logic.
# Normal operation: USE_TACTICAL_SHORT_ENGINE = False.

USE_TACTICAL_SHORT_ENGINE         = False   # master switch
TACTICAL_SHORT_SIZE_MULT          = 0.50    # fraction of VOL_SCALE_MAX base
TACTICAL_SHORT_MAX_HOLD           = 3       # max holding days
TACTICAL_SHORT_STOP               = -0.015  # exit if cum_ret < this (i.e., +1.5% adverse)
TACTICAL_SHORT_TRAIL_ARM          = 0.010   # arm trailing stop when profitable ≥+1.0%
TACTICAL_SHORT_TRAIL_DRAWDOWN     = 0.007   # fire trail when drawdown from peak ≥ 0.7%
TACTICAL_SHORT_REQUIRE_CONFIRMATION = True  # require price below rolling low at entry

# Which setups to enable (monkey-patched in experiments)
TACTICAL_SHORT_USE_DM  = True   # downside momentum short
TACTICAL_SHORT_USE_FR  = True   # failed rally short
TACTICAL_SHORT_USE_CB  = True   # compression breakdown short
TACTICAL_SHORT_STRONG_ONLY = False  # if True: only trade STRONG bucket signals

# Size fractions by bucket (applied to VOL_SCALE_MAX × TACTICAL_SHORT_SIZE_MULT)
TACTICAL_SHORT_BUCKET_FRACS = {
    "STRONG":   1.00,   # full allocated size
    "MODERATE": 0.50,   # half size
    "WEAK":     0.00,   # skip (WEAK is not traded)
    "NONE":     0.00,
}

# ══════════════════════════════════════════════════════════════
# SHOCK SHORT SLEEVE  [NEW]
# ══════════════════════════════════════════════════════════════
# SHOCK-regime-only short sleeve. Only activates during genuine
# market stress (SHOCK/post-SHOCK), not normal pullbacks.
# Normal operation: USE_SHOCK_SHORT_ENGINE = False.

USE_SHOCK_SHORT_ENGINE                = False   # master switch
SHOCK_SHORT_SYMBOLS                   = ["ES", "NQ"]
SHOCK_SHORT_SIZE                      = 0.10    # base weight (fraction of capital)
SHOCK_SHORT_MAX_SIZE                  = 0.15    # hard cap per symbol
SHOCK_SHORT_MAX_HOLD                  = 3       # default max hold (days)
SHOCK_SHORT_ALLOW_SHOCK_ENTRIES       = True    # allow entries during SHOCK (KEY fix vs v1)
SHOCK_SHORT_USE_ATR_STOP              = True    # use ATR-based stop (False = fixed pct)
SHOCK_SHORT_ATR_MULT                  = 2.5     # stop = entry + ATR_MULT × ATR
SHOCK_SHORT_FIXED_STOP                = -0.030  # fixed pct stop (if ATR stop disabled)
SHOCK_SHORT_CLOSE_ONLY_STOP           = True    # evaluate stop on daily close (always true here)
SHOCK_SHORT_REQUIRE_DOWNSIDE_CONFIRMATION = True  # require price below rolling low

# Which setups to enable (monkey-patched per experiment)
SHOCK_SHORT_USE_CONT      = True   # shock continuation
SHOCK_SHORT_USE_BOUNCE    = True   # post-shock failed bounce
SHOCK_SHORT_USE_BREAKDOWN = True   # compression breakdown

# Stop sweep variants (set via monkey-patch in experiment runner)
# stop_variant: "fixed_015" | "fixed_030" | "fixed_050" | "atr_20" | "atr_25" | "atr_30" | "none"
SHOCK_SHORT_STOP_VARIANT = "atr_25"  # default

# ══════════════════════════════════════════════════════════════
# STRUCTURAL MEAN-REVERSION SLEEVE  [NEW]
# ══════════════════════════════════════════════════════════════
# CWT/EMD-confirmed mean-reversion in CHOP/TRANSITION regimes.
# Separate optional module — does not modify TREND/STAT_ARB/SHOCK logic.
# Only active when USE_STRUCTURAL_MEAN_REVERSION = True.

USE_STRUCTURAL_MEAN_REVERSION    = False   # master switch

STRUCTURAL_MR_SIZE               = 0.10   # base position weight (fraction of capital)
STRUCTURAL_MR_MAX_SIZE           = 0.15   # hard cap per symbol
STRUCTURAL_MR_MAX_HOLD           = 2      # max hold days (MR must resolve quickly)
STRUCTURAL_MR_STOP_ATR_MULT      = 1.5    # stop = 1.5 × ATR beyond entry
STRUCTURAL_MR_TARGET             = "MIDPOINT"  # exit target: range midpoint

# Filter gates (monkey-patched per experiment)
STRUCTURAL_MR_REQUIRE_CWT        = True   # require CWT range oscillatory
STRUCTURAL_MR_REQUIRE_EMD        = True   # require EMD stable range
STRUCTURAL_MR_REQUIRE_RANGE      = True   # require range quality ≥ threshold

# Range quality threshold (monkey-patched per experiment)
STRUCTURAL_MR_RQ_THRESHOLD       = 0.42   # range_quality_score ≥ this

# Position-in-range thresholds for entry
STRUCTURAL_MR_PIR_LONG           = 0.22   # enter LONG when pos_in_range ≤ this
STRUCTURAL_MR_PIR_SHORT          = 0.78   # enter SHORT when pos_in_range ≥ this
STRUCTURAL_MR_ZSCORE_LONG        = -1.30  # or price_zscore ≤ this
STRUCTURAL_MR_ZSCORE_SHORT       =  1.30  # or price_zscore ≥ this

# ══════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME STRUCTURE SLEEVE  [NEW]
# ══════════════════════════════════════════════════════════════
# Intraday multi-timeframe structure engine.
# Uses genuine 1-minute Databento data resampled to 1H/4H/12H/24H.
# All flags default False — enabled per experiment only.
# Does not modify TREND / STAT_ARB / SHOCK_BOUNCE core logic.

USE_MTF_STRUCTURE              = False   # master switch
USE_MTF_BIAS_FILTER            = False   # filter TREND trades by MTF bias direction
USE_MTF_TREND_RESET            = False   # add 4H pullback/reset trades
USE_MTF_BREAKOUT_CONTINUATION  = False   # add 4H breakout continuation trades
USE_MTF_EXIT_OVERLAY           = False   # early exit when structure breaks

MTF_TREND_RESET_SIZE           = 0.05   # position size for reset trades
MTF_BREAKOUT_CONT_SIZE         = 0.05   # position size for breakout trades
MTF_MAX_NEW_TRADE_SIZE         = 0.10   # hard cap on any new MTF trade
MTF_MAX_HOLD_BARS              = 3      # max daily bars for intraday-derived trades

# Confirmation requirements
MTF_REQUIRE_12H_24H_CONFIRMATION = True  # 12H and 24H must agree on bias
MTF_REQUIRE_1H_TRIGGER           = True  # 1H must confirm direction
MTF_SKIP_CHAOTIC_STRUCTURE       = True  # no trades when ≥2 TFs are CHAOTIC

# Confidence floor — skip trades below this MTF confidence
MTF_CONFIDENCE_FLOOR           = 0.50

# Weights for MTF composite bias
MTF_WEIGHT_24H                 = 0.40
MTF_WEIGHT_12H                 = 0.30
MTF_WEIGHT_4H                  = 0.20
MTF_WEIGHT_1H                  = 0.10

# Bias thresholds
MTF_BIAS_BULLISH_THRESH        =  0.50
MTF_BIAS_BEARISH_THRESH        = -0.50
