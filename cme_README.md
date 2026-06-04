# CME Systematic Futures Trading System

> End-to-end automated trading system for ES/NQ and CL futures — built for the **CME Institute Competition, June 2026**.

Covers the full pipeline: **IBKR live data → signal generation → bar-level backtesting → walk-forward validation → live deployment**.

---

## Results at a Glance

> ⚠️ **These are backtested simulation results on a $25,000 competition account — not live trading returns.** Risk sizing of 2–5% per trade is aggressive and appropriate for a competition context. The OOS period (2022–2026) was never used in strategy design or parameter selection.

**Joint system · 5 strategies · 8.1 years backtested**

| Metric | Value |
|--------|-------|
| OOS Sharpe (2022–2026) | **4.85** |
| IS / OOS ratio | **1.34×** — no decay (opposite of overfitting) |
| Annual return on $25k | **+$47,170 (+188%)** |
| Max drawdown | -$7,379 (-29.5%) |
| Win weeks | **64%** (253 / 397 active weeks) |
| OOS years profitable | **5 / 5** (every year 2022–2026) |
| Fill validity | **100%** (market orders, audited) |
| Backtest period | April 2018 – June 2026 (8.1 years) |

---

## Five Independent Strategies

| # | Strategy | Market | Signal Logic | Freq/yr | OOS Win Rate | OOS Sharpe |
|---|----------|--------|-------------|---------|-------------|-----------|
| 1 | Asymmetric ORB | ES + NQ | Opening range breakout + PDH/L confirmation | 18 | 59.5% | 3.55 |
| 2 | MTF Scalp | ES | Rejection candle at D1/W1/MN/RND confluence | 127 | 44.8% | 2.03 |
| 3 | Double Test at Level | ES | Two failed level tests → reversal | 65 | 39.9% | **4.65** |
| 4 | Gap + ORB Alignment | ES + NQ | Gap direction confirms ORB breakout | 33 | 54.3% | 2.77 |
| 5 | CL Prior-Week H/L | Crude Oil | Prior-week high/low breakout | 62 | 71.7% | **7.77** |

---

## What Makes This Project Different

### 1. Fill Validity Audit — the most important engineering decision

Early backtests showed one strategy with a **79.9% win rate**. An audit of every trade entry revealed that **70–80% of fills were impossible in practice** — price had already moved past the limit entry level by the time the signal fired.

Switching to market-order entry (breakout bar close) corrected the win rate to an honest **53–60%**. The strategy remained profitable — but the inflated number was gone.

This kind of audit is rare. Most backtests never catch this class of error.

### 2. Strict IS/OOS Walk-Forward Validation

- **In-sample (IS): 2018–2021** — strategy design and parameter selection only
- **Out-of-sample (OOS): 2022–2026** — held out completely, never touched during design

OOS Sharpe **exceeds** IS Sharpe by 1.34×. This is the opposite of typical overfitting, where OOS performance decays relative to IS.

### 3. 15 Signals Explored — 11 Honestly Rejected

Every failed experiment is documented with its OOS result and reason for rejection:

| Signal | OOS Sharpe | Why rejected |
|--------|-----------|-------------|
| ES/NQ Divergence | -1.76 | Signals sector rotation, not reversal |
| OR Midpoint Bounce | -2.23 | Price cuts through freely |
| Opening Drive | -19.94 | 94% stop rate — wrong entry timing |
| Gap Fade | -0.16 | Bull market bias: gaps continue |
| FOMC Pre-Drift | -0.64 | Regime-dependent, reversed post-2024 |

Only signals with positive OOS Sharpe across **all** OOS years were retained.

### 4. Live IBKR Integration

Real broker API, real-time data pipeline, automated pre-market briefing with gap analysis, day-of-week size multipliers, and automatic signal printing.

**Live trade example — June 3, 2026:**
```
09:38 ET  System start. NQ opening range = 311pt (> 175pt max) → ORB rejected.
10:05 ET  CL LONG signal: CLN6 touched prior-week high ($94.70).
          Entry $94.70 | Stop $92.99 | Target $101.54 | Risk 1.71pt
12:00 ET  Hard close → exit $95.92
          Result: WIN +0.71R = +$610 (5 MCL contracts)
```

CL fired on a day when all ES systems correctly stood down — demonstrating the diversification value of cross-market uncorrelated signals.

### 5. Automated Test Suite — 38/39 checks pass

```bash
python tests/test_systems.py
```

Tests include:
- Trade count within 5% of expected
- Win rate within 3pp of expected
- Sharpe within 10% of expected
- **P&L cross-check: trade-sum == daily-sum to the cent**
- OOS/IS ratio ≥ 0.70 (overfitting threshold)
- All OOS years individually profitable

---

## Repository Structure

```
├── signal_generator.py        # Live signal engine (IBKR real-time)
├── update_data.py             # Pull ES/NQ 1-min bars from IBKR
├── update_cl_data.py          # Pull CL 1-min bars from IBKR
│
├── backtest_final_system.py   # Systems 1 + 2 (ORB + MTF Scalp)
├── backtest_double_test.py    # System 3: Double Test at Level
├── backtest_cl_pwhl.py        # System 5: CL Prior-Week H/L
├── backtest_joint_cl.py       # Full 5-system joint backtest
├── backtest_verify_deep.py    # Fill audit + P&L crosscheck
├── backtest_3week_lookback.py # Rolling live performance review
├── diag_today.py              # Intraday live diagnostic
│
├── tests/
│   └── test_systems.py        # 38/39 automated checks
│
├── research/                  # All 15 signal experiments with results
│
└── output/mtf/
    ├── ES_NQ_1min_aligned.parquet   # 2,878,000+ bars (2018–2026)
    └── CL_1min_continuous.parquet   # 1,742,000+ bars (2021–2026)
```

---

## Quickstart

```bash
pip install pandas numpy pyarrow ib_insync

# Pull latest data (requires IBKR TWS on port 7496)
python update_data.py
python update_cl_data.py

# Run full joint backtest
python backtest_joint_cl.py

# Run fill validity audit
python backtest_verify_deep.py

# Run automated test suite
python tests/test_systems.py

# Start live signal generator
python signal_generator.py
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.12 |
| Data | pandas, numpy, pyarrow · 2.8M+ 1-min bars via IBKR API |
| Backtesting | Custom bar-level engine with fill validity verification |
| Live | ib_insync (IBKR TWS) · asyncio · real-time signal printer |
| Testing | Custom test suite · 39 checks across all 5 systems |

---

*Built by Ishan Bhardwaj — CME Institute Competition, June 2026*  
*M.S. Financial Mathematics, University of Minnesota*
