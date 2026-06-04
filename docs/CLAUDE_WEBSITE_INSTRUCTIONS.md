# Instructions for Claude — Website Portfolio Integration

## Who You Are Helping
**Ishan Bhardwaj** — building a personal portfolio website to showcase
quantitative trading and data science projects for job search purposes
(targeting quant research, systematic trading, FinTech, and data science roles).

## Your Task
Help improve the portfolio website by:
1. Adding the CME Trading System project (fully documented below)
2. Adding any other projects Ishan uploads or describes
3. Making the site look professional, clean, and credible to a quant/tech hiring manager

---

## PROJECT 1 — CME Systematic Futures Trading System

### One-Line Description
*End-to-end systematic trading system for ES/NQ and CL futures —
data pipeline (IBKR API) → backtesting → live deployment — built
for the CME Institute Competition, June 2026.*

### Key Stats (verified, use these exact numbers)
| Metric | Value |
|--------|-------|
| OOS Sharpe (2022–2026) | 4.85 |
| IS/OOS ratio | 1.34× (no decay — opposite of overfitting) |
| Annual return on $25k | +188% (+$47,170) |
| Max Drawdown | -29.5% (-$7,379) |
| Win weeks | 64% (253/397 active weeks) |
| OOS years profitable | 5/5 (every year 2022–2026) |
| Fill validity | 100% (all market orders, audited) |
| Backtest period | April 2018 – June 2026 (8.1 years) |
| OOS period | 2022–2026 (never used in design) |

> **Important when displaying returns**: always note these are backtested
> simulation results on a $25k competition account — not live trading returns.
> The risk sizing is aggressive (2–5% per trade) appropriate for a competition context.

### What Makes It Stand Out (emphasise these)

1. **Fill validity audit** — Found and fixed a critical bug where 70–80% of
   backtested entries were impossible to fill in practice (price had already
   moved past the limit order level). Fixing this dropped WR from inflated 79.9%
   to honest 53–60%. This kind of rigour is rare and interviewers notice it.

2. **IS/OOS discipline** — 2018–2021 was in-sample, 2022–2026 was held out
   completely. OOS Sharpe actually exceeds IS Sharpe (1.34×), which is the
   opposite of typical overfitting.

3. **15 signal types explored, 11 rejected** — The research/ folder documents
   every failed experiment with honest reasons. This shows process, not just results.

4. **Live IBKR integration** — Real broker API, real data pipeline, real signal
   generation. Live trade example: June 3, 2026, CL LONG at $94.70 → closed
   $95.92 → +$610 profit (5 MCL contracts).

5. **Automated test suite** — 38/39 checks pass. Tests include: trade count
   verification, fill validity, P&L cross-check (trade-sum == daily-sum to the cent),
   IS/OOS ratio check, and year-by-year profitability.

### Five Strategies
| Strategy | Market | Freq | OOS WR | OOS Sharpe |
|----------|--------|------|--------|-----------|
| Asymmetric ORB | ES + NQ | 18/yr | 59.5% | 3.55 |
| MTF Scalp | ES | 127/yr | 44.8% | 2.03 |
| Double Test at Level | ES | 65/yr | 39.9% | 4.65 |
| Gap + ORB Alignment | ES + NQ | 33/yr | 54.3% | 2.77 |
| CL Prior-Week H/L | Crude Oil | 62/yr | 71.7% | 7.77 |

### Tech Stack
- Python 3.12: pandas, numpy, pyarrow, asyncio
- IBKR API via ib_insync
- Custom backtesting engine with bar-level fill verification
- 2.8M+ 1-minute bars from IBKR (ES, NQ, CL)

### GitHub / Code
*[Ishan to provide GitHub link — ask him for this when building the page]*

---

## How to Present Projects on the Website

### Tone & Style Guidelines
- **Be honest about limitations** — state "backtested" not "live" where applicable
- **Lead with methodology, not just returns** — sophisticated employers trust process
- **Keep stats precise** — use exact numbers from the tables above, not rounded
- **Show the process** — mention failed experiments, bug fixes, iterations
- **Professional, not salesy** — no hype words like "revolutionary" or "groundbreaking"

### Suggested Project Card Structure
```
[Project Name]
[One-line description]

Key Results:
• [2-3 most impressive verified stats]

What I Built:
• [Bullet points: specific technical contributions]
• [The bug I found / problem I solved]
• [Live deployment aspect if any]

Stack: [Technologies]
[GitHub link] [Demo/Screenshots if available]
```

### What NOT to Put on the Site
- Raw annual return percentages without context (188% looks suspicious alone)
- Sharpe ratios without noting they're computed on signal days
- Any stat that can't be verified by running the test suite
- Claims about live profitability (these are competition/simulation results)

---

## Adding New Projects

When Ishan uploads a new project, gather this information before writing anything:

1. **What does it do?** (one sentence)
2. **What problem does it solve?**
3. **What is the key technical contribution?** (not just "I built X", but "I found Y / fixed Z / improved W")
4. **What are the verified results?** (ask for specific numbers with methodology)
5. **Tech stack?**
6. **Is there a GitHub link or demo?**
7. **What audience is it for?** (quant, ML, general software, etc.)

Then write the project section following the same honest, methodology-first approach.

---

## Website Design Guidelines

### Overall
- Clean, minimal, professional — not flashy
- Dark or light theme (ask Ishan which he prefers)
- Fast loading — no heavy animations
- Mobile responsive

### Must-Have Sections
1. **Hero** — Name, tagline (1-2 sentences on focus area), contact links
2. **Projects** — Each project as a card or dedicated section
3. **Skills** — Brief, honest list of tools and languages
4. **About** — Short paragraph on background and what roles you're targeting
5. **Contact** — Email, GitHub, LinkedIn

### Project Display Recommendation
For quantitative projects: show a results table or chart screenshot if possible.
For the CME system specifically: the equity curve chart, the year-by-year table,
and a screenshot of the live terminal output (the signal generator printing a trade)
are all strong visual elements.

---

## Key Files Available

If Ishan shares any of these files, they contain the verified numbers:
- `README.md` — full project documentation with all stats
- `tests/test_systems.py` — run this to reproduce all results
- `backtest_verify_deep.py` — line-by-line verification of each system
- `backtest_joint_cl.py` — full 5-system joint performance

---

## Contact / Context

- Name: Ishan Bhardwaj
- Target roles: Quant research, systematic trading, FinTech, data science in finance
- Competition: CME Institute Competition, June 8–12, 2026
- Location: [Ishan to fill in]
- University/Background: [Ishan to fill in]

*This instruction file was created June 3, 2026.*
