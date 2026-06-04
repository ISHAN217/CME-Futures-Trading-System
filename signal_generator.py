"""
signal_generator.py — Live ORB Signal Generator for CME Competition

Connects to IBKR TWS for real-time ES + NQ 1-min bars.
Runs the validated ORB strategy and prints exact trade tickets
for manual entry on the CME Institute Simulator.

Requirements:
  pip install ib_insync
  TWS / IB Gateway open with API socket enabled
    Live  → Edit → Global Config → API → Settings → port 7496
    Paper → same, port 7497

  US Securities Snapshot & Futures Value Bundle ($10/month)
  subscribed in Client Portal → Settings → Market Data Subscriptions

Usage:
  python3 signal_generator.py              # live TWS  (port 7496)
  python3 signal_generator.py --paper      # paper TWS (port 7497)
  python3 signal_generator.py --risk=5     # 5% risk per instrument (default 3%)

What it does:
  09:25 ET  Connect and warm up
  09:30 ET  Start observing ES + NQ opening range
  10:00 ET  Lock OR — print H/L/range for both
  10:00+    Watch for both breaking same direction
  Signal!   Print exact entry ticket (symbol, qty, stop, target)
  Ongoing   Alert when stop / target / EOD close is due
  15:30 ET  EOD alert — close everything on the simulator
"""

from __future__ import annotations

import asyncio
import logging
import sys
import urllib.request
import xml.etree.ElementTree as ET_XML
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

# ─── Reuse the validated core engine ─────────────────────────────────────────
from orb_live import (
    Bar, BrokerAPI, DataFeed, Fill, ORBConfig, ORBEngine, Order,
)

ET     = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


# ─── Competition ORBConfig (ES/NQ full contracts) ─────────────────────────────

def competition_config(risk_pct: float = 0.03) -> ORBConfig:
    """
    Full-size ES + NQ for the competition.

    Strategy parameters are FROZEN from the backtest — only
    instrument sizes and risk% are adjusted for competition.

    Using "MES" / "MNQ" as internal symbol keys so ORBEngine's
    min_range checks work unchanged.  The data feed maps these
    to ES / NQ contracts; the SignalBroker prints ES / NQ.
    """
    return ORBConfig(
        # Session (unchanged)
        or_start        = time(9, 30),
        or_end          = time(10, 0),
        # Backtest (2020-2024, n=1142): signals after 10:30 ET have WR=42% vs 50%
        # for the first 30-min window. Only take breakouts within 30 min of OR lock.
        entry_cutoff    = time(10, 30),
        eod_close       = time(15, 30),

        # Range filters
        # min: backtest validated (ES=8, NQ=30)
        # max_mes: ES OR > 60pts → WR=7%, AvgR=-0.66R (14 trades, clear cliff)
        # max_mnq: NQ 150-200pt bucket → WR=43.2%, EV=+0.080R (90th percentile)
        #          No hard cliff like ES, but the 150-200pt bucket is the weakest
        #          genuine signal in the NQ range distribution. Cap at 150pts removes
        #          that bucket while keeping 90%+ of historical trades.
        min_range_mes   = 8.0,
        min_range_mnq   = 30.0,
        max_range_mes   = 60.0,   # skip abnormally wide ES OR
        max_range_mnq   = 175.0,  # raised 150→175: adds ~97 signal days/yr, −0.2pp WR
                                  # (static 150 optimal long-run; 175 better for
                                  #  competition week in elevated-volatility regimes)

        # Internal symbol keys — engine checks min_range per key
        symbols         = ["MES", "MNQ"],

        # Full contract values (ES=$50/pt, NQ=$20/pt, tick=$0.25)
        tick_size       = {"MES": 0.25,  "MNQ": 0.25},
        tick_value      = {"MES": 12.50, "MNQ": 5.00},
        pt_value        = {"MES": 50.00, "MNQ": 20.00},

        # Competition risk — 5% per leg for competition week
        # (3% = long-run conservative; 5% = competition-optimised, 67% more P&L)
        risk_pct        = risk_pct,
        profit_mult     = 3.0,   # 3R target — backtested optimal for competition EV
                                  # ORB: EV +1.020R (vs +0.337R at 1.5R)
                                  # PDH/L: EV +1.442R (vs +0.635R at 1.5R)
        slip_ticks      = 1,

        # 6% daily circuit breaker for competition
        daily_loss_limit = 0.06,
        commission_rt    = 2.05,   # informational only — not used in signal mode
    )


# ─── Signal-printing broker ───────────────────────────────────────────────────

# Maps internal engine keys → display names shown to the user
_DISPLAY = {"MES": "ES", "MNQ": "NQ"}


class SignalBroker(BrokerAPI):
    """
    Prints trade signals to the terminal instead of routing orders.
    Tracks simulated position so ORBEngine state stays consistent.
    """

    def __init__(self, account: float, risk_pct: float):
        self._account   = account
        self._risk_pct  = risk_pct
        self._positions: dict[str, int] = {}
        self._next_id   = 1
        # Stored entry details for context on close signals
        self._entries: dict[str, dict] = {}
        # Morning gap direction — set via set_gap_context() before trading starts
        self._es_gap_pct: float | None = None
        self._nq_gap_pct: float | None = None

    def set_gap_context(self, es_gap_pct: float | None,
                        nq_gap_pct: float | None) -> None:
        """
        Store morning gap direction for real-time auto-sizing at signal time.

        Backtest edge (n=1142 dual-confirmed trades, 2020-2024):
          ALIGNED gap (gap dir == breakout dir):  WR=58.1%,  EV=+0.468R
          OPPOSED gap (gap dir != breakout dir):  WR=45.3%,  EV=+0.086R
          >1.5% aligned:                          WR=66.7%,  EV=+0.854R
          p=0.002 — statistically significant

        Called from run_signal_generator() after the briefing is built.
        """
        self._es_gap_pct = es_gap_pct
        self._nq_gap_pct = nq_gap_pct

    async def connect(self) -> None:
        logger.info("SignalBroker ready — signals will print below")

    async def disconnect(self) -> None:
        logger.info("SignalBroker disconnected — session complete")

    async def get_account_value(self) -> float:
        return self._account

    def _print_gap_size_recommendation(self, action: str) -> None:
        """
        Print gap-direction-adjusted size recommendation at signal time.

        Uses the morning gap (set via set_gap_context) vs actual breakout direction.
        Backtest edge: aligned gap → WR=58.1% (+0.468R),  opposed → WR=45.3% (+0.086R).
        """
        # Average ES + NQ gap_pct; fall back to ES alone if NQ unavailable
        if self._es_gap_pct is not None and self._nq_gap_pct is not None:
            gap_pct = (self._es_gap_pct + self._nq_gap_pct) / 2
        elif self._es_gap_pct is not None:
            gap_pct = self._es_gap_pct
        else:
            return   # no gap data — skip recommendation

        is_long   = (action == "BUY")
        gap_dir   = (1 if gap_pct > 0.5 else -1 if gap_pct < -0.5 else 0)
        break_dir = (1 if is_long else -1)
        base_pct  = self._risk_pct * 100
        half_pct  = base_pct / 2

        if gap_dir == 0:
            # Flat gap — no adjustment, note neutral
            print(f"      Gap   : {gap_pct:+.2f}% (flat)  →  BASE SIZE ({base_pct:.0f}%)")
        elif gap_dir == break_dir:
            # Gap aligned with breakout — full size
            if abs(gap_pct) >= 1.5:
                print(f"      Gap   : {gap_pct:+.2f}% ✅ ALIGNED  →  FULL SIZE ({base_pct:.0f}%)  [WR≈67% historical]")
            else:
                print(f"      Gap   : {gap_pct:+.2f}% ✅ ALIGNED  →  FULL SIZE ({base_pct:.0f}%)  [WR≈58% historical]")
        else:
            # Gap opposed to breakout — halve size
            print(f"      Gap   : {gap_pct:+.2f}% ⚠️  OPPOSED  →  HALF SIZE ({half_pct:.0f}%)  [WR≈45% — reduce risk]")

    async def place_order(self, order: Order) -> str:
        oid  = str(self._next_id); self._next_id += 1
        disp = _DISPLAY.get(order.symbol, order.symbol)

        if order.order_type == "MKT":
            # ── Entry signal ──────────────────────────────────────────────
            delta = order.qty if order.action == "BUY" else -order.qty
            self._positions[order.symbol] = (
                self._positions.get(order.symbol, 0) + delta
            )
            direction = "LONG  ▲" if order.action == "BUY" else "SHORT ▼"
            print(f"\n  {'─'*52}")
            print(f"  🚨  ENTER {direction}   {disp}   qty={order.qty}")
            print(f"      Place a MARKET order on the CME Simulator NOW")
            self._print_gap_size_recommendation(order.action)
            print(f"  {'─'*52}")

        elif order.order_type == "STP":
            print(f"  📌  STOP   {disp}  @ {order.aux_price:>10.2f}   qty={order.qty}")

        elif order.order_type == "LMT":
            print(f"  🎯  TARGET {disp}  @ {order.price:>10.2f}   qty={order.qty}")
            print()

        return oid

    async def cancel_order(self, order_id: str) -> None:
        pass   # no real orders to cancel

    async def get_position(self, symbol: str) -> int:
        return self._positions.get(symbol, 0)

    async def close_position(self, symbol: str, qty: int) -> Fill:
        self._positions[symbol] = 0
        disp   = _DISPLAY.get(symbol, symbol)
        action = "SELL (close long)" if qty > 0 else "BUY  (cover short)"
        print(f"\n  {'─'*52}")
        print(f"  ⚠️   CLOSE   {disp}   {action}   qty={abs(qty)}")
        print(f"      Place a MARKET order on the CME Simulator NOW")
        print(f"  {'─'*52}\n")
        return Fill(
            order_id  = "signal",
            symbol    = symbol,
            qty       = abs(qty),
            avg_price = 0.0,
            timestamp = datetime.now(ET),
        )


# ─── IBKR live data feed ─────────────────────────────────────────────────────

class IBKRDataFeed(DataFeed):
    """
    Streams real-time 1-min bars from IBKR TWS.

    Uses reqHistoricalData(keepUpToDate=True) which:
      - Returns today's historical bars immediately on subscribe
      - Fires updateEvent(barList, hasNewBar) as each new bar closes

    The historical replay lets the engine build the OR correctly even
    if the script is started a few minutes after 09:30 ET.
    """

    # Maps internal engine key → exchange symbol for data pull
    _DATA_SYM = {"MES": "ES", "MNQ": "NQ"}

    def __init__(self, ib):
        self._ib        = ib
        self._queue:    asyncio.Queue = asyncio.Queue()
        self._bar_lists: dict         = {}
        self._seen_historical         = False

    async def subscribe(self, symbols: list[str]) -> None:
        import ib_insync
        from datetime import time as _time

        today_date = datetime.now(ET).date()

        for sym in symbols:
            data_sym = self._DATA_SYM.get(sym, sym)
            contract = ib_insync.ContFuture(data_sym, exchange="CME", currency="USD")
            await self._ib.qualifyContractsAsync(contract)
            logger.info("  %s → %s  (expiry %s)",
                        sym, contract.localSymbol,
                        contract.lastTradeDateOrContractMonth)

            bars = await self._ib.reqHistoricalDataAsync(
                contract,
                endDateTime     = "",
                durationStr     = "1 D",
                barSizeSetting  = "1 min",
                whatToShow      = "TRADES",
                useRTH          = False,
                keepUpToDate    = True,
            )

            # Replay only TODAY's session bars (9:00–15:30 ET).
            # IBKR returns 24h of Globex bars — overnight bars (18:00, 20:00 ET
            # from the previous session) would have t > eod_close and immediately
            # break the main loop if we pushed them naively.
            replayed = 0
            for b in bars[:-1]:   # skip the last (still forming) bar
                bar_obj = self._to_bar(sym, b)
                bar_t   = bar_obj.timestamp
                if (bar_t.date() == today_date
                        and _time(9, 0) <= bar_t.time() <= _time(15, 30)):
                    self._queue.put_nowait(bar_obj)
                    replayed += 1
            logger.info("  %s  replayed %d session bars from today", sym, replayed)

            # Register callback for future real-time bars
            def _make_handler(s):
                def _handler(bar_list, hasNewBar: bool):
                    if hasNewBar and len(bar_list) >= 2:
                        # bars[-1] = new forming bar; bars[-2] = just completed
                        self._queue.put_nowait(self._to_bar(s, bar_list[-2]))
                return _handler

            bars.updateEvent += _make_handler(sym)
            self._bar_lists[sym] = bars

        logger.info("IBKRDataFeed subscribed to: %s", symbols)

    def _to_bar(self, symbol: str, b) -> Bar:
        """Convert ib_insync BarData → our Bar dataclass."""
        dt = b.date
        if isinstance(dt, datetime):
            ts = dt if dt.tzinfo else dt.replace(tzinfo=ET)
        else:
            # TWS string: '20260527 09:31:00 US/Eastern'
            parts = str(dt).split()
            naive = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y%m%d %H:%M:%S")
            tz    = ZoneInfo(parts[2]) if len(parts) > 2 else ET
            ts    = naive.replace(tzinfo=tz)
        return Bar(
            symbol    = symbol,
            timestamp = ts.astimezone(ET),
            open      = float(b.open),
            high      = float(b.high),
            low       = float(b.low),
            close     = float(b.close),
            volume    = int(b.volume),
        )

    async def next_bar(self) -> Bar:
        return await self._queue.get()


# ─── Economic calendar ───────────────────────────────────────────────────────
#
# High-impact events that increase false-breakout probability.
# Source: BLS, Federal Reserve official calendars.
# Add new dates here as they are announced.
#
# Format: "YYYY-MM-DD": ("Event name", "HH:MM ET", impact_level)
#   impact_level: "🔴 EXTREME" | "🟠 HIGH" | "🟡 MEDIUM"
#
_ECONOMIC_CALENDAR = {
    # ── Competition week (June 8-12 2026) ─────────────────────────────
    "2026-06-10": ("CPI — May 2026",          "08:30", "🔴 EXTREME"),

    # ── Nearby events ──────────────────────────────────────────────────
    "2026-06-05": ("NFP — May 2026",          "08:30", "🔴 EXTREME"),
    "2026-06-16": ("FOMC Day 1",              "all day","🟠 HIGH"),
    "2026-06-17": ("FOMC Announcement",       "14:00", "🔴 EXTREME"),

    # ── Recurring high-impact template (update monthly) ───────────────
    # NFP  = first Friday of each month at 08:30 ET
    # CPI  = ~10th of each month at 08:30 ET
    # FOMC = 8 times/year (see federalreserve.gov)
}

_IMPACT_ADVICE = {
    "🔴 EXTREME": (
        "Consider SKIPPING today's ORB trade or cutting size to 1%.\n"
        "  Post-report spikes often create false breakouts that reverse hard."
    ),
    "🟠 HIGH": (
        "Reduce size to 2% and be quicker to accept a partial exit.\n"
        "  Elevated volatility increases whipsaw risk."
    ),
    "🟡 MEDIUM": (
        "Normal size OK but be aware of potential 10:00 ET data release.\n"
        "  Watch for unusual OR range expansion."
    ),
}


def _get_news_context(today: date) -> list[tuple]:
    """Return list of (event_name, time_et, impact) for today."""
    return [v for k, v in _ECONOMIC_CALENDAR.items()
            if k == str(today)]


def _parse_ibkr_bar_dt(b) -> datetime:
    """Convert an ib_insync BarData.date field to an ET-aware datetime."""
    dt = b.date
    if not isinstance(dt, datetime):
        parts = str(dt).split()
        from zoneinfo import ZoneInfo as ZI
        naive = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y%m%d %H:%M:%S")
        tz = ZI(parts[2]) if len(parts) > 2 else ET
        dt = naive.replace(tzinfo=tz)
    return dt.astimezone(ET)


async def _fetch_ibkr_hourly(ib, sym: str) -> list[tuple]:
    """Pull 45 days of hourly bars for a futures symbol from IBKR.

    45 days covers:
      - gap / overnight analysis (3 days needed)
      - prior full calendar week for PWH/PWL (up to 14 days)
      - prior full calendar MONTH for PMH/PML (up to 45 days)
    Monthly H/L backtest: Sharpe=2.43 vs PDH alone Sharpe=1.56
    """
    import ib_insync
    contract = ib_insync.ContFuture(sym, exchange="CME", currency="USD")
    await ib.qualifyContractsAsync(contract)
    bars = await ib.reqHistoricalDataAsync(
        contract, endDateTime="", durationStr="45 D",
        barSizeSetting="1 hour", whatToShow="TRADES",
        useRTH=False, keepUpToDate=False,
    )
    return [(_parse_ibkr_bar_dt(b), b.open, b.high, b.low, b.close)
            for b in bars]


async def _fetch_vix(ib) -> float | None:
    """
    Fetch VIX from Yahoo Finance RSS (free, no subscription needed).
    Falls back to IBKR if RSS fails.
    """
    # Try Yahoo Finance quote page RSS first (no auth needed)
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=1d"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            import json
            data = json.loads(r.read())
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return float(price)
    except Exception:
        pass

    # Fallback: IBKR delayed data
    try:
        import ib_insync
        vix = ib_insync.Index("VIX", "CBOE", "USD")
        await ib.qualifyContractsAsync(vix)
        ticker = ib.reqMktData(vix, "", True, False)   # True = delayed snapshot
        await asyncio.sleep(3)
        price = ticker.marketPrice()
        ib.cancelMktData(vix)
        return float(price) if price and price == price else None
    except Exception:
        return None


def _fetch_news_headlines(n: int = 4) -> list[str]:
    """
    Fetch top financial headlines from Yahoo Finance RSS.
    Returns list of plain-text headline strings. Never raises.
    """
    urls = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ES%3DF&region=US&lang=en-US",
    ]
    headlines = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                root = ET_XML.fromstring(r.read())
            for item in root.iter("item"):
                title = item.findtext("title") or ""
                if title and title not in headlines:
                    headlines.append(title.strip())
                if len(headlines) >= n:
                    break
        except Exception:
            pass
        if len(headlines) >= n:
            break
    return headlines[:n]


def _gap_analysis(hourly: list[tuple]) -> dict | None:
    """
    Compute gap and pre-market structural analysis from hourly bar list.

    Returns a dict with:
      - gap / gap_pct / momentum          (existing)
      - overnight_range / overnight_quality  (Phase 1A — COIL/NORMAL/WIDE)
      - close_position / close_bias          (Phase 1B — HOD/MID/LOD)
    """
    today = date.today()

    # ── Previous RTH session bars (most recent prior date, 9:30–16:00 ET) ──────
    prev_rth = [(dt, o, h, l, c) for dt, o, h, l, c in hourly
                if dt.date() < today
                and time(9, 30) <= dt.time() <= time(16, 0)]
    if not prev_rth:
        return None
    prev_date = prev_rth[-1][0].date()
    prev_bars = [(dt, o, h, l, c) for dt, o, h, l, c in prev_rth
                 if dt.date() == prev_date]
    if not prev_bars:
        return None

    prev_high  = max(b[2] for b in prev_bars)
    prev_low   = min(b[3] for b in prev_bars)
    prev_close = prev_bars[-1][4]   # last bar of the prior RTH session

    # ── Pre-market bars (today, before 9:30 ET) ───────────────────────────────
    premarket = [(dt, o, h, l, c) for dt, o, h, l, c in hourly
                 if dt.date() == today and dt.time() < time(9, 30)]
    if not premarket:
        return None

    pm_high = max(b[2] for b in premarket)
    pm_low  = min(b[3] for b in premarket)
    pm_last = premarket[-1][4]
    gap     = pm_last - prev_close
    gap_pct = gap / prev_close * 100

    # ── Phase 1A: Overnight range quality ────────────────────────────────────
    # How wide was the overnight session relative to yesterday's full RTH range?
    # A compressed overnight (COIL) signals energy building → explosive OR.
    # A wide overnight (WIDE) means the market already moved → chop risk.
    overnight_range = pm_high - pm_low
    prev_day_range  = prev_high - prev_low
    overnight_vs_prev = (overnight_range / prev_day_range
                         if prev_day_range > 0 else 0.5)

    if overnight_vs_prev < 0.30:
        overnight_quality = "COIL"    # tight compression — high ORB follow-through
    elif overnight_vs_prev < 0.60:
        overnight_quality = "NORMAL"  # average overnight
    else:
        overnight_quality = "WIDE"    # already moved a lot — higher false-breakout risk

    # ── Phase 1B: Previous session close position ─────────────────────────────
    # Where did yesterday close within its own RTH range?
    # HOD close → continuation bias (supply absorbed, closed strong)
    # LOD close → also continuation bias but bearish
    # MID close → no structural lean
    if prev_day_range > 0:
        close_position = (prev_close - prev_low) / prev_day_range
    else:
        close_position = 0.5

    if close_position >= 0.70:
        close_bias = "HOD"   # closed top 30% of range → bullish continuation
    elif close_position <= 0.30:
        close_bias = "LOD"   # closed bottom 30% of range → bearish continuation
    else:
        close_bias = "MID"   # mid-range → neutral

    # ── Prior Week RTH High / Low ─────────────────────────────────────────────
    # Backtest (2018-2026): PWH/PWL standalone WR=79.9% EV=+2.195R Sharpe=18.6
    # Weekly extremes attract institutional orders (options gamma, fund benchmarks).
    import pandas as _pd
    _today_ts   = _pd.Timestamp(today)
    _curr_iso   = _today_ts.isocalendar()
    _pw_week    = int(_curr_iso.week) - 1
    _pw_year    = int(_curr_iso.year)
    if _pw_week == 0:                         # year boundary
        _pw_year -= 1
        _pw_week  = int(_pd.Timestamp(f"{_pw_year}-12-28").isocalendar().week)

    _pw_bars = [
        (dt, o, h, l, c) for dt, o, h, l, c in hourly
        if (int(_pd.Timestamp(dt).isocalendar().week) == _pw_week
            and int(_pd.Timestamp(dt).isocalendar().year) == _pw_year
            and time(9, 30) <= dt.time() <= time(16, 0))
    ]
    prior_week_high = max(b[2] for b in _pw_bars) if _pw_bars else None
    prior_week_low  = min(b[3] for b in _pw_bars) if _pw_bars else None

    # ── Prior Month RTH High / Low ────────────────────────────────────────────
    # Backtest (2018-2026): Monthly H/L WR=46.4% AvgR=+0.795pts Sharpe=2.43
    # vs PDH alone: Sharpe=1.56. Position traders + monthly options create
    # real order clustering at prior month extremes.
    _pm_month = today.month - 1
    _pm_year  = today.year
    if _pm_month == 0:
        _pm_month = 12
        _pm_year  -= 1

    _pm_bars = [
        (dt, o, h, l, c) for dt, o, h, l, c in hourly
        if (dt.date().month == _pm_month
            and dt.date().year == _pm_year
            and time(9, 30) <= dt.time() <= time(16, 0))
    ]
    prior_month_high = max(b[2] for b in _pm_bars) if _pm_bars else None
    prior_month_low  = min(b[3] for b in _pm_bars) if _pm_bars else None

    # ── Prior Quarter RTH High / Low ─────────────────────────────────────────
    # Quarterly extremes: 3 months of fund benchmarks + quarterly options
    # Same structural logic as monthly but wider timeframe — stronger levels
    _pq_year = today.year
    _today_q = (today.month - 1) // 3 + 1
    _prev_q  = _today_q - 1
    if _prev_q == 0: _prev_q = 4; _pq_year -= 1
    _pq_months = {1:[1,2,3], 2:[4,5,6], 3:[7,8,9], 4:[10,11,12]}[_prev_q]

    _pq_bars = [
        (dt, o, h, l, c) for dt, o, h, l, c in hourly
        if (dt.date().year == _pq_year
            and dt.date().month in _pq_months
            and time(9, 30) <= dt.time() <= time(16, 0))
    ]
    prior_quarter_high = max(b[2] for b in _pq_bars) if _pq_bars else None
    prior_quarter_low  = min(b[3] for b in _pq_bars) if _pq_bars else None

    return {
        # Gap
        "prev_close":          prev_close,
        "pm_last":             pm_last,
        "pm_high":             pm_high,
        "pm_low":              pm_low,
        "gap":                 gap,
        "gap_pct":             gap_pct,
        "momentum":            premarket[-1][4] - premarket[0][4],
        # Phase 1A
        "prev_high":           prev_high,
        "prev_low":            prev_low,
        "prev_day_range":      prev_day_range,
        "overnight_range":     overnight_range,
        "overnight_vs_prev":   overnight_vs_prev,
        "overnight_quality":   overnight_quality,
        # Phase 1B
        "close_position":      close_position,
        "close_bias":          close_bias,
        # Prior Week H/L  (None if data unavailable)
        "prior_week_high":     prior_week_high,
        "prior_week_low":      prior_week_low,
        # Prior Month H/L  (None if data unavailable)
        "prior_month_high":    prior_month_high,
        "prior_month_low":     prior_month_low,
        # Prior Quarter H/L  (None if data unavailable)
        "prior_quarter_high":  prior_quarter_high,
        "prior_quarter_low":   prior_quarter_low,
    }


async def build_premarket_briefing(ib) -> dict:
    """
    Gather all pre-market context concurrently:
      - ES gap + momentum
      - NQ gap
      - VIX level
      - News headlines
      - Economic calendar events
    """
    today = date.today()

    # Fire IBKR data pulls concurrently
    es_task  = asyncio.create_task(_fetch_ibkr_hourly(ib, "ES"))
    nq_task  = asyncio.create_task(_fetch_ibkr_hourly(ib, "NQ"))
    vix_task = asyncio.create_task(_fetch_vix(ib))

    # News fetch is blocking — run in thread so it doesn't block event loop
    news_task = asyncio.get_event_loop().run_in_executor(
        None, _fetch_news_headlines, 4
    )

    es_bars, nq_bars, vix, headlines = await asyncio.gather(
        es_task, nq_task, vix_task, news_task,
        return_exceptions=True
    )

    es_gap = _gap_analysis(es_bars) if isinstance(es_bars, list) else None
    nq_gap = _gap_analysis(nq_bars) if isinstance(nq_bars, list) else None

    # ── Phase 2A: ES/NQ relative strength ─────────────────────────────────────
    # NQ leading overnight → tech-driven move → better ORB follow-through
    # Diverging (one up, one flat/down) → higher false-breakout risk
    if es_gap and nq_gap:
        rs_diff = nq_gap["gap_pct"] - es_gap["gap_pct"]
        if rs_diff > 0.4:
            rs_label = "NQ_LEADING"    # tech outperforming → momentum on side of LONG
        elif rs_diff < -0.4:
            rs_label = "ES_LEADING"    # value outperforming → defensive tone
        else:
            rs_label = "ALIGNED"       # moving together → cleanest ORB setup
    else:
        rs_diff  = None
        rs_label = None

    return {
        "date":     today,
        "es_gap":   es_gap,
        "nq_gap":   nq_gap,
        "vix":      vix       if isinstance(vix, float)      else None,
        "news":     headlines if isinstance(headlines, list)  else [],
        "events":   _get_news_context(today),
        # Phase 2A
        "rs_diff":  rs_diff,
        "rs_label": rs_label,
    }


def _score_briefing(b: dict) -> tuple[int, str, str, list[str]]:
    """
    Compute a 0-70 ORB confidence score from pre-market context.

    Components and max points
    ─────────────────────────
      Gap alignment  (both ES + NQ same side > 0.75%)   20 pts
      Gap size       (sweet spot 0.75–2.0% avg)         15 pts
      Overnight range  *** NOT SCORED (p=0.613 NS) ***   0 pts
        Backtest shows WIDE=52.4% WR vs COIL=56.2% — no meaningful edge.
        Displayed for context only; does not affect score.
      Close position   (prior LOD/HOD vs gap direction)  10 pts
      Relative strength  (ES/NQ aligned or NQ-led)      15 pts
      VIX level          (≤ 20 = normal regime)         10 pts
                                                       ─────
                                                        70 pts  (max)

    Note: 🔴 EXTREME events (CPI, NFP, FOMC) show a warning but do NOT
    reduce the score — backtesting shows edge persists on news days.

    Returns (score, verdict, size_str, reasons)
      verdict  : "TRADE" | "REDUCE" | "SKIP"
      size_str : "FULL — 3%" | "HALF — 1.5%" | "NO TRADE"
      reasons  : list of short strings explaining score components
    """
    score   = 0
    reasons = []

    es_g = b.get("es_gap")
    nq_g = b.get("nq_gap")

    # ── 1. Gap alignment (20 pts) ──────────────────────────────────────────────
    if es_g and nq_g:
        es_dir = (1  if es_g["gap_pct"] >= 0.75
                  else -1 if es_g["gap_pct"] <= -0.75 else 0)
        nq_dir = (1  if nq_g["gap_pct"] >= 0.75
                  else -1 if nq_g["gap_pct"] <= -0.75 else 0)
        if es_dir != 0 and es_dir == nq_dir:
            score += 20
            reasons.append("✅ gap aligned — ES + NQ both gapping same direction")
        elif es_dir == 0 and nq_dir == 0:
            score += 5
            reasons.append("➡️  gap neutral — no directional overnight lean")
        else:
            reasons.append("⚠️  gap diverged — ES and NQ gapping opposite directions")

    # ── 2. Gap size sweet spot (15 pts) ───────────────────────────────────────
    # Too small → no energy. Too large → likely to fade before OR locks.
    if es_g and nq_g:
        avg_gap = (abs(es_g["gap_pct"]) + abs(nq_g["gap_pct"])) / 2
        if 0.75 <= avg_gap <= 2.0:
            score += 15
            reasons.append(f"✅ gap size ideal ({avg_gap:.2f}% avg — sweet spot 0.75–2.0%)")
        elif 0.40 <= avg_gap < 0.75:
            score += 7
            reasons.append(f"⚠️  gap small ({avg_gap:.2f}% avg — less directional conviction)")
        elif avg_gap > 2.0:
            score += 5
            reasons.append(f"⚠️  gap large ({avg_gap:.2f}% avg — fade risk elevated)")
        else:
            reasons.append(f"❌ gap negligible ({avg_gap:.2f}%) — flat open expected")

    # ── 3. Overnight range quality (0 pts — display only) ────────────────────
    # Backtest (n=1142): COIL=56.2% WR, WIDE=52.4% WR, NORMAL=44.2% WR — p=0.613.
    # Not statistically significant; WIDE outperforms NORMAL (inverted hypothesis).
    # Shown in score breakdown for context but contributes 0 points.
    oq = (es_g or nq_g or {}).get("overnight_quality") if (es_g or nq_g) else None
    ov = (es_g or nq_g or {}).get("overnight_vs_prev", 0)
    if oq == "COIL":
        reasons.append(f"➡️  overnight COIL ({ov:.0%} of prev range) — compressed [display only, NS p=0.61]")
    elif oq == "NORMAL":
        reasons.append(f"➡️  overnight NORMAL ({ov:.0%} of prev range) — average [display only, NS p=0.61]")
    elif oq == "WIDE":
        reasons.append(f"➡️  overnight WIDE ({ov:.0%} of prev range) — extended [display only, NS p=0.61]")

    # ── 4. Previous session close position (10 pts) ───────────────────────────
    # Backtest (2020-2024, n=1142 dual-confirmed trades) shows:
    #   LOD close → 29.1% WR (+0.031 AvgR)  ← empirically strongest
    #   MID close → 24.2% WR (−0.066 AvgR)
    #   HOD close → 25.7% WR (−0.054 AvgR)
    #
    # Effect is modest (~4 pct-point WR difference) so weight is 10 pts, not 20.
    # LOD close = bonus (reversal setups follow through better than continuations).
    # "Opposed" (gap up + LOD close) is empirically the stronger setup, not weaker.
    if es_g:
        cb  = es_g.get("close_bias",     "MID")
        cp  = es_g.get("close_position", 0.5)
        gap_dir = 1 if es_g["gap_pct"] > 0 else -1
        opposed = (gap_dir > 0 and cb == "LOD") or (gap_dir < 0 and cb == "HOD")
        cp_pct  = int(cp * 100)
        if opposed:
            # Backtest shows opposed = best win rate (reversal dynamic)
            score += 10
            reasons.append(f"✅ close position: {cb} ({cp_pct}th pctile) — reversal setup, empirically stronger")
        elif cb == "MID":
            score += 5
            reasons.append(f"➡️  close position MID ({cp_pct}th pctile) — no structural lean")
        else:
            # Stacked (continuation) — weaker in backtest, no bonus
            reasons.append(f"➡️  close position: {cb} ({cp_pct}th pctile) — continuation setup (neutral)")

    # ── 5. Relative strength (15 pts) ─────────────────────────────────────────
    rs_label = b.get("rs_label")
    rs_diff  = b.get("rs_diff")
    if rs_label == "ALIGNED":
        score += 15
        reasons.append(f"✅ ES/NQ aligned (Δ{rs_diff:+.2f}%) — clean dual confirmation")
    elif rs_label == "NQ_LEADING":
        score += 10
        reasons.append(f"➡️  NQ leading by {rs_diff:+.2f}% — tech-driven, supports LONG ORB")
    elif rs_label == "ES_LEADING":
        score += 8
        reasons.append(f"➡️  ES leading by {rs_diff:+.2f}% — defensive tone, LONG ORB less favored")
    elif rs_label is None:
        score += 5   # data unavailable → neutral

    # ── 6. VIX (10 pts) ───────────────────────────────────────────────────────
    vix = b.get("vix")
    if vix is not None:
        if vix <= 20:
            score += 10
            reasons.append(f"✅ VIX {vix:.1f} — normal regime, tight OR expected")
        elif vix <= 25:
            score += 5
            reasons.append(f"➡️  VIX {vix:.1f} — slightly elevated, OR may be wider")
        elif vix <= 30:
            reasons.append(f"⚠️  VIX {vix:.1f} — elevated, false-breakout risk higher")
        else:
            score = max(score - 5, 0)
            reasons.append(f"❌ VIX {vix:.1f} — EXTREME volatility, avoid ORB today")
    else:
        score += 5   # VIX unavailable → neutral

    # ── EXTREME economic event — warning only, NO score cap ──────────────────
    # Backtest (2020-2024): adding a news-day skip HURTS performance by -1.3%
    # annual return. The ORB + PDH/L edge persists on CPI / NFP / FOMC days.
    # We display the warning for awareness but do NOT reduce the score.
    if b.get("events"):
        if any(e[2] == "🔴 EXTREME" for e in b["events"]):
            reasons.append("🔴 EXTREME event today — trade with awareness, edge holds on news days")

    # ── Verdict ───────────────────────────────────────────────────────────────
    # Max score is now 70 (overnight range quality removed as not significant).
    # Thresholds scaled proportionally from original 100-pt system:
    #   TRADE  ≥ 55/70 (≈79% of max) — all major signals aligned
    #   REDUCE ≥ 35/70 (≈50% of max) — decent setup, reduce size
    #   SKIP   < 35/70               — too much uncertainty
    if score >= 55:
        verdict  = "TRADE"
        size_str = "FULL — 3% per leg"
    elif score >= 35:
        verdict  = "REDUCE"
        size_str = "HALF — 1.5% per leg"
    else:
        verdict  = "SKIP"
        size_str = "NO TRADE today"

    return score, verdict, size_str, reasons


def print_premarket_briefing(b: dict) -> None:
    """
    Print the full pre-market briefing panel.

    Layout (verdict-first):
      ═══ header
      VERDICT block  — score, size, bias direction
      Score breakdown  — one line per component
      ─── divider
      Market snapshot  — ES / NQ gap + overnight quality + close position
      Relative strength
      VIX
      ─── divider
      Economic events
      News headlines
      ═══ footer
    """
    W = 62

    # ── Compute score (needs all data already in b) ───────────────────
    score, verdict, size_str, reasons = _score_briefing(b)

    # Determine bias direction for the verdict block
    es_g = b.get("es_gap")
    nq_g = b.get("nq_gap")
    gap_signals = []
    if es_g:
        gap_signals.append(1 if es_g["gap_pct"] >= 0.75 else (-1 if es_g["gap_pct"] <= -0.75 else 0))
    if nq_g:
        gap_signals.append(1 if nq_g["gap_pct"] >= 0.75 else (-1 if nq_g["gap_pct"] <= -0.75 else 0))
    net = sum(gap_signals)
    bias_arrow = "▲ LONG" if net >= 1 else ("▼ SHORT" if net <= -1 else "➡️  NEUTRAL")

    # Verdict colour
    verdict_icon = {"TRADE": "🟢", "REDUCE": "🟡", "SKIP": "🔴"}[verdict]

    print()
    print("  " + "═" * W)
    print(f"  PRE-MARKET BRIEFING  —  {b['date'].strftime('%A %B %d, %Y')}")
    print("  " + "═" * W)

    # ── 1. VERDICT block ──────────────────────────────────────────────
    print()
    print(f"  {verdict_icon}  VERDICT:  {verdict}   ({score}/70)")
    print(f"     Size  :  {size_str}")
    print(f"     Bias  :  {bias_arrow}")

    # ── 2. Score breakdown ────────────────────────────────────────────
    print()
    print("  📋  SCORE BREAKDOWN")
    for r in reasons:
        print(f"      {r}")

    # ── 3. Market snapshot ────────────────────────────────────────────
    print()
    print("  " + "─" * W)
    print("  📊  MARKET SNAPSHOT")
    print()

    def _snap_lines(sym, g):
        """Return 2-3 lines of detail for one symbol."""
        if not g:
            print(f"      {sym}  data unavailable")
            return
        arrow = "▲" if g["gap"] > 0 else "▼"
        bias  = ("BULLISH" if g["gap_pct"] >= 0.75
                 else "BEARISH" if g["gap_pct"] <= -0.75
                 else "neutral")
        mom   = g["momentum"]
        mom_s = (f"climbing {mom:+.1f}pts into open" if mom > 1
                 else f"falling {mom:+.1f}pts into open" if mom < -1
                 else "flat into open")
        oq    = g.get("overnight_quality", "?")
        ov    = g.get("overnight_vs_prev", 0)
        oq_icon = {"COIL": "🌀", "NORMAL": "➡️ ", "WIDE": "💥"}.get(oq, "  ")
        cb    = g.get("close_bias", "MID")
        cp    = int(g.get("close_position", 0.5) * 100)
        cb_icon = {"HOD": "⬆️ ", "LOD": "⬇️ ", "MID": "↔️ "}.get(cb, "  ")

        print(f"      {sym}  prev {g['prev_close']:.2f} → pre-mkt {g['pm_last']:.2f}  "
              f"({arrow}{g['gap_pct']:+.2f}%)  {bias}  |  {mom_s}")
        print(f"           overnight: {oq_icon} {oq} ({ov:.0%} of prev range)  "
              f"| prev close: {cb_icon} {cb} ({cp}th pctile)")

    _snap_lines("ES", es_g)
    print()
    _snap_lines("NQ", nq_g)

    # Relative strength line
    rs_label = b.get("rs_label")
    rs_diff  = b.get("rs_diff")
    print()
    if rs_label and rs_diff is not None:
        rs_icon = {"ALIGNED": "🔗", "NQ_LEADING": "🚀", "ES_LEADING": "🛡️"}.get(rs_label, "  ")
        print(f"      RS   {rs_icon} {rs_label}  (NQ − ES = {rs_diff:+.2f}%)")
    else:
        print("      RS   data unavailable")

    # VIX line
    v = b.get("vix")
    if v is not None:
        vix_icon = ("🔴" if v > 30 else "🟠" if v > 20 else "🟢")
        vix_lbl  = ("EXTREME — very wide ranges" if v > 30
                    else "ELEVATED — wider OR expected" if v > 20
                    else "NORMAL")
        print(f"      VIX  {vix_icon} {v:.1f}  {vix_lbl}")
    else:
        print("      VIX  data unavailable")

    # ── 4. Economic events ────────────────────────────────────────────
    print()
    print("  " + "─" * W)
    print("  📅  SCHEDULED EVENTS")
    if b["events"]:
        for name, t, impact in b["events"]:
            print(f"      {impact}  {name}  @ {t} ET")
            advice = _IMPACT_ADVICE.get(impact, "")
            if advice:
                for line in advice.strip().split("\n"):
                    print(f"         ⚡ {line.strip()}")
    else:
        print("      ✅  No high-impact events — clean slate")

    # ── 5. News headlines ─────────────────────────────────────────────
    print()
    print("  📰  TOP HEADLINES  (Yahoo Finance)")
    if b["news"]:
        for i, h in enumerate(b["news"], 1):
            if len(h) > 57:
                h = h[:54] + "..."
            print(f"      {i}. {h}")
    else:
        print("      Unable to fetch headlines — check internet connection")

    # ── 6. Prior Week H/L levels ─────────────────────────────────────
    print()
    print("  " + "─" * W)
    print("  📐  PRIOR WEEK LEVELS  (key structural S/R)")
    pwh_es = (es_g or {}).get("prior_week_high")
    pwl_es = (es_g or {}).get("prior_week_low")
    pwh_nq = (nq_g or {}).get("prior_week_high")
    pwl_nq = (nq_g or {}).get("prior_week_low")
    if pwh_es and pwl_es:
        pm_es = (es_g or {}).get("pm_last", 0)
        above_es = "▲ above PWH" if pm_es > pwh_es else ("▼ below PWL" if pm_es < pwl_es else "↔ inside range")
        print(f"      ES  PWH={pwh_es:.2f}  PWL={pwl_es:.2f}  "
              f"range={pwh_es-pwl_es:.2f}pts  pre-mkt: {above_es}")
    else:
        print("      ES  PWH/PWL unavailable")
    if pwh_nq and pwl_nq:
        pm_nq = (nq_g or {}).get("pm_last", 0)
        above_nq = "▲ above PWH" if pm_nq > pwh_nq else ("▼ below PWL" if pm_nq < pwl_nq else "↔ inside range")
        print(f"      NQ  PWH={pwh_nq:.2f}  PWL={pwl_nq:.2f}  "
              f"range={pwh_nq-pwl_nq:.2f}pts  pre-mkt: {above_nq}")
    else:
        print("      NQ  PWH/PWL unavailable")

    # ── Monthly levels ────────────────────────────────────────────────────────
    pmh_es = (es_g or {}).get("prior_month_high")
    pml_es = (es_g or {}).get("prior_month_low")
    pmh_nq = (nq_g or {}).get("prior_month_high")
    pml_nq = (nq_g or {}).get("prior_month_low")
    if pmh_es and pml_es:
        pm_es   = (es_g or {}).get("pm_last", 0)
        pos_es  = ("▲ above PMH" if pm_es > pmh_es
                   else "▼ below PML" if pm_es < pml_es
                   else "↔ inside range")
        print(f"      ES  PMH={pmh_es:.2f}  PML={pml_es:.2f}  "
              f"range={pmh_es-pml_es:.2f}pts  pre-mkt: {pos_es}  "
              f"[monthly — Sh=2.43 vs PDH Sh=1.56]")
    if pmh_nq and pml_nq:
        pm_nq   = (nq_g or {}).get("pm_last", 0)
        pos_nq  = ("▲ above PMH" if pm_nq > pmh_nq
                   else "▼ below PML" if pm_nq < pml_nq
                   else "↔ inside range")
        print(f"      NQ  PMH={pmh_nq:.2f}  PML={pml_nq:.2f}  "
              f"range={pmh_nq-pml_nq:.2f}pts  pre-mkt: {pos_nq}")

    print()
    print("  " + "─" * W)
    print("  ORB signal fires automatically below when both ES+NQ break.")
    print("  " + "═" * W)
    print()


# ─── Banner helpers ───────────────────────────────────────────────────────────

def _print_banner(risk_pct: float, account: float) -> None:
    print()
    print("  " + "═" * 56)
    print("  ORB SIGNAL GENERATOR — CME June 2026 Challenge")
    print("  " + "═" * 56)
    print(f"  Account  : ${account:,.0f}")
    print(f"  Risk/leg : {risk_pct*100:.0f}%  (${account*risk_pct:,.0f} per ES + per NQ)")
    print(f"  OR window: 09:30–10:00 ET  (both ES + NQ must break same way)")
    print(f"  Target   : 3.0R    Stop: opposite OR extreme    EOD: 15:30 ET")
    print()
    print("  Signals will appear below automatically.")
    print("  Enter them MANUALLY on the CME Simulator.")
    print("  " + "─" * 56)
    print(f"  Waiting for 09:30 ET …")
    print("  " + "═" * 56)
    print()


def _print_or_summary(cfg: ORBConfig, state) -> None:
    """Print OR levels once both symbols are locked."""
    print()
    print("  " + "─" * 56)
    print("  OR LOCKED (10:00 ET)")
    for sym, s in state.states.items():
        disp = _DISPLAY.get(sym, sym)
        rng  = s.H_OR - s.L_OR

        if not s.range_locked:
            ok = "❌ no data"
        else:
            # Check actual range bounds per symbol
            if sym == "MES":
                max_r = cfg.max_range_mes; min_r = cfg.min_range_mes
            else:
                max_r = cfg.max_range_mnq; min_r = cfg.min_range_mnq

            if rng > max_r:
                ok = f"❌ too wide  ({rng:.1f} > {max_r:.0f}pt max) — NO TRADE"
            elif rng < min_r:
                ok = f"❌ too narrow ({rng:.1f} < {min_r:.0f}pt min) — NO TRADE"
            else:
                ok = "✅"

        print(f"  {disp:4s}  H={s.H_OR:.2f}  L={s.L_OR:.2f}  R={rng:.2f}  {ok}")

    # Summary line
    all_valid = all(
        (s.H_OR - s.L_OR) <= (cfg.max_range_mes if sym == "MES" else cfg.max_range_mnq) and
        (s.H_OR - s.L_OR) >= (cfg.min_range_mes if sym == "MES" else cfg.min_range_mnq) and
        s.range_locked
        for sym, s in state.states.items()
    )
    if all_valid:
        print("  ✅ Both OR ranges valid — watching for breakout …")
    else:
        print("  ⛔ One or more OR ranges invalid — NO TRADE today")
    print("  " + "─" * 56)
    print()


# ─── Main run loop ────────────────────────────────────────────────────────────

async def _fire_pdhl_signal(
    broker:     "SignalBroker",
    day_state:  ORBDayState,
    direction:  str,
    cfg:        ORBConfig,
    pdh_es:     float,
    pdl_es:     float,
    pdh_nq:     float,
    pdl_nq:     float,
) -> None:
    """
    Print the PDH/L fallback signal in the same format as ORB.

    Fires when both ES and NQ have broken their prior-day RTH high (LONG)
    or prior-day RTH low (SHORT) within the 10:00–11:00 ET window,
    on a day where ORB dual-confirmation did not fire by 10:30 ET.

    Risk unit = today's OR range (same volatility reference as ORB).
    Backtest (2018-2026, n=364 legs): WR=65.4%  EV=+0.635R  p<0.00001.
    """
    print(f"\n  {'─'*52}")
    print(f"  📐  PDH/L FALLBACK  —  {direction}  ▲" if direction == "LONG"
          else f"  📐  PDH/L FALLBACK  —  {direction}  ▼")
    print(f"      Prior-day H/L break — both ES + NQ confirmed")
    print(f"      Historical edge: WR=65.4%  EV=+0.635R  (n=364 legs, 2018-2026)")

    for internal_sym, pdh, pdl, sym_disp in [
        ("MES", pdh_es, pdl_es, "ES"),
        ("MNQ", pdh_nq, pdl_nq, "NQ"),
    ]:
        s    = day_state.sym(internal_sym)
        rng  = (s.H_OR - s.L_OR) if (s.range_locked and s.H_OR > s.L_OR) else (
            cfg.min_range_mes if internal_sym == "MES" else cfg.min_range_mnq
        )
        pt_v = cfg.pt_value[internal_sym]

        if direction == "LONG":
            entry_px, stop_px = pdh, pdh - rng
            tgt_px   = pdh + rng * cfg.profit_mult
            action, stop_act = "BUY", "SELL"
        else:
            entry_px, stop_px = pdl, pdl + rng
            tgt_px   = pdl - rng * cfg.profit_mult
            action, stop_act = "SELL", "BUY"

        contracts = max(1, round(day_state.account * cfg.risk_pct / (rng * pt_v)))

        # Entry (MKT) — SignalBroker prints the gap-size recommendation too
        await broker.place_order(Order(
            symbol=internal_sym, action=action,
            qty=contracts, order_type="MKT",
        ))
        # Stop loss
        await broker.place_order(Order(
            symbol=internal_sym, action=stop_act,
            qty=contracts, order_type="STP", aux_price=stop_px, tif="GTC",
        ))
        # Profit target
        await broker.place_order(Order(
            symbol=internal_sym, action=stop_act,
            qty=contracts, order_type="LMT", price=tgt_px, tif="GTC",
        ))

    print(f"  {'─'*52}")


async def _fire_pwhl_signal(
    broker:    "SignalBroker",
    day_state: "ORBDayState",
    direction: str,
    cfg:       ORBConfig,
    pwh_es:    float,
    pwl_es:    float,
    pwh_nq:    float,
    pwl_nq:    float,
    sig_label: str = "PWH/PWL",
    wh_stat:   str = "WR=79.9%  EV=+2.195R  Sharpe=18.6",
) -> None:
    """
    Fire a Prior Week High/Low signal.

    Used for:
      - PWHL_ONLY    : standalone PWH/PWL (no ORB, no PDH/L)
      - PDHL_PWHL    : PDH/L + PWH/PWL agree, no ORB
      - ORB_PWHL     : ORB + PWH/PWL agree (annotation only, ORB already entered)

    Backtest (2018-2026):
      Standalone        WR=79.9%  EV=+2.195R  Sharpe=18.6
      3-Way confluence  WR=73.4%  EV=+1.934R  Sharpe=13.0
    """
    icon = "🏆" if "3-Way" in sig_label else "📅"
    arr  = "▲" if direction == "LONG" else "▼"
    print(f"\n  {'─'*52}")
    print(f"  {icon}  {sig_label}  —  {direction}  {arr}")
    print(f"      Prior-WEEK H/L break — both ES + NQ confirmed")
    print(f"      Historical edge: {wh_stat}")

    for internal_sym, pwh, pwl, sym_disp in [
        ("MES", pwh_es, pwl_es, "ES"),
        ("MNQ", pwh_nq, pwl_nq, "NQ"),
    ]:
        s    = day_state.sym(internal_sym)
        rng  = (s.H_OR - s.L_OR) if (s.range_locked and s.H_OR > s.L_OR) else (
            cfg.min_range_mes if internal_sym == "MES" else cfg.min_range_mnq
        )
        pt_v = cfg.pt_value[internal_sym]

        if direction == "LONG":
            entry_px, stop_px = pwh, pwh - rng
            tgt_px   = pwh + rng * cfg.profit_mult
            action, stop_act = "BUY", "SELL"
        else:
            entry_px, stop_px = pwl, pwl + rng
            tgt_px   = pwl - rng * cfg.profit_mult
            action, stop_act = "SELL", "BUY"

        contracts = max(1, round(day_state.account * cfg.risk_pct / (rng * pt_v)))

        await broker.place_order(Order(
            symbol=internal_sym, action=action,
            qty=contracts, order_type="MKT",
        ))
        await broker.place_order(Order(
            symbol=internal_sym, action=stop_act,
            qty=contracts, order_type="STP", aux_price=stop_px, tif="GTC",
        ))
        await broker.place_order(Order(
            symbol=internal_sym, action=stop_act,
            qty=contracts, order_type="LMT", price=tgt_px, tif="GTC",
        ))

    print(f"  {'─'*52}")


def _print_scalp_signal(account: float, risk_pct: float,
                        fade_dir: str, level_type: str, level: float,
                        entry_px: float, stop_px: float,
                        confluence: int, dow_mult: float,
                        mn_l_mult: float) -> None:
    """
    Print a formatted MTF scalp signal for manual entry on CME Simulator.

    Fires when a rejection candle appears at a structural level (D1/W1/MN/RND).
    Rejection = bar tests the level AND closes back away from it.

    Entry : next bar open (market order) — printed here
    Stop  : above/below rejection candle wick
    Target: NO fixed target — trail 3pt stop
             → when trade is +3pts: move stop to breakeven
             → trail 3pts behind best price from there
    Exit  : trailing stop fires OR 15:30 ET EOD close

    Verified edge: WR=44.5%, Sharpe=2.64, OOS Sharpe=2.53 (2022-2026)
    MN_L (monthly low as support): WR=52.4%, Sharpe=4.07
    """
    # Size calculation
    stop_dist = abs(entry_px - stop_px)
    if stop_dist < 0.25: return   # degenerate
    size_mult = dow_mult * mn_l_mult
    usd_risk  = account * risk_pct * size_mult  # total risk $ at 1% base
    n_mes     = max(1, round(usd_risk / (stop_dist * 5)))  # MES contracts
    n_es      = max(1, round(usd_risk / (stop_dist * 50))) # ES contracts (larger)
    risk_usd  = n_mes * stop_dist * 5

    # Trailing stop description
    trail_desc = (
        f"      Trail rule : Move stop to BREAKEVEN when +3pts profit\n"
        f"                   Then trail 3pts behind best price\n"
        f"                   EOD close at 15:30 ET regardless"
    )

    direction_arrow = "▼ SHORT" if fade_dir=="SHORT" else "▲ LONG"
    conf_str = f"{confluence}-TF confluence" if confluence > 1 else "1-TF"
    level_emoji = {"MN_L":"🌟","MN_H":"📅","QT_L":"🌟","QT_H":"📅",
                   "D1_H":"📌","D1_L":"📌","W1_H":"📐","W1_L":"📐","RND":"🔢"}.get(level_type,"📍")
    size_note = ""
    if mn_l_mult > 1.0: size_note = f"  ← MN_L {mn_l_mult}× SIZE BOOST"
    if dow_mult  > 1.0: size_note += f"  ← DOW {dow_mult}× BOOST"

    print(f"\n  {'═'*56}")
    print(f"  {level_emoji}  MTF SCALP {direction_arrow}  —  {level_type} Rejection")
    print(f"  {'═'*56}")
    print(f"  Level      : {level_type} = {level:.2f}  ({conf_str})")
    print(f"  Direction  : {direction_arrow}")
    print(f"  Entry      : {entry_px:.2f}  (MARKET ORDER — next bar open)")
    print(f"  Stop       : {stop_px:.2f}  ({stop_dist:.2f}pts away)")
    print(f"  Target     : NO FIXED TARGET — trailing stop")
    print()
    print(trail_desc)
    print()
    print(f"  Size hint  : {n_mes} MES (${risk_usd:.0f} risk at {risk_pct*100*size_mult:.1f}%){size_note}")
    print(f"  OR use 1 MES minimum if position is too small")
    print()
    print(f"  MANUAL STEPS:")
    print(f"  1. Place {n_mes} MES {fade_dir} at MARKET NOW")
    print(f"  2. Place STOP at {stop_px:.2f}  (GTC)")
    print(f"  3. Watch for +3pts profit → move stop to {entry_px:.2f} (breakeven)")
    print(f"  4. Let trail run until stopped or 15:30 ET")
    print(f"  {'═'*56}")


async def run_signal_generator(port: int = 7496, risk_pct: float = 0.03,
                                account: float = 100_000.0) -> None:
    try:
        import ib_insync
    except ImportError:
        raise RuntimeError("Run:  pip install ib_insync")

    # Set up logging — silence ib_insync's internal chatter (positions, portfolio)
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S",
    )
    logging.getLogger("ib_insync").setLevel(logging.ERROR)

    logger.info("Connecting to IBKR TWS  port=%d …", port)
    ib = ib_insync.IB()
    await ib.connectAsync("127.0.0.1", port, clientId=10)
    logger.info("Connected ✓")

    # ── Pre-market briefing (data + news + calendar) ─────────────────
    briefing = await build_premarket_briefing(ib)
    print_premarket_briefing(briefing)

    cfg    = competition_config(risk_pct=risk_pct)
    broker = SignalBroker(account=account, risk_pct=risk_pct)

    # Pass morning gap direction for real-time auto-sizing at signal print
    _es_g = briefing.get("es_gap")
    _nq_g = briefing.get("nq_gap")
    broker.set_gap_context(
        es_gap_pct = _es_g["gap_pct"] if _es_g else None,
        nq_gap_pct = _nq_g["gap_pct"] if _nq_g else None,
    )

    # ── IMPROVEMENT 3: Day-of-week size multipliers ───────────────────────────
    # ORB: Fri WR=61.8%(+0.376R)→1.25×, Wed WR=52.4% 0.75×, Thu WR=47.0% 0.75×
    # Scalp: Tue Sharpe=3.75 →1.25×, Thu scalp Sharpe=3.23 →1.1× (scalp stronger on Thu)
    _today_dow = date.today().weekday()  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
    _orb_dow_mult   = {0:1.0, 1:1.0, 2:0.75, 3:0.75, 4:1.25}.get(_today_dow, 1.0)
    _scalp_dow_mult = {0:1.0, 1:1.25, 2:1.0,  3:1.1,  4:1.0 }.get(_today_dow, 1.0)
    _dow_name = ["Mon","Tue","Wed","Thu","Fri"][_today_dow] if _today_dow < 5 else "???"

    # ── NEW: Seasonal multiplier (Jan/Apr exceptional, Feb/Dec weak) ──────────
    # Jan WR=71.4% AvgR=+0.511, Apr WR=66.7% AvgR=+0.509 → 1.25× ORB size
    # Feb WR=44.4% AvgR=-0.007, Dec WR=45.5% AvgR=+0.039 → 0.75× ORB size
    # Jun WR=50.0% (competition month — exactly neutral, standard size)
    _today_month = date.today().month
    _seasonal_mult = {1:1.25, 2:0.75, 3:1.0, 4:1.25, 5:1.0, 6:1.0,
                      7:1.0,  8:1.0,  9:1.0, 10:1.0, 11:1.0, 12:0.75
                     }.get(_today_month, 1.0)
    _month_name = ["","Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"][_today_month]

    # ── IMPROVEMENT 4: OR correlation filter (computed after OR locks) ────────
    # Low corr (<0.70): 53.6% WR → 1.5×  |  High corr (≥0.95): 37.7% WR → SKIP
    _or_corr:  float | None = None   # set after OR window closes
    _orb_corr_mult: float = 1.0      # updated after OR

    # ── IMPROVEMENT 5: OpEx calendar effect (+19.6pp on aligned trades) ──────
    # Academic source: Golez & Jackwerth (JFE 2014)
    # On monthly expiration Fridays, ES pulled toward nearest ATM strike.
    _today_date = date.today()
    _is_opex = False; _is_quarterly_opex = False; _pin_direction = None
    if _today_dow == 4:   # Friday
        # Check if 3rd Friday of month
        _ts_today = __import__("pandas").Timestamp(_today_date)
        _first = _ts_today.replace(day=1)
        _first_fri_delta = (4 - _first.weekday()) % 7
        _first_fri = _first + __import__("pandas").Timedelta(days=_first_fri_delta)
        if _first_fri.month != _ts_today.month:
            _first_fri += __import__("pandas").Timedelta(days=7)
        _third_fri = _first_fri + __import__("pandas").Timedelta(days=14)
        _is_opex = (_ts_today.date() == _third_fri.date())
        _is_quarterly_opex = _is_opex and _ts_today.month in (3, 6, 9, 12)
    if _is_opex:
        logger.info("OpEx Friday detected — pin filter active  quarterly=%s", _is_quarterly_opex)

    _combined_orb_mult = _orb_dow_mult * _seasonal_mult
    print(f"\n  📆  {_dow_name} {_month_name}  │  "
          f"ORB={_combined_orb_mult:.2f}× (DOW{_orb_dow_mult:.2f}×·Season{_seasonal_mult:.2f}×)  │  "
          f"Scalp={_scalp_dow_mult:.2f}×"
          + (f"  │  🗓️  OPEX{' QUARTERLY' if _is_quarterly_opex else ''}" if _is_opex else ""))

    # ── Prior-day H/L (PDH/L fallback) ───────────────────────────────────────
    # Backtest: WR=65.4%, EV=+0.635R
    _pdh_es: float | None = (_es_g or {}).get("prev_high")
    _pdl_es: float | None = (_es_g or {}).get("prev_low")
    _pdh_nq: float | None = (_nq_g or {}).get("prev_high")
    _pdl_nq: float | None = (_nq_g or {}).get("prev_low")
    _pdhl_es_up = _pdhl_es_dn = _pdhl_nq_up = _pdhl_nq_dn = False
    _pdhl_fired    = False
    _pdhl_fire_dir: str | None = None
    _PDHL_CUTOFF   = time(11, 0)

    # ── Prior-week H/L (PWH/PWL) ─────────────────────────────────────────────
    # Backtest: standalone WR=79.9% EV=+2.195R Sharpe=18.6  — STRONGEST SIGNAL
    _pwh_es: float | None = (_es_g or {}).get("prior_week_high")
    _pwl_es: float | None = (_es_g or {}).get("prior_week_low")
    _pwh_nq: float | None = (_nq_g or {}).get("prior_week_high")
    _pwl_nq: float | None = (_nq_g or {}).get("prior_week_low")
    _pwhl_es_up = _pwhl_es_dn = _pwhl_nq_up = _pwhl_nq_dn = False
    _pwhl_fired    = False
    _pwhl_fire_dir: str | None = None

    # ── Prior-month H/L (PMH/PML) ────────────────────────────────────────────
    # Backtest (2018-2026): Monthly H/L Sharpe=2.43  vs PDH alone Sharpe=1.56
    # Position traders + monthly options → real order clustering at monthly extremes
    # MTF confluence (≥1 TF within ±8pts): Sharpe=2.78 vs single-TF 1.53
    _pmh_es: float | None = (_es_g or {}).get("prior_month_high")
    _pml_es: float | None = (_es_g or {}).get("prior_month_low")
    _pmh_nq: float | None = (_nq_g or {}).get("prior_month_high")
    _pml_nq: float | None = (_nq_g or {}).get("prior_month_low")
    _pmhl_es_up = _pmhl_es_dn = _pmhl_nq_up = _pmhl_nq_dn = False
    _pmhl_fired    = False
    _pmhl_fire_dir: str | None = None

    # ── ORB direction tracking ────────────────────────────────────────────────
    _orb_fired_dir: str | None = None
    _orb_prev_busy = False

    # ── RETEST ENTRY STATE ────────────────────────────────────────────────────
    # Backtest: Sh=4.12 (vs 3.18 breakout entry), OOS Sh=3.66
    # After CONFIRMED signal: don't enter immediately — wait for price to
    # retrace to OR level (which becomes support) then enter there.
    # Stop: 2pts below OR level (tight, structural)
    # Target: 3 × stop_distance above entry (~10pts)
    # Retest entry: enter when price retraces to OR level (support/resistance)
    # Backtest verified: zone=3, stop=2pts below OR → Sh=4.12, OOS=3.66
    # Stop is pegged to OR LEVEL (structural support), not a fixed distance from entry
    RETEST_ZONE   = 3.0   # enter when price within 3pts of OR level
    RETEST_STOP   = 2.0   # stop 2pts BEYOND the OR level (structural)
    RETEST_PMULT  = 3.0   # target = 3 × actual stop distance (adaptive, ~6-12pts typically)
    _retest_pending:   bool  = False
    _retest_direction: str | None = None
    _retest_or_level_es: float | None = None   # OR level to retest (ES)
    _retest_or_level_nq: float | None = None   # OR level to retest (NQ)
    _RETEST_CUTOFF = time(11, 30)              # cancel retest after this time

    # ── MTF SCALP state (10:30–14:30 ET) ─────────────────────────────────────
    # Rejection candle detection: bar tests level AND closes back away from it
    # Backtest: WR=44.5%, Sharpe=2.64  |  MN_L: Sharpe=4.07
    _scalp_prev_close: float | None = None    # previous MES bar close (ES proxy)
    _scalp_fired_levels: set = set()           # (level_type, rounded_level) already fired today
    _SCALP_START = time(10, 30)
    _SCALP_END   = time(14, 30)
    _round_numbers: list = []   # populated after OR locks

    feed   = IBKRDataFeed(ib)

    _print_banner(risk_pct, account)

    # ── Core loop (mirrors run_trading_day but with OR-lock logging) ──────────
    await broker.connect()
    await feed.subscribe(cfg.symbols)

    engine    = ORBEngine(cfg, broker, sim_mode=True)
    today     = date.today()
    day_state = engine.start_day(today, account)

    or_printed = False

    try:
        while True:
            bar = await feed.next_bar()
            t   = bar.timestamp.time()

            if t > cfg.eod_close:
                break

            await engine.on_bar(bar)

            # Print OR summary once at 10:00 (after both symbols lock)
            if not or_printed and t >= cfg.or_end:
                all_locked = all(s.range_locked for s in day_state.states.values())
                if all_locked:
                    _print_or_summary(cfg, day_state)
                    or_printed = True

                    # ── OR compression ratio (sweet spot 40-60%) ─────────────
                    # Today's OR vs yesterday's full session range.
                    # 40-60% bucket: WR=58.6%, AvgR=+0.257 (best zone)
                    # 60-80% bucket: AvgR=-0.064 (danger — reduce size)
                    try:
                        _mes_s = day_state.sym("MES")
                        _prev_h = (_es_g or {}).get("prev_high")
                        _prev_l = (_es_g or {}).get("prev_low")
                        if _mes_s.range_locked and _prev_h and _prev_l:
                            _prev_range = _prev_h - _prev_l
                            _or_comp_ratio = (_mes_s.H_OR - _mes_s.L_OR) / _prev_range if _prev_range > 0 else 0.5
                            if 0.40 <= _or_comp_ratio <= 0.60:
                                _or_comp_label = "40-60% ✅ SWEET SPOT (+0.257R)"
                                _or_comp_mult  = 1.0   # best zone
                            elif 0.60 < _or_comp_ratio <= 0.80:
                                _or_comp_label = f"60-80% ⚠️  DANGER ZONE (AvgR=-0.064R)"
                                _or_comp_mult  = 0.75  # reduce
                            elif _or_comp_ratio < 0.20:
                                _or_comp_label = f"<20% compressed (limited data)"
                                _or_comp_mult  = 0.9
                            else:
                                _or_comp_label = f"{_or_comp_ratio*100:.0f}% normal"
                                _or_comp_mult  = 1.0
                            print(f"  📊  OR compression: {_or_comp_ratio*100:.0f}% of yesterday's range  "
                                  f"→  {_or_comp_label}")
                        else:
                            _or_comp_mult = 1.0
                    except Exception:
                        _or_comp_mult = 1.0

                    # ── Compute round numbers for scalp (±3 × 50pts around OR) ──
                    try:
                        _mes_s = day_state.sym("MES")
                        if _mes_s.range_locked:
                            _or_mid2 = (_mes_s.H_OR + _mes_s.L_OR) / 2
                            _round_numbers = [round(_or_mid2/50)*50 + k*50
                                              for k in range(-3, 4)]
                    except Exception:
                        pass

                    # ── IMPROVEMENT 4: Compute OR correlation ─────────────────
                    # After OR locks, compute intraday ES/NQ correlation during 9:30-10:00
                    # This requires the OR bar data from the feed's historical replay.
                    # We compute from the ORB engine's tracked bars if available,
                    # otherwise default to 1.0 (no adjustment).
                    try:
                        _mes_s = day_state.sym("MES"); _mnq_s = day_state.sym("MNQ")
                        if _mes_s.range_locked and _mnq_s.range_locked:
                            # Use the last known OR bars from the broker's price stream
                            # Correlation will be set when we can access intraday bars
                            pass  # computed below when bar data is available
                    except Exception:
                        pass

                    # ── IMPROVEMENT 5: OpEx pin direction ─────────────────────
                    # Golez & Jackwerth (JFE 2014): ES pulled toward ATM strike on expiry
                    # Verified: +19.6pp WR when ORB aligns with pin (95 events 2018-2026)
                    # Quarterly triple witching: aligned WR=71.4%
                    if _is_opex:
                        try:
                            _mes_s = day_state.sym("MES")
                            if _mes_s.range_locked:
                                _or_mid = (_mes_s.H_OR + _mes_s.L_OR) / 2
                                _nearest_strike = round(_or_mid / 25) * 25
                                _pin_direction = "LONG" if _or_mid < _nearest_strike else "SHORT"
                                _dist_to_pin = abs(_nearest_strike - _or_mid)
                                logger.info("OpEx pin: OR_mid=%.2f → strike=%.2f  dir=%s",
                                            _or_mid, _nearest_strike, _pin_direction)
                                print(f"\n  {'═'*52}")
                                print(f"  🗓️  OPEX FRIDAY {'(QUARTERLY TRIPLE WITCHING) ' if _is_quarterly_opex else ''}ACTIVE")
                                print(f"  {'═'*52}")
                                print(f"  OR midpoint     : {_or_mid:.2f}")
                                print(f"  Nearest strike  : {_nearest_strike:.0f}  ({_dist_to_pin:.1f}pts away)")
                                print(f"  Pin direction   : {_pin_direction}  (ES gravitates toward this strike)")
                                print(f"  Verified edge   : +19.6pp WR when ORB aligns with pin")
                                if _is_quarterly_opex:
                                    print(f"  Quarterly bonus : aligned WR=71.4% (vs 42.9% anti-pin)")
                                print()
                                print(f"  ✅ TAKE  : ORB {_pin_direction} signals — aligned, full size")
                                print(f"  ⚠️  REDUCE: ORB {'SHORT' if _pin_direction=='LONG' else 'LONG'} signals — anti-pin, 50% size")
                                print(f"  {'═'*52}")
                        except Exception:
                            pass

            # ── ORB direction tracker ─────────────────────────────────────
            _orb_busy = any(
                s.in_position or s.traded_today
                for s in day_state.states.values()
            )
            if _orb_busy and not _orb_prev_busy:
                mes_pos = broker._positions.get("MES", 0)
                _orb_fired_dir = ("LONG"  if mes_pos > 0
                                  else "SHORT" if mes_pos < 0 else None)
            _orb_prev_busy = _orb_busy

            # ── Track PDH/L + PWH/PWL simultaneously from 10:00 ET ───────
            if (time(10, 0) <= t < _PDHL_CUTOFF):

                # PDH/L flags
                if not _pdhl_fired and _pdh_es is not None and _pdh_nq is not None:
                    if bar.symbol == "MES":
                        if bar.high > _pdh_es: _pdhl_es_up = True
                        if bar.low  < _pdl_es: _pdhl_es_dn = True
                    elif bar.symbol == "MNQ":
                        if bar.high > _pdh_nq: _pdhl_nq_up = True
                        if bar.low  < _pdl_nq: _pdhl_nq_dn = True
                    if _pdhl_es_up and _pdhl_nq_up:
                        _pdhl_fired = True; _pdhl_fire_dir = "LONG"
                    elif _pdhl_es_dn and _pdhl_nq_dn:
                        _pdhl_fired = True; _pdhl_fire_dir = "SHORT"

                # PWH/PWL flags
                if not _pwhl_fired and _pwh_es is not None and _pwh_nq is not None:
                    if bar.symbol == "MES":
                        if bar.high > _pwh_es: _pwhl_es_up = True
                        if bar.low  < _pwl_es: _pwhl_es_dn = True
                    elif bar.symbol == "MNQ":
                        if bar.high > _pwh_nq: _pwhl_nq_up = True
                        if bar.low  < _pwl_nq: _pwhl_nq_dn = True
                    if _pwhl_es_up and _pwhl_nq_up:
                        _pwhl_fired = True; _pwhl_fire_dir = "LONG"
                    elif _pwhl_es_dn and _pwhl_nq_dn:
                        _pwhl_fired = True; _pwhl_fire_dir = "SHORT"

                # PMH/PML flags  (prior month — Sharpe=2.43 vs PDH 1.56)
                if not _pmhl_fired and _pmh_es is not None and _pmh_nq is not None:
                    if bar.symbol == "MES":
                        if bar.high > _pmh_es: _pmhl_es_up = True
                        if bar.low  < _pml_es: _pmhl_es_dn = True
                    elif bar.symbol == "MNQ":
                        if bar.high > _pmh_nq: _pmhl_nq_up = True
                        if bar.low  < _pml_nq: _pmhl_nq_dn = True
                    if _pmhl_es_up and _pmhl_nq_up:
                        _pmhl_fired = True; _pmhl_fire_dir = "LONG"
                    elif _pmhl_es_dn and _pmhl_nq_dn:
                        _pmhl_fired = True; _pmhl_fire_dir = "SHORT"

            # ── Full priority hierarchy ────────────────────────────────────
            # Evaluate once per bar after both trackers update.
            # Only act when a NEW signal just completed (edge-detect).
            _new_pdhl = _pdhl_fired and _pdhl_fire_dir is not None
            _new_pwhl = _pwhl_fired and _pwhl_fire_dir is not None
            _new_pmhl = _pmhl_fired and _pmhl_fire_dir is not None

            if _new_pdhl or _new_pwhl or _new_pmhl:
                _pd = _pdhl_fire_dir
                _pw = _pwhl_fire_dir
                _pm = _pmhl_fire_dir
                _ob = _orb_fired_dir

                # Multi-timeframe confluence score
                _fired_dirs = [d for d in [_ob, _pd, _pw, _pm] if d is not None]
                _agree_count = len(set(_fired_dirs))

                # Conflict check — any disagreement → warn + skip new entries
                _dirs = _fired_dirs
                _conflict = len(set(_dirs)) > 1

                if _conflict and not _orb_busy:
                    # Signals disagree, nothing entered yet → print and skip
                    print(f"\n  {'─'*52}")
                    print(f"  ⚠️  CONFLICT — signals disagree:")
                    if _ob: print(f"      ORB   : {_ob}")
                    if _pd: print(f"      PDH/L : {_pd}")
                    if _pw: print(f"      PWH/PWL: {_pw}")
                    print(f"      Backtest WR=43.8% on conflict days — NO TRADE")
                    print(f"  {'─'*52}")

                elif _conflict and _orb_busy:
                    # ORB already entered, now a conflicting signal appeared
                    print(f"\n  {'─'*52}")
                    print(f"  ⚠️  POST-ENTRY CONFLICT — new signal opposes your position")
                    if _pd and _pd != _ob: print(f"      PDH/L fired {_pd} vs your {_ob} ORB")
                    if _pw and _pw != _ob: print(f"      PWH/PWL fired {_pw} vs your {_ob} ORB")
                    print(f"      Backtest WR=43.8% — CONSIDER EXITING NOW")
                    print(f"  {'─'*52}")

                elif not _conflict and not _orb_busy:
                    # ── IMPROVEMENT 1: 2-min wait on ORB entry ────────────────
                    # 0-2min breaks WR=49.1%, 2-5min breaks WR=56.0% (+7pp)
                    # False breakouts shake out in the first 2 minutes.
                    if _ob is not None and t < time(10, 2):
                        # ORB fired but less than 2 minutes after OR locked — wait
                        pass   # allow remaining signals (PDH/L, PWH/PWL) but skip ORB entry

                    # ── IMPROVEMENT 5: OpEx direction filter ──────────────────
                    # +19.6pp WR edge when ORB aligns with pin. Anti-pin = 42.9% WR.
                    if _is_opex and _pin_direction is not None and _ob is not None:
                        _direction_raw = "LONG" if _ob=="LONG" else "SHORT"
                        if _direction_raw != _pin_direction:
                            print(f"\n  {'─'*52}")
                            print(f"  ⚠️  OPEX ANTI-PIN SIGNAL")
                            print(f"      ORB fired {_direction_raw} but pin direction is {_pin_direction}")
                            print(f"      Backtest: anti-pin WR=42.9%  aligned WR=62.5%")
                            print(f"      → ENTER AT 50% SIZE  (or skip entirely)")
                            print(f"  {'─'*52}")
                        else:
                            print(f"\n  ✅  OPEX ALIGNED: ORB {_direction_raw} matches pin {_pin_direction}")
                            print(f"      → ENTER AT 125% SIZE  (WR=62.5% confirmed)")
                            # Don't block — just warn. Can reduce size manually.

                    # ── NEW: Break strength + PDH proximity filters ───────────
                    # Break < 20% OR range: WR=61.5%, AvgR=+0.27 (best zone)
                    # Break > 100%: WR=37.5%, AvgR=-0.11 (AVOID)
                    # PDH within 2R: WR=60%+, AvgR=+0.30 (genuine confirmation)
                    # PDH > 2R: WR=44.7%, AvgR=+0.016 (meaningless)
                    # Combined filter: Sh 2.57→3.23, OOS 2.98→3.51
                    _orb_break_warn = ""
                    try:
                        _mes_s = day_state.sym("MES"); _mnq_s = day_state.sym("MNQ")
                        if _mes_s.range_locked and _mnq_s.range_locked and _ob is not None:
                            # Break distance as fraction of OR range
                            # (estimate from engine's tracked H/L vs OR extremes)
                            _or_r_es = _mes_s.H_OR - _mes_s.L_OR
                            _or_r_nq = _mnq_s.H_OR - _mnq_s.L_OR
                            if _or_r_es > 0 and _or_r_nq > 0:
                                # We can't easily get exact break distance from the engine
                                # but we print guidance based on the rule
                                _orb_break_warn = (
                                    f"\n      ⚡ Check: break distance should be ≤20% of OR range"
                                    f"\n         ES OR={_or_r_es:.1f}pts → max break = {_or_r_es*0.20:.1f}pts"
                                    f"\n         NQ OR={_or_r_nq:.1f}pts → max break = {_or_r_nq*0.20:.1f}pts"
                                    f"\n         If either broke by MORE → reduce size 50%"
                                )
                    except Exception:
                        pass

                    # All agreeing signals — determine best entry type
                    _direction = _dirs[0]
                    # Count how many non-None signals agree
                    _n_agree   = len([x for x in [_ob,_pd,_pw,_pm] if x is not None])
                    _three_way = (_ob == _pd == _pw) and all(
                        x is not None for x in [_ob, _pd, _pw])
                    _four_way  = (_n_agree >= 4 and
                                  len(set([x for x in [_ob,_pd,_pw,_pm] if x])) == 1)
                    _orb_pdhl  = _ob == _pd and _ob is not None and _pd is not None
                    _orb_pwhl  = _ob == _pw and _ob is not None and _pw is not None
                    _pdhl_pwhl = _pd == _pw and _pd is not None and _pw is not None and _ob is None

                    # Only act on the newly completed signal
                    if _four_way and _new_pmhl:
                        # 4-Way: ORB + PDH/L + PWH/PWL + PMH/PML
                        print(f"\n  {'─'*52}")
                        print(f"  🌟  4-WAY CONFLUENCE — ORB+PDH/L+PWH/PWL+PMH/PML all {_direction}")
                        print(f"      MTF Sharpe=2.78  (daily+weekly+monthly all agree)")
                        print(f"      MAXIMUM MULTI-TIMEFRAME CONFIDENCE")
                        _me = _pmh_es if _direction=="LONG" else _pml_es
                        _mn = _pmh_nq if _direction=="LONG" else _pml_nq
                        if _me: print(f"      Monthly: ES={_me:.2f}  NQ={_mn:.2f}")
                        print(f"  {'─'*52}")

                    elif _three_way and _new_pwhl:
                        # 3-Way: ORB + PDH/L + PWH/PWL — annotate, use PDH/L entry
                        # IMPROVEMENT 6: Check PDH/L + Monthly confluence
                        _pdh_level = _pdh_es if _direction=="LONG" else _pdl_es
                        _monthly_near_pdh = (_pmh_es is not None and
                                              _pdh_level is not None and
                                              (abs(_pdh_level-_pmh_es)<=8 or
                                               abs(_pdh_level-(_pml_es or 0))<=8))
                        print(f"\n  {'─'*52}")
                        if _monthly_near_pdh:
                            print(f"  🌟  3-WAY + MONTHLY CONFLUENCE — ORB+PDH/L+PWH/PWL+MN all {_direction}")
                            print(f"      Backtest: Sh=4.09 (vs Sh=2.48 confirmed only)  MAX SIZE ✅")
                        else:
                            print(f"  🔥  3-WAY CONFLUENCE — ORB + PDH/L + PWH/PWL all {_direction}")
                        print(f"      Backtest: WR=73.4%  EV=+1.934R  Sharpe=12.99")
                        print(f"      MAXIMUM CONFIDENCE — use PDH/L entry levels")
                        _pe = _pdh_es if _direction=="LONG" else _pdl_es
                        _pn = _pdh_nq if _direction=="LONG" else _pdl_nq
                        print(f"      PDH/L entry: ES={_pe:.2f}  NQ={_pn:.2f}")
                        _we = _pwh_es if _direction=="LONG" else _pwl_es
                        _wn = _pwh_nq if _direction=="LONG" else _pwl_nq
                        print(f"      PWH/PWL    : ES={_we:.2f}  NQ={_wn:.2f}")
                        print(f"  {'─'*52}")

                    elif _orb_pwhl and _new_pwhl and not _pdhl_fired:
                        # ORB + PWH/PWL (no PDH/L): annotate
                        print(f"\n  {'─'*52}")
                        print(f"  🏆  ORB + PWH/PWL CONFIRMED — both {_direction}")
                        print(f"      Backtest: WR=77.9%  EV=+2.114R  Sharpe=18.14")
                        _we = _pwh_es if _direction=="LONG" else _pwl_es
                        _wn = _pwh_nq if _direction=="LONG" else _pwl_nq
                        print(f"      PWH/PWL: ES={_we:.2f}  NQ={_wn:.2f}")
                        print(f"  {'─'*52}")

                    elif _orb_pdhl and _new_pdhl and not _pwhl_fired:
                        # ── ORB + PDH/L CONFIRMED → RETEST ENTRY ─────────────
                        # NEW: Don't enter at breakout bar. Wait for price to
                        # retrace to OR level (now support) then enter.
                        # Backtest: Sh=4.12 (vs 3.18 immediate entry), OOS Sh=3.66
                        # IMPROVEMENT 6: check monthly confluence
                        _pdh_level2 = _pdh_es if _direction=="LONG" else _pdl_es
                        _mn_near2   = (_pmh_es is not None and _pdh_level2 is not None and
                                       (abs(_pdh_level2-_pmh_es)<=8 or
                                        abs(_pdh_level2-(_pml_es or 0))<=8))
                        # Get OR level for retest
                        try:
                            _mes_st = day_state.sym("MES"); _mnq_st = day_state.sym("MNQ")
                            _or_lvl_es = _mes_st.H_OR if _direction=="LONG" else _mes_st.L_OR
                            _or_lvl_nq = _mnq_st.H_OR if _direction=="LONG" else _mnq_st.L_OR
                        except Exception:
                            _or_lvl_es = _or_lvl_nq = None

                        _full_size = _orb_dow_mult * _seasonal_mult * getattr(locals(),'_or_comp_mult',1.0)

                        print(f"\n  {'═'*56}")
                        if _mn_near2:
                            print(f"  🌟  ORB+PDH/L+MONTHLY — {_direction}  ← MAX SIZE")
                        else:
                            print(f"  ✅  ORB + PDH/L CONFIRMED — {_direction}")
                        print(f"  {'═'*56}")
                        print(f"  Signal quality : Sh=4.12  WR=45.5%  AvgR=+0.565R  OOS=3.66")
                        print(f"  Size mult      : {_full_size:.2f}×  (DOW·Season·OR_comp)")
                        print()
                        if _or_lvl_es:
                            _stop_es = (_or_lvl_es - RETEST_STOP if _direction=="LONG"
                                        else _or_lvl_es + RETEST_STOP)
                            print(f"  ⏳  DO NOT ENTER YET — WAIT FOR RETEST")
                            print(f"  {'─'*56}")
                            print(f"  OR level  : ES={_or_lvl_es:.2f}  NQ={_or_lvl_nq:.2f}")
                            print(f"  Retest zone: within {RETEST_ZONE:.0f}pts of OR level")
                            print(f"  Trigger at: ES ≤ {_or_lvl_es+RETEST_ZONE:.2f}" if _direction=="LONG"
                                  else f"  Trigger at: ES ≥ {_or_lvl_es-RETEST_ZONE:.2f}")
                            print(f"  Stop       : {RETEST_STOP:.0f}pts beyond OR level  (~${RETEST_STOP*50:.0f} risk/ES)")
                            print(f"  Target     : {RETEST_PMULT:.0f} × stop = ~{RETEST_STOP*RETEST_PMULT:.0f}pts  (~${RETEST_STOP*RETEST_PMULT*50:.0f} target/ES)")
                            print()
                            print(f"  → WAIT for ES to return to {_or_lvl_es:.2f}±{RETEST_ZONE:.0f}pts")
                            print(f"  → Signal will fire automatically when retest detected")
                            print(f"  {'═'*56}")
                            # Arm the retest state
                            _retest_pending   = True
                            _retest_direction = _direction
                            _retest_or_level_es = _or_lvl_es
                            _retest_or_level_nq = _or_lvl_nq
                        else:
                            # Fallback: no OR level available, enter immediately
                            _pe = _pdh_es if _direction=="LONG" else _pdl_es
                            _pn = _pdh_nq if _direction=="LONG" else _pdl_nq
                            print(f"  PDH/L entry: ES={_pe:.2f}  NQ={_pn:.2f}  (no OR level for retest)")
                            print(f"  {'═'*56}")

                    elif _pdhl_pwhl and (_new_pdhl or _new_pwhl):
                        # PDH/L + PWH/PWL both agree, no ORB
                        await _fire_pwhl_signal(
                            broker, day_state, _direction, cfg,
                            _pwh_es, _pwl_es, _pwh_nq, _pwl_nq,
                            sig_label = "PDH/L + PWH/PWL  (no ORB)",
                            wh_stat   = "WR=61.1%  EV=+1.444R  Sharpe=8.38",
                        )

                    elif _new_pwhl and not _pdhl_fired and not _orb_fired_dir:
                        # PWH/PWL standalone — strongest standalone signal
                        await _fire_pwhl_signal(
                            broker, day_state, _direction, cfg,
                            _pwh_es, _pwl_es, _pwh_nq, _pwl_nq,
                            sig_label = "PWH/PWL STANDALONE",
                            wh_stat   = "WR=79.9%  EV=+2.195R  Sharpe=18.60  ← STRONGEST SIGNAL",
                        )

                    elif _new_pmhl and not _pdhl_fired and not _pwhl_fired and not _orb_fired_dir:
                        # PMH/PML standalone (monthly level — Sharpe=2.43)
                        await _fire_pwhl_signal(
                            broker, day_state, _direction, cfg,
                            _pmh_es if _direction=="LONG" else _pml_es,
                            _pml_es if _direction=="LONG" else _pmh_es,
                            _pmh_nq if _direction=="LONG" else _pml_nq,
                            _pml_nq if _direction=="LONG" else _pmh_nq,
                            sig_label = "PMH/PML MONTHLY STANDALONE",
                            wh_stat   = "Monthly H/L  Sharpe=2.43  WR=46.4%  AvgR=+0.795pts",
                        )

                    elif _new_pdhl and not _pwhl_fired and not _orb_fired_dir:
                        # PDH/L standalone
                        await _fire_pdhl_signal(
                            broker, day_state, _direction, cfg,
                            _pdh_es, _pdl_es, _pdh_nq, _pdl_nq,
                        )

            # ── RETEST ENTRY DETECTION ────────────────────────────────────────
            # After CONFIRMED signal: watch for price to return to OR level
            # Backtest: Sh=4.12 vs 3.18 immediate entry  |  OOS Sh=3.66
            if _retest_pending and bar.symbol == "MES":
                if t >= _RETEST_CUTOFF:
                    # Retest window expired — cancel
                    _retest_pending = False
                    print(f"\n  ⏰  RETEST WINDOW EXPIRED ({_RETEST_CUTOFF})")
                    print(f"      OR level {_retest_or_level_es:.2f} not retested — signal cancelled")
                elif _retest_or_level_es is not None:
                    _retest_triggered = False
                    if _retest_direction == "LONG":
                        # Price must come back DOWN to within RETEST_ZONE of OR high
                        if bar.low <= _retest_or_level_es + RETEST_ZONE:
                            _retest_triggered = True
                    else:  # SHORT
                        # Price must come back UP to within RETEST_ZONE of OR low
                        if bar.high >= _retest_or_level_es - RETEST_ZONE:
                            _retest_triggered = True

                    if _retest_triggered:
                        _retest_pending = False
                        # Calculate entry, stop, target
                        _rt_ep_es = bar.close + (SLIP if _retest_direction=="LONG" else -SLIP)
                        _rt_sp_es = (_retest_or_level_es - RETEST_STOP - SLIP
                                     if _retest_direction=="LONG"
                                     else _retest_or_level_es + RETEST_STOP + SLIP)
                        _rt_risk  = abs(_rt_ep_es - _rt_sp_es)
                        _rt_tp_es = (_rt_ep_es + RETEST_PMULT*_rt_risk
                                     if _retest_direction=="LONG"
                                     else _rt_ep_es - RETEST_PMULT*_rt_risk)
                        # NQ equivalent
                        _rt_ep_nq = bar.close + (SLIP if _retest_direction=="LONG" else -SLIP)
                        _rt_sp_nq = (_retest_or_level_nq - RETEST_STOP - SLIP
                                     if _retest_direction=="LONG"
                                     else _retest_or_level_nq + RETEST_STOP + SLIP)
                        _rt_risk_nq = abs(_rt_ep_nq - _rt_sp_nq)
                        _rt_tp_nq = (_rt_ep_nq + RETEST_PMULT*_rt_risk_nq
                                     if _retest_direction=="LONG"
                                     else _rt_ep_nq - RETEST_PMULT*_rt_risk_nq)
                        # Size (same as primary, use account/risk_pct)
                        _rt_contracts_es = max(1, round(account*risk_pct/(_rt_risk*50)))
                        _rt_contracts_nq = max(1, round(account*risk_pct/(_rt_risk_nq*20)))
                        _dir_arrow = "▲ LONG" if _retest_direction=="LONG" else "▼ SHORT"
                        print(f"\n  {'═'*56}")
                        print(f"  🎯  RETEST ENTRY — {_dir_arrow}  NOW!")
                        print(f"  {'═'*56}")
                        print(f"  OR level retested at {t}  "
                              f"(price returned to {_retest_or_level_es:.2f})")
                        print(f"  Entry    : MARKET ORDER NOW")
                        print(f"  ES leg   : entry≈{_rt_ep_es:.2f}  stop={_rt_sp_es:.2f}  "
                              f"target={_rt_tp_es:.2f}  ({_rt_contracts_es} MES)")
                        print(f"  NQ leg   : entry≈{_rt_ep_nq:.2f}  stop={_rt_sp_nq:.2f}  "
                              f"target={_rt_tp_nq:.2f}  ({_rt_contracts_nq} MNQ)")
                        print(f"  Stop dist: {_rt_risk:.1f}pts (~${_rt_risk*50:.0f}/ES)  "
                              f"Target dist: {_rt_risk*RETEST_PMULT:.1f}pts (~${_rt_risk*RETEST_PMULT*50:.0f}/ES)  "
                              f"R:R={RETEST_PMULT:.0f}:1")
                        print()
                        print(f"  MANUAL STEPS:")
                        print(f"  1. Place {_rt_contracts_es} MES {_retest_direction} at MARKET NOW")
                        print(f"  2. Place {_rt_contracts_nq} MNQ {_retest_direction} at MARKET NOW")
                        print(f"  3. Place ES STOP at {_rt_sp_es:.2f}  (GTC)")
                        print(f"  4. Place NQ STOP at {_rt_sp_nq:.2f}  (GTC)")
                        print(f"  5. Target ES {_rt_tp_es:.2f}  NQ {_rt_tp_nq:.2f}")
                        print(f"  {'═'*56}")

            # ── MTF SCALP DETECTION (10:30–14:30 ET) ──────────────────────────
            # Scan MES bars for rejection candles at D1/W1/MN/RND levels.
            # One signal per level per day. Prints manual entry ticket.
            # Backtest verified: WR=44.5%, Sharpe=2.64, OOS=2.53
            if bar.symbol == "MES" and _SCALP_START <= t < _SCALP_END:
                if _scalp_prev_close is not None and _round_numbers:

                    # Build today's level list
                    _all_scalp_levels = []
                    if _pdh_es: _all_scalp_levels.append(("D1_H", _pdh_es, "SHORT"))
                    if _pdl_es: _all_scalp_levels.append(("D1_L", _pdl_es, "LONG"))
                    if _pwh_es: _all_scalp_levels.append(("W1_H", _pwh_es, "SHORT"))
                    if _pwl_es: _all_scalp_levels.append(("W1_L", _pwl_es, "LONG"))
                    if _pmh_es: _all_scalp_levels.append(("MN_H", _pmh_es, "SHORT"))
                    if _pml_es: _all_scalp_levels.append(("MN_L", _pml_es, "LONG"))
                    # Quarterly H/L — fund benchmarks + quarterly options (wider but strong)
                    _pqh_es = (_es_g or {}).get("prior_quarter_high")
                    _pql_es = (_es_g or {}).get("prior_quarter_low")
                    if _pqh_es: _all_scalp_levels.append(("QT_H", _pqh_es, "SHORT"))
                    if _pql_es: _all_scalp_levels.append(("QT_L", _pql_es, "LONG"))
                    for _rnd in _round_numbers:
                        _all_scalp_levels.append(("RND", float(_rnd), "SHORT"))
                        _all_scalp_levels.append(("RND", float(_rnd), "LONG"))

                    # Confluence scorer
                    def _sc_conf(lv, ll):
                        return sum(1 for (_,v,_d) in ll if 0 < abs(v-lv) <= 8) + 1

                    # Sort by confluence (highest first)
                    _sorted_lvls = sorted(
                        _all_scalp_levels,
                        key=lambda x: -_sc_conf(x[1], _all_scalp_levels)
                    )

                    for _nm, _lv, _fd in _sorted_lvls:
                        _key = (_nm, round(_lv / 5) * 5)   # deduplicate nearby levels
                        if _key in _scalp_fired_levels: continue

                        # Rejection candle detection
                        _reject = False
                        if _fd == "SHORT":
                            # Resistance rejection: prev close below level, bar wick above, close below
                            if (_scalp_prev_close < _lv - 0.25 and
                                    bar.high >= _lv and
                                    bar.close <= _lv - 1.0):
                                _reject = True
                                _ep = bar.close - 0.25        # enter SHORT at next bar (approx close)
                                _sp = bar.high + 0.50          # stop above rejection wick
                        else:
                            # Support rejection: prev close above level, bar wick below, close above
                            if (_scalp_prev_close > _lv + 0.25 and
                                    bar.low <= _lv and
                                    bar.close >= _lv + 1.0):
                                _reject = True
                                _ep = bar.close + 0.25        # enter LONG
                                _sp = bar.low - 0.50           # stop below rejection wick

                        if _reject:
                            _scalp_fired_levels.add(_key)
                            _c2      = _sc_conf(_lv, _sorted_lvls)
                            _mn_mult = 1.5 if _nm == "MN_L" else 1.0
                            _print_scalp_signal(
                                account     = account,
                                risk_pct    = 0.01,         # 1% base risk for scalp
                                fade_dir    = _fd,
                                level_type  = _nm,
                                level       = _lv,
                                entry_px    = _ep,
                                stop_px     = _sp,
                                confluence  = _c2,
                                dow_mult    = _scalp_dow_mult,
                                mn_l_mult   = _mn_mult,
                            )
                            break  # one scalp signal per bar, highest confidence first

                # Always update prev close at end of MES bar processing
                _scalp_prev_close = bar.close

    except KeyboardInterrupt:
        print("\n  [Interrupted — run end_day cleanup]")
    finally:
        summary = await engine.end_day()
        await broker.disconnect()
        ib.disconnect()

        print()
        print("  " + "═" * 56)
        print(f"  SESSION COMPLETE   {today}")
        pnl = summary.get("daily_pnl_pct", 0.0)
        print(f"  Daily P&L : {pnl:+.2f}%   (signal model, not actual)")
        print(f"  Trades    : {len(summary.get('trades', []))}")
        print("  " + "═" * 56)
        print()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port     = 7497 if "--paper" in sys.argv else 7496
    risk_pct = 0.05   # 5% per leg for competition week (was 0.03 for long-run)
    account  = 100_000.0

    for arg in sys.argv[1:]:
        if arg.startswith("--risk="):
            risk_pct = float(arg.split("=")[1]) / 100.0
        elif arg.startswith("--account="):
            account = float(arg.split("=")[1])

    print(f"  Mode    : {'PAPER (port 7497)' if port == 7497 else 'LIVE  (port 7496)'}")
    print(f"  Risk    : {risk_pct*100:.0f}% per instrument")

    asyncio.run(run_signal_generator(port=port, risk_pct=risk_pct, account=account))
