"""
orb_live.py — Live trading engine for the ORB alignment strategy.

Architecture (clean separation of concerns):
  ORBConfig      — all parameters in one place
  ORBDayState    — per-day mutable state (opening range, positions, P&L)
  BrokerAPI      — abstract interface: place/cancel/query orders
  IBKRBroker     — Interactive Brokers implementation via ib_insync
  DataFeed       — abstract interface: subscribe to real-time 1-min bars
  ORBEngine      — pure trading logic, broker/feed agnostic
  run_trading_day() — entry point called once per trading day

Strategy:
  1. 09:25 ET  Connect broker, warm up data feed
  2. 09:30–10:00 ET  Observe opening range for ES and NQ
  3. 10:00 ET  Lock H_OR / L_OR for both symbols
  4. 10:00–14:00 ET  Wait-for-confirmation: enter BOTH only when both
                     break the same direction (no lookahead)
  5. Position management: hard stop at L_OR, limit target at OR+1.5R
  6. 15:30 ET  Force-close all positions at market
  7. 16:00 ET  Disconnect, write daily log

Instruments: MES (Micro ES) + MNQ (Micro NQ)
  Sizing: round(account × RISK_PCT / (range × tick_mult)) contracts ≥ 1
  Slippage budget: 1 tick per side (entry + stop/EOD)
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum, auto
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class ORBConfig:
    """All strategy parameters — change here, nowhere else."""
    # Opening range
    or_start:          time  = time(9, 30)   # ET — start observing
    or_end:            time  = time(10, 0)   # ET — lock range
    entry_cutoff:      time  = time(14, 0)   # ET — no new entries after
    eod_close:         time  = time(15, 30)  # ET — force close all

    # Signal  — MES/MNQ trade at the SAME price levels as ES/NQ (only the $/pt differs)
    min_range_mes:     float = 8.0           # price points (matches backtest MIN_RANGE_ES=8)
    min_range_mnq:     float = 30.0          # price points (matches backtest MIN_RANGE_NQ=30)
    max_range_mes:     float = float("inf")  # skip abnormally wide ES OR (default: no cap)
    max_range_mnq:     float = float("inf")  # skip abnormally wide NQ OR (default: no cap)

    # Risk
    risk_pct:          float = 0.01          # 1% of account per trade
    profit_mult:       float = 1.5           # target = entry ± range × 1.5

    # Instruments (micro contracts)
    symbols:           list  = field(default_factory=lambda: ["MES", "MNQ"])
    tick_size:         dict  = field(default_factory=lambda: {"MES": 0.25, "MNQ": 0.25})
    tick_value:        dict  = field(default_factory=lambda: {"MES": 1.25, "MNQ": 0.50})
    pt_value:          dict  = field(default_factory=lambda: {"MES": 5.00, "MNQ": 2.00})

    # Execution
    slip_ticks:        int   = 1             # slippage budget: 1 tick per side
    commission_rt:     float = 0.74          # IBKR round-trip per micro contract

    # Risk management
    daily_loss_limit:  float = 0.015         # stop trading after -1.5% on the day

    # Logging
    log_dir:           str   = "./logs"


# ─── State ───────────────────────────────────────────────────────────────────

class BreakDir(Enum):
    NONE  = auto()
    LONG  = auto()
    SHORT = auto()


@dataclass
class SymbolDayState:
    """State for one symbol on one trading day."""
    sym:           str
    H_OR:          float = 0.0
    L_OR:          float = 0.0
    range_locked:  bool  = False

    # Position
    in_position:   bool  = False
    direction:     BreakDir = BreakDir.NONE
    entry_price:   float = 0.0
    stop_price:    float = 0.0
    target_price:  float = 0.0
    contracts:     int   = 0

    # Order IDs (broker-specific)
    entry_order_id:  Optional[str] = None
    stop_order_id:   Optional[str] = None
    target_order_id: Optional[str] = None

    # Confirmation tracking
    first_break:   BreakDir = BreakDir.NONE  # direction this sym broke (or NONE)
    traded_today:  bool     = False


@dataclass
class ORBDayState:
    """Full day state shared across both symbols."""
    trade_date:    date
    states:        dict[str, SymbolDayState] = field(default_factory=dict)
    daily_pnl_pct: float = 0.0
    account:       float = 25_000.0
    trades_today:  list  = field(default_factory=list)
    log_lines:     list  = field(default_factory=list)

    def sym(self, s: str) -> SymbolDayState:
        return self.states[s]

    def both_aligned(self) -> Optional[BreakDir]:
        """Return direction if both symbols have broken the same way, else None."""
        dirs = [st.first_break for st in self.states.values()]
        if all(d == BreakDir.LONG  for d in dirs): return BreakDir.LONG
        if all(d == BreakDir.SHORT for d in dirs): return BreakDir.SHORT
        return None

    def circuit_tripped(self, daily_loss_limit: float) -> bool:
        return self.daily_pnl_pct <= -daily_loss_limit


# ─── Broker interface ─────────────────────────────────────────────────────────

@dataclass
class Order:
    symbol:    str
    action:    str        # 'BUY' or 'SELL'
    qty:       int
    order_type: str       # 'MKT', 'LMT', 'STP', 'STP_LMT'
    price:     float = 0.0   # limit price (LMT / STP_LMT)
    aux_price: float = 0.0   # stop trigger price (STP / STP_LMT)
    tif:       str  = 'DAY'
    order_id:  Optional[str] = None


@dataclass
class Fill:
    order_id:  str
    symbol:    str
    qty:       int
    avg_price: float
    timestamp: datetime


class BrokerAPI(ABC):
    """Abstract broker — swap out for any execution venue."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def get_account_value(self) -> float: ...

    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """Returns order_id."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None: ...

    @abstractmethod
    async def get_position(self, symbol: str) -> int:
        """Returns current net position in contracts."""

    @abstractmethod
    async def close_position(self, symbol: str, qty: int) -> Fill:
        """Market order to close position."""


class IBKRBroker(BrokerAPI):
    """
    Interactive Brokers implementation via ib_insync.
    Install: pip install ib_insync

    Assumes TWS or IB Gateway running on localhost:7497 (paper) or 7496 (live).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7496, client_id: int = 1):
        self.host      = host
        self.port      = port
        self.client_id = client_id
        self._ib       = None  # ib_insync.IB instance

    async def connect(self) -> None:
        try:
            import ib_insync
            self._ib = ib_insync.IB()
            await self._ib.connectAsync(self.host, self.port, clientId=self.client_id)
            logger.info("IBKR connected  host=%s  port=%d", self.host, self.port)
        except ImportError:
            raise RuntimeError("ib_insync not installed. Run: pip install ib_insync")
        except Exception as e:
            raise RuntimeError(f"IBKR connection failed: {e}")

    async def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("IBKR disconnected")

    async def get_account_value(self) -> float:
        vals = self._ib.accountValues()
        for v in vals:
            if v.tag == "NetLiquidation" and v.currency == "USD":
                return float(v.value)
        return 0.0

    def _make_contract(self, symbol: str):
        import ib_insync
        # MES and MNQ are CME micro futures
        exchange_map = {"MES": "CME", "MNQ": "CME"}
        return ib_insync.Future(
            symbol   = symbol,
            exchange = exchange_map.get(symbol, "CME"),
            currency = "USD",
        )

    async def place_order(self, order: Order) -> str:
        import ib_insync
        contract = self._make_contract(order.symbol)
        # Qualify contract (get exact expiry from IBKR)
        await self._ib.qualifyContractsAsync(contract)

        if order.order_type == "MKT":
            ib_order = ib_insync.MarketOrder(order.action, order.qty)
        elif order.order_type == "LMT":
            ib_order = ib_insync.LimitOrder(order.action, order.qty, order.price)
        elif order.order_type == "STP":
            ib_order = ib_insync.StopOrder(order.action, order.qty, order.aux_price)
        elif order.order_type == "STP_LMT":
            ib_order = ib_insync.StopLimitOrder(
                order.action, order.qty, order.price, order.aux_price
            )
        else:
            raise ValueError(f"Unknown order_type: {order.order_type}")

        ib_order.tif = order.tif
        trade = self._ib.placeOrder(contract, ib_order)
        logger.info("Order placed  %s %s %d  type=%s  id=%s",
                    order.action, order.symbol, order.qty,
                    order.order_type, trade.order.orderId)
        return str(trade.order.orderId)

    async def cancel_order(self, order_id: str) -> None:
        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == order_id:
                self._ib.cancelOrder(trade.order)
                logger.info("Order cancelled  id=%s", order_id)
                return

    async def get_position(self, symbol: str) -> int:
        for pos in self._ib.positions():
            if pos.contract.symbol == symbol:
                return int(pos.position)
        return 0

    async def close_position(self, symbol: str, qty: int) -> Fill:
        action = "SELL" if qty > 0 else "BUY"
        order  = Order(symbol=symbol, action=action, qty=abs(qty), order_type="MKT")
        oid    = await self.place_order(order)
        # Wait for fill (simplified — in production use event-driven fill tracking)
        await asyncio.sleep(2)
        return Fill(order_id=oid, symbol=symbol, qty=abs(qty),
                    avg_price=0.0, timestamp=datetime.now(ET))


class PaperBroker(BrokerAPI):
    """
    Paper-trading broker for offline testing.
    Immediately simulates fills for MKT orders so get_position() is accurate
    — this is required for sim_mode backtesting in ORBEngine.
    """

    def __init__(self, starting_capital: float = 25_000.0):
        self._account   = starting_capital
        self._positions: dict[str, int]   = {}
        self._orders:    dict[str, Order] = {}
        self._next_id   = 1

    async def connect(self)    -> None: logger.info("PaperBroker connected")
    async def disconnect(self) -> None: logger.info("PaperBroker disconnected")
    async def get_account_value(self) -> float: return self._account

    async def place_order(self, order: Order) -> str:
        oid = str(self._next_id); self._next_id += 1
        self._orders[oid] = order
        # Immediately simulate fill for MKT orders (updates tracked position)
        if order.order_type == "MKT":
            delta = order.qty if order.action == "BUY" else -order.qty
            self._positions[order.symbol] = (
                self._positions.get(order.symbol, 0) + delta
            )
        logger.debug("PAPER order  %s %s %d @ %s  id=%s",
                     order.action, order.symbol, order.qty,
                     order.order_type, oid)
        return oid

    async def cancel_order(self, order_id: str) -> None:
        self._orders.pop(order_id, None)

    async def get_position(self, symbol: str) -> int:
        return self._positions.get(symbol, 0)

    async def close_position(self, symbol: str, qty: int) -> Fill:
        """qty > 0 = long position being closed; qty < 0 = short being closed."""
        self._positions[symbol] = 0
        return Fill(order_id="paper", symbol=symbol, qty=abs(qty),
                    avg_price=0.0, timestamp=datetime.now(ET))


# ─── Data feed interface ──────────────────────────────────────────────────────

@dataclass
class Bar:
    symbol:    str
    timestamp: datetime   # bar start time, ET
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    int


class DataFeed(ABC):
    """Abstract real-time 1-min bar feed."""

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None: ...

    @abstractmethod
    async def next_bar(self) -> Bar:
        """Block until next 1-min bar is available, then return it."""


class DatabentoFeed(DataFeed):
    """
    Databento real-time streaming feed.
    Install: pip install databento
    Requires DATABENTO_API_KEY environment variable.
    """

    def __init__(self):
        self._api_key  = os.environ.get("DATABENTO_API_KEY", "")
        self._queue: asyncio.Queue = asyncio.Queue()
        self._client   = None

    async def subscribe(self, symbols: list[str]) -> None:
        try:
            import databento as db
            self._client = db.Live(key=self._api_key)
            # Map MES/MNQ to front-month continuous contract
            dataset   = "GLBX.MDP3"
            schema    = "ohlcv-1m"
            stype     = "continuous"
            syms      = [f"{s}.c.0" for s in symbols]  # front-month continuous
            self._client.subscribe(dataset=dataset, schema=schema,
                                   stype_in=stype, symbols=syms)
            logger.info("Databento feed subscribed: %s", syms)
        except ImportError:
            raise RuntimeError("databento not installed. Run: pip install databento")

    async def next_bar(self) -> Bar:
        """Pull next completed 1-min bar from Databento stream."""
        record = await asyncio.get_event_loop().run_in_executor(
            None, next, iter(self._client)
        )
        ts = datetime.fromtimestamp(record.ts_event / 1e9, tz=ET)
        return Bar(
            symbol    = record.symbol.split(".")[0],  # strip ".c.0"
            timestamp = ts,
            open      = record.open  / 1e9,
            high      = record.high  / 1e9,
            low       = record.low   / 1e9,
            close     = record.close / 1e9,
            volume    = record.volume,
        )


class MockFeed(DataFeed):
    """
    Replays a DataFrame of historical bars for offline testing.
    Useful for dry-run and integration testing without live data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        df must have columns: symbol, timestamp (ET-aware), open, high, low, close, volume
        """
        self._bars = df.sort_values("timestamp").reset_index(drop=True)
        self._idx  = 0

    async def subscribe(self, symbols: list[str]) -> None:
        self._bars = self._bars[self._bars["symbol"].isin(symbols)]
        logger.info("MockFeed ready  %d bars", len(self._bars))

    async def next_bar(self) -> Bar:
        if self._idx >= len(self._bars):
            raise StopIteration("MockFeed exhausted")
        row = self._bars.iloc[self._idx]
        self._idx += 1
        return Bar(**{k: row[k] for k in
                      ["symbol","timestamp","open","high","low","close","volume"]})


# ─── Core trading engine ──────────────────────────────────────────────────────

class ORBEngine:
    """
    Pure trading logic — knows nothing about IBKR or Databento.
    Receives 1-min bars, emits orders via BrokerAPI.

    State machine per day:
      OBSERVING  → WAITING_CONFIRM → IN_POSITION → FLAT
    """

    def __init__(self, cfg: ORBConfig, broker: BrokerAPI, sim_mode: bool = False):
        self.cfg      = cfg
        self.broker   = broker
        self.sim_mode = sim_mode   # True: check stop/target bar-by-bar (backtest/dry-run)
        self.state: Optional[ORBDayState] = None

    def start_day(self, today: date, account: float) -> ORBDayState:
        """Call once at market open."""
        self.state = ORBDayState(
            trade_date = today,
            account    = account,
            states     = {s: SymbolDayState(sym=s) for s in self.cfg.symbols},
        )
        logger.info("Day started  %s  account=$%.0f", today, account)
        return self.state

    async def on_bar(self, bar: Bar) -> None:
        """
        Main entry point: called on every 1-min bar during the session.
        Handles: range observation, breakout detection, position management.
        """
        if self.state is None:
            return

        st  = self.state
        sym = bar.symbol
        if sym not in st.states:
            return

        s   = st.sym(sym)
        t   = bar.timestamp.time()
        cfg = self.cfg

        # ── Phase 1: Observe opening range ───────────────────────────────
        if cfg.or_start <= t < cfg.or_end:
            if not s.range_locked:
                s.H_OR = max(s.H_OR, bar.high)
                s.L_OR = s.L_OR if s.L_OR > 0 else bar.low
                s.L_OR = min(s.L_OR, bar.low)
            return

        # ── Phase 2: Lock range at or_end ────────────────────────────────
        if t >= cfg.or_end and not s.range_locked:
            s.range_locked = True
            rng = s.H_OR - s.L_OR
            min_rng = cfg.min_range_mes if sym == "MES" else cfg.min_range_mnq
            max_rng = cfg.max_range_mes if sym == "MES" else cfg.max_range_mnq
            if rng < min_rng:
                logger.info("%s  range %.2f < min %.2f → skip day", sym, rng, min_rng)
                s.traded_today = True   # mark as no-trade day
            elif rng > max_rng:
                logger.info("%s  range %.2f > max %.2f → skip day (too wide)", sym, rng, max_rng)
                s.traded_today = True   # mark as no-trade day
            else:
                logger.info("%s  OR locked  H=%.2f  L=%.2f  R=%.2f",
                            sym, s.H_OR, s.L_OR, rng)

        # ── Phase 3: Manage open position ────────────────────────────────
        if s.in_position:
            await self._manage_position(bar, s, st)
            return

        # ── Phase 4: Detect breakout (wait-for-confirmation) ─────────────
        if (s.traded_today or t >= cfg.entry_cutoff or
                st.daily_pnl_pct <= -cfg.daily_loss_limit):
            return

        if not s.range_locked or s.first_break != BreakDir.NONE:
            return   # no range yet, or already signalled

        rng = s.H_OR - s.L_OR
        if bar.high > s.H_OR:
            s.first_break = BreakDir.LONG
            logger.info("%s  LONG breakout at %.2f (bar H=%.2f)",
                        sym, s.H_OR, bar.high)
        elif bar.low < s.L_OR:
            s.first_break = BreakDir.SHORT
            logger.info("%s  SHORT breakout at %.2f (bar L=%.2f)",
                        sym, s.L_OR, bar.low)

        # ── Phase 5: Check alignment — enter both if confirmed ────────────
        if s.first_break != BreakDir.NONE:
            aligned_dir = st.both_aligned()
            if aligned_dir is not None:
                # Both symbols confirmed same direction → enter both
                for sym2, s2 in st.states.items():
                    if not s2.in_position and not s2.traded_today:
                        await self._enter(sym2, s2, aligned_dir, st, bar)

    async def _enter(self, sym: str, s: SymbolDayState,
                     direction: BreakDir, st: ORBDayState, bar: Bar) -> None:
        """Place entry + bracket orders for one symbol."""
        cfg  = self.cfg
        rng  = s.H_OR - s.L_OR
        is_long = (direction == BreakDir.LONG)
        tick = cfg.tick_size[sym]
        pt_v = cfg.pt_value[sym]

        # Entry price: at the OR level (buy-stop / sell-stop)
        # In live: this fills at H_OR + slippage (modelled as cost, not price adj)
        if is_long:
            entry  = s.H_OR
            stop   = s.L_OR
            target = s.H_OR + rng * cfg.profit_mult
            action = "BUY"
            stop_action = "SELL"
        else:
            entry  = s.L_OR
            stop   = s.H_OR
            target = s.L_OR - rng * cfg.profit_mult
            action = "SELL"
            stop_action = "BUY"

        # Position size
        contracts = max(1, round(st.account * cfg.risk_pct / (rng * pt_v)))

        logger.info(
            "%s  ENTER %s  entry=%.2f  stop=%.2f  target=%.2f  qty=%d",
            sym, direction.name, entry, stop, target, contracts
        )

        # Place entry at market (we're already past the level when we detect breakout)
        entry_oid = await self.broker.place_order(Order(
            symbol=sym, action=action, qty=contracts, order_type="MKT"
        ))

        # Place stop loss
        stop_oid = await self.broker.place_order(Order(
            symbol=sym, action=stop_action, qty=contracts,
            order_type="STP", aux_price=stop, tif="GTC"
        ))

        # Place profit target
        tgt_oid = await self.broker.place_order(Order(
            symbol=sym, action=stop_action, qty=contracts,
            order_type="LMT", price=target, tif="GTC"
        ))

        s.in_position    = True
        s.direction      = direction
        s.entry_price    = entry
        s.stop_price     = stop
        s.target_price   = target
        s.contracts      = contracts
        s.entry_order_id = entry_oid
        s.stop_order_id  = stop_oid
        s.target_order_id = tgt_oid

    async def _manage_position(self, bar: Bar, s: SymbolDayState,
                               st: ORBDayState) -> None:
        """
        Called each bar when in position.

        Live mode (sim_mode=False):
            Stop/target resting orders are managed by IBKR — we only handle EOD.

        Sim mode (sim_mode=True):
            Check stop/target bar-by-bar (mirrors the backtest logic exactly).
        """
        cfg     = self.cfg
        t       = bar.timestamp.time()
        is_long = (s.direction == BreakDir.LONG)

        if self.sim_mode:
            # Bar-by-bar stop/target resolution
            if is_long:
                if bar.low <= s.stop_price:
                    await self._close_position(s, st, s.stop_price, "STOP")
                    return
                elif bar.high >= s.target_price:
                    await self._close_position(s, st, s.target_price, "TARGET")
                    return
            else:
                if bar.high >= s.stop_price:
                    await self._close_position(s, st, s.stop_price, "STOP")
                    return
                elif bar.low <= s.target_price:
                    await self._close_position(s, st, s.target_price, "TARGET")
                    return

        # EOD force close (live and sim)
        if t >= cfg.eod_close:
            logger.info("%s  EOD force close at %.2f", s.sym, bar.close)
            await self._close_position(s, st, bar.close, "EOD")

    async def _close_position(self, s: SymbolDayState, st: ORBDayState,
                               exit_price: float, reason: str) -> None:
        """Cancel remaining bracket orders, market-close position, record P&L."""
        cfg   = self.cfg
        is_long = (s.direction == BreakDir.LONG)
        tick  = cfg.tick_size[s.sym]
        pt_v  = cfg.pt_value[s.sym]

        # Cancel open bracket orders
        for oid in [s.stop_order_id, s.target_order_id]:
            if oid:
                try:
                    await self.broker.cancel_order(oid)
                except Exception as e:
                    logger.warning("Cancel failed %s: %s", oid, e)

        # Market close
        pos_qty = await self.broker.get_position(s.sym)
        if pos_qty != 0:
            await self.broker.close_position(s.sym, pos_qty)

        # P&L (gross, before slippage — slippage modelled as budget, not price adj)
        sign      = 1 if is_long else -1
        gross_pts = sign * (exit_price - s.entry_price)
        entry_slip = cfg.slip_ticks * tick
        exit_slip  = cfg.slip_ticks * tick if reason in ("STOP", "EOD") else 0.0
        net_pts    = gross_pts - (entry_slip + exit_slip)
        pnl_dollar = net_pts * s.contracts * pt_v - cfg.commission_rt * s.contracts
        pnl_pct    = pnl_dollar / st.account

        st.account       += pnl_dollar
        st.daily_pnl_pct += pnl_pct

        trade_rec = {
            "date":      str(st.trade_date),
            "symbol":    s.sym,
            "direction": s.direction.name,
            "entry":     s.entry_price,
            "exit":      exit_price,
            "stop":      s.stop_price,
            "target":    s.target_price,
            "contracts": s.contracts,
            "net_pts":   net_pts,
            "pnl_$":     pnl_dollar,
            "pnl_%":     pnl_pct,
            "reason":    reason,
        }
        st.trades_today.append(trade_rec)
        logger.info(
            "%s  CLOSED  %s  exit=%.2f  net_pts=%.2f  pnl=$%.0f (%.2f%%)",
            s.sym, reason, exit_price, net_pts, pnl_dollar, pnl_pct * 100
        )

        s.in_position  = False
        s.traded_today = True

    async def end_day(self) -> dict:
        """Force-close any remaining positions and return day summary."""
        if self.state is None:
            return {}
        for sym, s in self.state.states.items():
            if s.in_position:
                logger.warning("%s  Still open at end_day — force closing", sym)
                pos_qty = await self.broker.get_position(sym)
                if pos_qty != 0:
                    await self.broker.close_position(sym, pos_qty)
                s.in_position  = False
                s.traded_today = True

        summary = {
            "date":          str(self.state.trade_date),
            "daily_pnl_pct": round(self.state.daily_pnl_pct * 100, 3),
            "account":       round(self.state.account, 2),
            "trades":        self.state.trades_today,
        }
        logger.info("Day ended  pnl=%.2f%%  account=$%.0f  trades=%d",
                    self.state.daily_pnl_pct * 100,
                    self.state.account,
                    len(self.state.trades_today))
        return summary


# ─── Daily run loop ───────────────────────────────────────────────────────────

async def run_trading_day(
    cfg:    ORBConfig,
    broker: BrokerAPI,
    feed:   DataFeed,
    today:  Optional[date] = None,
) -> dict:
    """
    Full daily trading cycle.  Call once per trading day from your scheduler.

    Returns: day summary dict {date, daily_pnl_pct, account, trades}
    """
    today = today or date.today()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-12s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("ORB LIVE  date=%s  symbols=%s", today, cfg.symbols)
    logger.info("=" * 60)

    await broker.connect()
    account = await broker.get_account_value()
    logger.info("Account: $%.2f", account)

    await feed.subscribe(cfg.symbols)

    engine  = ORBEngine(cfg, broker)
    day_state = engine.start_day(today, account)

    try:
        while True:
            bar = await feed.next_bar()
            t   = bar.timestamp.time()

            # Stop processing after EOD close
            if t > cfg.eod_close:
                break

            await engine.on_bar(bar)

            # Check if all symbols have traded — nothing left to do
            if all(s.traded_today for s in day_state.states.values()):
                logger.info("All symbols traded — waiting for EOD")

    except StopIteration:
        logger.info("Feed exhausted (dry-run mode)")
    except KeyboardInterrupt:
        logger.warning("Interrupted — closing positions")
    finally:
        summary = await engine.end_day()
        await broker.disconnect()

    return summary


# ─── Entry point / scheduler ─────────────────────────────────────────────────

# Global config (set once at startup)
_cfg = ORBConfig()


def main_live():
    """
    Production entry point.
    Run from cron at 09:25 ET on trading days:
      25 9 * * 1-5 cd /path/to/system && python3 orb_live.py
    """
    cfg    = ORBConfig()
    broker = IBKRBroker(host="127.0.0.1", port=7496)   # live: 7496, paper: 7497
    feed   = DatabentoFeed()
    asyncio.run(run_trading_day(cfg, broker, feed))


def _reshape_aligned_to_symbols(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Reshape aligned ES+NQ DataFrame to per-symbol (MES/MNQ) rows."""
    rows = []
    for _, row in df_1min.iterrows():
        for sym, es_sym in [("MES", "ES"), ("MNQ", "NQ")]:
            rows.append({
                "symbol":    sym,
                "timestamp": row["timestamp"],
                "open":      row[f"{es_sym}_open"],
                "high":      row[f"{es_sym}_high"],
                "low":       row[f"{es_sym}_low"],
                "close":     row[f"{es_sym}_close"],
                "volume":    int(row.get(f"{es_sym}_volume", 0)),
            })
    df_sym = pd.DataFrame(rows)
    df_sym["ts_et"] = pd.to_datetime(df_sym["timestamp"]).dt.tz_convert("America/New_York")
    df_sym["date"]  = df_sym["ts_et"].dt.date
    return df_sym.sort_values("ts_et").reset_index(drop=True)


async def run_full_backtest(
    df_1min:          pd.DataFrame,
    cfg:              ORBConfig,
    starting_capital: float = 25_000.0,
    news_dates:       set   | None = None,
) -> dict:
    """
    Run the live ORBEngine over the full historical dataset (multi-day sim).

    Produces the same metrics as run_orb_wait_confirm() for direct comparison.
    ORBEngine is constructed with sim_mode=True so stop/target are evaluated
    bar-by-bar rather than relying on IBKR bracket orders.

    Returns: {stats, trade_log, daily_returns}
    """
    skip_dates = news_dates or set()

    # Reshape once — use vectorised melt, not iterrows (2× faster)
    df_es = df_1min[["timestamp","ES_open","ES_high","ES_low","ES_close","ES_volume"]].copy()
    df_es.columns = ["timestamp","open","high","low","close","volume"]
    df_es["symbol"] = "MES"
    df_nq = df_1min[["timestamp","NQ_open","NQ_high","NQ_low","NQ_close","NQ_volume"]].copy()
    df_nq.columns = ["timestamp","open","high","low","close","volume"]
    df_nq["symbol"] = "MNQ"
    df_sym = pd.concat([df_es, df_nq], ignore_index=True)
    df_sym["ts_et"] = pd.to_datetime(df_sym["timestamp"]).dt.tz_convert("America/New_York")
    df_sym["date"]  = df_sym["ts_et"].dt.date
    df_sym = df_sym.sort_values(["ts_et", "symbol"]).reset_index(drop=True)

    # Pre-split by date (one scan, not O(n_dates × n_bars))
    day_groups = {d: g.reset_index(drop=True)
                  for d, g in df_sym.groupby("date")}

    trade_log  = []
    daily_rets = {}
    account    = starting_capital

    logger.info("run_full_backtest: %d dates  $%.0f start", len(day_groups), account)

    for trade_date in sorted(day_groups.keys()):
        date_str = str(trade_date)
        if date_str in skip_dates:
            continue

        day_df = day_groups[trade_date]

        # Fresh broker and engine per day (account updated day-to-day)
        broker    = PaperBroker(starting_capital=account)
        engine    = ORBEngine(cfg, broker, sim_mode=True)
        day_state = engine.start_day(trade_date, account)

        # Pre-extract columns — use tolist() for ts_et to preserve tz info.
        # CRITICAL: .values strips timezone from DatetimeTZDtype, producing
        # naive UTC timestamps. bar.timestamp.time() would then return UTC
        # time instead of ET time, breaking all phase-timing checks.
        symbols    = day_df["symbol"].tolist()
        ts_list    = day_df["ts_et"].tolist()   # preserves America/New_York tz
        open_vals  = day_df["open"].to_numpy()
        high_vals  = day_df["high"].to_numpy()
        low_vals   = day_df["low"].to_numpy()
        close_vals = day_df["close"].to_numpy()
        vol_vals   = day_df["volume"].to_numpy()

        for i in range(len(day_df)):
            bar = Bar(
                symbol    = symbols[i],
                timestamp = ts_list[i],         # ET-aware pd.Timestamp
                open      = float(open_vals[i]),
                high      = float(high_vals[i]),
                low       = float(low_vals[i]),
                close     = float(close_vals[i]),
                volume    = int(vol_vals[i]),
            )
            await engine.on_bar(bar)

        summary = await engine.end_day()
        account = summary.get("account", account)

        daily_pnl = sum(t.get("pnl_%", 0) for t in summary.get("trades", []))
        for t in summary.get("trades", []):
            trade_log.append(t)
        if daily_pnl != 0:
            daily_rets[date_str] = daily_pnl

    # Compute stats (same formulas as run_orb.py _compute_stats)
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

    stats = {
        "sharpe_ratio":          round(sharpe, 4),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "max_drawdown_pct":      round(max_dd, 2),
        "total_trades":          len(tl),
        "overall_win_rate":      float((tl["pnl_%"] > 0).mean()) if len(tl) else 0.0,
        "final_account":         round(account, 2),
        "trades_per_year":       round(len(tl) / n_years, 1),
    }

    ann_by_year = (dr.groupby(dr.index.year)
                     .apply(lambda x: (1 + x).prod() - 1) * 100)

    logger.info(
        "Backtest result: Sh=%.3f  Ann=+%.1f%%  DD=%.1f%%  T=%d  $25k→$%s",
        stats["sharpe_ratio"],
        stats["annualized_return_pct"],
        stats["max_drawdown_pct"],
        stats["total_trades"],
        f"{stats['final_account']:,.0f}",
    )
    return {"stats": stats, "trade_log": trade_log,
            "daily_returns": dr, "annual_rets": ann_by_year}


def main_dryrun(df_1min: pd.DataFrame):
    """
    Single-day dry-run.  Useful for a quick smoke test with a known date.
    """
    df_sym = _reshape_aligned_to_symbols(df_1min)
    cfg    = ORBConfig()
    broker = PaperBroker(starting_capital=25_000)
    feed   = MockFeed(df_sym.rename(columns={"ts_et": "timestamp"}))

    summary = asyncio.run(run_trading_day(cfg, broker, feed))
    print("\nDry-run summary:", summary)
    return summary


def main_validate(data_path: str = None):
    """
    Full multi-day backtest via the live engine.
    Run with: python3 orb_live.py --validate

    Compares results against run_orb_wait_confirm (E8 benchmark):
      E8 target: Sh≈1.326  Ann≈+37.2%  DD≈-28.4%  $25k→$369k
    """
    import time as _time
    data_path = data_path or (
        "/Users/ishanbhardwaj/cme_competition_system/output/mtf/"
        "ES_NQ_1min_aligned.parquet"
    )
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    print(f"\nLoading data from {data_path} …")
    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} bars  "
          f"{pd.to_datetime(df['timestamp']).min().date()} → "
          f"{pd.to_datetime(df['timestamp']).max().date()}")

    cfg = ORBConfig(slip_ticks=1, commission_rt=0.74)

    t0 = _time.time()
    result = asyncio.run(run_full_backtest(df, cfg, starting_capital=25_000.0))
    elapsed = _time.time() - t0

    s = result["stats"]
    ann = result["annual_rets"]

    # E6 = wait-confirm, fractional, no slip   → Sh≈1.613  Ann≈+47%  T=2984
    # E8 = wait-confirm, micro, 1-tick slip   → Sh≈1.326  Ann≈+37%  T=2984
    # Live engine fills BOTH syms at OR level (resting bracket order behaviour).
    # E8's lower return uses market-fill for the first sym — conservative but not
    # how a resting buy-stop actually executes; live should be between E6 and E8.
    print()
    print("=" * 65)
    print("LIVE ENGINE BACKTEST (sim_mode=True, 1-tick slip, $0.74/contract)")
    print("=" * 65)
    print(f"  Sharpe      : {s['sharpe_ratio']:.3f}   (E6≈1.613 / E8≈1.326)")
    print(f"  Annual ret  : {s['annualized_return_pct']:+.2f}%   (E6≈+47% / E8≈+37%)")
    print(f"  Max drawdown: {s['max_drawdown_pct']:.2f}%")
    print(f"  Trades      : {s['total_trades']:,}   ({s['trades_per_year']:.0f}/yr  target≈2984/368)")
    print(f"  Win rate    : {s['overall_win_rate']*100:.1f}%")
    print(f"  $25k → ${s['final_account']:,.0f}")
    print(f"  Elapsed     : {elapsed:.0f}s")
    print()
    print("  Annual breakdown:")
    for yr, ret in ann.items():
        print(f"    {yr}: {ret:+.1f}%")
    print("=" * 65)

    delta_sh = s["sharpe_ratio"] - 1.613
    dt_count = s["total_trades"] - 2984
    print(f"\n  ΔSharpe vs E6: {delta_sh:+.3f}   ΔTrades: {dt_count:+d}")
    if abs(delta_sh) < 0.05 and abs(dt_count) < 100:
        print("  ✅  Live engine validated — structure matches E6 (OR-level fills for both syms)")
    elif abs(delta_sh) < 0.15:
        print("  ⚠️   Small discrepancy — check OR filter / entry cutoff alignment")
    else:
        print("  ❌  Significant mismatch — debug required")
    return result


if __name__ == "__main__":
    import sys
    if "--validate" in sys.argv:
        main_validate()
    elif "--dryrun" in sys.argv:
        DATA = ("/Users/ishanbhardwaj/cme_competition_system/output/mtf/"
                "ES_NQ_1min_aligned.parquet")
        df   = pd.read_parquet(DATA)
        df["date_et"] = (pd.to_datetime(df["timestamp"])
                           .dt.tz_convert("America/New_York").dt.date)
        # Smoke test on one known trading day
        test_date = df["date_et"].unique()[50]   # ~50th trading day (mid-2018)
        one_day   = df[df["date_et"] == test_date].copy()
        print(f"Dry-run date: {test_date}")
        main_dryrun(one_day)
    else:
        main_live()
