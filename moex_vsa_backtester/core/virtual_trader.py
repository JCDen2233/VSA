from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd
from loguru import logger

from config import config


@dataclass
class VirtualPosition:
    ticket: str
    entry_time: int
    entry_price: float
    sl_price: float
    tp_price: float
    size: int
    side: str
    status: str = "OPEN"
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    rr: float = 0.0
    signal_data: Dict = field(default_factory=dict)

    @property
    def duration_bars(self) -> int:
        if self.exit_time:
            return (self.exit_time - self.entry_time) // 3600
        return (int(datetime.now().timestamp()) - self.entry_time) // 3600


class VirtualTrader:
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        risk_pct: float = 0.01,
        rr_ratio: float = 2.0,
        max_positions: int = 5,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.max_positions = max_positions
        self.positions: Dict[str, VirtualPosition] = {}
        self.journal: List[VirtualPosition] = []

    def open_position(
        self,
        ticker: str,
        signal: Dict,
        entry_price: float,
        sl_price: float,
        tp_price: Optional[float] = None,
        size: Optional[int] = None,
    ) -> Optional[VirtualPosition]:
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Max positions reached ({self.max_positions})")
            return None

        if ticker in self.positions:
            logger.warning(f"Position already exists for {ticker}")
            return None

        if tp_price is None:
            risk = abs(entry_price - sl_price)
            if signal.get("signal_type") == "LONG":
                tp_price = entry_price + risk * self.rr_ratio
            else:
                tp_price = entry_price - risk * self.rr_ratio

        if size is None:
            size = self._calculate_size(entry_price, sl_price)

        if size <= 0:
            logger.warning(f"Invalid position size: {size}")
            return None

        pos = VirtualPosition(
            ticket=ticker,
            entry_time=signal.get("timestamp", int(datetime.now().timestamp())),
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            size=size,
            side=signal.get("signal_type", "LONG"),
            signal_data=signal,
        )

        self.positions[ticker] = pos

        logger.info(
            f"{'LONG' if pos.side == 'LONG' else 'SHORT'} {ticker}: "
            f"entry={entry_price:.4f}, sl={sl_price:.4f}, "
            f"tp={tp_price:.4f}, size={size}"
        )

        return pos

    def _calculate_size(self, entry: float, sl: float) -> int:
        risk_amount = self.capital * self.risk_pct
        price_risk = abs(entry - sl)

        if price_risk <= 0:
            return 0

        commission = config.get("COMMISSION_PCT", 0.001)
        slippage = config.get("SLIPPAGE_PCT", 0.0005)

        adjusted_entry = entry * (1 + commission + slippage)
        adjusted_sl = sl * (1 - commission - slippage)
        actual_risk = abs(adjusted_entry - adjusted_sl)

        if actual_risk <= 0:
            return 0

        size = int(risk_amount / actual_risk)
        size = max(1, size)

        max_capital_usage = self.capital * 0.20
        position_value = size * entry
        if position_value > max_capital_usage:
            size = int(max_capital_usage / entry)
            size = max(1, size)

        return size

    def check_exits(self, ticker: str, current_price: float, current_ts: int) -> Optional[VirtualPosition]:
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        if pos.status != "OPEN":
            return None

        entry = pos.entry_price
        sl = pos.sl_price
        tp = pos.tp_price
        side = pos.side
        size = pos.size
        pnl = 0.0
        status = "OPEN"
        exit_price = None

        if side == "LONG":
            if current_price <= sl:
                pnl = (sl - entry) * size
                status = "SL"
                exit_price = sl
            elif current_price >= tp:
                pnl = (tp - entry) * size
                status = "TP"
                exit_price = tp
            elif current_price >= entry + (tp - entry) * 0.5:
                pos.status = "HALF_CLOSED"
                pnl_half = (entry + (tp - entry) * 0.5 - entry) * (size // 2)
                pnl = pnl_half + (current_price - entry) * (size - size // 2)
                self.capital += pnl_half
                pos.size = size - size // 2
                pos.sl_price = entry
                logger.info(f"Half position closed at 1.5R for {ticker}")
        else:
            if current_price >= sl:
                pnl = (entry - sl) * size
                status = "SL"
                exit_price = sl
            elif current_price <= tp:
                pnl = (entry - tp) * size
                status = "TP"
                exit_price = tp
            elif current_price <= entry - (entry - tp) * 0.5:
                pos.status = "HALF_CLOSED"
                pnl_half = (entry - (entry - tp) * 0.5 - entry) * (size // 2)
                pnl = pnl_half + (entry - current_price) * (size - size // 2)
                self.capital += pnl_half
                pos.size = size - size // 2
                pos.sl_price = entry
                logger.info(f"Half position closed at 1.5R for {ticker}")

        if status != "OPEN":
            pos.exit_time = current_ts
            pos.exit_price = exit_price or current_price
            pos.pnl = pnl
            pos.status = status

            if pos.side == "LONG":
                reward = pos.pnl / size / abs(entry - sl)
            else:
                reward = pos.pnl / size / abs(entry - sl)
            pos.rr = reward

            self.capital += pnl
            self.journal.append(pos)
            del self.positions[ticker]

            logger.info(
                f"Closed {ticker}: {status}, PnL={pnl:.2f}, "
                f"RR={reward:.2f}, Capital={self.capital:.2f}"
            )

        return pos

    def close_position(
        self, ticker: str, exit_price: float, status: str = "MANUAL"
    ) -> Optional[VirtualPosition]:
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        size = pos.size
        entry = pos.entry_price
        side = pos.side

        if side == "LONG":
            pnl = (exit_price - entry) * size
        else:
            pnl = (entry - exit_price) * size

        pos.exit_time = int(datetime.now().timestamp())
        pos.exit_price = exit_price
        pos.pnl = pnl
        pos.status = status

        risk = abs(entry - pos.sl_price)
        if risk > 0:
            pos.rr = pnl / (size * risk)

        self.capital += pnl
        self.journal.append(pos)
        del self.positions[ticker]

        logger.info(
            f"Closed {ticker}: {status}, PnL={pnl:.2f}, Capital={self.capital:.2f}"
        )

        return pos

    def close_all(self, prices: Dict[str, float] = None) -> List[VirtualPosition]:
        closed = []
        for ticker in list(self.positions.keys()):
            if prices and ticker in prices:
                self.close_position(ticker, prices[ticker], "EOD_CLOSE")
            else:
                pos = self.positions[ticker]
                self.close_position(ticker, pos.entry_price, "EOD_CLOSE")
            closed.append(self.journal[-1])
        return closed

    def get_stats(self) -> Dict:
        if not self.journal:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "capital": self.capital,
                "profit_factor": 0.0,
                "avg_rr": 0.0,
            }

        winning = [t for t in self.journal if t.pnl > 0]
        losing = [t for t in self.journal if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in self.journal)
        wins_pnl = sum(t.pnl for t in winning)
        loss_pnl = sum(t.pnl for t in losing)

        avg_rr = sum(t.rr for t in self.journal) / len(self.journal) if self.journal else 0.0

        return {
            "total_trades": len(self.journal),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(self.journal) if self.journal else 0.0,
            "total_pnl": total_pnl,
            "capital": self.capital,
            "profit_factor": abs(wins_pnl / loss_pnl) if loss_pnl != 0 else float("inf"),
            "avg_rr": avg_rr,
        }

    def get_open_positions(self) -> List[Dict]:
        result = []
        for ticker, pos in self.positions.items():
            result.append({
                "ticker": ticker,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "current_price": pos.signal_data.get("current_price", pos.entry_price),
                "sl_price": pos.sl_price,
                "tp_price": pos.tp_price,
                "size": pos.size,
                "unrealized_pnl": self._calc_unrealized_pnl(pos),
                "entry_time": pos.entry_time,
            })
        return result

    def _calc_unrealized_pnl(self, pos: VirtualPosition) -> float:
        current_price = pos.signal_data.get("current_price", pos.entry_price)
        if pos.side == "LONG":
            return (current_price - pos.entry_price) * pos.size
        else:
            return (pos.entry_price - current_price) * pos.size


class VirtualTradingMonitor:
    def __init__(self, output_dir: Path = Path("logs")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.traders: Dict[str, VirtualTrader] = {}

    def get_or_create_trader(self, ticker: str) -> VirtualTrader:
        if ticker not in self.traders:
            self.traders[ticker] = VirtualTrader()
        return self.traders[ticker]

    def get_all_positions(self) -> List[Dict]:
        all_positions = []
        for ticker, trader in self.traders.items():
            all_positions.extend(trader.get_open_positions())
        return all_positions

    def close_all_positions(self) -> List[VirtualPosition]:
        all_closed = []
        for trader in self.traders.values():
            all_closed.extend(trader.close_all())
        return all_closed

    def get_total_stats(self) -> Dict:
        total_trades = sum(len(t.journal) for t in self.traders.values())
        total_pnl = sum(sum(tr.pnl for tr in t.journal) for t in self.traders.values())
        capital = sum(t.capital for t in self.traders.values())

        winning = sum(1 for t in trader.journal if t.pnl > 0 for trader in self.traders.values())
        win_rate = winning / total_trades if total_trades > 0 else 0.0

        return {
            "total_trades": total_trades,
            "winning_trades": winning,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "capital": capital,
        }