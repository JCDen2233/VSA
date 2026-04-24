import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional
import csv
from loguru import logger

from core.virtual_trader import VirtualTrader, VirtualTradingMonitor
from core.trade_journal import TradeJournal, DailyReportGenerator


class VirtualTradingService:
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        risk_pct: float = 0.01,
        rr_ratio: float = 2.0,
        max_positions: int = 5,
        output_dir: Path = Path("logs/virtual_trading"),
    ):
        self.monitor = VirtualTradingMonitor(output_dir)
        self.journal = TradeJournal(output_dir)
        self.reporter = DailyReportGenerator(output_dir / "reports")

        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.max_positions = max_positions

        self._last_report_date: Optional[date] = None
        self._last_report_hour: int = -1  # For hourly reporting

    def execute_signal(self, signal: Dict, current_price: float) -> Optional[Dict]:
        ticker = signal.get("ticker")
        if not ticker:
            return None

        # Get the trader for this specific ticker
        trader = self.monitor.get_or_create_trader(ticker)

        # Check for existing open positions FIRST - manage exits before new entries
        if ticker in trader.positions:
            pos = trader.positions[ticker]
            pos.signal_data["current_price"] = current_price
            
            # Check exits and exit if needed
            result = trader.check_exits(ticker, current_price, int(datetime.now().timestamp()))

            if result:
                self._log_closed_trade(result)
                # After closing, continue to check if we can re-enter on the same signal
                if result.status in ("TP", "SL"):
                    logger.info(f"[{ticker}] Position closed ({result.status}), checking for re-entry")
                else:
                    return {"action": "CLOSED", "ticker": ticker, "pnl": result.pnl, "status": result.status}

        # Check if we're allowed to open a new position
        if len(trader.positions) >= self.max_positions:
            logger.warning(f"Max positions reached ({self.max_positions}) for {ticker}")
            return None

        entry_price = signal.get("entry_price", current_price)
        sl_price = signal.get("sl_price", entry_price * 0.9985)
        tp_price = signal.get("tp_price")

        # Prevent duplicate trades by checking signal timestamp
        if not self._is_duplicate_signal(ticker, signal):
            pos = trader.open_position(
                ticker=ticker,
                signal=signal,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
            )

            if pos:
                pos.signal_data["current_price"] = current_price
                logger.info(f"[{ticker}] ✅ POSITION OPENED: {pos.side} @ {entry_price:.4f}, SL={sl_price:.4f}, TP={tp_price:.4f}")
                return {"action": "OPENED", "ticker": ticker, "position": pos}
            else:
                return None
        else:
            logger.debug(f"[{ticker}] Duplicate signal detected, skipping trade")
            return None

    def _log_closed_trade(self, pos):
        if pos.exit_time:
            self.journal.add_trade(
                ticker=pos.ticket,
                side=pos.side,
                entry_time=pos.entry_time,
                exit_time=pos.exit_time,
                entry_price=pos.entry_price,
                exit_price=pos.exit_price or pos.entry_price,
                sl_price=pos.sl_price,
                tp_price=pos.tp_price,
                size=pos.size,
                pnl=pos.pnl,
                rr=pos.rr,
                status=pos.status,
            )

    def _is_duplicate_signal(self, ticker: str, signal: Dict) -> bool:
        """
        Check if this signal is a duplicate by comparing timestamp to existing open positions
        """
        trader = self.monitor.get_or_create_trader(ticker)
        
        # If no existing position, not a duplicate
        if ticker not in trader.positions:
            return False
            
        pos = trader.positions[ticker]
        # Compare with signal timestamp
        signal_timestamp = signal.get("timestamp", 0)
        if signal_timestamp and signal_timestamp == pos.entry_time:
            return True
        
        return False

    def check_daily_report(self):
        today = date.today()
        if self._last_report_date != today:
            yesterday = today - timedelta(days=1)
            self.reporter.generate_daily_report(yesterday)
            self._last_report_date = today

    def check_interim_report(self):
        """
        Generate interim report every hour at 30 minutes past every hour
        """
        now = datetime.now()
        
        # Check if it's 30 minutes past the hour and hour changed
        if now.minute == 30 and self._last_report_hour != now.hour:
            # Get open/closed trades for this time window
            report_content = self._generate_interim_report()
            logger.info("Interim report generated at " + now.strftime("%Y-%m-%d %H:%M"))
            logger.info(report_content[:200] + "..." if len(report_content) > 200 else report_content)
            
            self._last_report_hour = now.hour

    def _generate_interim_report(self) -> str:
        """
        Generate interim report of opened and closed trades
        """
        now = datetime.now()
        start_of_hour = now.replace(minute=0, second=0, microsecond=0)
        end_of_hour = start_of_hour + timedelta(hours=1)
        
        # Get all trades for last hour
        all_trades = self.journal.get_trades()  # This will get all trades from journal
        
        # For now, return basic structure - we can enhance later
        content = f"""
INTERIM REPORT - {now.strftime('%Y-%m-%d %H:%M')}
Open Positions: {len(self.monitor.get_all_positions())}
Total Trades Today: {len(self.journal.get_trades(start_date=date.today(), end_date=date.today()))}
Today's PnL: {self.monitor.get_total_stats().get('total_pnl', 0):,.2f}
        """
        return content

    def get_status(self) -> Dict:
        stats = self.monitor.get_total_stats()
        positions = self.monitor.get_all_positions()

        return {
            "capital": stats.get("capital", self.initial_capital),
            "total_pnl": stats.get("total_pnl", 0),
            "total_trades": stats.get("total_trades", 0),
            "win_rate": stats.get("win_rate", 0),
            "open_positions": len(positions),
            "positions": positions,
        }

    def close_all(self):
        closed = self.monitor.close_all_positions()
        for pos in closed:
            self._log_closed_trade(pos)

        logger.info(f"Closed {len(closed)} positions")

    def generate_report(self, report_date: Optional[date] = None) -> str:
        return self.reporter.generate_daily_report(report_date)

    def get_today_trades(self) -> List[Dict]:
        today = date.today()
        return self.journal.get_trades(start_date=today, end_date=today)