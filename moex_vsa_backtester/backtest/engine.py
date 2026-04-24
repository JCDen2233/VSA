"""Backtest engine for VSA strategies."""

from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
from loguru import logger

from domain import (
    Trade,
    TradeDirection,
    Position,
    Signal,
    DEFAULT_RISK_PER_TRADE,
    DEFAULT_COMMISSION,
    DEFAULT_SLIPPAGE,
)
from core.risk_manager import apply_rr_exits, calculate_position_size, calculate_risk_reward


class VSABacktester:
    """VSA strategy backtester with risk management.
    
    Attributes:
        capital: Starting capital
        risk_pct: Risk percentage per trade
        rr_ratio: Risk-reward ratio
        commission: Trading commission rate
        slippage: Slippage rate
    """
    
    def __init__(
        self,
        capital: float = 1_000_000,
        risk_pct: float = DEFAULT_RISK_PER_TRADE,
        rr_ratio: float = 2.0,
        commission: float = DEFAULT_COMMISSION,
        slippage: float = DEFAULT_SLIPPAGE,
    ):
        self.capital = capital
        self.initial_capital = capital
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.commission = commission
        self.slippage = slippage
        
        self.position: Optional[Dict[str, Any]] = None
        self.trades: List[Trade] = []
        self.equity: List[Dict[str, Any]] = []

    def run(self, signals: pd.DataFrame, prices: pd.DataFrame) -> List[Trade]:
        """Run backtest on given signals and prices.
        
        Args:
            signals: DataFrame with trading signals
            prices: DataFrame with OHLCV price data
            
        Returns:
            List of completed trades
        """
        if signals.empty or prices.empty:
            logger.warning("Empty signals or prices")
            return []

        prices = prices.set_index("timestamp")
        signal_timestamps = signals["timestamp"].tolist()
        
        for i, ts in enumerate(signal_timestamps):
            if ts not in prices.index:
                continue

            price_row = prices.loc[ts]
            self._open_position(signals.iloc[i], price_row)

            if self.position is None:
                continue

            for next_ts in list(prices.index):
                if next_ts <= ts:
                    continue
                self._check_exits(prices, next_ts)
                if self.position is None:
                    break

            self._close_expired_positions(prices, cutoff_hour=18)

        self._close_open_positions(prices)
        logger.info(f"Backtest completed: {len(self.trades)} trades")
        return self.trades

    def _open_position(self, signal: pd.Series, price_row: pd.Series) -> None:
        """Open a new position based on signal.
        
        Args:
            signal: Signal data series
            price_row: Price data at signal time
        """
        if self.position is not None:
            return

        entry = signal.get("entry_price", price_row.get("Close"))
        sl = signal.get("sl_price", entry * 0.9985)
        tp = signal.get("tp_price") or apply_rr_exits(entry, sl, self.rr_ratio)

        size = calculate_position_size(self.capital, self.risk_pct, entry, sl)
        if size <= 0:
            return

        # Convert signal_type string to TradeDirection
        signal_type_str = signal.get("signal_type", "LONG")
        if isinstance(signal_type_str, str):
            direction = TradeDirection.BUY if signal_type_str == "LONG" else TradeDirection.SELL
        else:
            direction = signal_type_str

        self.position = {
            "entry_time": signal.get("timestamp"),
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "size": size,
            "direction": direction,
            "entry_bar": signal,
        }
        logger.info(
            f"Opened {direction.value}: entry={entry}, "
            f"sl={sl}, tp={tp}, size={size}"
        )

    def _check_exits(self, prices: pd.DataFrame, current_ts: int) -> None:
        """Check and execute position exits based on price levels.
        
        Args:
            prices: Price DataFrame
            current_ts: Current timestamp
        """
        if self.position is None:
            return

        pos = self.position
        entry = pos["entry"]
        sl = pos["sl"]
        tp = pos["tp"]
        direction = pos["direction"]
        size = pos["size"]

        if current_ts not in prices.index:
            return
        price = prices.loc[current_ts]
        current_price = price.get("Close")

        pnl = 0.0
        status = "OPEN"
        exit_price = None

        if direction == TradeDirection.BUY:
            # Check for SL first (per requirement: if both TP and SL hit, count SL)
            if current_price <= sl:
                pnl = (sl - entry) * size
                status = "SL"
                exit_price = sl
            elif current_price >= tp:
                pnl = (tp - entry) * size
                status = "TP"
                exit_price = tp
            elif current_price >= entry + (tp - entry) * 0.5:
                partial_size = size // 2
                pnl_partial = (entry + (tp - entry) * 0.5 - entry) * partial_size
                self.capital += pnl_partial
                pos["size"] -= partial_size
                pos["sl"] = entry
                logger.info("Partial close at 1.5R, moved SL to BE")
        else:  # SELL
            # Check for SL first (per requirement: if both TP and SL hit, count SL)
            if current_price >= sl:
                pnl = (entry - sl) * size
                status = "SL"
                exit_price = sl
            elif current_price <= tp:
                pnl = (entry - tp) * size
                status = "TP"
                exit_price = tp
            elif current_price <= entry - (entry - tp) * 0.5:
                partial_size = size // 2
                pnl_partial = (entry - (entry - tp) * 0.5 - entry) * partial_size
                self.capital += pnl_partial
                pos["size"] -= partial_size
                pos["sl"] = entry
                logger.info("Partial close at 1.5R, moved SL to BE")

        if status != "OPEN":
            self._close_position(current_ts, exit_price or current_price, pnl, status)

        self.equity.append({"timestamp": current_ts, "equity": self.capital + pnl})

    def _close_position(
        self, exit_ts: int, exit_price: float, pnl: float, status: str
    ) -> None:
        """Close position and record trade.
        
        Args:
            exit_ts: Exit timestamp
            exit_price: Exit price
            pnl: Profit/loss
            status: Exit status (SL, TP, EOD_CLOSE, etc.)
        """
        pos = self.position
        entry = pos["entry"]
        size = pos["size"]
        direction = pos["direction"]
        
        # Calculate costs
        trade_value = entry * size
        commission = trade_value * self.commission
        slippage_cost = trade_value * self.slippage
        
        # Create Trade record
        trade = Trade(
            ticker="",  # Will be filled by caller if needed
            direction=direction,
            entry_price=entry,
            exit_price=exit_price,
            quantity=size,
            entry_time=pos["entry_time"],
            exit_time=exit_ts,
            pnl=pnl,
            pnl_percent=pnl / (entry * size) if entry > 0 else 0,
            commission=commission,
            slippage=slippage_cost,
        )
        
        self.trades.append(trade)
        self.capital += pnl - commission - slippage_cost
        logger.info(f"Closed position: {status}, PnL={pnl:.2f}")
        self.position = None

    def _close_expired_positions(self, prices: pd.DataFrame, cutoff_hour: int = 18) -> None:
        """Close positions that have exceeded holding period.
        
        Args:
            prices: Price DataFrame
            cutoff_hour: Hour at which to close positions
        """
        if self.position is None:
            return

        entry_ts = self.position.get("entry_time")
        if entry_ts is None:
            return

        entry_idx = prices.index.get_loc(entry_ts) if entry_ts in prices.index else 0

        for i in range(entry_idx, min(entry_idx + 24, len(prices))):
            ts = prices.index[i]
            dt = pd.to_datetime(ts, unit="s")
            if dt.hour >= cutoff_hour:
                price = prices.loc[ts]["Close"]
                pnl = self._calculate_pnl(price)
                self._close_position(ts, price, pnl, "EOD_CLOSE")
                break

    def _close_open_positions(self, prices: pd.DataFrame) -> None:
        """Close any remaining open positions at end of backtest.
        
        Args:
            prices: Price DataFrame with final prices
        """
        if self.position is None:
            return

        last_ts = prices.index[-1]
        last_price = prices.iloc[-1]["Close"]
        pnl = self._calculate_pnl(last_price)
        self._close_position(last_ts, last_price, pnl, "END")

    def _calculate_pnl(self, current_price: float) -> float:
        """Calculate current PnL for open position.
        
        Args:
            current_price: Current market price
            
        Returns:
            Profit/loss amount
        """
        if self.position is None:
            return 0.0
        
        pos = self.position
        entry = pos["entry"]
        size = pos["size"]
        direction = pos["direction"]
        
        if direction == TradeDirection.BUY:
            return (current_price - entry) * size
        else:
            return (entry - current_price) * size

    def save_trades(self, ticker: str, tf: str, output_dir: Path = Path("logs")):
        if not self.trades:
            logger.warning("No trades to save")
            return

        df = pd.DataFrame(self.trades)
        path = output_dir / f"trades_{ticker}_{tf}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved trades to {path}")