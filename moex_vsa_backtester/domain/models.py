"""Domain models for VSA backtester."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .enums import SignalType, PositionStatus, TradeDirection


@dataclass
class Bar:
    """OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    ticker: str = ""
    
    @property
    def range(self) -> float:
        """Bar price range."""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """Bar body size."""
        return abs(self.close - self.open)
    
    def is_bullish(self) -> bool:
        """Check if bar is bullish."""
        return self.close > self.open
    
    def is_bearish(self) -> bool:
        """Check if bar is bearish."""
        return self.close < self.open


@dataclass
class Signal:
    """VSA trading signal."""
    ticker: str
    signal_type: SignalType
    timestamp: datetime
    price: float
    volume_spike: float
    confidence: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Position:
    """Trading position."""
    ticker: str
    direction: TradeDirection
    entry_price: float
    quantity: int
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    
    def update_pnl(self, current_price: float) -> None:
        """Update PnL based on current price."""
        if self.direction == TradeDirection.BUY:
            self.pnl = (current_price - self.entry_price) * self.quantity
            self.pnl_percent = (current_price - self.entry_price) / self.entry_price
        else:
            self.pnl = (self.entry_price - current_price) * self.quantity
            self.pnl_percent = (self.entry_price - current_price) / self.entry_price
    
    def close(self, exit_price: float, exit_time: datetime) -> None:
        """Close position with given exit price."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = PositionStatus.CLOSED
        self.update_pnl(exit_price)


@dataclass
class Trade:
    """Completed trade record."""
    ticker: str
    direction: TradeDirection
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    commission: float = 0.0
    slippage: float = 0.0
    
    @property
    def net_pnl(self) -> float:
        """PnL after costs."""
        return self.pnl - self.commission - self.slippage


@dataclass
class RiskMetrics:
    """Risk management metrics."""
    total_capital: float
    current_exposure: float
    positions_count: int
    max_position_size: float
    risk_per_trade: float
    drawdown: float = 0.0
    var_95: float = 0.0
