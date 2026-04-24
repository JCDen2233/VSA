"""Domain enums for type safety."""

from enum import Enum


class SignalType(Enum):
    """VSA signal types."""
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Position lifecycle status."""
    OPEN = "open"
    CLOSED = "closed"
    STOPPED = "stopped"


class TradeDirection(Enum):
    """Trade direction."""
    BUY = "buy"
    SELL = "sell"


class ModelArchitecture(Enum):
    """Supported ML model architectures."""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
