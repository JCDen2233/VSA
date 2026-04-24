"""VSA (Volume Spread Analysis) signal generation engine."""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from domain import (
    Signal,
    SignalType,
    Bar,
    VOLUME_SPIKE_THRESHOLD,
    PRICE_CHANGE_THRESHOLD,
    LOOKBACK_PERIOD,
    MARKET_OPEN_HOUR,
    MARKET_CLOSE_HOUR,
)

RR_RATIO = 2.0


def detect_sr_levels(df: pd.DataFrame, window: int = LOOKBACK_PERIOD) -> Dict[str, List[float]]:
    """Detect support and resistance levels from price data.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size for level detection
        
    Returns:
        Dictionary with 'support' and 'resistance' level lists
    """
    if df.empty or len(df) < window:
        return {"support": [], "resistance": []}

    highs = df["High"].rolling(window=window, center=False).max()
    lows = df["Low"].rolling(window=window, center=False).min()

    resistance = highs.dropna().unique().tolist()[-10:]
    support = lows.dropna().unique().tolist()[-10:]

    return {"support": support, "resistance": resistance}


def _check_session(dt: pd.Timestamp) -> bool:
    """Check if timestamp is within trading session hours.
    
    Args:
        dt: Timestamp to check
        
    Returns:
        True if within session, False otherwise
    """
    hour = dt.hour
    minute = dt.minute
    if hour < MARKET_OPEN_HOUR or hour >= MARKET_CLOSE_HOUR:
        return False
    if hour == MARKET_CLOSE_HOUR - 1 and minute >= 25:
        return False
    return True


def _is_volume_spike(volume: float, sma_vol: float, threshold: float = VOLUME_SPIKE_THRESHOLD) -> bool:
    """Check if volume represents a spike above average.
    
    Args:
        volume: Current bar volume
        sma_vol: Simple moving average of volume
        threshold: Multiplier threshold for spike detection
        
    Returns:
        True if volume spike detected
    """
    return volume > sma_vol * threshold if sma_vol else False


def _is_spread_normal(spread: float, avg_spread: float) -> bool:
    """Check if spread is within normal range.
    
    Args:
        spread: Current bar spread
        avg_spread: Average spread over lookback period
        
    Returns:
        True if spread is normal
    """
    return spread >= avg_spread if avg_spread else False


def generate_vsa_signals(
    df_h1: pd.DataFrame, 
    df_d1: Optional[pd.DataFrame] = None, 
    levels: Optional[Dict[str, List[float]]] = None,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Generate VSA trading signals from OHLCV data.
    
    Args:
        df_h1: Hourly OHLCV DataFrame
        df_d1: Daily OHLCV DataFrame for trend context
        levels: Support/resistance levels dictionary
        allow_short: Whether to allow SHORT signals
        
    Returns:
        DataFrame with generated signals
    """
    if df_h1.empty:
        return pd.DataFrame()

    signals: List[Signal] = []
    df = df_h1.copy().reset_index(drop=True)

    df_d1_indexed = None
    if df_d1 is not None and not df_d1.empty and "SMA_Close_50" in df_d1.columns:
        df_d1_indexed = df_d1.set_index("timestamp")

    for i in range(1, len(df) - 1):
        curr = df.iloc[i]
        next_bar = df.iloc[i + 1]
        
        curr_ts = curr.get("timestamp", 0)

        # Determine trend direction from daily chart
        trend_long = False
        if df_d1_indexed is not None:
            d1_before = df_d1_indexed[df_d1_indexed.index <= curr_ts]
            if not d1_before.empty:
                last_d1 = d1_before.iloc[-1]
                trend_long = last_d1["Close"] > last_d1["SMA_Close_50"]

        # Check session hours
        try:
            session_dt = pd.to_datetime(curr.get("Date", "") + " " + curr.get("Time", ""))
            if not _check_session(session_dt):
                continue
        except (ValueError, TypeError):
            continue

        spread = curr["Spread"]
        volume = curr["Volume"]
        sma_vol = curr.get("SMA_Vol_20", 0)
        avg_spread = curr.get("Avg_Spread_20", 0)

        # Validate volume spike and spread
        if not _is_volume_spike(volume, sma_vol):
            continue
        if not _is_spread_normal(spread, avg_spread):
            continue

        low = curr["Low"]
        high = curr["High"]
        close = curr["Close"]
        mid = (high + low) / 2

        signal_type: Optional[SignalType] = None
        entry_price: Optional[float] = None
        sl_price: Optional[float] = None
        confirmed = False

        # LONG signal logic
        if trend_long and levels:
            for level in levels.get("support", []):
                if low < level and close > low + 0.66 * spread:
                    signal_type = SignalType.LONG
                    sl_price = level * 0.9985
                    entry_price = close
                    if next_bar["Close"] > mid:
                        confirmed = True
                    break

        # SHORT signal logic
        elif not trend_long and levels:
            for level in levels.get("resistance", []):
                if high > level and close < high - 0.66 * spread:
                    if allow_short:
                        signal_type = SignalType.SHORT
                        sl_price = level * 1.0015
                        entry_price = close
                        if next_bar["Close"] < mid:
                            confirmed = True
                        break
                    else:
                        logger.warning(
                            f"⚠️ SHORT SIGNAL DETECTED (blocked by config): "
                            f"{curr.get('Date')} {curr.get('Time')} | "
                            f"Entry: {close:.4f} | "
                            f"Recommendation: DO NOT ENTER LONG - market may be overbought"
                        )
                        break

        if signal_type and entry_price and sl_price:
            risk = abs(entry_price - sl_price)
            tp_price = entry_price + risk * RR_RATIO if signal_type == SignalType.LONG else entry_price - risk * RR_RATIO
            
            signals.append(Signal(
                ticker=curr.get("Ticker", ""),
                signal_type=signal_type,
                timestamp=pd.to_datetime(curr["timestamp"]) if "timestamp" in curr else session_dt,
                price=entry_price,
                volume_spike=volume / sma_vol if sma_vol else 0,
                confidence=1.0 if confirmed else 0.5,
                stop_loss=sl_price,
                take_profit=tp_price,
                metadata={"spread": spread, "confirmed": confirmed},
            ))

    logger.info(f"Generated {len(signals)} VSA signals")
    
    # Convert to DataFrame for backward compatibility
    if not signals:
        return pd.DataFrame()
    
    return pd.DataFrame([
        {
            "timestamp": s.timestamp,
            "Date": s.timestamp.strftime("%Y-%m-%d") if hasattr(s.timestamp, 'strftime') else "",
            "Time": s.timestamp.strftime("%H:%M") if hasattr(s.timestamp, 'strftime') else "",
            "signal_type": s.signal_type.value,
            "entry_price": s.price,
            "sl_price": s.stop_loss,
            "tp_price": s.take_profit,
            "confirmed": s.metadata.get("confirmed", False),
        }
        for s in signals
    ])