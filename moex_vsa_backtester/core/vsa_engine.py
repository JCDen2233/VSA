from typing import Dict, Optional

import pandas as pd
from loguru import logger

from config import config

RR_RATIO = 2.0


def detect_sr_levels(df: pd.DataFrame, window: int = 20) -> Dict:
    if df.empty or len(df) < window:
        return {"support": [], "resistance": []}

    highs = df["High"].rolling(window=window, center=False).max()
    lows = df["Low"].rolling(window=window, center=False).min()

    resistance = highs.dropna().unique().tolist()[-10:]
    support = lows.dropna().unique().tolist()[-10:]

    return {"support": support, "resistance": resistance}


def _check_session(dt: pd.Timestamp) -> bool:
    hour = dt.hour
    minute = dt.minute
    if hour < 10 or hour >= 18:
        return False
    if hour == 18 and minute >= 25:
        return False
    return True


def _is_volume_spike(volume: float, sma_vol_20: float) -> bool:
    return volume > sma_vol_20 * 1.5 if sma_vol_20 else False


def _is_spread_normal(spread: float, avg_spread_20: float) -> bool:
    return spread >= avg_spread_20 if avg_spread_20 else False


def generate_vsa_signals(
    df_h1: pd.DataFrame, df_d1: Optional[pd.DataFrame] = None, levels: Dict = None
) -> pd.DataFrame:
    if df_h1.empty:
        return pd.DataFrame()

    signals = []
    df = df_h1.copy()
    df = df.reset_index(drop=True)

    df_d1_indexed = None
    if df_d1 is not None and not df_d1.empty and "SMA_Close_50" in df_d1.columns:
        df_d1_indexed = df_d1.set_index("timestamp")

    for i in range(1, len(df) - 1):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        next_bar = df.iloc[i + 1]
        
        curr_ts = curr.get("timestamp", 0)

        trend_long = False
        if df_d1_indexed is not None:
            d1_before = df_d1_indexed[df_d1_indexed.index <= curr_ts]
            if not d1_before.empty:
                last_d1 = d1_before.iloc[-1]
                trend_long = last_d1["Close"] > last_d1["SMA_Close_50"]

        if not _check_session(pd.to_datetime(curr.get("Date") + " " + curr.get("Time"))):
            continue

        spread = curr["Spread"]
        volume = curr["Volume"]
        sma_vol = curr.get("SMA_Vol_20", 0)
        avg_spread = curr.get("Avg_Spread_20", 0)

        if not _is_volume_spike(volume, sma_vol):
            continue
        if not _is_spread_normal(spread, avg_spread):
            continue

        low = curr["Low"]
        high = curr["High"]
        close = curr["Close"]
        mid = (high + low) / 2

        signal_type = None
        entry_price = None
        sl_price = None
        confirmed = False

        if trend_long and levels:
            for level in levels.get("support", []):
                if low < level and close > low + 0.66 * spread:
                    signal_type = "LONG"
                    sl_price = level * 0.9985
                    entry_price = close
                    if next_bar["Close"] > mid:
                        confirmed = True
                    break

        elif not trend_long and levels:
            allow_short = config.get("ALLOW_SHORT", False)
            
            for level in levels.get("resistance", []):
                if high > level and close < high - 0.66 * spread:
                    if allow_short:
                        signal_type = "SHORT"
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

        if signal_type:
            risk = abs(entry_price - sl_price)
            tp_price = entry_price + risk * RR_RATIO if signal_type == "LONG" else entry_price - risk * RR_RATIO
            signals.append(
                {
                    "timestamp": curr["timestamp"],
                    "Date": curr["Date"],
                    "Time": curr["Time"],
                    "signal_type": signal_type,
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "confirmed": confirmed,
                }
            )

    df_signals = pd.DataFrame(signals)
    logger.info(f"Generated {len(df_signals)} VSA signals")
    return df_signals