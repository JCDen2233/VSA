from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger

CONTEXT_WINDOW = 24


class DatasetGenerator:
    def __init__(self, context_window: int = CONTEXT_WINDOW):
        self.context_window = context_window

    def generate_from_trades(
        self,
        trades: List[dict],
        prices: pd.DataFrame,
        include_vsa: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not trades or prices.empty:
            logger.warning("No trades or prices provided")
            return np.array([]), np.array([])

        prices = prices.sort_values("timestamp").reset_index(drop=True)
        prices = self._add_technicals(prices)

        X, y = [], []
        for trade in trades:
            features = self._extract_context(prices, trade, include_vsa)
            if features is not None:
                X.append(features)
                y.append(1 if trade.get("pnl", 0) > 0 else 0)

        if not X:
            logger.warning("No samples generated")
            return np.array([]), np.array([])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y

    def _add_technicals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        df["returns"] = df["Close"].pct_change()
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

        for window in [5, 10, 20]:
            df[f"SMA_{window}"] = df["Close"].rolling(window).mean()
            df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
            df[f"VOL_SMA_{window}"] = df["Volume"].rolling(window).mean()

        for window in [7, 14, 21]:
            df[f"RSI_{window}"] = self._calculate_rsi(df["Close"], window)
            df[f"ATR_{window}"] = self._calculate_atr(df, window)

        df["BB_upper"], df["BB_middle"], df["BB_lower"] = self._calculate_bollinger(df["Close"])
        df["MACD"], df["MACD_signal"] = self._calculate_macd(df["Close"])

        df["Stoch_K"], df["Stoch_D"] = self._calculate_stochastic(df)
        df["OBV"] = self._calculate_obv(df)

        df = df.ffill().bfill()
        return df

    def _calculate_rsi(self, series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, window: int) -> pd.Series:
        high_low = df["High"] - df["Low"]
        high_close = abs(df["High"] - df["Close"].shift(1))
        low_close = abs(df["Low"] - df["Close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def _calculate_bollinger(self, series: pd.Series, window: int = 20, std_dev: float = 2.0):
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    def _calculate_macd(
        self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal

    def _calculate_stochastic(
        self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        low_min = df["Low"].rolling(k_period).min()
        high_max = df["High"].rolling(k_period).max()
        k = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-10)
        d = k.rolling(d_period).mean()
        return k, d

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
        return obv

    def _extract_context(
        self, prices: pd.DataFrame, trade: dict, include_vsa: bool
    ) -> Optional[np.ndarray]:
        entry_ts = trade.get("entry_time")
        if entry_ts is None:
            return None

        idx = prices[prices["timestamp"] == entry_ts].index
        if len(idx) == 0:
            return None

        idx = idx[0]
        start_idx = idx - self.context_window

        if start_idx < 0:
            return None

        window_df = prices.iloc[start_idx:idx].copy()
        if len(window_df) < self.context_window:
            return None

        features = []

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in window_df.columns:
                vals = window_df[col].values
                features.extend(self._normalize(vals))
                features.append(vals[-1] / (vals.mean() + 1e-10))

        for col in ["returns", "log_returns"]:
            if col in window_df.columns:
                vals = window_df[col].fillna(0).values
                features.extend(self._normalize(vals))
                features.append(vals[-1])

        feature_cols = [
            "SMA_5", "SMA_10", "SMA_20",
            "EMA_5", "EMA_10", "EMA_20",
            "VOL_SMA_5", "VOL_SMA_10", "VOL_SMA_20",
            "RSI_7", "RSI_14", "RSI_21",
            "ATR_7", "ATR_14", "ATR_21",
            "BB_upper", "BB_middle", "BB_lower",
            "MACD", "MACD_signal",
            "Stoch_K", "Stoch_D", "OBV",
        ]
        for col in feature_cols:
            if col in window_df.columns:
                vals = window_df[col].fillna(0).values
                features.extend(self._normalize(vals))
                features.append(vals[-1])

        if include_vsa:
            features.append(self._count_volume_spikes(window_df))
            features.append(self._count_spread_anomalies(window_df))
            features.append(self._get_trend_direction(window_df))
            features.append(self._get_volatility_regime(window_df))

        side = trade.get("side", "LONG")
        features.append(1.0 if side == "LONG" else 0.0)

        return np.array(features, dtype=np.float32)

    def _normalize(self, arr: np.ndarray) -> List[float]:
        arr = np.array(arr)
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val < 1e-10:
            return [0.0] * len(arr)
        return ((arr - min_val) / (max_val - min_val)).tolist()

    def _count_volume_spikes(self, df: pd.DataFrame) -> float:
        if "VOL_SMA_20" not in df.columns:
            return 0.0
        avg_vol = df["VOL_SMA_20"].mean()
        if avg_vol <= 0:
            return 0.0
        spikes = (df["Volume"] > avg_vol * 1.5).sum()
        return spikes / self.context_window

    def _count_spread_anomalies(self, df: pd.DataFrame) -> float:
        if "ATR_14" not in df.columns:
            return 0.0
        avg_spread = df["ATR_14"].mean()
        if avg_spread <= 0:
            return 0.0
        count = 0
        for _, row in df.iterrows():
            spread = row.get("High", 0) - row.get("Low", 0)
            if spread > avg_spread * 1.3:
                count += 1
        return count / self.context_window

    def _get_trend_direction(self, df: pd.DataFrame) -> float:
        if "SMA_20" not in df.columns or "Close" not in df.columns:
            return 0.0
        sma20 = df["SMA_20"].iloc[-1]
        close = df["Close"].iloc[-1]
        if sma20 <= 0:
            return 0.0
        return 1.0 if close > sma20 else -1.0

    def _get_volatility_regime(self, df: pd.DataFrame) -> float:
        if "ATR_14" not in df.columns or "Close" not in df.columns:
            return 0.0
        atr = df["ATR_14"].iloc[-1]
        close = df["Close"].iloc[-1]
        if close <= 0:
            return 0.0
        volatility = atr / close * 100
        if volatility > 2.0:
            return 1.0
        elif volatility > 1.0:
            return 0.0
        else:
            return -1.0

    def save_dataset(self, X: np.ndarray, y: np.ndarray, path: Path):
        np.savez(path, X=X, y=y)
        logger.info(f"Saved dataset to {path}")

    def load_dataset(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(path)
        return data["X"], data["y"]