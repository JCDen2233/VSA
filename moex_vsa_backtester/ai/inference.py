from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ai.dataset import DatasetGenerator, CONTEXT_WINDOW
from ai.trainer import ModelTrainer


class TradePredictor:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_type: str = "mlp",
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        self.dataset_generator = DatasetGenerator(context_window=CONTEXT_WINDOW)
        
        self.trainer: Optional[ModelTrainer] = None
        
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            logger.warning(f"Model path {model_path} not found, using untrained predictor")

    def load_model(self, model_path: Path):
        self.trainer = ModelTrainer()
        self.trainer.load(model_path)
        logger.debug(f"Model loaded from {model_path}")

    def predict(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        if signals.empty or prices.empty:
            return signals

        if self.trainer is None:
            logger.warning("No trained model, returning signals without AI filter")
            signals["ai_probability"] = 0.5
            signals["ai_predicted_success"] = False
            return signals

        prices = prices.sort_values("timestamp").reset_index(drop=True)
        
        predictions = []
        
        for idx, signal in signals.iterrows():
            trade = {
                "entry_time": signal.get("timestamp"),
                "side": signal.get("signal_type", "LONG"),
                "pnl": 0,
            }
            
            features = self._extract_single_context(prices, trade)
            
            if features is None:
                predictions.append({"ai_probability": 0.5, "ai_predicted_success": False})
                continue
            
            features = features.reshape(1, -1)
            
            prob = self.trainer.predict_proba(features)
            if prob.ndim > 0:
                prob = prob[0]
            else:
                prob = float(prob)
            predicted_success = prob >= self.threshold
            
            predictions.append({
                "ai_probability": float(prob),
                "ai_predicted_success": bool(predicted_success),
            })
        
        pred_df = pd.DataFrame(predictions)
        signals = pd.concat([signals.reset_index(drop=True), pred_df], axis=1)
        
        logger.info(f"AI predictions added to {len(signals)} signals")
        return signals

    def filter_signals(
        self,
        signals: pd.DataFrame,
        min_probability: float = 0.6,
    ) -> pd.DataFrame:
        if "ai_probability" not in signals.columns:
            logger.warning("No AI predictions found, returning all signals")
            return signals
        
        filtered = signals[signals["ai_probability"] >= min_probability].copy()
        
        logger.info(
            f"Filtered {len(signals)} signals to {len(filtered)} "
            f"(min_prob={min_probability})"
        )
        return filtered

    def rank_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        if "ai_probability" not in signals.columns:
            return signals
        
        signals = signals.sort_values("ai_probability", ascending=False).reset_index(drop=True)
        return signals

    def _extract_single_context(
        self, prices: pd.DataFrame, trade: dict
    ) -> Optional[np.ndarray]:
        entry_ts = trade.get("entry_time")
        if entry_ts is None:
            logger.debug("No entry_ts in trade")
            return None

        prices = prices[prices["timestamp"] <= entry_ts].copy()
        
        if prices.empty:
            logger.debug(f"No prices data available before signal timestamp {entry_ts}")
            return None

        prices = prices.sort_values("timestamp").reset_index(drop=True)

        idx = prices[prices["timestamp"] == entry_ts].index
        if len(idx) == 0:
            logger.debug(f"Signal timestamp {entry_ts} not found in filtered prices (range: {prices['timestamp'].min()} - {prices['timestamp'].max()})")
            return None

        idx = idx[0]
        start_idx = idx - CONTEXT_WINDOW

        if start_idx < 0:
            logger.debug(f"Not enough bars before signal: idx={idx}, need {CONTEXT_WINDOW}")
            return None

        window_df = prices.iloc[start_idx:idx].copy()
        if len(window_df) < CONTEXT_WINDOW:
            logger.debug(f"Not enough context bars: {len(window_df)} < {CONTEXT_WINDOW}")
            return None

        return self._extract_features(window_df, trade)

    def _extract_features(self, window_df: pd.DataFrame, trade: dict) -> np.ndarray:
        df = window_df.copy()
        
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
        
        features = []
        
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                vals = df[col].values
                features.extend(self._normalize(vals))
                features.append(vals[-1] / (vals.mean() + 1e-10))
        
        for col in ["returns", "log_returns"]:
            if col in df.columns:
                vals = df[col].fillna(0).values
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
            if col in df.columns:
                vals = df[col].fillna(0).values
                features.extend(self._normalize(vals))
                features.append(vals[-1])
        
        features.append(self._count_volume_spikes(df))
        features.append(self._count_spread_anomalies(df))
        features.append(self._get_trend_direction(df))
        features.append(self._get_volatility_regime(df))
        
        side = trade.get("side", "LONG")
        features.append(1.0 if side == "LONG" else 0.0)
        
        return np.array(features, dtype=np.float32)

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

    def _calculate_bollinger(self, series: pd.Series, window: int = 20):
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, sma, lower

    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14):
        low_min = df["Low"].rolling(k_period).min()
        high_max = df["High"].rolling(k_period).max()
        k = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-10)
        d = k.rolling(3).mean()
        return k, d

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
        return obv

    def _normalize(self, arr: np.ndarray) -> list:
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
        return spikes / CONTEXT_WINDOW

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
        return count / CONTEXT_WINDOW

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