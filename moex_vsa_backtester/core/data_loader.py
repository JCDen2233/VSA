from typing import List, Tuple

import pandas as pd
from loguru import logger

from db import fetch_ohlcv


class DataPreparator:
    def __init__(self, tickers: List[str], timeframes: List[str] = None):
        self.tickers = tickers
        self.timeframes = timeframes or ["D1", "H1"]
        self.cache = {}

    def load_and_prepare(
        self, ticker: str, start_ts: int, end_ts: int, min_bars: int = 100
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_d1 = self._load_tf(ticker, "D1", start_ts, end_ts, min_bars)
        df_h1 = self._load_tf(ticker, "H1", start_ts, end_ts, min_bars)

        df_d1 = self._add_indicators_d1(df_d1)
        df_h1 = self._add_indicators_h1(df_h1)

        return df_d1, df_h1

    def _load_tf(self, ticker: str, tf: str, start_ts: int, end_ts: int, min_bars: int = 100) -> pd.DataFrame:
        # Сначала пробуем загрузить по временному диапазону
        df = fetch_ohlcv(ticker, tf, start_ts, end_ts)
        
        # Если данных недостаточно, загружаем последние N баров независимо от времени
        if len(df) < min_bars:
            logger.debug(f"{ticker}_{tf}: Загружено {len(df)} баров (< {min_bars}), загружаю последние {min_bars} баров")
            df = fetch_ohlcv_last_bars(ticker, tf, min_bars)
        
        if df.empty:
            logger.warning(f"Empty data for {ticker}_{tf}")
            return df

        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.ffill().bfill()
        return df

    def _add_indicators_d1(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "Close" not in df.columns:
            return df
        df["SMA_Close_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
        return df

    def _add_indicators_h1(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df["SMA_Vol_20"] = df["Volume"].rolling(window=20, min_periods=1).mean()
        df["Spread"] = df["High"] - df["Low"]
        df["Avg_Spread_20"] = df["Spread"].rolling(window=20, min_periods=1).mean()
        return df

    def merge_context(self, df_h1: pd.DataFrame, df_d1: pd.DataFrame) -> pd.DataFrame:
        if df_h1.empty or df_d1.empty:
            return df_h1
        df_d1 = df_d1.set_index("timestamp")
        df_h1 = df_h1.set_index("timestamp")
        df_h1["D1_Close"] = df_d1["Close"]
        df_h1["SMA_Close_50"] = df_d1["SMA_Close_50"]
        df_h1 = df_h1.reset_index()
        df_h1["D1_Close"] = df_h1["D1_Close"].ffill()
        df_h1["SMA_Close_50"] = df_h1["SMA_Close_50"].ffill()
        return df_h1