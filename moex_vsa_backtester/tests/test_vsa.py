import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from core.vsa_engine import detect_sr_levels, generate_vsa_signals
from core.data_loader import DataPreparator


@pytest.fixture
def sample_df():
    dates = pd.date_range(start="2024-01-01", periods=50, freq="h")
    data = {
        "timestamp": [int(d.timestamp()) for d in dates],
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Time": [d.strftime("%H:%M") for d in dates],
        "Open": np.random.uniform(100, 110, 50),
        "High": np.random.uniform(110, 120, 50),
        "Low": np.random.uniform(90, 100, 50),
        "Close": np.random.uniform(100, 110, 50),
        "Volume": np.random.uniform(1000, 5000, 50).astype(int),
    }
    df = pd.DataFrame(data)
    df["SMA_Vol_20"] = df["Volume"].rolling(20).mean()
    df["Spread"] = df["High"] - df["Low"]
    df["Avg_Spread_20"] = df["Spread"].rolling(20).mean()
    return df


@pytest.fixture
def sample_d1():
    dates = pd.date_range(start="2024-01-01", periods=30, freq="d")
    data = {
        "timestamp": [int(d.timestamp()) for d in dates],
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Time": ["00:00"] * 30,
        "Open": np.random.uniform(100, 110, 30),
        "High": np.random.uniform(110, 120, 30),
        "Low": np.random.uniform(90, 100, 30),
        "Close": np.random.uniform(100, 110, 30),
        "Volume": np.random.uniform(10000, 50000, 30).astype(int),
    }
    df = pd.DataFrame(data)
    df["SMA_Close_50"] = df["Close"].rolling(50).mean()
    return df


class TestVSAEngine:
    def test_detect_sr_levels(self, sample_df):
        levels = detect_sr_levels(sample_df, window=20)
        assert "support" in levels
        assert "resistance" in levels
        assert isinstance(levels["support"], list)
        assert isinstance(levels["resistance"], list)

    def test_detect_sr_levels_empty(self):
        df = pd.DataFrame()
        levels = detect_sr_levels(df, window=20)
        assert levels == {"support": [], "resistance": []}

    def test_generate_vsa_signals_empty(self):
        signals = generate_vsa_signals(pd.DataFrame())
        assert signals.empty

    def test_generate_vsa_signals_no_spike(self, sample_df):
        df = sample_df.copy()
        df["Volume"] = 100
        df["SMA_Vol_20"] = 100
        signals = generate_vsa_signals(df)
        assert signals.empty

    def test_generate_vsa_signals_with_trend_long(self, sample_df, sample_d1):
        df = sample_df.copy()
        df = df.reset_index(drop=True)
        df["SMA_Vol_20"] = df["Volume"].rolling(20).mean()
        df["Avg_Spread_20"] = df["Spread"].rolling(20).mean()
        
        sample_d1.iloc[-1, sample_d1.columns.get_loc("Close")] = 150
        sample_d1.iloc[-1, sample_d1.columns.get_loc("SMA_Close_50")] = 100
        
        levels = {"support": [95], "resistance": [115]}
        
        signals = generate_vsa_signals(df, sample_d1, levels)
        assert isinstance(signals, pd.DataFrame)


class TestDataLoader:
    def test_data_preparator_init(self):
        prep = DataPreparator(["SBER"], ["D1", "H1"])
        assert prep.tickers == ["SBER"]
        assert prep.timeframes == ["D1", "H1"]

    def test_data_preparator_custom_tf(self):
        prep = DataPreparator(["PLZL"], ["H4"])
        assert prep.timeframes == ["H4"]