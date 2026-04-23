import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from config import config


class DatabaseManager:
    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(
                config.db_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
        return self._engine

    def close(self):
        if self._engine:
            self._engine.dispose()
            self._engine = None


_db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    return _db_manager


def fetch_ohlcv(ticker: str, tf: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    from loguru import logger

    table_name = f"{ticker.upper()}_{tf.upper()}"
    query = text(f"""
        SELECT timestamp, Date, Time, Open, High, Low, Close, Volume
        FROM {table_name}
        WHERE timestamp >= :start_ts AND timestamp < :end_ts
        ORDER BY timestamp ASC
    """)

    try:
        with _db_manager.engine.connect() as conn:
            result = conn.execute(query, {"start_ts": start_ts, "end_ts": end_ts})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            if df.empty:
                return df
            df["timestamp"] = df["timestamp"].astype("int64")
            df["Open"] = df["Open"].astype(float)
            df["High"] = df["High"].astype(float)
            df["Low"] = df["Low"].astype(float)
            df["Close"] = df["Close"].astype(float)
            df["Volume"] = df["Volume"].astype(int)
            logger.info(f"Loaded {len(df)} rows from {table_name}")
            return df
    except Exception as e:
        logger.error(f"Error fetching {table_name}: {e}")
        raise