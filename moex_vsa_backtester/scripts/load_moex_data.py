#!/usr/bin/env python3
"""
Скрипт для загрузки исторических данных MOEX в базу данных.
Использует официальный API Московской Биржи.
"""

import argparse
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import requests
from loguru import logger
from db import get_db_manager
from sqlalchemy import text


# Список популярных инструментов MOEX
DEFAULT_TICKERS = [
    "SBER", "GAZP", "LKOH", "ROSN", "VTBR", "NVTK", "YNDX", "TCSG",
    "BSPB", "MGNT", "X5", "PLZL", "GMKN", "NLMK", "CHMF", "SNGSP",
    "RTKM", "URKA", "POLY", "MTSS", "AFKS", "PIKK", "MRKH", "MOEX"
]

# Криптовалюты и другие инструменты
CRYPTO_TICKERS = ["BITCOIN", "BITCOINC", "EURUSD"]

TIMEFRAMES = {
    "D1": 86400,  # 1 день в секундах
    "H1": 3600,   # 1 час
    "H4": 14400,  # 4 часа
    "M30": 1800,  # 30 минут
    "M15": 900,   # 15 минут
    "M5": 300,    # 5 минут
}


def fetch_moex_data(
    ticker: str,
    timeframe: str = "D1",
    from_date: str = "2020-01-01",
    to_date: str = None,
) -> pd.DataFrame:
    """
    Загружает данные с API MOEX ISS.
    
    Args:
        ticker: Тикер инструмента (например, SBER)
        timeframe: Таймфрейм (D1, H1, H4, M30, M15, M5)
        from_date: Дата начала в формате YYYY-MM-DD
        to_date: Дата окончания (по умолчанию сегодня)
    
    Returns:
        DataFrame с колонками: timestamp, Date, Time, Open, High, Low, Close, Volume
    """
    if to_date is None:
        to_date = datetime.now().strftime("%Y-%m-%d")
    
    # MOEX API endpoint - используем правильный формат URL
    base_url = "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities"
    
    # Для криптовалют и forex используем другие борды или источники
    if ticker in CRYPTO_TICKERS:
        logger.warning(f"Инструмент {ticker} требует отдельного источника данных")
        return pd.DataFrame()
    
    params = {
        "from": from_date,
        "to": to_date,
        "interval": {"D1": 1, "H1": 4, "H4": 5}.get(timeframe, 1),
        "sort_order": "TRADEDATE",
        "limit": 10000,
    }
    
    try:
        all_data = []
        start = 0
        
        while True:
            url = f"{base_url}/{ticker}.json?start={start}"
            for key, value in params.items():
                url += f"&{key}={value}"
            
            logger.debug(f"Запрос: {url[:100]}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "history" not in data:
                logger.warning(f"Нет ключа 'history' в ответе для {ticker}")
                break
            
            history = data["history"]
            if not history:
                break
            
            all_data.extend(history)
            
            if len(history) < 100:
                break
            
            start += 100
        
        if not all_data:
            logger.warning(f"Нет данных для {ticker}")
            return pd.DataFrame()
        
        # MOEX возвращает данные в формате {columns: [...], data: [[...], ...]}
        # Преобразуем в DataFrame
        if isinstance(all_data, dict) and 'columns' in all_data and 'data' in all_data:
            columns = all_data['columns']
            data = all_data['data']
            df = pd.DataFrame(data, columns=columns)
        else:
            df = pd.DataFrame(all_data)
        
        # Преобразуем в нужный формат
        result = pd.DataFrame()
        result["Date"] = pd.to_datetime(df["TRADEDATE"])
        result["timestamp"] = result["Date"].astype("int64") // 10**9
        
        if "WAPRICE" in df.columns and "OPEN" not in df.columns:
            # Для некоторых таймфреймов MOEX возвращает только одну цену
            result["Open"] = df["WAPRICE"]
            result["High"] = df["WAPRICE"]
            result["Low"] = df["WAPRICE"]
            result["Close"] = df["WAPRICE"]
        else:
            result["Open"] = df["OPEN"]
            result["High"] = df["HIGH"]
            result["Low"] = df["LOW"]
            result["Close"] = df["CLOSE"]
        
        result["Volume"] = df.get("VOLUME", pd.Series([0] * len(df))).fillna(0).astype(int)
        result["Time"] = result["Date"].dt.strftime("%H:%M:%S")
        
        # Сортируем по времени
        result = result.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"Загружено {len(result)} баров для {ticker}_{timeframe}")
        return result
        
    except Exception as e:
        logger.error(f"Ошибка загрузки {ticker}: {e}")
        return pd.DataFrame()


def create_table_if_not_exists(ticker: str, timeframe: str):
    """Создает таблицу для инструмента если она не существует."""
    db = get_db_manager()
    table_name = f"{ticker.upper()}_{timeframe.upper()}"
    
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        timestamp BIGINT NOT NULL UNIQUE,
        Date DATETIME NOT NULL,
        Time VARCHAR(20),
        Open DECIMAL(18,8),
        High DECIMAL(18,8),
        Low DECIMAL(18,8),
        Close DECIMAL(18,8),
        Volume BIGINT,
        INDEX idx_timestamp (timestamp),
        INDEX idx_date (Date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    try:
        with db.engine.connect() as conn:
            conn.execute(text(create_sql))
            conn.commit()
        logger.debug(f"Таблица {table_name} создана или уже существует")
    except Exception as e:
        logger.error(f"Ошибка создания таблицы {table_name}: {e}")
        raise


def save_to_db(df: pd.DataFrame, ticker: str, timeframe: str):
    """Сохраняет данные в базу данных."""
    if df.empty:
        return
    
    db = get_db_manager()
    table_name = f"{ticker.upper()}_{timeframe.upper()}"
    
    try:
        with db.engine.connect() as conn:
            # Используем INSERT ... ON DUPLICATE KEY UPDATE для избежания дубликатов
            for _, row in df.iterrows():
                insert_sql = f"""
                INSERT INTO {table_name} 
                (timestamp, Date, Time, Open, High, Low, Close, Volume)
                VALUES 
                (:timestamp, :Date, :Time, :Open, :High, :Low, :Close, :Volume)
                ON DUPLICATE KEY UPDATE
                Date=VALUES(Date), Time=VALUES(Time), Open=VALUES(Open),
                High=VALUES(High), Low=VALUES(Low), Close=VALUES(Close), Volume=VALUES(Volume)
                """
                conn.execute(text(insert_sql), row.to_dict())
            conn.commit()
        
        logger.info(f"Сохранено {len(df)} записей в {table_name}")
    except Exception as e:
        logger.error(f"Ошибка сохранения в {table_name}: {e}")
        raise


def load_ticker_data(
    ticker: str,
    timeframes: List[str] = None,
    from_date: str = "2020-01-01",
    to_date: str = None,
):
    """Загружает данные для одного инструмента."""
    if timeframes is None:
        timeframes = ["D1", "H1"]
    
    logger.info(f"Загрузка данных для {ticker}...")
    
    for tf in timeframes:
        try:
            # Создаем таблицу
            create_table_if_not_exists(ticker, tf)
            
            # Загружаем данные
            df = fetch_moex_data(ticker, tf, from_date, to_date)
            
            if not df.empty:
                # Сохраняем в БД
                save_to_db(df, ticker, tf)
                
        except Exception as e:
            logger.error(f"Ошибка загрузки {ticker}_{tf}: {e}")


def load_all_data(
    tickers: List[str] = None,
    timeframes: List[str] = None,
    from_date: str = "2020-01-01",
    to_date: str = None,
):
    """Загружает данные для всех инструментов."""
    if tickers is None:
        tickers = DEFAULT_TICKERS
    
    if timeframes is None:
        timeframes = ["D1", "H1"]
    
    total = len(tickers) * len(timeframes)
    current = 0
    
    logger.info(f"Начало загрузки данных для {len(tickers)} инструментов...")
    
    for ticker in tickers:
        for tf in timeframes:
            current += 1
            logger.info(f"[{current}/{total}] Загрузка {ticker}_{tf}...")
            load_ticker_data(ticker, [tf], from_date, to_date)
    
    logger.info("Загрузка данных завершена!")


def main():
    parser = argparse.ArgumentParser(description="Загрузка данных MOEX в базу данных")
    
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS[:5]),
        help="Список тикеров через запятую (по умолчанию: SBER,GAZP,LKOH,ROSN,VTBR)",
    )
    
    parser.add_argument(
        "--timeframes",
        type=str,
        default="D1,H1",
        help="Таймфреймы через запятую (по умолчанию: D1,H1)",
    )
    
    parser.add_argument(
        "--from-date",
        type=str,
        default="2020-01-01",
        help="Дата начала в формате YYYY-MM-DD",
    )
    
    parser.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="Дата окончания в формате YYYY-MM-DD (по умолчанию сегодня)",
    )
    
    args = parser.parse_args()
    
    tickers = [t.strip() for t in args.tickers.split(",")]
    timeframes = [tf.strip() for tf in args.timeframes.split(",")]
    
    # Настройка логирования
    logger.add("logs/data_loader.log", rotation="10 MB", level="INFO")
    
    try:
        load_all_data(
            tickers=tickers,
            timeframes=timeframes,
            from_date=args.from_date,
            to_date=args.to_date,
        )
    except KeyboardInterrupt:
        logger.warning("Загрузка прервана пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise


if __name__ == "__main__":
    main()
