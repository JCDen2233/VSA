import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

from db import get_db_manager
from core.data_loader import DataPreparator
from core.vsa_engine import detect_sr_levels, generate_vsa_signals
from ai.inference import TradePredictor
from utils.market_hours import is_in_session_range

_MS = ZoneInfo("Europe/Moscow")


class SignalScanner:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        ai_threshold: float = 0.6,
    ):
        self.ai_threshold = ai_threshold
        self.default_model_path = model_path
        self.predictors = {}
        
        if model_path and model_path.exists():
            self.predictors["default"] = TradePredictor(
                model_path=model_path,
                threshold=ai_threshold,
            )
            logger.info(f"AI модель загружена из {model_path}")

    def get_predictor(self, ticker: str) -> Optional[TradePredictor]:
        if ticker in self.predictors:
            return self.predictors[ticker]
        
        model_path = Path(f"models/{ticker}_model.pt")
        if model_path.exists():
            predictor = TradePredictor(
                model_path=model_path,
                threshold=self.ai_threshold,
            )
            self.predictors[ticker] = predictor
            logger.info(f"AI модель для {ticker} загружена")
            return predictor
        
        if "default" in self.predictors:
            return self.predictors["default"]
        
        return None

    def get_all_instruments(self) -> List[str]:
        from sqlalchemy import text
        
        db = get_db_manager()
        
        try:
            with db.engine.connect() as conn:
                result = conn.execute(text("SHOW TABLES"))
                tables = [row[0] for row in result.fetchall()]
            
            instruments = set()
            for table in tables:
                parts = table.upper().split("_")
                if len(parts) >= 2:
                    tf = parts[-1]
                    if tf in ("D1", "H1", "H4", "M5", "M15", "M30"):
                        ticker = "_".join(parts[:-1])
                        instruments.add(ticker)
            
            instruments = sorted(list(instruments))
            logger.info(f"Найдено {len(instruments)} инструментов: {instruments}")
            return instruments
        except Exception as e:
            logger.error(f"Ошибка получения инструментов: {e}")
            return []

    def scan_instrument(
        self,
        ticker: str,
        lookback_hours: int = 24,
    ) -> List[Dict]:
        from utils.market_hours import is_in_session_range
        now_ms = datetime.now(_MS)
        
        # Get the latest bar timestamp
        latest_ts = self.get_latest_timestamp(ticker)
        if not latest_ts:
            return []
            
        latest_dt = datetime.fromtimestamp(latest_ts, _MS)
        if not is_in_session_range(latest_dt):
            return []
        
        # Use shorter lookback for faster signal detection
        start_ts = int((now_ms - timedelta(hours=lookback_hours)).timestamp())
        end_ts = int(now_ms.timestamp())

        try:
            prep = DataPreparator([ticker], ["H1", "D1"])
            
            import logging
            logging.getLogger("db").setLevel(logging.WARNING)
            
            df_d1, df_h1 = prep.load_and_prepare(ticker, start_ts, end_ts)
            
            if df_h1.empty or df_d1.empty:
                return []
            
            df_h1 = prep.merge_context(df_h1, df_d1)
            
            levels = detect_sr_levels(df_h1)
            
            signals = generate_vsa_signals(df_h1, df_d1, levels)
            
            if signals.empty:
                return []
            
            # Take ONLY the most recent signal - immediate execution
            latest_signal = signals.iloc[-1]
            
            # Check signal age - only process signals from last 2 hours
            signal_age_hours = (now_ms.timestamp() - latest_signal["timestamp"]) / 3600
            if signal_age_hours > 2:
                logger.debug(f"[{ticker}] Signal too old ({signal_age_hours:.1f}h), skipping")
                return []
            
            predictor = self.get_predictor(ticker)
            if predictor is not None:
                latest_signal = predictor.predict(
                    latest_signal.to_frame().T, df_h1
                ).iloc[0]
            
            return self._format_signals(latest_signal, ticker)
            
        except Exception as e:
            logger.error(f"Ошибка сканирования {ticker}: {e}")
            return []

    def _format_signals(self, signal: pd.Series, ticker: str) -> List[Dict]:
        signal_time = datetime.fromtimestamp(signal["timestamp"])
        
        formatted = {
            "ticker": ticker,
            "signal_time": signal_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": signal["timestamp"],
            "signal_type": signal.get("signal_type", "UNKNOWN"),
            "entry_price": signal.get("entry_price", 0),
            "sl_price": signal.get("sl_price", 0),
            "tp_price": signal.get("tp_price", 0),
        }
        
        if "ai_probability" in signal:
            formatted["ai_probability"] = signal["ai_probability"]
            formatted["ai_predicted_success"] = signal.get("ai_predicted_success", False)
        
        return [formatted]

    def log_signal(self, signal: Dict):
        current_price = self.get_latest_price(signal['ticker'])
        
        crossed_entry = ""
        crossed_sl = ""
        crossed_tp = ""
        
        if current_price is not None:
            entry = signal['entry_price']
            sl = signal['sl_price']
            tp = signal['tp_price']
            
            if signal['signal_type'] == 'LONG':
                if current_price >= entry:
                    crossed_entry = " ✅"
                if current_price >= tp:
                    crossed_tp = " ✅"
                if current_price <= sl:
                    crossed_sl = " ✅"
            else:
                if current_price <= entry:
                    crossed_entry = " ✅"
                if current_price <= tp:
                    crossed_tp = " ✅"
                if current_price >= sl:
                    crossed_sl = " ✅"
        
        logger.info("=" * 60)
        logger.info(f"📊 ОБНАРУЖЕН СИГНАЛ: {signal['ticker']}")
        logger.info("=" * 60)
        logger.info(f"⏰ Время:      {signal['signal_time']}")
        logger.info(f"📈 Тип:        {signal['signal_type']}")
        logger.info(f"💵 Вход:      {signal['entry_price']:.4f}{crossed_entry}")
        logger.info(f"🛡️  SL:         {signal['sl_price']:.4f}{crossed_sl}")
        logger.info(f"🎯 TP:          {signal['tp_price']:.4f}{crossed_tp}")
        
        if "ai_probability" in signal:
            prob = signal["ai_probability"]
            logger.info(f"🤖 AI Вероятность: {prob:.2%}")
            status = "ПРОЙДЕН" if signal.get('ai_predicted_success') else "НЕ ПРОЙДЕН"
            icon = "✅" if signal.get('ai_predicted_success') else "❌"
            logger.info(f"{icon} AI Фильтр:   {status}")
        
        logger.info("=" * 60)

    def scan_all_instruments(
        self,
        lookback_hours: int = 336,
        instruments: List[str] = None,
    ) -> List[Dict]:
        if instruments is None:
            instruments = self.get_all_instruments()
        
        all_signals = []
        
        for ticker in instruments:
            signals = self.scan_instrument(ticker, lookback_hours)
            for sig in signals:
                self.log_signal(sig)
                all_signals.append(sig)
        
        return all_signals

    def get_latest_timestamp(self, ticker: str) -> Optional[int]:
        from sqlalchemy import text
        from db import get_db_manager
        from utils.market_hours import is_in_session_range
        
        db = get_db_manager()
        table_name = f"{ticker.upper()}_H1"
        
        try:
            with db.engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT MAX(timestamp) as max_ts FROM {table_name}")
                )
                row = result.fetchone()
                ts = row[0] if row else None
                if ts is not None:
                    bar_dt = datetime.fromtimestamp(int(ts), _MS)
                    if is_in_session_range(bar_dt):
                        return int(ts)
                    return None  # Last bar was not in session
                return None
        except Exception as e:
            logger.debug(f"Error getting latest timestamp for {ticker}: {e}")
            return None

    def get_latest_price(self, ticker: str) -> Optional[float]:
        from sqlalchemy import text
        from db import get_db_manager
        
        db = get_db_manager()
        table_name = f"{ticker.upper()}_H1"
        
        try:
            with db.engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT Close FROM {table_name} ORDER BY timestamp DESC LIMIT 1")
                )
                row = result.fetchone()
                return float(row[0]) if row else None
        except Exception as e:
            logger.debug(f"Error getting latest price for {ticker}: {e}")
            return None

    def run_instrument_analysis_for_signal(self, ticker: str, signal: Dict):
        signal_ts = signal.get("timestamp")
        if not signal_ts:
            return
        
        signal_time = datetime.fromtimestamp(signal_ts)
        lookback_hours = 168 * 2
        start_ts = int((signal_time - timedelta(hours=lookback_hours)).timestamp())
        end_ts = int(signal_ts + 3600)

        try:
            prep = DataPreparator([ticker], ["H1", "D1"])
            import logging
            logging.getLogger("db").setLevel(logging.WARNING)
            
            df_d1, df_h1 = prep.load_and_prepare(ticker, start_ts, end_ts)
            
            if df_h1.empty or df_d1.empty:
                logger.debug(f"[{ticker}] No historical data for analysis at {signal_time}")
                return
            
            df_h1 = prep.merge_context(df_h1, df_d1)
            
            predictor = self.get_predictor(ticker)
            if predictor is None:
                return
            
            signal_df = pd.DataFrame([signal])
            result = predictor.predict(signal_df, df_h1)
            
            ai_prob = 0.5
            ai_passed = False
            
            if "ai_probability" in result.columns and len(result) > 0:
                ai_prob_val = result["ai_probability"].iloc[0]
                try:
                    ai_prob = float(ai_prob_val)
                except Exception:
                    try:
                        ai_prob = float(np.asarray(ai_prob_val).item())
                    except:
                        ai_prob = 0.5
            
            if "ai_predicted_success" in result.columns and len(result) > 0:
                ai_passed_val = result["ai_predicted_success"].iloc[0]
                try:
                    ai_passed = bool(ai_passed_val)
                except Exception:
                    try:
                        ai_passed = bool(np.asarray(ai_passed_val).item())
                    except:
                        ai_passed = False
            
            logger.info(f"[{ticker}] AI анализ на момент сигнала: вероятность={ai_prob*100:.1f}%, пройден={ai_passed}")
            
        except Exception as e:
            import traceback
            logger.error(f"[{ticker}] Ошибка анализа: {e}\n{traceback.format_exc()}")

    def run_instrument_analysis(self, ticker: str):
        logger.debug(f"[{ticker}] Running instrument analysis...")