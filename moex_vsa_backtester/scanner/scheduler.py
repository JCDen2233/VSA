import time
from datetime import datetime, timedelta
from threading import Thread, Event, Lock
from typing import Dict
from zoneinfo import ZoneInfo

from loguru import logger

_MS = ZoneInfo("Europe/Moscow")
from utils.market_hours import is_in_session_range


class HourlyScheduler:
    def __init__(self, scanner, interval_seconds: int = 60):
        self.scanner = scanner
        self.interval_seconds = interval_seconds
        self.running = False
        self.stop_event = Event()
        self.candle_timestamps: Dict[str, int] = {}
        self.lock = Lock()
        self._last_report_hour = -1

    def start(self):
        self.running = True
        self.stop_event.clear()

        logger.info("Starting hourly scheduler")

        self._full_scan()

        def run_loop():
            counter = 0
            while self.running and not self.stop_event.is_set():
                now_ms = datetime.now(_MS)
                
                # Skip scanning outside market hours
                if not is_in_session_range(now_ms):
                    self.stop_event.wait(self.interval_seconds)
                    continue
                
                try:
                    self._check_new_candles(now_ms)
                    self._check_interim_report(now_ms)
                except Exception as e:
                    logger.error(f"Error in scan loop: {e}")

                counter += 1

                if counter % 10 == 0:
                    with self.lock:
                        ts_copy = dict(self.candle_timestamps)
                    logger.info(
                        f"[{now_ms.strftime('%H:%M:%S')}] Scanner alive | "
                        f"Latest candles: {len(ts_copy)} tickers scanned"
                    )

                self.stop_event.wait(self.interval_seconds)
            logger.info("Scanner loop stopped")

        return Thread(target=run_loop, daemon=True).start()

    def _check_interim_report(self, now_ms: datetime):
        if now_ms.minute >= 30 and self._last_report_hour != now_ms.hour:
            if hasattr(self.scanner, 'generate_interim_report'):
                report = self.scanner.generate_interim_report()
                logger.info("=" * 60)
                logger.info(f"INTERIM REPORT - {now_ms.strftime('%Y-%m-%d %H:%M')}")
                logger.info("=" * 60)
                logger.info(report)
                logger.info("=" * 60)
            
            self._last_report_hour = now_ms.hour

    def stop(self):
        self.running = False
        self.stop_event.set()

    def _full_scan(self):
        now_ms = datetime.now(_MS)
        if not is_in_session_range(now_ms):
            logger.info(f"[{now_ms.strftime('%H:%M')}] Market closed, skipping full scan")
            return
        
        logger.info("Running full instrument scan...")
        instruments = self.scanner.get_all_instruments()
        signal_count = 0

        for ticker in instruments:
            ts = self.scanner.get_latest_timestamp(ticker)
            if ts:
                with self.lock:
                    self.candle_timestamps[ticker] = ts

            signals = self.scanner.scan_instrument(ticker, lookback_hours=24)
            if signals:
                self.scanner.log_signal(signals[-1])
                logger.info(f"[{ticker}] Signal detected: {signals[-1]}")
                signal_count += 1

        logger.info(
            f"Full scan complete: {len(self.candle_timestamps)} initialized, "
            f"{signal_count} signals found"
        )

    def _check_new_candles(self, now_ms: datetime):
        instruments = self.scanner.get_all_instruments()

        for ticker in instruments:
            current_ts = self.scanner.get_latest_timestamp(ticker)
            if not current_ts:
                continue

            # Only process new candles during session hours
            current_bar_dt = datetime.fromtimestamp(current_ts, _MS)
            if not is_in_session_range(current_bar_dt):
                continue

            with self.lock:
                last_ts = self.candle_timestamps.get(ticker)

            if last_ts is None or current_ts != last_ts:
                with self.lock:
                    self.candle_timestamps[ticker] = current_ts

                new_ts = current_ts
                logger.info(
                    f"[{ticker}] New candle at {datetime.fromtimestamp(new_ts, _MS).strftime('%H:%M')}"
                )

                # Use shorter lookback for immediate signal detection
                signals = self.scanner.scan_instrument(ticker, lookback_hours=24)
                if signals:
                    latest_signal = signals[-1]
                    
                    self.scanner.log_signal(latest_signal)

                    if hasattr(self.scanner, 'vt_service'):
                        current_price = self.scanner.get_latest_price(ticker)
                        if current_price:
                            try:
                                result = self.scanner.vt_service.execute_signal(latest_signal, current_price)
                                logger.info(f"[{ticker}] Trade execution result: {result}")
                            except Exception as e:
                                logger.error(f"[{ticker}] Trade execution error: {e}")

                    if hasattr(self.scanner, "run_instrument_analysis"):
                        try:
                            self.scanner.run_instrument_analysis_for_signal(ticker, latest_signal)
                        except Exception as e:
                            logger.error(f"[{ticker}] Analysis error: {e}")

    def _init_timestamps(self):
        instruments = self.scanner.get_all_instruments()
        for ticker in instruments:
            ts = self.scanner.get_latest_timestamp(ticker)
            if ts:
                with self.lock:
                    self.candle_timestamps[ticker] = ts
