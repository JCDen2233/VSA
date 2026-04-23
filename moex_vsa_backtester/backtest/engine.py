from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from core.risk_manager import apply_rr_exits, calculate_position_size, calculate_risk_reward


class VSA_Backtester:
    def __init__(
        self,
        capital: float = 1_000_000,
        risk_pct: float = 0.01,
        rr_ratio: float = 2.0,
    ):
        self.capital = capital
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.position = None
        self.trades: List[dict] = []
        self.equity = []

    def run(self, signals: pd.DataFrame, prices: pd.DataFrame) -> List[dict]:
        if signals.empty or prices.empty:
            logger.warning("Empty signals or prices")
            return []

        prices = prices.set_index("timestamp")
        signal_timestamps = signals["timestamp"].tolist()
        
        for i, ts in enumerate(signal_timestamps):
            if ts not in prices.index:
                continue

            price_row = prices.loc[ts]
            self._open_position(signals.iloc[i], price_row)

            if self.position is None:
                continue

            for next_ts in list(prices.index):
                if next_ts <= ts:
                    continue
                self._check_exits(prices, next_ts)
                if self.position is None:
                    break

            self._close_expired_positions(prices, cutoff_hour=18)

        self._close_open_positions(prices)
        logger.info(f"Backtest completed: {len(self.trades)} trades")
        return self.trades

    def _open_position(self, signal: pd.Series, price_row: pd.Series):
        if self.position is not None:
            return

        entry = signal.get("entry_price", price_row.get("Close"))
        sl = signal.get("sl_price", entry * 0.9985)
        tp = signal.get("tp_price") or apply_rr_exits(entry, sl, self.rr_ratio)

        size = calculate_position_size(self.capital, self.risk_pct, entry, sl)
        if size <= 0:
            return

        self.position = {
            "entry_time": signal.get("timestamp"),
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "size": size,
            "side": signal.get("signal_type", "LONG"),
            "entry_bar": signal,
        }
        logger.info(
            f"Opened {self.position['side']}: entry={entry}, "
            f"sl={sl}, tp={tp}, size={size}"
        )

    def _check_exits(self, prices: pd.DataFrame, current_ts: int):
        if self.position is None:
            return

        pos = self.position
        entry = pos["entry"]
        sl = pos["sl"]
        tp = pos["tp"]
        side = pos["side"]

        if current_ts not in prices.index:
            return
        price = prices.loc[current_ts]
        current_price = price.get("Close")

        pnl = 0
        status = "OPEN"
        exit_price = None

        if side == "LONG":
            # Check for SL first (per requirement: if both TP and SL hit, count SL)
            if current_price <= sl:
                pnl = (sl - entry) * pos["size"]
                status = "SL"
                exit_price = sl
            elif current_price >= tp:
                pnl = (tp - entry) * pos["size"]
                status = "TP"
                exit_price = tp
            elif current_price >= entry + (tp - entry) * 0.5:
                partial_size = pos["size"] // 2
                pnl_partial = (entry + (tp - entry) * 0.5 - entry) * partial_size
                self.capital += pnl_partial
                pos["size"] -= partial_size
                pos["sl"] = entry
                logger.info(f"Partial close at 1.5R, moved SL to BE")
        else:
            # Check for SL first (per requirement: if both TP and SL hit, count SL)
            if current_price >= sl:
                pnl = (entry - sl) * pos["size"]
                status = "SL"
                exit_price = sl
            elif current_price <= tp:
                pnl = (entry - tp) * pos["size"]
                status = "TP"
                exit_price = tp
            elif current_price <= entry - (entry - tp) * 0.5:
                partial_size = pos["size"] // 2
                pnl_partial = (entry - (entry - tp) * 0.5 - entry) * partial_size
                self.capital += pnl_partial
                pos["size"] -= partial_size
                pos["sl"] = entry
                logger.info(f"Partial close at 1.5R, moved SL to BE")

        if status != "OPEN":
            self._close_position(current_ts, exit_price or current_price, pnl, status)

        rr = calculate_risk_reward(entry, sl, tp, side)
        self.equity.append({"timestamp": current_ts, "equity": self.capital + pnl})

    def _close_position(
        self, exit_ts: int, exit_price: float, pnl: float, status: str
    ):
        pos = self.position
        self.trades.append(
            {
                "entry_time": pos["entry_time"],
                "exit_time": exit_ts,
                "side": pos["side"],
                "entry": pos["entry"],
                "sl": pos["sl"],
                "tp": pos["tp"],
                "size": pos["size"],
                "pnl": pnl,
                "rr": calculate_risk_reward(pos["entry"], pos["sl"], pos["tp"], pos["side"]),
                "status": status,
            }
        )
        self.capital += pnl
        logger.info(f"Closed position: {status}, PnL={pnl:.2f}")
        self.position = None

    def _close_expired_positions(self, prices: pd.DataFrame, cutoff_hour: int = 18):
        if self.position is None:
            return

        entry_ts = self.position.get("entry_time")
        if entry_ts is None:
            return

        entry_idx = prices.index.get_loc(entry_ts) if entry_ts in prices.index else 0

        for i in range(entry_idx, min(entry_idx + 24, len(prices))):
            ts = prices.index[i]
            dt = pd.to_datetime(ts, unit="s")
            if dt.hour >= cutoff_hour:
                price = prices.loc[ts]["Close"]
                pnl = 0
                if self.position["side"] == "LONG":
                    pnl = (price - self.position["entry"]) * self.position["size"]
                else:
                    pnl = (self.position["entry"] - price) * self.position["size"]
                self._close_position(ts, price, pnl, "EOD_CLOSE")
                break

    def _close_open_positions(self, prices: pd.DataFrame):
        if self.position is None:
            return

        last_ts = prices.index[-1]
        last_price = prices.iloc[-1]["Close"]
        pnl = 0
        if self.position["side"] == "LONG":
            pnl = (last_price - self.position["entry"]) * self.position["size"]
        else:
            pnl = (self.position["entry"] - last_price) * self.position["size"]
        self._close_position(last_ts, last_price, pnl, "END")

    def save_trades(self, ticker: str, tf: str, output_dir: Path = Path("logs")):
        if not self.trades:
            logger.warning("No trades to save")
            return

        df = pd.DataFrame(self.trades)
        path = output_dir / f"trades_{ticker}_{tf}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved trades to {path}")