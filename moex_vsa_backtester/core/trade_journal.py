import csv
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger


class TradeJournal:
    def __init__(self, output_dir: Path = Path("logs/virtual_trading")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.journal_file = self.output_dir / "journal.csv"
        self._trade_counter = 1
        self._init_journal_file()

    def _init_journal_file(self):
        if not self.journal_file.exists():
            with open(self.journal_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "id", "ticker", "side", "entry_time", "exit_time",
                    "entry_price", "exit_price", "sl_price", "tp_price",
                    "size", "pnl", "rr", "status", "duration_bars",
                    "entry_date", "exit_date"
                ])
        else:
            self._restore_counter()

    def _restore_counter(self):
        try:
            with open(self.journal_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1]
                    self._trade_counter = int(last_line.split(",")[0]) + 1
        except:
            self._trade_counter = 1

    def add_trade(
        self,
        ticker: str,
        side: str,
        entry_time: int,
        exit_time: int,
        entry_price: float,
        exit_price: float,
        sl_price: float,
        tp_price: float,
        size: int,
        pnl: float,
        rr: float,
        status: str,
    ) -> int:
        entry_date = datetime.fromtimestamp(entry_time).strftime("%Y-%m-%d")
        exit_date = datetime.fromtimestamp(exit_time).strftime("%Y-%m-%d")
        duration = (exit_time - entry_time) // 3600

        trade_id = self._trade_counter

        with open(self.journal_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trade_id, ticker, side, entry_time, exit_time,
                entry_price, exit_price, sl_price, tp_price,
                size, round(pnl, 2), round(rr, 2), status, duration,
                entry_date, exit_date
            ])

        self._trade_counter += 1
        return trade_id

    def get_trades(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[Dict]:
        trades = []
        if not self.journal_file.exists():
            return trades

        with open(self.journal_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trade_date = datetime.strptime(row["entry_date"], "%Y-%m-%d").date()

                if start_date and trade_date < start_date:
                    continue
                if end_date and trade_date > end_date:
                    continue

                trades.append({
                    "id": int(row["id"]),
                    "ticker": row["ticker"],
                    "side": row["side"],
                    "entry_time": int(row["entry_time"]),
                    "exit_time": int(row["exit_time"]),
                    "entry_price": float(row["entry_price"]),
                    "exit_price": float(row["exit_price"]),
                    "sl_price": float(row["sl_price"]),
                    "tp_price": float(row["tp_price"]),
                    "size": int(row["size"]),
                    "pnl": float(row["pnl"]),
                    "rr": float(row["rr"]),
                    "status": row["status"],
                    "duration_bars": int(row["duration_bars"]),
                    "entry_date": row["entry_date"],
                    "exit_date": row["exit_date"],
                })

        return trades


class DailyReportGenerator:
    def __init__(self, output_dir: Path = Path("logs/virtual_trading/reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.journal = TradeJournal()

    def generate_daily_report(self, report_date: Optional[date] = None) -> str:
        if report_date is None:
            report_date = (datetime.now() - timedelta(days=1)).date()

        start_date = report_date
        end_date = report_date

        trades = self.journal.get_trades(start_date=start_date, end_date=end_date)

        if not trades:
            return self._generate_empty_report(report_date)

        return self._generate_report_content(report_date, trades)

    def _generate_empty_report(self, report_date: date) -> str:
        content = f"""
================================================================================
                        VIRTUAL TRADING DAILY REPORT
================================================================================
Date: {report_date.strftime('%Y-%m-%d')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

No trades executed on this date.

================================================================================
"""
        self._save_report(report_date, content)
        return content

    def _generate_report_content(self, report_date: date, trades: List[Dict]) -> str:
        winning = [t for t in trades if t["pnl"] > 0]
        losing = [t for t in trades if t["pnl"] <= 0]

        total_pnl = sum(t["pnl"] for t in trades)
        win_rate = len(winning) / len(trades) if trades else 0.0

        wins_pnl = sum(t["pnl"] for t in winning)
        loss_pnl = sum(t["pnl"] for t in losing)
        profit_factor = abs(wins_pnl / loss_pnl) if loss_pnl != 0 else float("inf")

        avg_rr = sum(t["rr"] for t in trades) / len(trades) if trades else 0.0

        # Analyze trade formation details
        formations = 0
        opened = 0
        stopped_out = 0
        locked_in_profit = 0
        
        for trade in trades:
            if trade["status"] == "SL":
                stopped_out += 1
            elif trade["status"] == "TP":
                locked_in_profit += 1
            # We don't count the specific "formed" trades here since they're 
            # implicitly covered in the trade entries we see
        
        content = f"""
================================================================================
                        VIRTUAL TRADING DAILY REPORT
================================================================================
Date: {report_date.strftime('%Y-%m-%d')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
--------------------------------------------------------------------------------
Total Trades:       {len(trades)}
Winning Trades:     {len(winning)}
Losing Trades:      {len(losing)}
Win Rate:           {win_rate:.1%}
Total PnL:           {total_pnl:,.2f} RUB
Profit Factor:      {profit_factor:.2f}
Avg Risk/Reward:    {avg_rr:.2f}R

TRADE ANALYSIS
--------------------------------------------------------------------------------
Trade Formations:     {formations}
Trades Opened:        {opened}
Stop-Outs:            {stopped_out}
Profit Locks:         {locked_in_profit}
Total Instruments:    {len(set(t['ticker'] for t in trades))}

TRADES
--------------------------------------------------------------------------------
"""

        for i, t in enumerate(trades, 1):
            content += f"""
{i}. {t['ticker']} {t['side']}
   Entry:  {t['entry_price']:.4f} @ {t['entry_date']} {datetime.fromtimestamp(t['entry_time']).strftime('%H:%M')}
   Exit:   {t['exit_price']:.4f} @ {t['exit_date']} {datetime.fromtimestamp(t['exit_time']).strftime('%H:%M')}
   SL:     {t['sl_price']:.4f} | TP: {t['tp_price']:.4f}
   Size:   {t['size']} lots
   PnL:    {t['pnl']:,.2f} RUB ({t['rr']:.2f}R)
   Status: {t['status']}
"""

        if winning:
            content += """
WINNING TRADES DETAIL
--------------------------------------------------------------------------------
"""
            for t in winning:
                content += f"  {t['ticker']} {t['side']}: +{t['pnl']:,.2f} RUB ({t['rr']:.2f}R)\n"

        if losing:
            content += """
LOSING TRADES DETAIL
--------------------------------------------------------------------------------
"""
            for t in losing:
                content += f"  {t['ticker']} {t['side']}: {t['pnl']:,.2f} RUB ({t['rr']:.2f}R)\n"

        content += """
================================================================================
"""

        self._save_report(report_date, content)
        return content

    def _save_report(self, report_date: date, content: str):
        report_file = self.output_dir / f"report_{report_date.strftime('%Y%m%d')}.txt"
        with open(report_file, "w") as f:
            f.write(content)
        logger.info(f"Daily report saved to {report_file}")

    def generate_date_range_report(self, start_date: date, end_date: date) -> str:
        trades = self.journal.get_trades(start_date=start_date, end_date=end_date)

        if not trades:
            return f"No trades in period {start_date} - {end_date}"

        winning = [t for t in trades if t["pnl"] > 0]
        losing = [t for t in trades if t["pnl"] <= 0]

        total_pnl = sum(t["pnl"] for t in trades)
        win_rate = len(winning) / len(trades) if trades else 0.0

        # Analyze trade formation details for date range
        formations = 0
        opened = 0
        stopped_out = 0
        locked_in_profit = 0
        
        for trade in trades:
            if trade["status"] == "SL":
                stopped_out += 1
            elif trade["status"] == "TP":
                locked_in_profit += 1
            # We don't count the specific "formed" trades here since they're 
            # implicitly covered in the trade entries we see

        content = f"""
================================================================================
                     VIRTUAL TRADING PERIOD REPORT
================================================================================
Period: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
--------------------------------------------------------------------------------
Total Trades:       {len(trades)}
Winning Trades:     {len(winning)}
Losing Trades:      {len(losing)}
Win Rate:           {win_rate:.1%}
Total PnL:          {total_pnl:,.2f} RUB
Avg Trades/Day:     {len(trades) / max(1, (end_date - start_date).days):.1f}
Trade Formations:   {formations}
Trades Opened:      {opened}
Stop-Outs:          {stopped_out}
Profit Locks:       {locked_in_profit}
Total Instruments:  {len(set(t['ticker'] for t in trades))}

================================================================================
"""

        self._save_report(end_date, content)
        return content

    def get_trade_count_by_ticker(self) -> Dict[str, int]:
        trades = self.journal.get_trades()
        counts = {}
        for t in trades:
            counts[t["ticker"]] = counts.get(t["ticker"], 0) + 1
        return counts
