import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

from loguru import logger

from config import config
from core.data_loader import DataPreparator
from core.vsa_engine import detect_sr_levels, generate_vsa_signals
from core.risk_manager import calculate_position_size
from backtest.engine import VSABacktester
from backtest.metrics import calculate_metrics, print_report
from ai.dataset import DatasetGenerator
from ai.trainer import ModelTrainer
from ai.inference import TradePredictor
from scanner.scanner import SignalScanner
from scanner.scheduler import HourlyScheduler
from scanner.virtual_trading import VirtualTradingService


def setup_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        Path(__file__).parent / "logs" / "vsa_backtest.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
    )
    logger.add(
        Path(__file__).parent / "logs" / "signals.log",
        rotation="10 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="VSA Backtester for MOEX")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker symbol (required for backtest)")
    parser.add_argument(
        "--tf", type=str, default="H1", help="Timeframe (D1 or H1)"
    )
    parser.add_argument(
        "--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default="2024-01-01", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--capital", type=float, default=1_000_000, help="Initial capital for backtest/virtual"
    )
    parser.add_argument(
        "--risk", type=float, default=0.01, help="Risk per trade (0.01 = 1%)"
    )
    parser.add_argument(
        "--rr", type=float, default=2.0, help="Risk-Reward ratio"
    )
    parser.add_argument(
        "--train-ai", action="store_true", help="Train AI model for ticker"
    )
    parser.add_argument(
        "--train-global", action="store_true",
        help="Train global AI model on ALL instruments (including SHORT)"
    )
    parser.add_argument(
        "--allow-short", action="store_true",
        help="Allow SHORT trades in backtest (ignored if --train-global)"
    )
    parser.add_argument(
        "--ai-model", type=str, default="models/trade_model.pt", help="AI model path"
    )
    parser.add_argument(
        "--ai-threshold", type=float, default=0.6, help="AI probability threshold"
    )
    parser.add_argument(
        "--ai-filter", action="store_true", help="Filter signals by AI probability"
    )
    parser.add_argument(
        "--backtest", action="store_true", 
        help="Run backtest instead of scanner"
    )
    parser.add_argument(
        "--scan", "--scanner", action="store_true", 
        help="Run in scanner mode (scan all instruments)"
    )
    parser.add_argument(
        "--interval", type=int, default=60, 
        help="Scanner check interval in seconds"
    )
    parser.add_argument(
        "--virtual", "--virtual-trading", action="store_true",
        help="Run in virtual trading mode (monitor and execute trades)"
    )
    parser.add_argument(
        "--max-positions", type=int, default=5,
        help="Maximum concurrent positions in virtual trading"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate daily report for virtual trading"
    )
    return parser.parse_args()


def train_ai_model(trades, prices, model_path, ticker: str = None):
    logger.info("Training AI model...")

    dataset_gen = DatasetGenerator(context_window=24)
    X, y = dataset_gen.generate_from_trades(trades, prices, include_vsa=True)

    if X.size == 0 or y.size == 0:
        logger.error("No data for training")
        return None

    if ticker:
        model_path = Path(f"models/{ticker}_model.pt")
    
    model_path.parent.mkdir(parents=True, exist_ok=True)

    trainer = ModelTrainer(
        model_type="mlp",
        hidden_sizes=[128, 64, 32],
        dropout=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        early_stopping_patience=15,
        use_attention=True,
    )

    metrics = trainer.fit(X, y, val_size=0.2, scale=True)

    trainer.save(model_path)

    logger.info(f"AI model trained and saved to {model_path}")
    logger.info(f"Metrics: {metrics}")
    return trainer


def train_global_ai_model(args):
    from core.data_loader import DataPreparator
    from core.vsa_engine import detect_sr_levels, generate_vsa_signals
    from backtest.engine import VSABacktester
    from db import fetch_ohlcv
    from config import config

    logger.info("Training GLOBAL AI model on ALL instruments...")

    all_trades = []
    all_prices = {}
    all_tickers = set()

    try:
        from sqlalchemy import text
        from db import get_db_manager
        db = get_db_manager()
        with db.engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result.fetchall()]
        
        for table in tables:
            parts = table.upper().split("_")
            if len(parts) >= 2:
                tf = parts[-1]
                if tf == "H1":
                    ticker = "_".join(parts[:-1])
                    all_tickers.add(ticker)
    except Exception as e:
        logger.error(f"Error getting instruments: {e}")
        all_tickers = set(config.get("TICKER_LIST", ["SBER", "VTBR", "LKOH", "NVTK"]))

    logger.info(f"Found {len(all_tickers)} instruments: {all_tickers}")

    start_ts = int(datetime.fromisoformat(args.start).timestamp())
    end_ts = int(datetime.fromisoformat(args.end).timestamp())

    original_allow_short = config.get("ALLOW_SHORT", False)
    config._config["ALLOW_SHORT"] = True

    for ticker in all_tickers:
        try:
            prep = DataPreparator([ticker], ["H1", "D1"])
            df_d1, df_h1 = prep.load_and_prepare(ticker, start_ts, end_ts)
            
            if df_h1.empty or df_d1.empty:
                logger.warning(f"No data for {ticker}, skipping")
                continue

            df_h1 = prep.merge_context(df_h1, df_d1)
            levels = detect_sr_levels(df_h1)
            signals = generate_vsa_signals(df_h1, df_d1, levels)
            
            if signals.empty:
                logger.debug(f"No signals for {ticker}")
                continue

            all_prices[ticker] = df_h1

            backtester = VSABacktester(
                capital=args.capital, risk_pct=args.risk, rr_ratio=args.rr
            )
            trades = backtester.run(signals, df_h1)
            
            if trades:
                all_trades.extend(trades)
                logger.info(f"{ticker}: {len(trades)} trades")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            continue

    config._config["ALLOW_SHORT"] = original_allow_short

    if not all_trades:
        logger.error("No trades generated for training")
        return None

    logger.info(f"Total trades: {len(all_trades)}")

    prices_combined = pd.concat(all_prices.values(), ignore_index=True)
    prices_combined = prices_combined.sort_values("timestamp").reset_index(drop=True)

    model_path = Path(args.ai_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    trainer = train_ai_model(all_trades, prices_combined, model_path, ticker=None)

    if trainer:
        logger.info(f"GLOBAL AI model trained on {len(all_trades)} trades from {len(all_tickers)} instruments")
    
    return trainer


def run_backtest(args):
    logger.info(f"Starting VSA Backtester for {args.ticker}_{args.tf}")
    logger.info(f"Period: {args.start} - {args.end}")

    original_allow_short = config.get("ALLOW_SHORT", False)
    if args.allow_short:
        config._config["ALLOW_SHORT"] = True
        logger.info("SHORT trades ENABLED")
    elif not original_allow_short:
        logger.info("SHORT trades DISABLED (using config)")

    start_ts = int(datetime.fromisoformat(args.start).timestamp())
    end_ts = int(datetime.fromisoformat(args.end).timestamp())

    prep = DataPreparator([args.ticker], [args.tf, "D1"])
    df_d1, df_h1 = prep.load_and_prepare(args.ticker, start_ts, end_ts)

    if df_h1.empty:
        logger.error(f"No data for {args.ticker}")
        return

    df_h1 = prep.merge_context(df_h1, df_d1)

    levels = detect_sr_levels(df_h1)

    signals = generate_vsa_signals(df_h1, df_d1, levels)

    if signals.empty:
        logger.warning("No VSA signals generated")
        return

    predictor = None
    model_path = Path(args.ai_model)

    if args.train_ai:
        backtester_for_labels = VSABacktester(
            capital=args.capital, risk_pct=args.risk, rr_ratio=args.rr
        )

        trades = backtester_for_labels.run(signals, df_h1)

        if trades:
            train_ai_model(trades, df_h1, model_path, ticker=args.ticker)

    if model_path.exists():
        predictor = TradePredictor(
            model_path=model_path,
            threshold=args.ai_threshold,
        )
        signals = predictor.predict(signals, df_h1)

        if args.ai_filter:
            signals = predictor.filter_signals(
                signals,
                min_probability=args.ai_threshold,
            )

    backtester = VSABacktester(
        capital=args.capital, risk_pct=args.risk, rr_ratio=args.rr
    )

    trades = backtester.run(signals, df_h1)

    if trades:
        metrics = calculate_metrics(trades)
        print_report(metrics)
        backtester.save_trades(args.ticker, args.tf)

        if predictor and "ai_probability" in signals.columns:
            ai_filtered = signals[signals["ai_probability"] >= args.ai_threshold]
            ai_trades = len([
                t for t in trades
                if any(t.get("entry_time") == s.get("timestamp") for _, s in ai_filtered.iterrows())
            ])
            logger.info(f"AI-filtered trades: {ai_trades}/{len(trades)}")
    else:
        logger.warning("No trades executed")

    config._config["ALLOW_SHORT"] = original_allow_short


def run_scanner(args):
    model_path = Path(args.ai_model)
    
    scanner = SignalScanner(
        model_path=model_path if model_path.exists() else None,
        ai_threshold=args.ai_threshold,
    )
    
    if args.virtual:
        run_virtual_trading(args, scanner)
        return
    
    logger.info("Starting VSA Scanner in live monitoring mode")
    logger.info(f"AI model: {'Enabled' if scanner.predictors else 'Disabled'}")
    logger.info("Press Ctrl+C to stop")
    
    scheduler = HourlyScheduler(scanner, interval_seconds=args.interval)
    scheduler.start()
    
    try:
        while scheduler.running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nReceived Ctrl+C, stopping scanner...")
        scheduler.stop()
        time.sleep(1)
        logger.info("Scanner stopped")


def run_virtual_trading(args, scanner=None):
    if args.report:
        from core.trade_journal import DailyReportGenerator
        reporter = DailyReportGenerator()
        content = reporter.generate_daily_report()
        print(content)
        return

    if scanner is None:
        model_path = Path(args.ai_model)
        scanner = SignalScanner(
            model_path=model_path if model_path.exists() else None,
            ai_threshold=args.ai_threshold,
        )

    vt_service = VirtualTradingService(
        initial_capital=args.capital,
        risk_pct=args.risk,
        rr_ratio=args.rr,
        max_positions=args.max_positions,
    )

    vt_scanner = VirtualTradingScanner(scanner, vt_service)

    logger.info("Starting VSA Scanner in VIRTUAL TRADING mode")
    logger.info(f"Initial Capital: {args.capital:,.2f} RUB")
    logger.info(f"Risk per Trade: {args.risk * 100}%")
    logger.info(f"RR Ratio: {args.rr}")
    logger.info(f"Max Positions: {args.max_positions}")
    logger.info("Press Ctrl+C to stop")

    scheduler = HourlyScheduler(vt_scanner, interval_seconds=args.interval)
    scheduler.start()

    try:
        while scheduler.running:
            status = vt_scanner.get_status()
            logger.info(
                f"Status: Capital={status['capital']:,.2f} | "
                f"PnL={status['total_pnl']:,.2f} | "
                f"Trades={status['total_trades']} | "
                f"Open={status['open_positions']}"
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("\nReceived Ctrl+C, closing positions...")
        vt_scanner.close_all()
        vt_service.check_daily_report()
        status = vt_scanner.get_status()
        logger.info(f"Final Status: Capital={status['capital']:,.2f}, PnL={status['total_pnl']:,.2f}")
        scheduler.stop()
        logger.info("Virtual trading stopped")


def main():
    setup_logging()
    args = parse_args()

    if args.train_global:
        train_global_ai_model(args)
    elif args.backtest:
        if not args.ticker:
            logger.error("--ticker is required for backtest mode")
            return
        run_backtest(args)
    else:
        run_scanner(args)


if __name__ == "__main__":
    main()