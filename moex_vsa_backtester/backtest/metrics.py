from typing import List, Optional

import pandas as pd


def calculate_metrics(trades: List[dict]) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_rr": 0.0,
            "total_pnl": 0.0,
        }

    df = pd.DataFrame(trades)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0

    total_wins = wins["pnl"].sum() if len(wins) > 0 else 0
    total_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    df_sorted = df.sort_values("exit_time")
    equity = df_sorted["pnl"].cumsum().values
    running_max = pd.Series(equity).cummax()
    drawdown = equity - running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

    avg_rr = df["rr"].mean() if len(df) > 0 else 0

    total_pnl = df["pnl"].sum()

    return {
        "total_trades": len(df),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_drawdown, 2),
        "avg_rr": round(avg_rr, 2),
        "total_pnl": round(total_pnl, 2),
    }


def print_report(metrics: dict):
    print("\n" + "=" * 40)
    print("BACKTEST RESULTS")
    print("=" * 40)
    print(f"Total Trades:     {metrics['total_trades']}")
    print(f"Winning:          {metrics['winning_trades']}")
    print(f"Losing:           {metrics['losing_trades']}")
    print(f"Win Rate:         {metrics['win_rate']:.2f}%")
    print(f"Profit Factor:   {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']:.2f}")
    print(f"Avg RR:           {metrics['avg_rr']:.2f}")
    print(f"Total PnL:        {metrics['total_pnl']:.2f}")
    print("=" * 40 + "\n")


def save_equity_curve(equity_data: List[dict], output_path: str):
    if not equity_data:
        return

    df = pd.DataFrame(equity_data)
    df.to_csv(output_path, index=False)