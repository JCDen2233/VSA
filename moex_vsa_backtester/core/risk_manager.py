from loguru import logger

from config import config


def calculate_position_size(
    capital: float, risk_pct: float, entry: float, sl: float
) -> int:
    if entry <= 0 or sl <= 0 or entry == sl:
        logger.warning("Invalid entry or SL price")
        return 0

    risk_amount = capital * risk_pct
    price_risk = abs(entry - sl)

    commission = config.get("COMMISSION_PCT", 0.001)
    slippage = config.get("SLIPPAGE_PCT", 0.0005)

    adjusted_entry = entry * (1 + commission + slippage)
    adjusted_sl = sl * (1 - commission - slippage)

    actual_risk = abs(adjusted_entry - adjusted_sl)
    if actual_risk <= 0:
        return 0

    size = risk_amount / actual_risk
    size = int(size)

    max_capital_usage = capital * 0.20
    position_value = size * entry
    if position_value > max_capital_usage:
        size = int(max_capital_usage / entry)

    logger.info(
        f"Position size: {size} lots (risk={risk_pct*100}%, "
        f"entry={entry}, sl={sl}, risk_amount={risk_amount})"
    )
    return size


def apply_rr_exits(entry: float, sl: float, rr_ratio: float = 2.0) -> float:
    if entry <= 0 or sl <= 0 or rr_ratio <= 0:
        logger.warning("Invalid parameters for RR exits")
        return 0

    risk = abs(entry - sl)
    reward = risk * rr_ratio

    tp_long = entry + reward
    tp_short = entry - reward

    tp_price = tp_long
    logger.info(f"TP calculation: entry={entry}, sl={sl}, tp={tp_price}, RR={rr_ratio}")
    return tp_price


def calculate_risk_reward(entry: float, sl: float, tp: float, side: str) -> float:
    if entry <= 0 or sl <= 0 or tp <= 0:
        return 0.0

    risk = abs(entry - sl)
    if side == "LONG":
        reward = tp - entry
    else:
        reward = entry - tp

    if risk <= 0:
        return 0.0
    return reward / risk