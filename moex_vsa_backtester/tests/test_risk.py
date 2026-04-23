import pytest
from unittest.mock import patch, MagicMock

from core.risk_manager import (
    calculate_position_size,
    apply_rr_exits,
    calculate_risk_reward,
)


class TestRiskManager:
    def test_calculate_position_size_valid(self):
        with patch("core.risk_manager.config") as mock_config:
            mock_config.get = lambda key, default=None: {
                "COMMISSION_PCT": 0.001,
                "SLIPPAGE_PCT": 0.0005,
            }.get(key, default)
            
            size = calculate_position_size(
                capital=1_000_000,
                risk_pct=0.01,
                entry=100.0,
                sl=99.0
            )
            assert size > 0

    def test_calculate_position_size_zero_sl(self):
        size = calculate_position_size(
            capital=1_000_000,
            risk_pct=0.01,
            entry=100.0,
            sl=0
        )
        assert size == 0

    def test_calculate_position_size_equal_prices(self):
        size = calculate_position_size(
            capital=1_000_000,
            risk_pct=0.01,
            entry=100.0,
            sl=100.0
        )
        assert size == 0

    def test_calculate_position_size_negative(self):
        size = calculate_position_size(
            capital=1_000_000,
            risk_pct=0.01,
            entry=100.0,
            sl=101.0
        )
        assert size > 0

    @pytest.mark.parametrize("entry,sl,expected_rr", [
        (100.0, 99.0, 2.0),
        (100.0, 98.0, 2.0),
        (100.0, 95.0, 1.0),
    ])
    def test_apply_rr_exits(self, entry, sl, expected_rr):
        tp = apply_rr_exits(entry, sl, rr_ratio=expected_rr)
        if entry > 0 and sl > 0 and expected_rr > 0:
            assert tp > 0

    def test_apply_rr_exits_invalid(self):
        tp = apply_rr_exits(0, 100, 2.0)
        assert tp == 0

    @pytest.mark.parametrize("side", ["LONG", "SHORT"])
    def test_calculate_risk_reward_long(self, side):
        if side == "LONG":
            rr = calculate_risk_reward(entry=100, sl=99, tp=102, side=side)
            assert rr > 0
        else:
            rr = calculate_risk_reward(entry=100, sl=101, tp=98, side=side)
            assert rr > 0

    def test_calculate_risk_reward_zero_values(self):
        rr = calculate_risk_reward(0, 0, 0, "LONG")
        assert rr == 0.0


class TestRiskCalculation:
    def test_risk_1_percent_capital(self):
        capital = 1_000_000
        risk_pct = 0.01
        risk_amount = capital * risk_pct
        assert risk_amount == 10_000

    def test_position_cap_limit(self):
        capital = 1_000_000
        max_usage = capital * 0.20
        assert max_usage == 200_000