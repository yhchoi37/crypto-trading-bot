import pytest
from datetime import datetime
from src.portfolio_manager import MultiCoinPortfolioManager


def test_buy_and_sell_flow():
    pm = MultiCoinPortfolioManager(initial_balance=100000)
    # buy 1 unit at 1000
    success = pm.execute_trade('TEST', 'BUY', 1, 1000, datetime.now())
    assert success
    assert pm.coins['TEST']['quantity'] == 1
    assert pm.cash < 100000

    # sell 0.5 unit
    success = pm.execute_trade('TEST', 'SELL', 0.5, 1200, datetime.now())
    assert success
    # remaining quantity should be 0.5
    assert pytest.approx(pm.coins['TEST']['quantity'], rel=1e-6) == 0.5
