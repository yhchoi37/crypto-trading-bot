# -*- coding: utf-8 -*-
"""
다중 코인 포트폴리오 관리 모듈
"""
import logging
from config.settings import TradingConfig

logger = logging.getLogger(__name__)

class MultiCoinPortfolioManager:
    """다중 자산(코인+현금) 포트폴리오 할당/리밸런싱/평가"""
    def __init__(self):
        self.config = TradingConfig()
        self.cash = self.config.INITIAL_BALANCE
        self.coins = {}  # {symbol: quantity}
        self.target_allocation = self.config.TARGET_ALLOCATION
        self.trade_history = []

    def set_target_allocation(self, allocation: dict):
        """목표 자산 비중 설정"""
        self.target_allocation = allocation

    def get_current_allocation(self, prices: dict):
        """현재 포트폴리오의 비중 계산"""
        total_value = self.cash + sum(prices.get(sym, 0)*qty for sym, qty in self.coins.items())
        alloc = {}
        for sym in self.target_allocation.keys():
            if sym == 'CASH':
                alloc[sym] = self.cash / total_value if total_value > 0 else 0
            else:
                alloc[sym] = (self.coins.get(sym, 0) * prices.get(sym, 0)) / total_value if total_value > 0 else 0
        return alloc

    def get_portfolio_value(self, prices: dict):
        """전체 포트폴리오 가치 계산"""
        return self.cash + sum(prices.get(sym,0)*qty for sym, qty in self.coins.items())
    
    def perform_rebalancing(self, prices: dict):
        """리밸런싱 예시 로직(단순 목표 비율 맞추기)"""
        total_value = self.get_portfolio_value(prices)
        logger.info(f"리밸런싱 시작 (포트폴리오 가치: {total_value})")
        for sym, target_ratio in self.target_allocation.items():
            if sym == 'CASH': 
                continue
            target_value = total_value * target_ratio
            current_value = self.coins.get(sym, 0) * prices.get(sym, 0)
            diff_value = target_value - current_value
            qty_to_trade = diff_value // prices.get(sym, 1)
            if abs(qty_to_trade) > 0:
                self.coins[sym] = self.coins.get(sym, 0) + qty_to_trade
                self.cash -= qty_to_trade * prices.get(sym, 1)
                self.trade_history.append({
                    'symbol': sym,
                    'qty': qty_to_trade,
                    'price': prices.get(sym, 1),
                })
                logger.info(f"{sym} 리밸런싱 거래: {qty_to_trade}, 가격: {prices.get(sym, 1)}")

    def get_portfolio_summary(self, prices=None):
        """간단한 포트폴리오 요약"""
        value = self.get_portfolio_value(prices or {})
        cash_balance = self.cash
        total_return = ((value - self.config.INITIAL_BALANCE)/self.config.INITIAL_BALANCE) * 100 if value else None
        return {
            'metrics':{
                'total_value': value,
                'total_return': total_return,
                'cash_balance': cash_balance,
                'trades_today': len(self.trade_history),
                'total_positions': len([q for q in self.coins.values() if q!=0])
            }
        }

    def export_trade_history(self, filename):
        """거래 히스토리 파일로 저장"""
        import pandas as pd
        df = pd.DataFrame(self.trade_history)
        df.to_csv(filename, index=False)
        return filename
