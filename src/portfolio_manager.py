# -*- coding: utf-8 -*-
"""
다중 코인 포트폴리오 관리 모듈
"""
import logging
from datetime import datetime
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
        if not prices:
            # 가격 정보가 없을 경우, 코인 가치를 0으로 계산
            return self.cash
        return self.cash + sum(prices.get(sym, 0) * qty for sym, qty in self.coins.items())
    
    def execute_trade(self, symbol: str, action: str, quantity: float, price: float):
        """중앙화된 거래 실행 및 리스크 관리"""
        if quantity <= 0 or price <= 0:
            logger.warning(f"유효하지 않은 거래 시도: {symbol} 수량={quantity}, 가격={price}")
            return False

        trade_value = quantity * price
        fee = trade_value * self.config.TRADING_CONFIG.get('transaction_fee_percent', 0.001)
        total_portfolio_value = self.get_portfolio_value({symbol: price})

        if action.upper() == 'BUY':
            # 1. 최대 포지션 크기 확인
            max_position_value = total_portfolio_value * self.config.MAX_POSITION_SIZE
            current_position_value = self.coins.get(symbol, 0) * price
            if current_position_value + trade_value > max_position_value:
                logger.warning(
                    f"{symbol} 매수 불가: 최대 포지션 크기 초과. "
                    f"(현재: ${current_position_value:,.2f}, 추가: ${trade_value:,.2f}, "
                    f"한도: ${max_position_value:,.2f})"
                )
                return False

            # 2. 현금 잔고 확인
            if self.cash < trade_value + fee:
                logger.warning(f"{symbol} 매수 불가: 현금 부족 (필요: ${trade_value + fee:,.2f}, 보유: ${self.cash:,.2f})")
                return False

            # 3. 포트폴리오 업데이트
            self.cash -= (trade_value + fee)
            self.coins[symbol] = self.coins.get(symbol, 0) + quantity

        elif action.upper() == 'SELL':
            # 1. 보유 수량 확인
            if self.coins.get(symbol, 0) < quantity:
                logger.warning(f"{symbol} 매도 불가: 보유 수량 부족 (필요: {quantity}, 보유: {self.coins.get(symbol, 0)})")
                return False

            # 2. 포트폴리오 업데이트
            self.cash += (trade_value - fee)
            self.coins[symbol] -= quantity
            if self.coins[symbol] == 0:
                del self.coins[symbol]

        else:
            logger.error(f"알 수 없는 거래 유형: {action}")
            return False

        # 거래 기록
        self.trade_history.append({
            'timestamp': datetime.now(), 'symbol': symbol, 'action': action.upper(),
            'qty': quantity, 'price': price, 'value': trade_value, 'fee': fee
        })
        logger.info(
            f"거래 실행: {symbol} {'매수' if action.upper() == 'BUY' else '매도'} | "
            f"수량: {quantity:.6f} | 가격: {price:,.2f} | 수수료: {fee:,.2f}"
        )
        return True

    def perform_rebalancing(self, prices: dict):
        """리밸런싱 로직 (execute_trade 사용)"""
        total_value = self.get_portfolio_value(prices)
        logger.info(f"리밸런싱 시작 (포트폴리오 가치: ${total_value:,.2f})")

        for sym, target_ratio in self.target_allocation.items():
            if sym == 'CASH' or prices.get(sym, 0) <= 0:
                continue

            price = prices.get(sym)
            target_value = total_value * target_ratio
            current_value = self.coins.get(sym, 0) * price
            diff_value = target_value - current_value

            min_trade_amount = self.config.TRADING_CONFIG.get('min_trade_amount', 10000)
            if abs(diff_value) < min_trade_amount:
                continue

            quantity_to_trade = abs(diff_value) / price
            if diff_value > 0:
                self.execute_trade(sym, 'BUY', quantity_to_trade, price)
            else:
                self.execute_trade(sym, 'SELL', quantity_to_trade, price)

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
```

