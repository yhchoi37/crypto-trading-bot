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
        self.coins = {}  # {symbol: {'quantity': float, 'avg_buy_price': float}}
        self.target_allocation = self.config.TARGET_ALLOCATION
        self.trade_history = []

    def set_target_allocation(self, allocation: dict):
        """목표 자산 비중 설정"""
        self.target_allocation = allocation

    def get_current_allocation(self, prices: dict):
        """현재 포트폴리오의 비중 계산"""
        total_value = self.get_portfolio_value(prices)
        alloc = {}
        for sym in self.target_allocation.keys():
            if sym == 'CASH':
                alloc[sym] = self.cash / total_value if total_value > 0 else 0
            else:
                current_value = self.coins.get(sym, {}).get('quantity', 0) * prices.get(sym, 0)
                alloc[sym] = current_value / total_value if total_value > 0 else 0
        return alloc
        
    def get_portfolio_value(self, prices: dict):
        """전체 포트폴리오 가치 계산"""
        if not prices:
            # 가격 정보가 없을 경우, 코인 가치를 0으로 계산
            return self.cash
        return self.cash + sum(prices.get(sym, 0) * pos.get('quantity', 0) for sym, pos in self.coins.items())

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
            current_position_value = self.coins.get(symbol, {}).get('quantity', 0) * price
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

            # 3. 포트폴리오 업데이트 및 평균 매수 단가 재계산
            self.cash -= (trade_value + fee)

            if symbol in self.coins:
                # 기존 포지션에 추가 매수
                position = self.coins[symbol]
                old_quantity = position['quantity']
                old_value = old_quantity * position['avg_buy_price']

                new_quantity = old_quantity + quantity
                new_value = old_value + trade_value

                position['avg_buy_price'] = new_value / new_quantity
                position['quantity'] = new_quantity
            else:
                # 신규 포지션
                self.coins[symbol] = {
                    'quantity': quantity,
                    'avg_buy_price': price
                }

        elif action.upper() == 'SELL':
            # 1. 보유 수량 확인
            if self.coins.get(symbol, {}).get('quantity', 0) < quantity:
                logger.warning(f"{symbol} 매도 불가: 보유 수량 부족 (필요: {quantity}, 보유: {self.coins.get(symbol, {}).get('quantity', 0)})")
                return False

            # 2. 포트폴리오 업데이트
            self.cash += (trade_value - fee)

            position = self.coins[symbol]
            position['quantity'] -= quantity
            if position['quantity'] < 1e-8: # 부동소수점 오차 감안
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

    def check_risk_management(self, prices: dict):
        """보유 포지션에 대한 손절/익절 조건 확인 및 실행"""
        stop_loss_pct = self.config.TRADING_CONFIG.get('stop_loss_percent')
        take_profit_pct = self.config.TRADING_CONFIG.get('take_profit_percent')

        # 반복 중 딕셔너리 변경을 피하기 위해 키 목록 복사
        for symbol in list(self.coins.keys()):
            position = self.coins.get(symbol)
            if not position or 'avg_buy_price' not in position:
                continue

            current_price = prices.get(symbol)
            avg_buy_price = position['avg_buy_price']
            quantity = position['quantity']

            if not current_price or avg_buy_price <= 0:
                continue

            # 수익률 계산
            pnl_percent = (current_price - avg_buy_price) / avg_buy_price

            # 손절 조건 확인
            if stop_loss_pct and pnl_percent <= -stop_loss_pct:
                logger.info(
                    f"🚨 손절매 실행: {symbol} | "
                    f"수익률: {pnl_percent:.2%} (목표: -{stop_loss_pct:.2%})"
                )
                self.execute_trade(symbol, 'SELL', quantity, current_price)
                continue # 다음 코인으로

            # 이익 실현 조건 확인
            if take_profit_pct and pnl_percent >= take_profit_pct:
                logger.info(
                    f"💰 이익 실현 실행: {symbol} | "
                    f"수익률: {pnl_percent:.2%} (목표: +{take_profit_pct:.2%})"
                )
                self.execute_trade(symbol, 'SELL', quantity, current_price)

    def perform_rebalancing(self, prices: dict):
        """리밸런싱 로직 (execute_trade 사용)"""
        total_value = self.get_portfolio_value(prices)
        logger.info(f"리밸런싱 시작 (포트폴리오 가치: ${total_value:,.2f})")

        for sym, target_ratio in self.target_allocation.items():
            if sym == 'CASH' or prices.get(sym, 0) <= 0:
                continue

            price = prices.get(sym)
            target_value = total_value * target_ratio
            current_value = self.coins.get(sym, {}).get('quantity', 0) * price
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
                'trades_today': len(self.trade_history), # TODO: 날짜별 거래 필터링 필요
                'total_positions': len([p for p in self.coins.values() if p.get('quantity', 0) > 0])
            }
        }

    def export_trade_history(self, filename):
        """거래 히스토리 파일로 저장"""
        import pandas as pd
        df = pd.DataFrame(self.trade_history)
        df.to_csv(filename, index=False)
        return filename

