# -*- coding: utf-8 -*-
"""
다중 코인 포트폴리오 관리 모듈
"""
import logging
from datetime import datetime, timedelta
from config.settings import TradingConfig
from src.utils import validate_price_data, safe_divide, calculate_percentage_change

logger = logging.getLogger(__name__)

class MultiCoinPortfolioManager:
    """다중 자산(코인+현금) 포트폴리오 할당/리밸런싱/평가"""
    def __init__(self):
        self.config = TradingConfig()
        self.cash = self.config.INITIAL_BALANCE
        self.coins = {}  # {symbol: {'quantity': float, 'avg_buy_price': float}}
        self.target_allocation = self.config.TARGET_ALLOCATION
        self.trade_history = []
        self.last_trade_times = {} # 코인별 마지막 거래 시간 기록

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

    def is_cooldown_active(self, symbol: str, current_time: datetime) -> bool:
        """주어진 코인이 현재 거래 쿨다운 상태인지 확인"""
        cd_config = self.config.COOLDOWN_CONFIG
        default_cd = cd_config.get('default', {'enabled': False})
        coin_cd = cd_config.get(symbol, default_cd)

        if not coin_cd.get('enabled', False):
            return False # 쿨다운 비활성화

        last_trade_time = self.last_trade_times.get(symbol)
        if not last_trade_time:
            return False # 거래 기록 없음

        # TODO:데이터 최소 시간에 맞춰 수정 필요
        cooldown_period = coin_cd.get('period', 4)
        time_since_last_trade = current_time - last_trade_time

        if time_since_last_trade < timedelta(hours=cooldown_period):
            logger.debug(f"[{symbol} 쿨다운 활성화 중. "
                        f"마지막 거래 후 {time_since_last_trade.total_seconds() / 3600:.0f}시간 경과 "
                        f"(필요: {cooldown_period}시간)")
            return True
        return False

    def calculate_trading_fee(self, trade_value: float, order_type: str) -> float:
        """주문 유형에 따른 수수료 계산"""
        if order_type == 'LIMIT':  # 메이커 주문
            fee_rate = self.config.TRADING_CONFIG.get('maker_fee_percent', 0.0005)
        else:  # 'MARKET' - 테이커 주문
            fee_rate = self.config.TRADING_CONFIG.get('taker_fee_percent', 0.001)
        
        return trade_value * fee_rate

    def execute_trade(self, symbol: str, action: str, quantity: float, price: float, current_time: datetime):
        """중앙화된 거래 실행 및 리스크 관리"""
        if quantity <= 0 or price <= 0:
            logger.warning(f"유효하지 않은 거래 시도: {symbol} 수량={quantity}, 가격={price}")
            return False
        if not validate_price_data(price, symbol):
            return False

        trade_value = quantity * price
        # TODO: 테이커 메이커에 따른 설정 수정
        fee = self.calculate_trading_fee(trade_value, 'MARKET')
        total_portfolio_value = self.get_portfolio_value({symbol: price})

        if action.upper() == 'BUY':
            # 1. 최대 포지션 크기 확인 및 조정
            max_position_value = total_portfolio_value * self.config.MAX_POSITION_SIZE
            current_position_value = self.coins.get(symbol, {}).get('quantity', 0) * price

            if current_position_value + trade_value > max_position_value:
                # 매수 가능한 최대 금액 계산
                adjusted_trade_value = max_position_value - current_position_value

                # 조정된 금액이 최소 거래 금액보다 큰지 확인
                min_trade_amount = self.config.TRADING_CONFIG.get('min_trade_amount', 5000)
                if adjusted_trade_value < min_trade_amount:
                    logger.warning(
                        f"{symbol} 매수 취소: 최대 포지션 크기 도달 후 조정된 금액이 최소 거래 금액 미만입니다. "
                        f"(가용 한도: ₩{adjusted_trade_value:,.0f}, 최소 거래액: ₩{min_trade_amount:,.0f})"
                    )
                    return False
                logger.info(
                    f"INFO: {symbol} 매수 수량 조정: 최대 포지션 크기 초과. "
                    f"(요청액: ₩{trade_value:,.0f} -> 조정액: ₩{adjusted_trade_value:,.0f})"
                )
                # 조정된 값으로 수량, 거래액, 수수료를 다시 계산
                quantity = adjusted_trade_value / price
                trade_value = adjusted_trade_value
                # TODO: 테이커 메이커에 따른 설정 수정
                fee = self.calculate_trading_fee(trade_value, 'MARKET')

            # 2. 현금 잔고 확인
            if self.cash < trade_value + fee:
                logger.warning(f"{symbol} 매수 불가: 현금 부족 (필요: ₩{trade_value + fee:,.0f}, 보유: ₩{self.cash:,.0f})")
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
            'timestamp': current_time, 'symbol': symbol, 'action': action.upper(),
            'qty': quantity, 'price': price, 'value': trade_value, 'fee': fee
        })
        # 거래 성공 시, 마지막 거래 시간 기록
        self.last_trade_times[symbol] = current_time

        logger.info(
            f"거래 실행: {symbol} {'매수' if action.upper() == 'BUY' else '매도'} | "
            f"수량: {quantity:.6f} | 가격: {price:,.0f} | 수수료: {fee:,.0f}"
        )
        return True

    def check_risk_management(self, prices: dict, current_time: datetime = None):
        if current_time is None:
            current_time = datetime.now()
        """보유 포지션에 대한 손절/익절 조건 확인 및 실행"""
        rm_config = self.config.RISK_MANAGEMENT
        default_rm = rm_config.get('default', {'enabled': False})
        # 반복 중 딕셔너리 변경을 피하기 위해 키 목록 복사
        for symbol in list(self.coins.keys()):
            position = self.coins.get(symbol)
            if not position or 'avg_buy_price' not in position:
                continue

            # 코인별 설정 가져오기 (없으면 기본 설정 사용)
            coin_rm = rm_config.get(symbol, default_rm)

            # 리스크 관리가 비활성화된 경우 다음 코인으로 건너뛰기
            if not coin_rm.get('enabled', False):
                continue

            stop_loss_pct = coin_rm.get('stop_loss_percent')
            take_profit_pct = coin_rm.get('take_profit_percent')

            current_price = prices.get(symbol)
            avg_buy_price = position['avg_buy_price']
            quantity = position['quantity']

            if not current_price or avg_buy_price <= 0:
                continue
            # 수익률 계산
            pnl_percent = calculate_percentage_change(avg_buy_price, current_price)

            # 손절 조건 확인 (None이 아니고, 조건 충족 시)
            if stop_loss_pct is not None and pnl_percent <= -stop_loss_pct:
                logger.info(
                    f"🚨 손절매 실행 ({symbol}): "
                    f"수익률 {pnl_percent:.2%} (목표: -{stop_loss_pct:.2%})"
                )
                self.execute_trade(symbol, 'SELL', quantity, current_price, current_time)
                continue # 손절매 실행 후에는 추가 익절 검사 없이 다음 코인으로

            # 이익 실현 조건 확인 (None이 아니고, 조건 충족 시)
            if take_profit_pct is not None and pnl_percent >= take_profit_pct:
                logger.info(
                    f"💰 이익 실현 실행 ({symbol}): "
                    f"수익률 {pnl_percent:.2%} (목표: +{take_profit_pct:.2%})"
                )
                self.execute_trade(symbol, 'SELL', quantity, current_price, current_time)

    def perform_rebalancing(self, prices: dict, current_time: datetime):
        """리밸런싱 로직 (execute_trade 사용)"""
        total_value = self.get_portfolio_value(prices)
        logger.info(f"리밸런싱 시작 (포트폴리오 가치: ₩{total_value:,.0f})")

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
                self.execute_trade(sym, 'BUY', quantity_to_trade, price, current_time)
            else:
                self.execute_trade(sym, 'SELL', quantity_to_trade, price, current_time)

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

