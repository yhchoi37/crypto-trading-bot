# -*- coding: utf-8 -*-
"""
메인 트레이딩 시스템 모듈
"""
import logging
from datetime import datetime
import pandas as pd
import pandas_ta as ta # pandas-ta 임포트

from .data_manager import MultiCoinDataManager
from .portfolio_manager import MultiCoinPortfolioManager
from .social_sentiment import SocialSentimentBasedAlgorithm, TwitterSentimentCollector, RedditSentimentCollector
from config.settings import TradingConfig # TradingConfig import 추가

logger = logging.getLogger(__name__)

class TechnicalAnalysisAlgorithm:
    """동적 지표 조합 및 파라미터 기반 기술적 분석 알고리즘"""
    def __init__(self):
        pass

    # 클래스 시작 부분에 헬퍼 메서드 추가:
    def _find_indicator_column(self, df: pd.DataFrame, indicator: str, *params) -> str:
        """동적으로 지표 컬럼명 찾기"""
        if indicator == 'SMA':
            period = params[0]
            candidates = [f'SMA_{period}', f'sma_{period}', f'SMA{period}']
        elif indicator == 'RSI':
            period = params[0]
            candidates = [f'RSI_{period}', f'rsi_{period}', f'RSI{period}']
        elif indicator == 'BBL':
            window, std = params
            candidates = [f'BBL_{window}_{std}.0', f'BBL_{window}_{int(std)}', f'bbl_{window}_{std}']
        elif indicator == 'BBU':
            window, std = params
            candidates = [f'BBU_{window}_{std}.0', f'BBU_{window}_{int(std)}', f'bbu_{window}_{std}']
        
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def evaluate_signal(self, indicator, params, latest, previous, df, mode):
        """지표별 매수/매도 조건 평가를 통합 처리"""
        if indicator == 'MA_Cross':
            ma_s_period, ma_l_period = params.get("ma_short_period"), params.get("ma_long_period")
            if ma_s_period and ma_l_period:
                ma_s_col = self._find_indicator_column(df, 'SMA', ma_s_period)
                ma_l_col = self._find_indicator_column(df, 'SMA', ma_l_period)
                if ma_s_col and ma_l_col:
                    if mode == 'buy':
                        return (latest[ma_s_col] > latest[ma_l_col]) and (previous[ma_s_col] <= previous[ma_l_col])
                    else:
                        return (latest[ma_s_col] < latest[ma_l_col]) and (previous[ma_s_col] >= previous[ma_l_col])
        elif indicator == 'RSI':
            rsi_period = params.get("rsi_period")
            if mode == 'buy':
                rsi_threshold = params.get('rsi_oversold_threshold')
            else:
                rsi_threshold = params.get('rsi_overbought_threshold')
            if rsi_period and rsi_threshold:
                rsi_col = self._find_indicator_column(df, 'RSI', rsi_period)
                if rsi_col:
                    if mode == 'buy':
                        return latest[rsi_col] < rsi_threshold
                    else:
                        return latest[rsi_col] > rsi_threshold
        elif indicator == 'BollingerBand':
            bb_window, bb_std = params.get("bollinger_window"), params.get("bollinger_std_dev")
            if bb_window and bb_std:
                if mode == 'buy':
                    bbl_col = self._find_indicator_column(df, 'BBL', bb_window, bb_std)
                    if bbl_col:
                        return latest['close'] < latest[bbl_col]
                else:
                    bbu_col = self._find_indicator_column(df, 'BBU', bb_window, bb_std)
                    if bbu_col:
                        return latest['close'] > latest[bbu_col]
        return False
    
    def generate_signal(self, historical_data: pd.DataFrame, indicator_combo: tuple, buy_params: dict, sell_params: dict, weights: dict) -> dict:
        """주어진 지표 조합과 파라미터로 신호를 생성합니다."""
        required_period = max(
            buy_params.get('MA_Cross', {}).get('ma_long_period', 20),
            sell_params.get('MA_Cross', {}).get('ma_long_period', 20)
        )
        if historical_data.empty or len(historical_data) < required_period:
            return {'action': 'HOLD', 'strength': 0}

        df = historical_data
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        log_msg_details = []
        buy_conditions = {}
        sell_conditions = {}
        # 반복적으로 각 지표별 조건 평가
        for indicator in indicator_combo:
            # Buy 조건
            if indicator in buy_params:
                cond = self.evaluate_signal(indicator, buy_params[indicator], latest, previous, df, mode='buy')
                buy_conditions[indicator] = cond
                log_msg_details.append(f"Buy {indicator}: {cond}")
            # Sell 조건
            if indicator in sell_params:
                cond = self.evaluate_signal(indicator, sell_params[indicator], latest, previous, df, mode='sell')
                sell_conditions[indicator] = cond
                log_msg_details.append(f"Sell {indicator}: {cond}")
        # 점수 일괄 계산
        buy_score = sum(weights.get(f'{ind}_buy', 1) for ind, cond in buy_conditions.items() if cond and ind in buy_params)
        sell_score = sum(weights.get(f'{ind}_sell', 1) for ind, cond in sell_conditions.items() if cond and ind in sell_params)

        # --- 최종 결정 ---
        buy_trigger = buy_params.get('buy_trigger_threshold', 99)
        sell_trigger = sell_params.get('sell_trigger_threshold', 99)
        is_buy_signal = buy_score >= buy_trigger
        is_sell_signal = sell_score >= sell_trigger

        action = 'HOLD'
        strength = 0
        if is_buy_signal and is_sell_signal:
            action = 'CONFLICT'
            strength = max(buy_score, sell_score) # 둘 중 더 강한 신호의 점수를 강도로 사용
        elif is_buy_signal:
            action, strength = 'BUY', buy_score
        elif is_sell_signal:
            action, strength = 'SELL', sell_score

        # 상세 로그 출력 (점수가 0보다 클 때만)
        if buy_score > 0 or sell_score > 0:
            coin_symbol = df['coin'].iloc[-1] # 데이터프레임에서 코인 심볼 가져오기
            # latest.name이 timestamp 객체인지 확인하고 포맷팅
            if isinstance(latest.name, pd.Timestamp):
                timestamp_str = latest.name.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_str = str(latest.name)

            logger.debug(
                f"[{timestamp_str}][{coin_symbol}] Signal Eval: "
                f"Scores(Buy:{buy_score}/Sell:{sell_score}) | "
                f"Triggers(Buy:{buy_trigger}/Sell:{sell_trigger}) | "
                f"Final Action: {action} | Details: {', '.join(log_msg_details)}"
            )

        return {'action': action, 'strength': strength}


class MultiCoinTradingSystem:
    """다중 코인 통합 트레이딩 시스템"""
    def __init__(self, initial_balance: float = 10000000, config: TradingConfig = None):
        """
        시스템을 초기화합니다.
        config 객체가 주입되지 않으면 새로 생성합니다.
        """
        logger.info(f"🚀 트레이딩 시스템 초기화 - 초기 자본: ￦{initial_balance:,.0f}")
        self.config = config if config else TradingConfig()
        # Pass initial_balance to portfolio manager if available
        try:
            self.portfolio_manager = MultiCoinPortfolioManager(initial_balance=initial_balance)
        except TypeError:
            # backward compatibility: constructor may not accept parameter
            self.portfolio_manager = MultiCoinPortfolioManager()
        self.data_manager = MultiCoinDataManager()
        self.twitter_collector = TwitterSentimentCollector()
        self.reddit_collector = RedditSentimentCollector()
        self.algorithms = {}
        self.setup_algorithms()
        logger.info("✅ 트레이딩 시스템 초기화 완료")

    def setup_algorithms(self):
        """알고리즘 설정"""
        logger.info("🔧 거래 알고리즘 설정 중...")
        enabled_coins = [coin for coin in self.config.TARGET_ALLOCATION if coin != 'CASH']

        # backtest.py로 실행 시 항상 기술적 분석 사용
        if self.config.IS_BACKTEST_MODE:
            tech_algo = TechnicalAnalysisAlgorithm()
            self.algorithms['technical_analysis'] = {
                'algorithm': tech_algo, 'weight': 1.0, 'enabled_coins': enabled_coins
            }
            logger.info("📈 백테스트 모드 활성화. 기술적 분석 알고리즘을 사용합니다.")
        else:
            # main.py로 실행 시 소셜 센티멘트 등 다른 알고리즘 사용 가능 (현재는 기술적 분석으로 고정)
            # TODO: 실거래 시 사용할 알고리즘 선택 로직 추가
            tech_algo = TechnicalAnalysisAlgorithm()
            self.algorithms['technical_analysis'] = {
                'algorithm': tech_algo, 'weight': 1.0, 'enabled_coins': enabled_coins
            }
            logger.info("🤖 실시간/모의 거래 모드 활성화. 기술적 분석 알고리즘을 사용합니다.")
        logger.info(f"✅ {len(self.algorithms)}개 알고리즘 설정 완료. 대상 코인: {', '.join(enabled_coins)}")

    def setup_portfolio_allocation(self, target_allocation: dict):
        """포트폴리오 목표 배분 설정"""
        self.portfolio_manager.set_target_allocation(target_allocation)
        logger.info("🎯 포트폴리오 목표 배분 설정:")
        for asset, weight in target_allocation.items():
            logger.info(f"  - {asset}: {weight:.1%}")

    def analyze_coin_signals(self, coin: str, data: pd.DataFrame, job_config: dict = None) -> dict:
        """특정 코인에 대한 종합 신호 분석 (백테스트 시 파라미터 주입 가능)"""
        tech_algo_info = self.algorithms.get('technical_analysis')
        if not tech_algo_info or coin not in tech_algo_info.get('enabled_coins', []):
            return {'decision': {'action': 'HOLD', 'strength': 0}}

        algo = tech_algo_info['algorithm']

        # 백테스트 모드일 경우에만 job_config(최적화 파라미터)를 사용
        if self.config.IS_BACKTEST_MODE and job_config:
            # 백테스트: job_config에서 파라미터 추출
            signal_weights = job_config.get('signal_weights', {})
            buy_params = {
                'buy_trigger_threshold': job_config.get('buy_trigger_threshold'),
                **job_config.get('buy_indicators', {})
            }
            sell_params = {
                'sell_trigger_threshold': job_config.get('sell_trigger_threshold'),
                **job_config.get('sell_indicators', {})
            }
            buy_indicator_combo = tuple(job_config.get('buy_indicators', {}).keys())
            sell_indicator_combo = tuple(job_config.get('sell_indicators', {}).keys())
            indicator_combo = tuple(set(buy_indicator_combo) | set(sell_indicator_combo))
        else:
            # 실시간/모의 거래: config 파일의 기본 설정 사용
            config_params = self.config.TECHNICAL_ANALYSIS_CONFIG
            signal_weights = config_params.get('signal_weights', {})
            buy_params = {
                'buy_trigger_threshold': config_params.get('buy_trigger_threshold'),
                **config_params.get('buy_indicators', {})
            }
            sell_params = {
                'sell_trigger_threshold': config_params.get('sell_trigger_threshold'),
                **config_params.get('sell_indicators', {})
            }
            buy_indicator_combo = tuple(config_params.get('buy_indicators', {}).keys())
            sell_indicator_combo = tuple(config_params.get('sell_indicators', {}).keys())
            indicator_combo = tuple(set(buy_indicator_combo) | set(sell_indicator_combo))

        signal = algo.generate_signal(data, indicator_combo, buy_params, sell_params, signal_weights)
        return {'decision': signal}

    def run_trading_cycle(self) -> dict:
        """한 번의 거래 사이클 실행"""
        logger.info(f"🔄 거래 사이클 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        coins = [coin for coin in self.config.TARGET_ALLOCATION if coin != 'CASH']
        current_prices = self.data_manager.get_coin_prices(coins)

        active_signals = []
        all_coin_data = self.data_manager.generate_multi_coin_data(coins, days=50)

        for coin in coins:
            if all_coin_data.empty:
                logger.warning(f"{coin}에 대한 데이터를 가져올 수 없습니다.")
                continue

            coin_data = all_coin_data[all_coin_data['coin'] == coin]
            if not coin_data.empty:
                # 실시간 거래에서는 job_config 없이 호출
                analysis = self.analyze_coin_signals(coin, coin_data)
                decision = analysis['decision']
                if decision['action'] != 'HOLD':
                    active_signals.append({
                        'coin': coin, 'decision': decision,
                        'price': current_prices.get(coin, 0)
                    })

        # --- 거래 실행 로직 (main.py로 실행 시에만 해당) ---
        if active_signals and not self.config.IS_BACKTEST_MODE:
            portfolio_value = self.portfolio_manager.get_portfolio_value(current_prices)
            current_allocations = self.portfolio_manager.get_current_allocation(current_prices)
            target_allocations = self.config.TARGET_ALLOCATION

            log_prefix = "[모의 거래]" if self.config.SIMULATION_MODE else "[실거래]"
            logger.info(f"📊 {log_prefix} {len(active_signals)}개의 활성 신호를 기반으로 거래 검토...")
            for signal in active_signals:
                coin, decision, price = signal['coin'], signal['decision'], signal['price']
                if not price or price <= 0: continue
                action = decision['action']
                position = self.portfolio_manager.coins.get(coin)
                has_position = position and position.get('quantity', 0) > 0

                # CONFLICT 신호 처리 로직
                if action == 'CONFLICT':
                    # 포지션이 있으면 매도, 없으면 매수
                    action = 'SELL' if has_position else 'BUY'
                    logger.info(f"{log_prefix} {coin}의 신호 충돌 발생. 포지션 보유 여부({has_position})에 따라 '{action}'으로 결정.")

                if action == 'BUY':
                    target_ratio = target_allocations.get(coin, 0)
                    current_ratio = current_allocations.get(coin, 0)
                    if current_ratio < target_ratio:
                        amount_to_invest = (target_ratio - current_ratio) * portfolio_value
                        min_trade_amount = self.config.TRADING_CONFIG.get('min_trade_amount', 10000)

                        if amount_to_invest > min_trade_amount:
                            quantity = amount_to_invest / price
                            logger.info(f"{log_prefix} {coin} 매수 실행: 수량={quantity:.6f}, 가격={price:,.0f}")
                            if not self.config.SIMULATION_MODE:
                                self.portfolio_manager.execute_trade(coin, 'BUY', quantity, price)
                        else:
                            logger.warning(f"{log_prefix} {coin} 최소 거래 금액 {min_trade_amount} 미만으로 실패: {amount_to_invest}")

                elif action == 'SELL':
                    if has_position:
                        quantity_to_sell = position['quantity'] * 0.5 # 예시: 50% 매도
                        logger.info(f"{log_prefix} {coin} 매도 실행: 수량={quantity_to_sell:.6f}, 가격={price:,.0f}")
                        if not self.config.SIMULATION_MODE:
                            self.portfolio_manager.execute_trade(coin, 'SELL', quantity_to_sell, price)

        elif not active_signals:
            logger.info("📊 활성 거래 신호 없음.")
            
        portfolio_value = self.portfolio_manager.get_portfolio_value(current_prices)
        return {
            'timestamp': datetime.now(),
            'prices': current_prices,
            'active_signals': active_signals,
            'portfolio_value': portfolio_value
        }
    
    def perform_rebalancing(self, prices: dict):
        """포트폴리오 리밸런싱 실행"""
        logger.info("⚖️ 포트폴리오 리밸런싱 확인 중...")
        self.portfolio_manager.perform_rebalancing(prices)

