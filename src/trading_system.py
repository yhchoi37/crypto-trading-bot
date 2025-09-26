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

    def generate_signal(self, historical_data: pd.DataFrame, indicator_combo: tuple, buy_params: dict, sell_params: dict) -> dict:
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
        buy_score = 0
        sell_score = 0
        weights = buy_params.get('signal_weights', {})

        # --- 매수 신호 점수 계산 (사전 계산된 컬럼 사용) ---
        if 'MA_Cross' in indicator_combo:
            ma_params = buy_params.get('MA_Cross', {})
            ma_s_col = f'SMA_{ma_params.get("ma_short_period")}'
            ma_l_col = f'SMA_{ma_params.get("ma_long_period")}'
            if ma_s_col in df.columns and ma_l_col in df.columns:
                if latest[ma_s_col] > latest[ma_l_col] and previous[ma_s_col] <= previous[ma_l_col]:
                    buy_score += weights.get('MA_Cross_buy', 1)

        if 'RSI' in indicator_combo:
            rsi_params = buy_params.get('RSI', {})
            rsi_col = f'RSI_{rsi_params.get("rsi_period")}'
            if rsi_col in df.columns:
                if latest[rsi_col < rsi_params.get('rsi_oversold_threshold'):
                    buy_score += weights.get('RSI_buy', 1)

        if 'BollingerBand' in indicator_combo:
            bb_params = buy_params.get('BollingerBand', {})
            bbl_col = f'BBL_{bb_params.get("bollinger_window")}_{bb_params.get("bollinger_std_dev")}.0'
            if bbl_col in df.columns:
                if latest['close'] < latest[bbl_col]:
                    buy_score += weights.get('BollingerBand_buy', 1)

        # --- 매도 신호 점수 계산 (사전 계산된 컬럼 사용) ---
        if 'MA_Cross' in indicator_combo:
            ma_params = sell_params.get('MA_Cross', {})
            ma_s_col = f'SMA_{ma_params.get("ma_short_period")}'
            ma_l_col = f'SMA_{ma_params.get("ma_long_period")}'
            if ma_s_col in df.columns and ma_l_col in df.columns:
                if latest[ma_s_col] < latest[ma_l_col] and previous[ma_s_col] >= previous[ma_l_col]:
                    sell_score += weights.get('MA_Cross_sell', 1)

        if 'RSI' in indicator_combo:
            rsi_params = sell_params.get('RSI', {})
            rsi_col = f'RSI_{rsi_params.get("rsi_period")}'
            if rsi_col in df.columns:
                if latest[rsi_col] > rsi_params.get('rsi_overbought_threshold'):
                    sell_score += weights.get('RSI_sell', 1)

        if 'BollingerBand' in indicator_combo:
            bb_params = sell_params.get('BollingerBand', {})
            bbu_col = f'BBU_{bb_params.get("bollinger_window")}_{bb_params.get("bollinger_std_dev")}.0'
            if bbu_col in df.columns:
                if latest['close'] > latest[bbu_col]:
                    sell_score += weights.get('BollingerBand_sell', 1)

        # --- 최종 결정 ---
        buy_trigger = buy_params.get('buy_trigger_threshold', 99)
        sell_trigger = sell_params.get('sell_trigger_threshold', 99)
        is_buy_signal = buy_score >= buy_trigger
        is_sell_signal = sell_score >= sell_trigger

        if is_buy_signal and not is_sell_signal:
            return {'action': 'BUY', 'strength': buy_score}
        elif is_sell_signal and not is_buy_signal:
            return {'action': 'SELL', 'strength': sell_score}
        else:
            return {'action': 'HOLD', 'strength': 0}


class MultiCoinTradingSystem:
    """다중 코인 통합 트레이딩 시스템"""
    def __init__(self, initial_balance: float = 100000):
        logger.info(f"🚀 트레이딩 시스템 초기화 - 초기 자본: ￦{initial_balance:,.2f}")
        self.config = TradingConfig()
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
        if self.config.BACKTEST_MODE:
            # 백테스트 모드: 기술적 분석 알고리즘 객체만 생성 (파라미터는 실행 시 주입)
            tech_algo = TechnicalAnalysisAlgorithm()
            self.algorithms['technical_analysis'] = {
                'algorithm': tech_algo,
                'weight': 1.0,
                'enabled_coins': enabled_coins
            }
            logger.info("📈 백테스트 모드 활성화. 기술적 분석 알고리즘을 사용합니다.")
        else:
            # 실시간 거래 모드: 소셜 센티멘트 알고리즘 사용
            social_algo = SocialSentimentBasedAlgorithm(self.twitter_collector, self.reddit_collector)
            self.algorithms['social_sentiment'] = {
                'algorithm': social_algo,
                'weight': 1.0,
                'enabled_coins': enabled_coins
            }
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

        if self.config.BACKTEST_MODE and job_config:
            # 백테스트: job_config에서 파라미터 추출
            buy_params = {
                'buy_trigger_threshold': job_config.get('buy_trigger_threshold'),
                'signal_weights': job_config.get('signal_weights', {}),
                **job_config.get('buy_indicators', {})
            }
            sell_params = {
                'sell_trigger_threshold': job_config.get('sell_trigger_threshold'),
                'signal_weights': job_config.get('signal_weights', {}),
                **job_config.get('sell_indicators', {})
            }
            buy_indicator_combo = tuple(job_config.get('buy_indicators', {}).keys())
            sell_indicator_combo = tuple(job_config.get('sell_indicators', {}).keys())
            indicator_combo = tuple(set(buy_indicator_combo) | set(sell_indicator_combo))
        else:
            # 실시간 거래 또는 기본 설정: config에서 파라미터 추출
            config_params = self.config.TECHNICAL_ANALYSIS_CONFIG
            buy_params = {
                'buy_trigger_threshold': config_params.get('buy_trigger_threshold'),
                'signal_weights': config_params.get('signal_weights', {}),
                **config_params.get('buy_indicators', {})
            }
            sell_params = {
                'sell_trigger_threshold': config_params.get('sell_trigger_threshold'),
                'signal_weights': config_params.get('signal_weights', {}),
                **config_params.get('sell_indicators', {})
            }
            buy_indicator_combo = tuple(config_params.get('buy_indicators', {}).keys())
            sell_indicator_combo = tuple(config_params.get('sell_indicators', {}).keys())
            indicator_combo = tuple(set(buy_indicator_combo) | set(sell_indicator_combo))

        signal = algo.generate_signal(data, indicator_combo, buy_params, sell_params)
        return {'decision': signal}

    def run_trading_cycle(self) -> dict:
        """한 번의 거래 사이클 실행"""
        logger.info(f"🔄 거래 사이클 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # TARGET_ALLOCATION에 설정된 코인 목록을 가져옴 (CASH 제외)
        coins = [coin for coin in self.config.TARGET_ALLOCATION if coin != 'CASH']
        current_prices = self.data_manager.get_coin_prices(coins)
        
        active_signals = []

        # 모든 코인 데이터를 한 번에 가져오도록 수정
        all_coin_data = self.data_manager.generate_multi_coin_data(coins, days=50) # MA 계산을 위해 충분한 데이터 확보

        for coin in coins:
            if all_coin_data.empty:
                logger.warning(f"{coin}에 대한 데이터를 가져올 수 없습니다.")
                continue

            coin_data = all_coin_data[all_coin_data['coin'] == coin]
            if not coin_data.empty:
                # 실시간 거래에서는 job_config 없이 호출
                analysis = self.analyze_coin_signals(coin, coin_data)

                if analysis['decision']['action'] != 'HOLD':
                    decision = analysis['decision'
                    active_signals.append({
                        'coin': coin,
                        'decision': decision,
                        'price': current_prices.get(coin, 0)
                    })

        if active_signals:
            logger.info(f"📊 {len(active_signals)}개의 활성 거래 신호 발견.")
        else:
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

