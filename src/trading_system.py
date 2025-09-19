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
    def generate_signal(self, historical_data: pd.DataFrame, indicator_combo: tuple, params: dict) -> dict:
        """
        주어진 지표 조합과 파라미터로 신호를 생성합니다.
        :param historical_data: 분석할 과거 데이터 (OHLCV)
        :param indicator_combo: 사용할 지표 이름의 튜플 (예: ('MA_Cross', 'RSI'))
        :param params: 지표 계산에 필요한 모든 파라미터 딕셔너리
        """
        # 데이터가 충분한지 기본 검사 (가장 긴 ma_long_period 기준)
        if historical_data.empty or len(historical_data) < params.get('ma_long_period', 20):
            return {'action': 'HOLD', 'strength': 0}

        df = historical_data.copy()
        buy_score = 0

        # --- 1. MA_Cross 신호 계산 ---
        if 'MA_Cross' in indicator_combo:
            df.ta.sma(length=params['ma_short_period'], append=True)
            df.ta.sma(length=params['ma_long_period'], append=True)

            ma_short_col = f'SMA_{params["ma_short_period"]}'
        ma_long_col = f'SMA_{params["ma_long_period"]}'

            latest = df.iloc[-1]
            previous = df.iloc[-2]
            if latest[ma_short_col] > latest[ma_long_col and previous[ma_short_col] <= previous[ma_long_col:
                buy_score += params['weights']['MA_Cross_buy']

        # --- 2. RSI 신호 계산 ---
        if 'RSI' in indicator_combo:
            df.ta.rsi(length=params['rsi_period'], append=True)
        rsi_col = f'RSI_{params["rsi_period"]}'
            if df.iloc[-1][rsi_col] < params['rsi_oversold_threshold']:
                buy_score += params['weights']['RSI_buy']

        # --- 3. BollingerBand 신호 계산 ---
        if 'BollingerBand' in indicator_combo:
            df.ta.bbands(length=params['bollinger_window'], std=params['bollinger_std_dev'], append=True)
            bb_lower_col = f'BBL_{params["bollinger_window"]}_{params["bollinger_std_dev"]}.0'
            if df.iloc[-1]['close'] < df.iloc[-1][bb_lower_col]:
                buy_score += params['weights']['BollingerBand_buy']
        # --- 최종 결정 ---
        if buy_score >= params['buy_trigger_threshold']:
            return {'action': 'BUY', 'strength': buy_score}
        return {'action': 'HOLD', 'strength': 0}

class MultiCoinTradingSystem:
    """다중 코인 통합 트레이딩 시스템"""
    def __init__(self, initial_balance: float = 100000):
        logger.info(f"🚀 트레이딩 시스템 초기화 - 초기 자본: ${initial_balance:,.2f}")
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

    def analyze_coin_signals(self, coin: str, data: pd.DataFrame, indicator_combo: tuple = None, params: dict = None) -> dict:
        """특정 코인에 대한 종합 신호 분석 (백테스트 시 파라미터 주입 가능)"""
        if self.config.BACKTEST_MODE:
            tech_algo_info = self.algorithms.get('technical_analysis')
            if tech_algo_info and coin in tech_algo_info['enabled_coins']:
                algo = tech_algo_info['algorithm']
                if indicator_combo and params:
                    return {'decision': algo.generate_signal(data, indicator_combo, params)}

        # 실시간 모드 또는 기본 백테스트 로직 (현재 최적화에서는 사용되지 않음)
        # Social sentiment logic was here, now defaults to HOLD for non-optimization runs
        return {'decision': {'action': 'HOLD', 'strength': 0}}
    def run_trading_cycle(self) -> dict:
        """한 번의 거래 사이클 실행"""
        logger.info(f"🔄 거래 사이클 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # TARGET_ALLOCATION에 설정된 코인 목록을 가져옴 (CASH 제외)
        coins = [coin for coin in self.config.TARGET_ALLOCATION if coin != 'CASH']
        current_prices = self.data_manager.get_coin_prices(coins)
        
        active_signals = []

        # 모든 코인 데이터를 한 번에 가져오도록 수정
        all_coin_data = self.data_manager.generate_multi_coin_data(coins, days=7)

        for coin in coins:
            if all_coin_data.empty:
                logger.warning(f"{coin}에 대한 데이터를 가져올 수 없습니다.")
                continue
            coin_data = all_coin_data[all_coin_data['coin'] == coin]
            if not coin_data.empty:
                analysis = self.analyze_coin_signals(coin, coin_data)

                if analysis['decision']['action'] != 'HOLD':
                    decision = analysis['decision']
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

