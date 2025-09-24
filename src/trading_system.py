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

    def generate_signal(self, historical_data: pd.DataFrame, job_config: dict) -> dict:
        """주어진 지표 조합과 파라미터로 신호를 생성합니다."""
        buy_combo = job_config.get('buy_indicator_combo', ())
        sell_combo = job_config.get('sell_indicator_combo', ())
        params = job_config.get('params', {})

        # --- 상세 로그: Job 정보 출력 ---
        logger.debug(f"--- Running Job ---")
        logger.debug(f"Buy Combo: {buy_combo}")
        logger.debug(f"Sell Combo: {sell_combo}")
        logger.debug(f"Params: {params}")

        min_period = max(params.get('buy_ma_long_period', 20), params.get('sell_ma_long_period', 20))
        if historical_data.empty or len(historical_data) < min_period:
            return {'action': 'HOLD', 'strength': 0}

        df = historical_data.copy()

        # --- 1. 필요한 모든 지표 계산을 미리 수행 ---
        try:
            # MA Cross
            if 'MA_Cross' in buy_combo:
                df.ta.sma(length=params['buy_ma_short_period'], append=True)
                df.ta.sma(length=params['buy_ma_long_period'], append=True)
            # RSI Buy
            if 'RSI' in buy_combo:
                df.ta.rsi(length=params['buy_rsi_period'], append=True)
            # Dead Cross
            if 'Dead_Cross' in sell_combo:
                df.ta.sma(length=params['sell_ma_short_period'], append=True)
                df.ta.sma(length=params['sell_ma_long_period'], append=True)
            # RSI Sell
            if 'RSI_Sell' in sell_combo:
                df.ta.rsi(length=params['sell_rsi_period'], append=True)
        except Exception as e:
            logger.error(f"지표 계산 중 오류 발생: {e}", exc_info=True)
        return {'action': 'HOLD', 'strength': 0}

        # --- 상세 로그: 생성된 컬럼 목록 출력 ---
        logger.debug(f"Available columns after TA: {df.columns.to_list()}")

        latest = df.iloc[-1]
        previous = df.iloc[-2]
        buy_score, sell_score = 0, 0
        weights = params.get('weights', {})

        # --- 2. 신호 점수 계산 (try-except로 감싸서 오류 추적) ---
        try:
            # 매수 신호
            if 'MA_Cross' in buy_combo:
                ma_short_col = f'SMA_{params["buy_ma_short_period"]}'
                ma_long_col = f'SMA_{params["buy_ma_long_period"]}'
                if latest[ma_short_col] > latest[ma_long_col] and previous[ma_short_col] <= previous[ma_long_col]:
                    buy_score += weights.get('MA_Cross_buy', 1)

            if 'RSI' in buy_combo:
                rsi_col = f'RSI_{params["buy_rsi_period"]}'
                if pd.notna(latest[rsi_col]) and latest[rsi_col] < params['buy_rsi_oversold_threshold']:
                    buy_score += weights.get('RSI_buy', 1)

            # 매도 신호
            if 'Dead_Cross' in sell_combo:
                ma_short_col = f'SMA_{params["sell_ma_short_period"]}'
                ma_long_col = f'SMA_{params["sell_ma_long_period"]}'
                if latest[ma_short_col] < latest[ma_long_col] and previous[ma_short_col] >= previous[ma_long_col]:
                    sell_score += weights.get('Dead_Cross_sell', 1)

            if 'RSI_Sell' in sell_combo:
                rsi_col = f'RSI_{params["sell_rsi_period"]}'
                if pd.notna(latest[rsi_col]) and latest[rsi_col] > params['sell_rsi_overbought_threshold']:
                    sell_score += weights.get('RSI_Sell_sell', 1)

        except KeyError as e:
            # --- 상세 로그: KeyError 발생 시점의 상세 정보 출력 ---
            logger.error(f"!!! KeyError while accessing signal data: {e}")
            logger.error(f"Failed to find key: {e.args[0]}")
            logger.error(f"Current available columns: {df.columns.to_list()}")
            logger.error(f"Job that caused error: {job_config}")
            return {'action': 'HOLD', 'strength': 0} # 오류 발생 시 거래 중단

        # --- 최종 결정 ---
        if buy_score >= params.get('buy_trigger_threshold', 99):
            return {'action': 'BUY', 'strength': buy_score}
        if sell_score >= params.get('sell_trigger_threshold', 99):
            return {'action': 'SELL', 'strength': sell_score}

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
        if self.config.BACKTEST_MODE:
            tech_algo_info = self.algorithms.get('technical_analysis')
            if tech_algo_info and coin in tech_algo_info['enabled_coins']:
                algo = tech_algo_info['algorithm']
                if job_config:
                    return {'decision': algo.generate_signal(data, job_config)}

        return {'decision': {'action': 'HOLD', 'strength': 0}}

    def run_trading_cycle(self) -> dict:
        """한 번의 거래 사이클 실행"""
        logger.info(f"🔄 거래 사이클 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # TARGET_ALLOCATION에 설정된 코인 목록을 가져옴 (CASH 제외)
        coins = [coin for coin in self.config.TARGET_ALLOCATION if coin != 'CASH']
        current_prices = self.data_manager.get_coin_prices(coins)
        
        active_signals = []

        # 모든 코인 데이터를 한 번에 가져오도록 수정
        all_coin_data = self.data_manager.generate_multi_coin_data(coins, days=30) # 백테스팅과 유사하게 데이터 기간 확보

        for coin in coins:
            if all_coin_data.empty:
                logger.warning(f"{coin}에 대한 데이터를 가져올 수 없습니다.")
                continue
            # 오류 수정: 닫는 괄호 추가
            coin_data = all_coin_data[all_coin_data['coin'] == coin]
            if not coin_data.empty:
                # 실시간 거래에서는 기본 파라미터로 신호 분석 (job_config 없음)
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

