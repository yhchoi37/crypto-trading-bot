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
        """
        주어진 매수/매도 지표 조합과 파라미터로 신호를 생성합니다.
        """
        params = job_config['params']
        buy_combo = job_config['buy_indicator_combo']
        sell_combo = job_config['sell_indicator_combo']
        # 데이터가 충분한지 기본 검사 (가장 긴 기간을 기준으로 동적 계산)
        required_periods = [p for p_name, p in params.items() if 'period' in p_name or 'window' in p_name]
        if not required_periods or historical_data.empty or len(historical_data) < max(required_periods):
            return {'action': 'HOLD', 'strength': 0}

        df = historical_data.copy()
        buy_score = 0
        sell_score = 0
        weights = params['weights']

        # --- 모든 필요한 지표를 한 번에 계산 ---
        # MA (short, long), RSI는 양쪽에서 모두 사용할 수 있으므로 미리 계산
        if any('MA' in ind or 'Dead' in ind for ind in buy_combo + sell_combo):
            df.ta.sma(length=params['ma_short_period'], append=True)
            df.ta.sma(length=params['ma_long_period'], append=True)
        if any('RSI' in ind for ind in buy_combo + sell_combo):
            df.ta.rsi(length=params['rsi_period'], append=True)

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        # --- 매수 신호 점수 계산 ---
        if 'MA_Cross' in buy_combo:



            # 오류 수정: f-string 따옴표 및 if 조건식
            ma_short_col = f"SMA_{params['ma_short_period']}"
            ma_long_col = f"SMA_{params['ma_long_period'}"
            if latest[ma_short_col] > latest[ma_long_col] and previous[ma_short_col] <= previous[ma_long_col:
                buy_score += weights['MA_Cross_buy']
        if 'RSI' in buy_combo:

            rsi_col = f"RSI_{params['rsi_period']}"
            if latest[rsi_col] < params['rsi_oversold_threshold']:
                buy_score += weights['RSI_buy']

        # --- 매도 신호 점수 계산 ---
        if 'Dead_Cross' in sell_combo:




            # 오류 수정: f-string 따옴표 및 if 조건식
            ma_short_col = f"SMA_{params['ma_short_period']}"
            ma_long_col = f"SMA_{params['ma_long_period']}"
            if latest[ma_short_col] < latest[ma_long_col] and previous[ma_short_col] >= previous[ma_long_col]:
                sell_score += weights['Dead_Cross_sell'
        if 'RSI_Sell' in sell_combo:

            rsi_col = f"RSI_{params['rsi_period']}"
            if latest[rsi_col] > params['rsi_overbought_threshold']:
                sell_score += weights['RSI_Sell_sell']

        # --- 최종 결정 (매도 신호 우선) ---
        if sell_score >= params['sell_trigger_threshold']:
            return {'action': 'SELL', 'strength': sell_score}
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

    def analyze_coin_signals(self, coin: str, data: pd.DataFrame, job_config: dict = None) -> dict:
        """특정 코인에 대한 종합 신호 분석 (백테스트 시 job_config 주입)"""
        if self.config.BACKTEST_MODE and job_config:
            tech_algo_info = self.algorithms.get('technical_analysis')
            if tech_algo_info and coin in tech_algo_info['enabled_coins']:
                algo = tech_algo_info['algorithm']
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

