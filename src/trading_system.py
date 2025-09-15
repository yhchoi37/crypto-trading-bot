# -*- coding: utf-8 -*-
"""
메인 트레이딩 시스템 모듈
"""
import logging
from datetime import datetime
import pandas as pd
from .data_manager import MultiCoinDataManager
from .portfolio_manager import MultiCoinPortfolioManager
from .social_sentiment import SocialSentimentBasedAlgorithm, TwitterSentimentCollector, RedditSentimentCollector

logger = logging.getLogger(__name__)

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
        social_algo = SocialSentimentBasedAlgorithm(self.twitter_collector, self.reddit_collector)
        self.algorithms['social_sentiment'] = {
            'algorithm': social_algo,
            'weight': 1.0,
            'enabled_coins': self.config.SUPPORTED_COINS[:6]
        }
        logger.info(f"✅ {len(self.algorithms)}개 알고리즘 설정 완료")

    def setup_portfolio_allocation(self, target_allocation: dict):
        """포트폴리오 목표 배분 설정"""
        self.portfolio_manager.set_target_allocation(target_allocation)
        logger.info("🎯 포트폴리오 목표 배분 설정:")
        for asset, weight in target_allocation.items():
            logger.info(f"  - {asset}: {weight:.1%}")

    def analyze_coin_signals(self, coin: str, data: pd.DataFrame) -> dict:
        """특정 코인에 대한 종합 신호 분석"""
        # (신호 분석 로직 구현부)
        # 여기서는 간단한 홀드 신호로 대체
        return {'decision': {'action': 'HOLD'}}

    def run_trading_cycle(self) -> dict:
        """한 번의 거래 사이클 실행"""
        logger.info(f"🔄 거래 사이클 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        coins = self.config.SUPPORTED_COINS[:6]
        current_prices = self.data_manager.get_coin_prices(coins)
        
        active_signals = []
        for coin in coins:
            coin_data = self.data_manager.generate_multi_coin_data([coin], days=7)
            if not coin_data.empty:
                analysis = self.analyze_coin_signals(coin, coin_data[coin_data['coin'] == coin])
                if analysis['decision']['action'] != 'HOLD':
                    active_signals.append({'coin': coin, 'decision': analysis['decision']})

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
