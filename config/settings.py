# -*- coding: utf-8 -*-
"""
트레이딩 시스템 설정 파일
"""
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class TradingConfig:
    """트레이딩 시스템 설정 클래스"""
    def __init__(self):
        # 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
        
        # 데이터 수집 간격 (초)
        self.DATA_COLLECTION_INTERVAL = int(os.getenv('DATA_COLLECTION_INTERVAL', 300))
        
        # 백테스트 모드 (True/False)
        self.BACKTEST_MODE = os.getenv('BACKTEST_MODE', 'false').lower() == 'true'

        # 기본 설정
        self.INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 100000))
        self.MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 0.15))
        self.REBALANCING_THRESHOLD = float(os.getenv('REBALANCING_THRESHOLD', 0.05))

        # API 키 설정
        self.TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
        self.REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
        self.REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
        self.UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY')
        self.UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY')

        # 알림 설정
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
        self.EMAIL_SETTINGS = {
            'smtp_server': os.getenv('EMAIL_SMTP_SERVER'),
            'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', 587)),
            'email': os.getenv('EMAIL_ADDRESS'),
            'password': os.getenv('EMAIL_PASSWORD'),
            'recipient': os.getenv('EMAIL_RECIPIENT')
        }

        # 지원 코인 설정
        self.SUPPORTED_COINS = [
            'BTC', 'ETH', 'XRP', 'ADA', 'DOGE',
            'SOL', 'DOT', 'LINK', 'LTC', 'MATIC'
        # 포트폴리오 목표 배분
        self.TARGET_ALLOCATION = {
            'BTC': 0.25, 'ETH': 0.20, 'XRP': 0.10, 'ADA': 0.05,
            'DOGE': 0.05, 'SOL': 0.05, 'CASH': 0.30
        }

        # 센티멘트 분석 설정
        self.SENTIMENT_CONFIG = {
            'twitter_max_tweets': 100,
            'reddit_max_posts': 50,
            'sentiment_threshold': 0.3,
            'cache_timeout': 300
        }
        
        # 거래 전략 설정
        self.TRADING_CONFIG = {
            'buy_threshold': 0.6, 'sell_threshold': 0.6,
            'min_trade_amount': 10000, 'max_slippage': 0.02,
            'stop_loss_percent': 0.05,   # 5% 손실 시 손절
            'take_profit_percent': 0.10, # 10% 이익 시 익절
            'transaction_fee_percent': 0.001
        }

        # 기술적 분석 설정
        self.TECHNICAL_ANALYSIS_CONFIG = {
            'ma_short_period': 5,
            'ma_long_period': 20,
            'rsi_period': 14,
            'bollinger_window': 20,
            'bollinger_std_dev': 2,

            'buy_signal_weights': {
                'golden_cross': 1,
                'rsi_oversold': 1,
                'bb_lower': 2,
            },
            'sell_signal_weights': {
                'dead_cross': 1,
                'rsi_overbought_weak': 1,  # RSI 70 이상
                'rsi_overbought_strong': 2, # RSI 80 이상
                'bb_upper': 2,
            },

            'buy_trigger_threshold': 2,
            'sell_trigger_threshold': 3
        }

        # 백테스팅 파라미터 최적화 설정 (Grid Search)
        self.OPTIMIZATION_CONFIG = {
            # 1. 매수 전략 최적화 설정
            'buy_indicators': {
                'MA_Cross': {
                    'buy_ma_short_period': {'min': 5, 'max': 15, 'step': 5},
                    'buy_ma_long_period': {'min': 20, 'max': 40, 'step': 10}
                },
                'RSI': {
                    'buy_rsi_period': {'min': 14, 'max': 21, 'step': 7},
                    'buy_rsi_oversold_threshold': {'min': 25, 'max': 35, 'step': 5}
                }
            },
            'buy_trigger_threshold': {'min': 1, 'max': 2, 'step': 1},
            'buy_signal_weights': {
                'MA_Cross_buy': 1,
                'RSI_buy': 1
            },

            # 2. 매도 전략 최적화 설정
            'sell_indicators': {
                'Dead_Cross': {
                    'sell_ma_short_period': {'min': 5, 'max': 15, 'step': 5},
                    'sell_ma_long_period': {'min': 20, 'max': 40, 'step': 10}
                },
                'RSI_Sell': {
                    'sell_rsi_period': {'min': 14, 'max': 21, 'step': 7},
                    'sell_rsi_overbought_threshold': {'min': 65, 'max': 75, 'step': 5}
                }
            },
            'sell_trigger_threshold': {'min': 1, 'max': 2, 'step': 1},
            'sell_signal_weights': {
                'Dead_Cross_sell': 1,
                'RSI_Sell_sell': 2
            }
        }

        # 전진 분석 (Walk-Forward Optimization) 설정
        self.WALK_FORWARD_CONFIG = {
            'enabled': True,  # 전진 분석 활성화 여부
            'training_period_months': 12, # 훈련 기간 (과거 12개월 데이터로 최적화)
            'testing_period_months': 3    # 검증 기간 (이후 3개월 데이터로 성과 검증)
        }

        # 성능 최적화 설정
        self.PERFORMANCE_CONFIG = {
            'parallel_cores': -1,  # 사용할 CPU 코어 수. -1이면 가능한 모든 코어 사용
        }

        self._validate_config()

    def _validate_config(self):
        """설정 값의 유효성을 검사합니다."""
        # 1. 포트폴리오 목표 배분 합계 검증
        total_allocation = sum(self.TARGET_ALLOCATION.values())
        if not (0.999 < total_allocation < 1.001): # 부동소수점 오차 감안
            raise ValueError(f"포트폴리오 목표 배분의 총합이 1이 아닙니다: {total_allocation}")

        # 2. 목표 배분 코인이 지원 코인 목록에 있는지 확인
        for coin in self.TARGET_ALLOCATION:
            if coin != 'CASH' and coin not in self.SUPPORTED_COINS:
                raise ValueError(f"목표 배분에 포함된 '{coin}'은(는) 지원하는 코인이 아닙니다.")

        # 3. 지원하지만 목표 배분에 없는 코인에 대해 경고
        unallocated_coins = [
            coin for coin in self.SUPPORTED_COINS
            if coin not in self.TARGET_ALLOCATION
        ]

        if unallocated_coins:
            logger.warning(
                f"다음 코인은 지원되지만 목표 배분이 설정되지 않았습니다: {', '.join(unallocated_coins)}"
            )

        logger.info("✅ 설정 파일 유효성 검사 완료")

