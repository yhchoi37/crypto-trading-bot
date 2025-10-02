# -*- coding: utf-8 -*-
"""
트레이딩 시스템 설정 파일
"""
import os
from dotenv import load_dotenv
import logging
import json
from .optimization import OptimizationSettings # optimization 설정 import
load_dotenv()
logger = logging.getLogger(__name__)

class TradingConfig:
    """트레이딩 시스템 설정 클래스"""
    def __init__(self):
        # 설정 버전 관리
        self.CONFIG_VERSION = '0.1.0'
        self.MIN_COMPATIBLE_VERSION = '0.1.0'

        # 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
        
        # 데이터 수집 간격 (초)
        self.DATA_COLLECTION_INTERVAL = int(os.getenv('DATA_COLLECTION_INTERVAL', 300))
        
        # 백테스트 실행 여부 (backtest.py에서 설정)
        self.IS_BACKTEST_MODE = os.getenv('IS_BACKTEST_MODE', 'false').lower() == 'true'

        # 모의 거래 모드 (main.py에서 사용)
        self.SIMULATION_MODE = os.getenv('SIMULATION_MODE', 'true').lower() == 'true'

        # 기본 설정
        self.INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 10000000))
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
            'BTC', 'ETH', 'XRP', 'ADA',
            'SOL', 'DOT', 'LINK',
        ]
        # 포트폴리오 목표 배분
        # self.TARGET_ALLOCATION = {
        #     'BTC': 0.30, 'ETH': 0.30, 'XRP': 0.20,
        #     'SOL': 0.19, 'CASH': 0.01
        # }
        self.TARGET_ALLOCATION = {
            'BTC': 0.50, 'ETH': 0.49, 'CASH': 0.01
        }

        # 센티멘트 분석 설정
        self.SENTIMENT_CONFIG = {
            'twitter_max_tweets': 100,
            'reddit_max_posts': 50,
            'sentiment_threshold': 0.3,
            'cache_timeout': 300
        }
        
        # 거래 전략 설정 (손절/익절 등)
        self.TRADING_CONFIG = {
            'buy_threshold': 0.6,
            'sell_threshold': 0.6,
            'min_trade_amount': 5000,
            'max_slippage': 0.002,          # 0.2% 슬리피지
            'maker_fee_percent': 0.0005,   # 메이커 수수료
            'taker_fee_percent': 0.001,    # 테이커 수수료
        }

        # 리스크 관리 설정 (코인별 손절/익절 및 쿨다운)
        self.RISK_MANAGEMENT = {
            'default': {
                'enabled': True,
                'stop_loss_percent': 0.05,   # 5% 손실 시 손절
                'take_profit_percent': 0.10, # 10% 이익 시 익절
                'daily_loss_limit': 0.03,    # 일일 3% 손실 시 거래 중단
                'weekly_loss_limit': 0.10,   # 주간 10% 손실 시 거래 중단
                'consecutive_loss_limit': 3,  # 연속 3회 손실 시 일시 중지
            },
            'BTC': {
                'enabled': True,
                'stop_loss_percent': 0.07,  # BTC는 손절 라인을 7%로 다르게 설정
            },
            'ETH': {
                'enabled': False  # ETH는 손절/익절 및 쿨다운을 적용하지 않음
            }
        }

        # 거래 쿨다운 설정 (기존 RISK_MANAGEMENT의 cooldown_period를 여기로 이동)
        self.COOLDOWN_CONFIG = {
            'default': {
                'enabled': True,
                'period': 4  # 4시간 (기존 cooldown_period는 시간 단위였음)
            },
            'BTC': {
                'enabled': True,
                'period': 2  # 2시간
            },
            'ETH': {
                'enabled': False
            }
        }

        # 시간 단위 설정
        # pyupbit : month, week, day, minute240, minute60,30,15,10,5,3,1
        self.INTERVAL = 'minute60' 

        # 기술적 분석 설정 (실시간/모의 거래 시 기본값)
        self.TECHNICAL_ANALYSIS_CONFIG = {
            'buy_indicators': {
                'MA_Cross': {'ma_short_period': 5, 'ma_long_period': 20},
                'RSI': {'rsi_period': 14, 'rsi_oversold_threshold': 30},
                'BollingerBand': {'bollinger_window': 20, 'bollinger_std_dev': 2}
            },
            'sell_indicators': {
                'MA_Cross': {'ma_short_period': 10, 'ma_long_period': 40},
                'RSI': {'rsi_period': 14, 'rsi_overbought_threshold': 70},
                'BollingerBand': {'bollinger_window': 20, 'bollinger_std_dev': 2}
            },
            'signal_weights': {
                'MA_Cross_buy': 1,
                'RSI_buy': 1,
                'BollingerBand_buy': 2,
                'MA_Cross_sell': 1,
                'RSI_sell': 1,
                'BollingerBand_sell': 2,
            },
            'buy_trigger_threshold': 2,
            'sell_trigger_threshold': 2
        }

        # 백테스트 vs 실거래 설정 분리
        if self.IS_BACKTEST_MODE:
            self.optimization = OptimizationSettings()
            self.BACKTEST_SPECIFIC = {
                'commission_model': 'percentage',
                'slippage_model': 'fixed',
                'use_bid_ask_spread': True
            }
            
        # 최적화된 파라미터 로드 시도
        self._load_optimized_params()

        # 데이터 캐시 설정
        self.DATA_CACHE_DIR = "data_cache"
        self.MARKET_DATA_CACHE_TIMEOUT = 60  # 추가: 60초
        
        self._validate_config()

    def _load_optimized_params(self):
        """'optimized_params.json' 파일이 있으면 로드하여 기술적 분석 설정을 덮어씁니다."""
        filepath = 'optimized_params.json'
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    optimized_params = json.load(f)

                # 이제 구조가 거의 동일하므로 전체를 업데이트
                self.TECHNICAL_ANALYSIS_CONFIG.update(optimized_params)
                logger.info(f"✅ '{filepath}'에서 최적화된 파라미터를 로드하여 적용했습니다.")
            except Exception as e:
                logger.error(f"❌ '{filepath}' 파일 로드 중 오류 발생: {e}", exc_info=True)
        else:
            logger.info("최적화된 파라미터 파일이 없습니다. 기본 설정을 사용합니다.")

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

        # 4. 리스크 관리 설정 검증
        for coin, settings in self.RISK_MANAGEMENT.items():
            if coin == 'default':
                continue
            if coin not in self.SUPPORTED_COINS:
                raise ValueError(f"리스크 관리에 설정된 '{coin}'은(는) 지원하는 코인이 아닙니다.")
            if not isinstance(settings, dict):
                raise ValueError(f"'{coin}'의 리스크 관리 설정이 올바른 형식이 아닙니다 (딕셔너리 필요).")
        logger.info("✅ 설정 파일 유효성 검사 완료")

