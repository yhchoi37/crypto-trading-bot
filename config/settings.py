# -*- coding: utf-8 -*-
"""
트레이딩 시스템 설정 파일
"""
import os
from dotenv import load_dotenv
import logging
import json # json 모듈 추가

load_dotenv()
logger = logging.getLogger(__name__)

class TradingConfig:
    """트레이딩 시스템 설정 클래스"""
    def __init__(self):
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
        self.TARGET_ALLOCATION = {
            'BTC': 0.30, 'ETH': 0.30, 'XRP': 0.20,
            'SOL': 0.19, 'CASH': 0.01
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
            'min_trade_amount': 5000, 'max_slippage': 0.02,
            'transaction_fee_percent': 0.001
        }

        # 리스크 관리 설정 (코인별 손절/익절 및 쿨다운)
        self.RISK_MANAGEMENT = {
            'default': {
                'enabled': True,
                'stop_loss_percent': 0.05,   # 5% 손실 시 손절
                'take_profit_percent': 0.10, # 10% 이익 시 익절
                'cooldown_period': 4   # 거래 후 4시간 동안 추가 거래 방지
            },
            'BTC': {
                'stop_loss_percent': 0.07,  # BTC는 손절 라인을 7%로 다르게 설정
                'cooldown_period': 2  # BTC는 쿨다운을 2시간으로 짧게 설정
            },
            'ETH': {
                'enabled': False  # ETH는 손절/익절 및 쿨다운을 적용하지 않음
            }
        }

        # 백테스팅 시간 단위 설정
        self.BACKTEST_INTERVAL = 'minute60' # 'day', 'minute240', 'minute60' 등 pyupbit에서 지원하는 interval

        # 기술적 분석 설정
        self.TECHNICAL_ANALYSIS_CONFIG = {
            'buy_indicators': {
                'MA_Cross': {
                    'ma_short_period': 5,
                    'ma_long_period': 20
                },
                'RSI': {
                    'rsi_period': 14,
                    'rsi_oversold_threshold': 30
                },
                'BollingerBand': {
                    'bollinger_window': 20,
                    'bollinger_std_dev': 2
                }
            },
            'sell_indicators': {
                'MA_Cross': {
                    'ma_short_period': 10,
                    'ma_long_period': 40
                },
                'RSI': {
                    'rsi_period': 14,
                    'rsi_overbought_threshold': 70
                },
                'BollingerBand': {
                    'bollinger_window': 20,
                    'bollinger_std_dev': 2
                }
            },

            # 가중치 구조를 하나로 통합
            'signal_weights': {
                'MA_Cross_buy': 1,
                'RSI_buy': 1,
                'BollingerBand_buy': 2,
                'MA_Cross_sell': 1,
                'RSI_sell': 1,
                'BollingerBand_sell': 2,
            },

            'buy_trigger_threshold': 2,
            'sell_trigger_threshold': 2 # 기본값을 2로 수정
        }

        # 최적화된 파라미터 로드 시도
        self._load_optimized_params()

        # 백테스팅 파라미터 최적화 설정 (Grid Search)
        self.OPTIMIZATION_CONFIG = {
            # 1. 매수/매도에 사용할 기술 지표와 파라미터 범위 정의
            # 'buy_indicators': {
            #     'MA_Cross': {
            #         'ma_short_period': {'min': 5, 'max': 15, 'step': 5},
            #         'ma_long_period': {'min': 20, 'max': 40, 'step': 10}
            #     },
            #     'RSI': {
            #         'rsi_period': {'min': 14, 'max': 28, 'step': 7},
            #         'rsi_oversold_threshold': {'min': 25, 'max': 35, 'step': 5}
            #     },
            #     'BollingerBand': {
            #         'bollinger_window': {'min': 20, 'max': 20, 'step': 5},
            #         'bollinger_std_dev': {'min': 2, 'max': 3, 'step': 1}
            #     }
            # },
            # 'sell_indicators': {
            #     'MA_Cross': {
            #         'ma_short_period': {'min': 5, 'max': 15, 'step': 5},
            #         'ma_long_period': {'min': 20, 'max': 40, 'step': 10}
            #     },
            #     'RSI': {
            #         'rsi_period': {'min': 14, 'max': 28, 'step': 7},
            #         'rsi_overbought_threshold': {'min': 65, 'max': 75, 'step': 5}
            #     },
            #     'BollingerBand': {
            #         'bollinger_window': {'min': 20, 'max': 20, 'step': 5},
            #         'bollinger_std_dev': {'min': 2, 'max': 3, 'step': 1}
            #     }
            # },
            'buy_indicators': {
                'MA_Cross': {
                    'ma_short_period': {'min': 10, 'max': 30, 'step': 5},
                    'ma_long_period': {'min': 60, 'max': 90, 'step': 10}
                },
                'BollingerBand': {
                    'bollinger_window': {'min': 20, 'max': 20, 'step': 5},
                    'bollinger_std_dev': {'min': 2, 'max': 3, 'step': 1}
                }
            },
            'sell_indicators': {
                'MA_Cross': {
                    'ma_short_period': {'min': 10, 'max': 30, 'step': 5},
                    'ma_long_period': {'min': 30, 'max': 40, 'step': 5}
                },
                'BollingerBand': {
                    'bollinger_window': {'min': 20, 'max': 20, 'step': 5},
                    'bollinger_std_dev': {'min': 2, 'max': 3, 'step': 1}
                }
            },

            # 2. 매수/매도 신호 발생을 위한 가중치 합계 임계값 범위
            'buy_trigger_threshold': {'min': 1, 'max': 3, 'step': 1},
            'sell_trigger_threshold': {'min': 1, 'max': 3, 'step': 1},

            # 3. 각 신호별 가중치 (고정값, 이름 통일)
            'signal_weights': {
                'MA_Cross_buy': 1,
                'RSI_buy': 1,
                'BollingerBand_buy': 2,
                'MA_Cross_sell': 1,
                'RSI_sell': 1,
                'BollingerBand_sell': 2,
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

        # 데이터 캐시 설정
        self.DATA_CACHE_DIR = "data_cache"

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

