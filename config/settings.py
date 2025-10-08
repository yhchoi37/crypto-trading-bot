# -*- coding: utf-8 -*-
"""
트레이딩 시스템 설정 파일
"""
import os
import sys
try:
    from dotenv import load_dotenv
except Exception:
    # allow running without python-dotenv installed (tests / CI without env file)
    def load_dotenv(*args, **kwargs):
        return None
import logging
import json
from .optimization import OptimizationSettings
from .risk_management import RiskManagementSettings
load_dotenv()
logger = logging.getLogger(__name__)

class TradingConfig:
    """트레이딩 시스템 설정 클래스"""
    def __init__(self, force_mode: str = None):
        """
        Args:
            force_mode: 강제 모드 설정 ('backtest', 'simulation', 'live', None)
                       None이면 자동 감지
        """
        # 설정 버전 관리
        self.CONFIG_VERSION = '0.3.0'
        self.MIN_COMPATIBLE_VERSION = '0.2.0'
        
        # ========== 실행 모드 자동 감지 ==========
        self._detect_execution_mode(force_mode)

        # 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
        
        # 데이터 수집 간격 (초)
        self.DATA_COLLECTION_INTERVAL = int(os.getenv('DATA_COLLECTION_INTERVAL', 300))

        # 모의 거래 모드 (main.py에서 사용)
        self.SIMULATION_MODE = os.getenv('SIMULATION_MODE', 'true').lower() == 'true'

        # 기본 설정
        if self.IS_BACKTEST_MODE:
            # 백테스트는 초기 자본을 직접 설정
            self.INITIAL_BALANCE = 10_000_000  # 1천만원 (기본값)
            logger.info(f"💰 백테스트 초기 자본: ₩{self.INITIAL_BALANCE:,}")
        else:
            # 실거래/시뮬레이션은 API로 잔고 조회 (INITIAL_BALANCE 불필요)
            self.INITIAL_BALANCE = None
            logger.info("💰 실거래 모드: API로 실제 잔고 조회 예정")
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
            'BTC': 0.40, 'ETH': 0.50, 'CASH': 0.10
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

        # 리스크 관리 설정은 외부 모듈(config/risk_management.py)에 정의되어 있음.
        try:
            rm = RiskManagementSettings()
            # 주요 설정을 TradingConfig에 노출
            self.RISK_MANAGEMENT = rm.RISK_MANAGEMENT
            self.COOLDOWN_CONFIG = rm.COOLDOWN_CONFIG
            self.POSITION_SIZING = rm.POSITION_SIZING
            self.TRADING_HOUR = rm.TRADING_HOUR
            self.DRAWDOWN_CONFIG = rm.DRAWDOWN_CONFIG
            self.VOLATILITY_FILTER = rm.VOLATILITY_FILTER
            logger.info("✅ config/risk_management.py에서 리스크 설정을 로드했습니다.")
        except Exception as e:
            # 실패 시 기존 기본값을 사용하도록 하여 backward compatibility 유지
            logger.warning(f"리스크 설정 로드 실패: {e}. 기본값을 사용합니다.")
            self.RISK_MANAGEMENT = {
                'default': {
                    'enabled': True,
                    'stop_loss_percent': 0.05,
                    'take_profit_percent': 0.10
                }
            }
            self.COOLDOWN_CONFIG = {'default': {'enabled': True, 'period': 4}}

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
        self._log_execution_mode()

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

    def _detect_execution_mode(self, force_mode: str = None):
        """실행 모드 자동 감지 및 설정"""
        if force_mode:
            # 명시적으로 지정된 경우
            mode = force_mode.lower()
            if mode == 'backtest':
                self.IS_BACKTEST_MODE = True
                self.SIMULATION_MODE = True  # 백테스트는 항상 시뮬레이션
            elif mode == 'simulation':
                self.IS_BACKTEST_MODE = False
                self.SIMULATION_MODE = True
            elif mode == 'live':
                self.IS_BACKTEST_MODE = False
                self.SIMULATION_MODE = False
            else:
                raise ValueError(f"알 수 없는 모드: {force_mode}")
        else:
            # 자동 감지
            script_name = os.path.basename(sys.argv[0])
            
            if script_name.startswith('backtest'):
                # backtest.py 실행 → 백테스트 모드
                self.IS_BACKTEST_MODE = True
                self.SIMULATION_MODE = True
                logger.info("🔍 백테스트 모드 자동 감지")
            else:
                # main.py 또는 기타 → 실시간 모드
                self.IS_BACKTEST_MODE = False
                # .env에서 SIMULATION_MODE 읽기
                self.SIMULATION_MODE = os.getenv('SIMULATION_MODE', 'true').lower() == 'true'
                logger.info("🔍 실시간 모드 자동 감지")
    
    def _log_execution_mode(self):
        """현재 실행 모드 로깅"""
        mode_str = self._get_mode_description()
        logger.info(f"{'='*60}")
        logger.info(f"🎯 실행 모드: {mode_str}")
        logger.info(f"{'='*60}")
    
    def _get_mode_description(self) -> str:
        """현재 모드 설명 반환"""
        if self.IS_BACKTEST_MODE:
            return "백테스트 모드 (과거 데이터 분석)"
        elif self.SIMULATION_MODE:
            return "시뮬레이션 모드 (실시간 데이터, 모의 거래)"
        else:
            return "⚠️  실거래 모드 (실제 주문 발생) ⚠️"
    
    def is_paper_trading(self) -> bool:
        """모의 거래 여부 (백테스트 또는 시뮬레이션)"""
        return self.IS_BACKTEST_MODE or self.SIMULATION_MODE

    def _validate_config(self):
        """설정 값의 유효성을 검사합니다."""
        # 실거래 모드인데 API 키가 없으면 오류
        if not self.is_paper_trading():
            if not self.UPBIT_ACCESS_KEY or not self.UPBIT_SECRET_KEY:
                raise ValueError(
                    "⚠️ 실거래 모드인데 Upbit API 키가 설정되지 않았습니다! "
                    ".env 파일을 확인하거나 SIMULATION_MODE=true로 변경하세요."
                )

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
        # 4. 리스크 관리 설정 검증
        # - 각 코인 키는 SUPPORTED_COINS에 포함되어야 함
        # - 각 설정은 dict여야 하며, stop_loss_percent, take_profit_percent 등은 0~1 범위
        for coin, settings in self.RISK_MANAGEMENT.items():
            if coin == 'default':
                if not isinstance(settings, dict):
                    raise ValueError("'default' 리스크 관리 설정은 딕셔너리여야 합니다.")
                continue
            if coin not in self.SUPPORTED_COINS:
                raise ValueError(f"리스크 관리에 설정된 '{coin}'은(는) 지원하는 코인이 아닙니다.")
            if not isinstance(settings, dict):
                raise ValueError(f"'{coin}'의 리스크 관리 설정이 올바른 형식이 아닙니다 (딕셔너리 필요).")
            # optional keys check
            sl = settings.get('stop_loss_percent')
            tp = settings.get('take_profit_percent')
            if sl is not None and not (0 <= float(sl) <= 1):
                raise ValueError(f"{coin} stop_loss_percent 값은 0~1 범위여야 합니다: {sl}")
            if tp is not None and not (0 <= float(tp) <= 1):
                raise ValueError(f"{coin} take_profit_percent 값은 0~1 범위여야 합니다: {tp}")

        # 4b. COOLDOWN_CONFIG 검증
        if not isinstance(self.COOLDOWN_CONFIG, dict):
            raise ValueError('COOLDOWN_CONFIG는 딕셔너리여야 합니다.')
        for coin, cfg in self.COOLDOWN_CONFIG.items():
            if not isinstance(cfg, dict):
                raise ValueError(f'COOLDOWN_CONFIG[{coin}]는 딕셔너리여야 합니다.')
            period = cfg.get('period')
            if cfg.get('enabled', False) and (period is None or int(period) < 0):
                raise ValueError(f'COOLDOWN_CONFIG[{coin}].period는 0 이상의 정수여야 합니다.')

        # 4c. POSITION_SIZING 검증
        if not hasattr(self, 'POSITION_SIZING'):
            logger.warning('POSITION_SIZING 설정이 없습니다. 기본값을 사용합니다.')
            self.POSITION_SIZING = {'method': 'fixed', 'fixed_percent': 0.15}
        else:
            ps = self.POSITION_SIZING
            if not isinstance(ps, dict):
                raise ValueError('POSITION_SIZING는 딕셔너리여야 합니다.')
            method = ps.get('method', 'fixed')
            if method not in ('fixed', 'kelly', 'atr_based'):
                raise ValueError("POSITION_SIZING.method는 'fixed'|'kelly'|'atr_based' 중 하나여야 합니다.")
            fixed_pct = ps.get('fixed_percent', 0.0)
            if method == 'fixed' and not (0 <= float(fixed_pct) <= 1):
                raise ValueError('POSITION_SIZING.fixed_percent는 0~1 범위여야 합니다.')

        # 4d. DRAWDOWN_CONFIG 검증
        if not hasattr(self, 'DRAWDOWN_CONFIG'):
            logger.warning('DRAWDOWN_CONFIG 설정이 없습니다. 기본값을 사용합니다.')
            self.DRAWDOWN_CONFIG = {'max_drawdown_percent': 0.2, 'warning_threshold': 0.15}
        else:
            dd = self.DRAWDOWN_CONFIG
            if not isinstance(dd, dict):
                raise ValueError('DRAWDOWN_CONFIG는 딕셔너리여야 합니다.')
            md = dd.get('max_drawdown_percent')
            wt = dd.get('warning_threshold')
            if md is not None and not (0 <= float(md) <= 1):
                raise ValueError('DRAWDOWN_CONFIG.max_drawdown_percent는 0~1 범위여야 합니다.')
            if wt is not None and not (0 <= float(wt) <= 1):
                raise ValueError('DRAWDOWN_CONFIG.warning_threshold는 0~1 범위여야 합니다.')

        # 4e. TRADING_HOUR 검증
        if not hasattr(self, 'TRADING_HOUR'):
            logger.warning('TRADING_HOUR 설정이 없습니다. 기본값을 사용합니다.')
            self.TRADING_HOUR = {'enabled': False}
        else:
            th = self.TRADING_HOUR
            if not isinstance(th, dict):
                raise ValueError('TRADING_HOUR는 딕셔너리여야 합니다.')
            if th.get('enabled', False):
                ah = th.get('allowed_hours')
                if not isinstance(ah, (list, tuple)):
                    raise ValueError('TRADING_HOUR.allowed_hours는 리스트/튜플의 (start,end) 쌍이어야 합니다.')

        # 4f. VOLATILITY_FILTER 검증
        if not hasattr(self, 'VOLATILITY_FILTER'):
            logger.warning('VOLATILITY_FILTER 설정이 없습니다. 기본값을 사용합니다.')
            self.VOLATILITY_FILTER = {'default': {'enabled': False}}
        else:
            vf = self.VOLATILITY_FILTER
            if not isinstance(vf, dict):
                raise ValueError('VOLATILITY_FILTER는 딕셔너리여야 합니다.')
            for coin, vfcfg in vf.items():
                if not isinstance(vfcfg, dict):
                    raise ValueError(f'VOLATILITY_FILTER[{coin}]는 딕셔너리여야 합니다.')
                if vfcfg.get('enabled', False):
                    atr = vfcfg.get('atr_threshold')
                    if atr is None or float(atr) < 0:
                        raise ValueError(f'VOLATILITY_FILTER[{coin}].atr_threshold는 0 이상의 숫자여야 합니다.')
        logger.info("✅ 설정 파일 유효성 검사 완료")

