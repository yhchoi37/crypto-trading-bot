# -*- coding: utf-8 -*-
"""
리스크 관리 관련 설정
"""

class RiskManagementSettings:
    """리스크 관리 설정을 보관하는 클래스.

    사용 예:
        rm = RiskManagementSettings()
        rm.RISK_MANAGEMENT
    """
    def __init__(self):
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
                'enabled': False,  # ETH는 손절/익절 및 쿨다운을 적용하지 않음
            }
        }

        # 트레일링 스톱 설정 (코인별)
        self.TRAILING_STOP_CONFIG = {
            'default': {
                'enabled': False,
                'percent': 0.05
            },
            'BTC': {
                'enabled': False,
                'percent': 0.07
            },
            'ETH': {
                'enabled': False,
                'percent': 0.05
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
                'period': 12
            },
            'ETH': {
                'enabled': False
            }
        }

        # 동적 포지션 사이징
        self.POSITION_SIZING = {
            'method': 'fixed',  # 'fixed', 'kelly', 'atr_based'
            'fixed_percent': 0.15,
            'buy_percent': 0.15,   # 매수 시 비율 (지정 시 fixed_percent 대신 사용)
            'sell_percent': 0.10,  # 매도 시 비율 (지정 시 fixed_percent 대신 사용)
            'kelly_fraction': 0.5,  # Kelly의 절반만 사용 (보수적)
            'atr_multiplier': 2.0,
            'max_position_percent': 0.25,  # 절대 최대값
        }

        # 거래 시간 제한: 암호화폐는 24/7 거래되지만, 특정 시간대(예: 미국 장 마감 후)에만 거래하는 옵션
        self.TRADING_HOUR = {
            'enabled': False,
            'timezone': 'Asia/Seoul',
            'allowed_hours': [(9, 18)],  # 9시-18시만 거래
            'excluded_days': ['Saturday', 'Sunday']
        }

        # 최대 드로다운 알림: 특정 임계값 초과 시 자동으로 거래를 중단하거나 알림을 보내는 기능
        self.DRAWDOWN_CONFIG = {
            'max_drawdown_percent': 0.20,  # 20% 낙폭 시 거래 중단
            'warning_threshold': 0.15,     # 15% 낙폭 시 경고 알림
            'recovery_mode': True,         # 회복 모드에서는 거래량 50% 감소
        }

        # 코인별 변동성 임계값: 변동성이 너무 높을 때 거래를 중단
        self.VOLATILITY_FILTER = {
            'default': {
                'enabled': True,
                'atr_threshold': 5.0,  # ATR이 5% 초과 시 거래 제한
                'lookback_period': 24  # 24시간 기준
            },
            'BTC': {
                'enabled': False,
            },
        }