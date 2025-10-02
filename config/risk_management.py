# -*- coding: utf-8 -*-
"""
리스크 관리 관련 설정
"""

class RiskManagementSettings:
        # 동적 포지션 사이징
        self.POSITION_SIZING = {
            'method': 'fixed',  # 'fixed', 'kelly', 'atr_based'
            'fixed_percent': 0.15,
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