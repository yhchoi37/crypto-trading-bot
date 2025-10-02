# -*- coding: utf-8 -*-
"""
백테스팅 및 파라미터 최적화 관련 설정
"""

class OptimizationSettings:
    """백테스팅, 최적화, 전진 분석 관련 설정을 담는 클래스"""
    def __init__(self):
        # 백테스팅 파라미터 최적화 설정 (Grid Search)
        self.OPTIMIZATION_CONFIG = {
            # 1. 매수/매도에 사용할 기술 지표와 파라미터 범위 정의
            'buy_indicators': {
                'MA_Cross': {
                    'ma_short_period': {'min': 12, 'max': 24, 'step': 12},
                    'ma_long_period': {'min': 48, 'max': 96, 'step': 24}
                },
                # 'RSI': {
                #     'rsi_period': {'min': 14, 'max': 28, 'step': 7},
                #     'rsi_oversold_threshold': {'min': 25, 'max': 35, 'step': 5}
                # },
                # 'BollingerBand': {
                #     'bollinger_window': {'min': 20, 'max': 20, 'step': 5},
                #     'bollinger_std_dev': {'min': 2, 'max': 3, 'step': 1}
                # }
            },
            'sell_indicators': {
                'MA_Cross': {
                    'ma_short_period': {'min': 12, 'max': 24, 'step': 12},
                    'ma_long_period': {'min': 48, 'max': 96, 'step': 24}
                },
                # 'RSI': {
                #     'rsi_period': {'min': 14, 'max': 28, 'step': 7},
                #     'rsi_overbought_threshold': {'min': 65, 'max': 75, 'step': 5}
                # },
                # 'BollingerBand': {
                #     'bollinger_window': {'min': 20, 'max': 20, 'step': 5},
                #     'bollinger_std_dev': {'min': 2, 'max': 3, 'step': 1}
                # }
            },

            # 2. 매수/매도 신호 발생을 위한 가중치 합계 임계값 범위
            'buy_trigger_threshold': {'min': 1, 'max': 3, 'step': 1},
            'sell_trigger_threshold': {'min': 1, 'max': 3, 'step': 1},

            # 3. 각 신호별 가중치 (고정값, 이름 통일)
            'signal_weights': {
                'MA_Cross_buy': 1, 'RSI_buy': 1, 'BollingerBand_buy': 2,
                'MA_Cross_sell': 1, 'RSI_sell': 1, 'BollingerBand_sell': 2,
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
