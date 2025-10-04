# -*- coding: utf-8 -*-
"""
백테스트 시각화 기능 테스트 스크립트
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
from datetime import datetime, timedelta

# backtest.py와 동일한 환경을 설정
os.environ['IS_BACKTEST_MODE'] = 'true'
matplotlib.use('Agg') # GUI가 없는 환경에서도 실행 가능하도록 설정

# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from backtest import WalkForwardOptimizer
from src.logging_config import setup_logging

def create_sample_portfolio_history() -> pd.DataFrame:
    """
    plot_results 함수를 테스트하기 위한 가짜 포트폴리오 이력 데이터를 생성합니다.
    """
    print("📊 가짜 포트폴리오 이력 데이터 생성 중...")
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(180)]
    n = len(dates)
    
    # 코인 가격 시뮬레이션
    btc_price = 30000 + np.cumsum(np.random.randn(n) * 300)
    eth_price = 2000 + np.cumsum(np.random.randn(n) * 50)
    
    # 포트폴리오 가치 및 현금 시뮬레이션
    portfolio_value = 1_000_000 + np.cumsum(np.random.randn(n) * 10000)
    cash = portfolio_value * (0.5 - 0.2 * np.sin(np.linspace(0, 2 * np.pi, n))) # 현금 비중 변동
    
    # 코인 보유 수량 시뮬레이션 (매수/매도 흉내)
    btc_quantity = np.zeros(n)
    btc_quantity[30:90] = 5 # 30일차에 매수
    btc_quantity[90:] = 2.5 # 90일차에 절반 매도
    
    eth_quantity = np.zeros(n)
    eth_quantity[60:150] = 20 # 60일차에 매수
    eth_quantity[150:] = 0 # 150일차에 전량 매도
    
    data = {
        'date': dates,
        'portfolio_value': portfolio_value,
        'cash': cash,
        'BTC_price': btc_price,
        'BTC_quantity': btc_quantity,
        'ETH_price': eth_price,
        'ETH_quantity': eth_quantity,
        'XRP_price': np.nan, # 데이터 없는 코인 테스트
        'XRP_quantity': 0,
    }
    
    df = pd.DataFrame(data)
    print("✅ 데이터 생성 완료")
    return df

def main():
    """
    메인 테스트 함수
    """
    setup_logging('DEBUG', 'test.log')
    
    # 1. WalkForwardOptimizer 인스턴스 생성 (초기화 값은 중요하지 않음)
    print("⚙️ WalkForwardOptimizer 초기화...")
    wfo = WalkForwardOptimizer(
        start_date_str="2023-01-01",
        end_date_str="2023-06-30",
        initial_balance=1_000_000
    )

    # 2. 샘플 데이터 생성
    history_df = create_sample_portfolio_history()

    # 3. plot_results 함수 테스트
    print("📈 시각화 함수(plot_results) 테스트 실행...")
    wfo.plot_results(history_df)
    
    print("\n🎉 테스트 완료! 'walk_forward_performance.png' 파일을 확인하세요.")

if __name__ == "__main__":
    main()
