# -*- coding: utf-8 -*-
"""
다중 코인 데이터 관리 모듈
"""
import pyupbit
import ccxt
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from config.settings import TradingConfig

logger = logging.getLogger(__name__)

class MultiCoinDataManager:
    """여러 거래소·코인 데이터 통합 관리 클래스"""

    def __init__(self):
        self.config = TradingConfig()
        self.upbit = None
        self.binance = None

    def get_coin_prices(self, coins=None):
        """주요 코인들의 현재가를 Upbit와 Binance 양쪽에서 받아옴"""
        if coins is None:
            coins = self.config.SUPPORTED_COINS
        prices = {}
        # Upbit
        try:
            for symbol in coins:
                ticker = f"KRW-{symbol}"
                price = pyupbit.get_current_price(ticker)
                prices[symbol] = price
        except Exception as e:
            logger.error(f"[Upbit] 가격 조회 에러: {e}")
        # Binance (참고용, 실제로는 글로벌 시세 비교 가능)
        try:
            if self.binance is None:
                self.binance = ccxt.binance()
            for symbol in coins:
                ticker = f"{symbol}/USDT"
                price = self.binance.fetch_ticker(ticker)['last']
                prices[f"{symbol}_USD"] = price
        except Exception as e:
            logger.warning(f"[Binance] 가격 조회 에러: {e}")
        return prices

    def generate_multi_coin_data(self, coins, days=7):
        """여러 코인에 대한 과거 캔들 데이터 통합 DataFrame 생성"""
        all_data = []
        until = datetime.now()
        since = until - timedelta(days=days)
        for symbol in coins:
            try:
                df = pyupbit.get_ohlcv(f"KRW-{symbol}", interval="day", count=days)
                df = df.reset_index()
                df['coin'] = symbol
                all_data.append(df)
            except Exception as e:
                logger.error(f"[{symbol}] 데이터 수집 실패: {e}")
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_historical_data_for_backtest(self, coins, start_date, end_date):
        """백테스트용 과거 데이터 반환 (코인별로 병합)"""
        all_data = []
        for symbol in coins:
            try:
                df = pyupbit.get_ohlcv(f"KRW-{symbol}", interval="day")
                df = df.loc[start_date:end_date]
                df = df.reset_index()
                df['coin'] = symbol
                all_data.append(df)
            except Exception as e:
                logger.error(f"[{symbol}] 백테스트 데이터 수집 에러: {e}")
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
