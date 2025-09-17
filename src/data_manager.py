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
        self._cache = {}
        self._last_updated = {}

    def _is_cache_valid(self, key):
        """캐시가 유효한지 확인"""
        last_time = self._last_updated.get(key)
        if not last_time:
            return False
        timeout = self.config.MARKET_DATA_CACHE_TIMEOUT
        return (time.time() - last_time) < timeout

    def get_coin_prices(self, coins=None):
        """주요 코인들의 현재가를 Upbit에서 받아옴 (캐싱 적용)"""
        cache_key = 'current_prices'
        if self._is_cache_valid(cache_key):
            logger.debug("가격 정보 캐시 사용")
            return self._cache[cache_key]

        if coins is None:
            coins = self.config.SUPPORTED_COINS
        prices = {}
        try:
            tickers = [f"KRW-{symbol}" for symbol in coins]
            all_ticker_data = pyupbit.get_current_price(tickers)

            if isinstance(all_ticker_data, dict):
                for ticker, price in all_ticker_data.items():
                    symbol = ticker.split('-')[1]
                prices[symbol] = price
            else: # 단일 코인 조회 시 float 반환 대응
                if len(coins) == 1 and isinstance(all_ticker_data, (int, float)):
                    prices[coins[0]] = all_ticker_data

        except Exception as e:
            logger.error(f"[Upbit] 전체 가격 조회 에러: {e}. 개별 조회 시도.")
            for symbol in coins:
            try:
                    ticker = f"KRW-{symbol}"
                    price = pyupbit.get_current_price(ticker)
                    if price:
                        prices[symbol] = price
                    time.sleep(0.1) # Rate limit 방지
                except Exception as e_ind:
                    logger.error(f"[Upbit] {symbol} 가격 조회 에러: {e_ind}")
        if prices:
            self._cache[cache_key] = prices
            self._last_updated[cache_key] = time.time()
            logger.info(f"{len(prices)}개 코인 가격 정보 업데이트 완료")

        return prices

    def generate_multi_coin_data(self, coins, days=7):
        """여러 코인에 대한 과거 캔들 데이터 통합 DataFrame 생성 (캐싱 적용)"""
        cache_key = f"ohlcv_{'_'.join(sorted(coins))}_{days}d"
        if self._is_cache_valid(cache_key):
            logger.debug(f"{cache_key} 캐시 사용")
            return self._cache[cache_key]
        all_data = []
        for symbol in coins:
            try:
                # pyupbit은 최대 200일치 데이터를 가져오므로, days가 200을 넘지 않도록 조정
                df = pyupbit.get_ohlcv(f"KRW-{symbol}", interval="day", count=min(days, 200))
                if df is not None:
                    df = df.reset_index()
                df['coin'] = symbol
                all_data.append(df)
                time.sleep(0.1) # Rate Limit 방지
            except Exception as e:
                logger.error(f"[{symbol} 데이터 수집 실패: {e}")
        if not all_data:
        return pd.DataFrame()

        result_df = pd.concat(all_data, ignore_index=True)
        self._cache[cache_key] = result_df
        self._last_updated[cache_key] = time.time()
        logger.info(f"{len(coins)}개 코인 과거 데이터 업데이트 완료")
        return result_df

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
```

