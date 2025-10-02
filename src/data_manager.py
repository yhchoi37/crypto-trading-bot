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
import os

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

    def generate_multi_coin_data(self, coins, min_period=7):
        """여러 코인에 대한 과거 캔들 데이터 통합 DataFrame 생성 (캐싱 적용)"""
        interval = self.config.INTERVAL
        interval_
        cache_key = f"ohlcv_{'_'.join(sorted(coins))}_{min_period}d"
        if self._is_cache_valid(cache_key):
            logger.debug(f"{cache_key} 캐시 사용")
            return self._cache[cache_key]
        all_data = []
        for symbol in coins:
            try:
                # pyupbit은 최대 200개 데이터를 개져오므로, count가 200을 넘지 않도록 조정
                df = pyupbit.get_ohlcv(
                    f"KRW-{symbol}", interval=interval, count=min(min_period, 200)
                )
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
        """
        백테스트용 과거 데이터 반환 (Parquet 캐싱 적용)
        설정된 interval(시간봉/일봉 등)에 따라 데이터를 수집합니다.
        """
        cache_dir = self.config.DATA_CACHE_DIR
        interval = self.config.INTERVAL
        os.makedirs(cache_dir, exist_ok=True)

        all_data = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        for symbol in coins:
            cache_file = f"{cache_dir}/{symbol}_{interval}_{start_date}_to_{end_date}.parquet"

            if os.path.exists(cache_file):
                logger.info(f"'{symbol}' ({interval}) 데이터 캐시 발견. 파일에서 로드합니다.")
                df = pd.read_parquet(cache_file)
                all_data.append(df)
                continue

            logger.info(f"'{symbol}' ({interval}) 데이터 캐시 없음. API를 통해 수집합니다...")
            try:
                df_list = []
                to_time = end_dt + timedelta(days=1)

                while True:
                    df_chunk = pyupbit.get_ohlcv(
                        f"KRW-{symbol}",
                        interval=interval,
                        to=to_time,
                        count=200
                    )
                    if df_chunk is None or df_chunk.empty:
                        break

                    df_list.append(df_chunk)
                    first_timestamp = df_chunk.index[0]
                    if first_timestamp <= start_dt:
                        break

                    to_time = first_timestamp
                    time.sleep(0.25) # API Rate Limit
                if not df_list:
                    logger.warning(f"[{symbol} 데이터 수집 실패.")
                    continue

                df = pd.concat(df_list).sort_index()
                df = df[~df.index.duplicated(keep='first')]

                df = df[(df.index >= start_dt) & (df.index < (end_dt + timedelta(days=1)))]
                if df.empty:
                    logger.warning(f"[{symbol}] 요청 기간에 해당하는 데이터가 없습니다.")
                    continue

                df = df.reset_index()
                df.rename(columns={'index': 'timestamp'}, inplace=True)
                df['coin'] = symbol

                df.to_parquet(cache_file)
                logger.info(f"'{symbol}' ({interval}) 데이터를 캐시에 저장했습니다.")
                all_data.append(df)
            except Exception as e:
                logger.error(f"[{symbol}] 백테스트 데이터 수집 중 에러: {e}", exc_info=True)

        if not all_data:
            return pd.DataFrame()

        final_df = pd.concat(all_data, ignore_index=True)
        if 'timestamp' in final_df.columns:
             final_df.rename(columns={'timestamp': 'index'}, inplace=True)
        return final_df

