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
        """
        백테스트용 과거 데이터 반환 (Parquet 캐싱 적용)
        먼저 로컬 캐시에서 데이터를 찾고, 없으면 API를 통해 다운로드 후 저장합니다.
        """
        # 캐시 디렉토리 생성
        cache_dir = self.config.DATA_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        all_data = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        for symbol in coins:
            # 캐시 파일 경로 생성
            cache_file = f"{cache_dir}/{symbol}_{start_date}_to_{end_date}.parquet"

            # 1. 캐시 확인
            if os.path.exists(cache_file):
                logger.info(f"'{symbol}' 데이터 캐시 발견. 파일에서 로드합니다: {cache_file}")
                df = pd.read_parquet(cache_file)
                all_data.append(df)
                continue

            # 2. 캐시 없으면 API 호출
            logger.info(f"'{symbol}' 데이터 캐시 없음. API를 통해 수집합니다 ({start_date} ~ {end_date})...")
            try:
                df = pyupbit.get_ohlcv(
                    f"KRW-{symbol}",
                    interval="day",
                    to=end_dt.strftime('%Y-%m-%d 23:59:59'), # 마지막 날을 포함하도록 시간 지정
                    count=(end_dt - start_dt).days + 2 # 넉넉하게 요청
                )
                if df is None:
                    logger.warning(f"[{symbol}] 데이터 수집 실패: API가 None을 반환했습니다.")
                    continue

                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                if df.empty:
                    logger.warning(f"[{symbol}] 요청 기간에 해당하는 데이터가 없습니다.")
                    continue

                df = df.reset_index()
                df.rename(columns={'index': 'timestamp'}, inplace=True)
                df['coin'] = symbol

                # 3. 다운로드 후 캐시에 저장
                df.to_parquet(cache_file)
                logger.info(f"'{symbol}' 데이터를 캐시에 저장했습니다: {cache_file}")

                all_data.append(df)
                time.sleep(0.2) # Rate limit 방지
            except Exception as e:
                logger.error(f"[{symbol}] 백테스트 데이터 수집 중 에러: {e}")

        if not all_data:
        return pd.DataFrame()

        # 'index' 컬럼명을 통일성 있게 변경
        final_df = pd.concat(all_data, ignore_index=True)
        if 'timestamp' in final_df.columns:
             final_df.rename(columns={'timestamp': 'index'}, inplace=True)
        return final_df

