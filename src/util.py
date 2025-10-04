# -*- coding: utf-8 -*-
"""
공통 유틸리티 함수 모듈
"""
import os
import sys
import time
import logging
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional, List, Tuple
from functools import wraps
from collections import deque

logger = logging.getLogger(__name__)

# ========== 데이터 타입 변환 ==========
def convert_numpy_types(obj: Any) -> Any:
    """Numpy 타입을 JSON 직렬화 가능한 파이썬 기본 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(i) for i in obj]
    return obj

# ========== 백테스트 결과 출력 ==========
def plot_backtest_results(history_df, filename):
    """백테스트 결과를 종합 성능 차트로 저장하는 함수"""
    if history_df is None or history_df.empty:
        logger.warning("성과 기록 데이터가 없어 그래프를 생성할 수 없습니다.")
        return

    # --- 1. 데이터 준비 ---
    # 코인 목록을 데이터프레임 컬럼에서 동적으로 추출
    qty_cols = [col for col in history_df.columns if col.endswith('_qty')]
    coins = [col.replace('_qty', '') for col in qty_cols]

    # 가격 정규화 (시작점을 100으로)
    price_cols = [f'{c}_price' for c in coins if f'{c}_price' in history_df.columns and not history_df[f'{c}_price'].isnull().all()]
    if price_cols:
        df_for_norm = history_df[price_cols].copy()
        first_valid_prices = df_for_norm.bfill().iloc[0]
        first_valid_prices[first_valid_prices == 0] = np.nan # 0으로 나누기 방지
        normalized_prices = (df_for_norm / first_valid_prices * 100)

        # 정규화된 가격 컬럼명을 만들 때, 원래 코인 이름만 사용
        norm_price_col_names = {orig_col: f'{coin}_norm_price' for orig_col, coin in zip(price_cols, coins)}
        normalized_prices.rename(columns=norm_price_col_names, inplace=True)
        history_df = pd.concat([history_df, normalized_prices], axis=1)

    # MDD 계산
    history_df['peak'] = history_df['portfolio_value'].cummax()
    history_df['drawdown'] = (history_df['portfolio_value'] - history_df['peak']) / history_df['peak'].replace(0, np.nan)

    # --- 2. 차트 그리기 ---
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(15, 20), sharex=True, gridspec_kw={'height_ratios': [3, 2, 2, 1]}
    )
    fig.suptitle('Comprehensive Backtest Performance Analysis', fontsize=16)

    # [차트 1 포트폴리오 자산 구성
    value_cols = [f'{c}_value' for c in coins if f'{c}_value' in history_df.columns]
    if value_cols or 'cash' in history_df.columns:
        history_df[value_cols + ['cash']].plot.area(ax=ax1, stacked=True, alpha=0.7)
    history_df['portfolio_value'].plot(ax=ax1, color='black', lw=2, label='Total Value', ls='--')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Asset Composition')
    ax1.legend()
    ax1.grid(True)

    # [차트 2 자산 가격 추이 (정규화)
    norm_price_cols = [f'{c}_norm_price' for c in coins if f'{c}_norm_price' in history_df.columns]
    if norm_price_cols:
        history_df[norm_price_cols].plot(ax=ax2, alpha=0.8, lw=1.5)
        ax2.legend([col.replace('_norm_price', '') for col in norm_price_cols])
    ax2.set_ylabel('Normalized Price (Start=100)')
    ax2.set_title('Asset Price Trend')
    ax2.grid(True)

    # [차트 3 코인별 보유 수량
    quantity_cols = [f'{c}_qty' for c in coins if f'{c}_qty' in history_df.columns]
    if quantity_cols:
        history_df[quantity_cols].plot(ax=ax3)
        ax3.set_ylabel('Quantity Held')
        ax3.set_title('Position Change by Coin')
        ax3.legend([col.replace('_qty', '') for col in quantity_cols])
    ax3.grid(True)

    # [차트 4] 최대 낙폭 (MDD)
    ax4.fill_between(history_df.index, history_df['drawdown'] * 100, 0, color='red', alpha=0.4)
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_title('Portfolio Drawdown (MDD)')
    ax4.grid(True)

    plt.xlabel('Date')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        plt.savefig(filename, dpi=300)
        logger.info(f"🎨 성과 그래프 저장 완료: {filename}")
    except Exception as e:
        logger.error(f"그래프 저장 중 오류 발생: {e}")
    finally:
        plt.close(fig)

def report_final_backtest_results(start_date, end_date, initial_balance, result: dict, prefix: str = ""):
    """백테스트 최종 결과를 리포팅하고 파일로 저장하는 범용 함수"""
    if not result or 'portfolio_history' not in result or result['portfolio_history'].empty:
        logger.error("백테스트 결과가 없어 리포트를 생성할 수 없습니다.")
        return

    summary = result['summary']
    portfolio_history_df = result['portfolio_history']
    trade_history_df = result.get('trade_history', pd.DataFrame())

    logger.info(f"\n{'='*80}\n ** {prefix} 최종 백테스트 결과 **\n{'='*80}")
    logger.info(f"전체 기간: {start_date.date()} ~ {end_date.date()}")
    logger.info(f"초기 자본: ${initial_balance:,.2f} | 최종 자산: ${summary['final_value']:,.2f}")
    logger.info(f"총 수익률: {summary['total_return']:.2f}% | 최대 낙폭 (MDD): {summary['mdd']:.2f}%")
    logger.info("="*80)

    # --- 파일 저장 로직 ---
    now_str = datetime.now().strftime("%y%m%d_%H%M%S")
    prefix_str = f"{prefix}_" if prefix else ""
    
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    portfolio_filename = os.path.join(output_dir, f'{prefix_str}portfolio_history_{now_str}.csv')
    trade_filename = os.path.join(output_dir, f'{prefix_str}trade_history_{now_str}.csv')
    plot_filename = os.path.join(output_dir, f'{prefix_str}performance_{now_str}.png')

    portfolio_history_df.to_csv(portfolio_filename)
    logger.info(f"📈 포트폴리오 상세 내역 저장 완료: {portfolio_filename}")
    if not trade_history_df.empty:
        trade_history_df.to_csv(trade_filename, index=False)
        logger.info(f"TRADE_LOG 거래 상세 내역 저장 완료: {trade_filename}")

    plot_backtest_results(portfolio_history_df, plot_filename)

# ========== 환경 감지 ==========
def detect_multiprocessing_mode() -> bool:
    """멀티프로세싱 사용 여부 자동 감지"""
    script_name = os.path.basename(sys.argv[0])
    return (
        script_name.startswith('backtest') or 
        os.getenv("IS_BACKTEST_MODE", "false").lower() == 'true'
    )

# ========== 파일/디렉토리 ==========
def ensure_dir_exists(directory: str) -> None:
    """디렉토리가 없으면 생성"""
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
def get_log_filename(prefix: str, extension: str = 'log') -> str:
    """로그 파일명 생성 (날짜 포함)"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


# ========== 시간/날짜 ==========
def parse_date_safe(date_str: str, default_format: str = '%Y-%m-%d') -> Optional[datetime]:
    """안전한 날짜 파싱 (오류 시 None 반환)"""
    try:
        return datetime.strptime(date_str, default_format)
    except (ValueError, TypeError):
        return None

def get_date_range(start_date: str, end_date: str, default_format: str = '%Y-%m-%d') -> int:
    """두 날짜 사이의 일수 계산"""
    start = parse_date_safe(start_date, default_format)
    end = parse_date_safe(end_date, default_format)
    if start and end:
        return (end - start).days
    return 0

def is_trading_hours(current_time: datetime = None, 
                     allowed_hours: list = None) -> bool:
    """거래 가능 시간인지 확인"""
    if current_time is None:
        current_time = datetime.now()
    
    if allowed_hours is None:
        return True  # 제한 없음
    
    current_hour = current_time.hour
    for start_hour, end_hour in allowed_hours:
        if start_hour <= current_hour < end_hour:
            return True
    return False


# ========== 데코레이터 ==========
def timing_decorator(func):
    """함수 실행 시간 측정 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - start) * 1000
        logger.debug(f"{func.__name__} 실행 시간: {elapsed_ms:.2f}ms")
        return result
    return wrapper

def log_exceptions(func):
    """예외 자동 로깅 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} 실행 중 오류: {e}", exc_info=True)
            raise
    return wrapper

# ========== 데이터 검증 ==========
def validate_price_data(price: float, symbol: str = "") -> bool:
    """가격 데이터 유효성 검증"""
    if price is None or price <= 0:
        logger.warning(f"{symbol} 잘못된 가격: {price}")
        return False
    return True

def is_valid_ohlcv_row(row: dict) -> bool:
    """OHLCV 데이터 한 행의 논리적 유효성 검증"""
    try:
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        # High가 최고가, Low가 최저가인지
        if h < max(o, l, c) or l > min(o, h, c):
            return False
        # 모두 양수인지
        if any(val <= 0 for val in [o, h, l, c]):
            return False
        return True
    except (KeyError, TypeError):
        return False

# ========== 수학/통계 ==========
def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """값을 0-1 범위로 정규화"""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """백분율 변화 계산"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """0으로 나누기 방지"""
    if denominator == 0:
        return default
    return numerator / denominator
    # ... (위 코드)

# ========== 문자열/해시 ==========
def generate_hash(text: str, algorithm: str = 'md5') -> str:
    """텍스트의 해시값 생성 (중복 감지용)"""
    hash_func = getattr(hashlib, algorithm)
    return hash_func(text.encode()).hexdigest()

def truncate_string(text: str, max_length: int = 50, suffix: str = '...') -> str:
    """긴 문자열 자르기"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

# ========== Rate Limiter ==========
class RateLimiter:
    """API Rate Limiting 관리 클래스"""
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """요청 권한 획득 (필요시 대기)"""
        async with self._lock:
            now = time.time()
            
            # 오래된 요청 제거
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()
            
            # 한도 초과 시 대기
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.time_window - now + 0.1
                await asyncio.sleep(sleep_time)
                return await self.acquire()
            
            self.requests.append(now)
    
    def reset(self):
        """Rate limiter 초기화"""
        self.requests.clear()


