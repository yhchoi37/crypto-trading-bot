# -*- coding: utf-8 -*-
"""
기술적 분석 전략 최적화 실행 스크립트 (Walk-Forward Optimization)
"""
import os
# TradingConfig가 import 되기 전에 환경 변수를 설정해야 합니다.
os.environ['IS_BACKTEST_MODE'] = 'true'

import sys
import logging
import itertools
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import multiprocessing as mp
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from config.settings import TradingConfig
from src.trading_system import MultiCoinTradingSystem
from src.data_manager import MultiCoinDataManager
from src.logging_config import setup_logging # 공통 로깅 함수 임포트

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
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

# multiprocessing을 위한 최상위 레벨 함수
def run_backtest_job(job_args: tuple) -> dict:
    """단일 백테스트 잡을 실행하는 워커 함수"""
    job_config, initial_balance, historical_data, config = job_args
    
    logger.debug(f"백테스트 실행: {job_config}")
    
    runner = BacktestRunner(initial_balance, historical_data, config)
    result = runner.run(job_config)
    
    # 상세 데이터는 용량이 크므로 멀티프로세싱 결과 반환 시 제외
    if 'portfolio_history' in result:
        del result['portfolio_history']
    if 'trade_history' in result:
        del result['trade_history']
        
    return {**job_config, **result}

def report_final_results(start_date, end_date, initial_balance, result: dict, prefix: str = ""):
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

    plot_results(portfolio_history_df, plot_filename)

def plot_results(history_df, filename):
    """백테스트 결과를 그래프로 저장하는 범용 함수"""
    if history_df is None or history_df.empty: return
    history_df['peak'] = history_df['portfolio_value'].cummax()
    peak = history_df['peak']
    portfolio_value = history_df['portfolio_value']
    history_df['drawdown'] = (portfolio_value - peak) / peak.replace(0, np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Backtest Performance', fontsize=16)

    ax1.plot(history_df.index, history_df['portfolio_value'], label='Portfolio Value', color='blue')
    ax1.set_ylabel('Portfolio Value ($)'); ax1.set_title('Portfolio Value Over Time'); ax1.grid(True)
    ax2.fill_between(history_df.index, history_df['drawdown'] * 100, 0, color='red', alpha=0.3)
    ax2.set_ylabel('Drawdown (%)'); ax2.set_title('Drawdown Over Time'); ax2.grid(True)

    plt.xlabel('Date'); plt.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        plt.savefig(filename, dpi=300)
        logger.info(f"🎨 성과 그래프 저장 완료: {filename}")
    except Exception as e:
        logger.error(f"그래프 저장 중 오류 발생: {e}")
    finally:
        plt.close(fig)

class BacktestRunner:
    """단일 전략 조합으로 백테스팅을 수행하고 결과를 반환하는 클래스"""
    def __init__(self, initial_balance: float, historical_data: pd.DataFrame, config: TradingConfig):
        self.initial_balance = initial_balance
        self.historical_data = historical_data
        self.config = config
        
    def run(self, job_config: dict) -> dict:
        trading_system = MultiCoinTradingSystem(initial_balance=self.initial_balance, config=self.config)
        if self.historical_data.empty:
            return self._analyze_results([])

        unique_timestamps = self.historical_data.index.unique().sort_values()
        target_allocations = self.config.TARGET_ALLOCATION
        portfolio_history = []

        for current_ts in unique_timestamps:
            current_data_slice = self.historical_data.loc[self.historical_data.index == current_ts]
            current_prices = {row['coin']: row['close'] for _, row in current_data_slice.iterrows()}
            
            trading_system.portfolio_manager.check_risk_management(current_prices)

            for coin in [c for c in target_allocations if c != 'CASH']:
                if coin not in current_prices: continue
                
                coin_history_until_today = self.historical_data[
                    (self.historical_data['coin'] == coin) & (self.historical_data.index <= current_ts)
                ]
                if coin_history_until_today.empty: continue

                analysis = trading_system.analyze_coin_signals(coin, coin_history_until_today, job_config)
                decision = analysis['decision']
                price = current_prices.get(coin)

                ts_str = current_ts.strftime('%Y-%m-%d %H:%M:%S')
                action = decision['action']
                position = trading_system.portfolio_manager.coins.get(coin)
                has_position = position and position.get('quantity', 0) > 0

                if action == 'CONFLICT':
                    logger.debug(f"[{ts_str}] CONFLICT signal for {coin}. Position: {'Yes' if has_position else 'No'}.")
                    action = 'SELL' if has_position else 'BUY'

                if action == 'BUY':
                    current_allocations = trading_system.portfolio_manager.get_current_allocation(current_prices)
                    target_ratio = target_allocations.get(coin, 0)
                    current_ratio = current_allocations.get(coin, 0)
                    if current_ratio < target_ratio:
                        portfolio_value_before_trade = trading_system.portfolio_manager.get_portfolio_value(current_prices)
                        amount_to_invest = (target_ratio - current_ratio) * portfolio_value_before_trade
                        max_investment_per_trade = trading_system.portfolio_manager.cash * 0.1
                        final_investment = min(amount_to_invest, max_investment_per_trade)
                        logger.debug(f"[{ts_str}] BUY signal for {coin}: Target({target_ratio:.2%}) > Current({current_ratio:.2%}). Attempting to invest ~${final_investment:,.2f}.")
                        trading_system.portfolio_manager.execute_trade(coin, 'BUY', final_investment / price, price)
                    else:
                        logger.debug(f"[{ts_str}] BUY signal for {coin} IGNORED: Target({target_ratio:.2%}) <= Current({current_ratio:.2%}).")
                elif action == 'SELL':
                    if has_position:
                        quantity_to_sell = position['quantity'] * 0.5
                        logger.debug(f"[{ts_str}] SELL signal for {coin}: Attempting to sell {quantity_to_sell:.6f} coins.")
                        trading_system.portfolio_manager.execute_trade(coin, 'SELL', quantity_to_sell, price)
                    else:
                        logger.debug(f"[{ts_str}] SELL signal for {coin} IGNORED: No position to sell.")

            # --- 포트폴리오 스냅샷 기록 ---
            portfolio_value = trading_system.portfolio_manager.get_portfolio_value(current_prices)
            snapshot = {'timestamp': current_ts, 'portfolio_value': portfolio_value, 'cash': trading_system.portfolio_manager.cash}
            for coin in target_allocations:
                if coin != 'CASH':
                    position = trading_system.portfolio_manager.coins.get(coin, {})
                    price = current_prices.get(coin, 0)
                    quantity = position.get('quantity', 0)
                    value = quantity * price
                    snapshot[f'{coin}_qty'] = quantity
                    snapshot[f'{coin}_price'] = price
                    snapshot[f'{coin}_value'] = value
                    snapshot[f'{coin}_alloc'] = (value / portfolio_value) if portfolio_value > 0 else 0
            portfolio_history.append(snapshot)

        trade_history = trading_system.portfolio_manager.trade_history
        return self._analyze_results(portfolio_history, trade_history)

    def _analyze_results(self, portfolio_history: list, trade_history: list = None) -> dict:
        if not portfolio_history:
            return {'summary': {'total_return': -100, 'mdd': -100, 'final_value': 0},
                    'portfolio_history': pd.DataFrame(), 'trade_history': pd.DataFrame()}

        portfolio_df = pd.DataFrame(portfolio_history).set_index('timestamp')
        trade_df = pd.DataFrame(trade_history) if trade_history is not None else pd.DataFrame()

        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value / self.initial_balance - 1) * 100
        peak = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - peak) / peak.replace(0, np.nan)
        mdd = 0 if peak.iloc[-1] == 0 else drawdown.min() * 100

        summary = {'total_return': total_return, 'mdd': mdd, 'final_value': final_value}

        return {'summary': summary, 'portfolio_history': portfolio_df, 'trade_history': trade_df}

class Optimizer:
    """주어진 기간 내에서 최적의 전략 파라미터를 찾는 클래스"""
    def __init__(self, initial_balance: float, config: TradingConfig, historical_data: pd.DataFrame):
        self.initial_balance = initial_balance
        self.config = config
        self.historical_data = historical_data

    def _generate_param_space(self, params_config: dict) -> dict:
        param_values = {}
        for name, config in params_config.items():
            param_values[name] = list(np.arange(config['min'], config['max'] + config['step'], config['step']))
        return param_values

    def _generate_jobs(self):
        """최적화를 위한 유효하고 논리적인 테스트 '잡(Job)' 생성"""
        jobs = []
        cfg = self.config.OPTIMIZATION_CONFIG

        # 1. 사용할 지표의 모든 조합 생성 (A, B, C, AB, AC, BC, ABC)
        buy_indicator_names = list(cfg['buy_indicators'].keys())
        buy_combos = [c for i in range(1, len(buy_indicator_names) + 1) for c in itertools.combinations(buy_indicator_names, i)]
        sell_indicator_names = list(cfg['sell_indicators'].keys())
        sell_combos = [c for i in range(1, len(sell_indicator_names) + 1) for c in itertools.combinations(sell_indicator_names, i)]
        for buy_combo in buy_combos:
            for sell_combo in sell_combos:
                buy_param_producers = []
                sell_param_producers = []

                # --- 매수 지표 파라미터 생성 준비 ---
                for ind_name in buy_combo:
                    space = self._generate_param_space(cfg['buy_indicators'][ind_name])
                    # 각 파라미터 조합을 (indicator_name, {param_name: value, ...}) 형태로 생성
                    param_combinations = [dict(zip(space.keys(), values)) for values in itertools.product(*space.values())]
                    buy_param_producers.append([(ind_name, p_comb) for p_comb in param_combinations])

                # --- 매도 지표 파라미터 생성 준비 ---
                for ind_name in sell_combo:
                    space = self._generate_param_space(cfg['sell_indicators'][ind_name])
                    param_combinations = [dict(zip(space.keys(), values)) for values in itertools.product(*space.values())]
                    sell_param_producers.append([(ind_name, p_comb) for p_comb in param_combinations])

                # 모든 매수/매도 파라미터 조합을 순회
                for buy_params_list in itertools.product(*buy_param_producers):
                    for sell_params_list in itertools.product(*sell_param_producers):
                        buy_indicators = dict(buy_params_list)
                        sell_indicators = dict(sell_params_list)

                        # --- 논리적 파라미터 검증 ---
                        valid = True
                        if 'MA_Cross' in buy_indicators:
                            if buy_indicators['MA_Cross']['ma_short_period'] >= buy_indicators['MA_Cross']['ma_long_period']:
                                valid = False
                        if 'MA_Cross' in sell_indicators:
                            if sell_indicators['MA_Cross']['ma_short_period'] >= sell_indicators['MA_Cross']['ma_long_period']:
                                valid = False
                        if not valid:
                            continue

                        # --- 유효한 Trigger 값 계산 ---
                        max_buy_score = sum(cfg['signal_weights'].get(f'{name}_buy', 1) for name in buy_combo)
                        buy_trigger_cfg = cfg['buy_trigger_threshold']
                        buy_valid_triggers = np.arange(buy_trigger_cfg['min'], buy_trigger_cfg['max'] + buy_trigger_cfg['step'], buy_trigger_cfg['step'])
                        valid_buy_triggers = [t for t in buy_valid_triggers if 0 < t <= max_buy_score]
                        if len(buy_combo) == 1: valid_buy_triggers = [1]

                        max_sell_score = sum(cfg['signal_weights'].get(f'{name}_sell', 1) for name in sell_combo)
                        sell_trigger_cfg = cfg['sell_trigger_threshold']
                        sell_valid_triggers = np.arange(sell_trigger_cfg['min'], sell_trigger_cfg['max'] + sell_trigger_cfg['step'], sell_trigger_cfg['step'])
                        valid_sell_triggers = [t for t in sell_valid_triggers if 0 < t <= max_sell_score]
                        if len(sell_combo) == 1: valid_sell_triggers = [1]

                        if not valid_buy_triggers or not valid_sell_triggers:
                            continue
                        for buy_trigger in valid_buy_triggers:
                            for sell_trigger in valid_sell_triggers:
                                jobs.append({
                                    'buy_indicators': buy_indicators,
                                    'sell_indicators': sell_indicators,
                                    'buy_trigger_threshold': buy_trigger,
                                    'sell_trigger_threshold': sell_trigger,
                                    'signal_weights': cfg.get('signal_weights', {})
                                })

        # 중복 제거 로직
        final_jobs = []
        temp_set = set()
        for job in jobs:
            # json.dumps가 numpy 타입을 처리할 수 없으므로, 먼저 파이썬 기본 타입으로 변환
            job_serializable = convert_numpy_types(job)
            job_repr = json.dumps(job_serializable, sort_keys=True)
            if job_repr not in temp_set:
                # 최종 잡 리스트에는 원본(numpy 타입 포함)을 추가해도 무방
                final_jobs.append(job)
                temp_set.add(job_repr)
        return final_jobs

    def run_optimization(self) -> dict:
        """최적화 실행 및 최고의 전략 반환 (병렬 처리 적용)"""
        jobs = self._generate_jobs()
        if not jobs:
            logger.warning("생성된 최적화 작업이 없습니다.")
            return None

        core_count = self.config.PERFORMANCE_CONFIG['parallel_cores']
        if core_count == -1:
            core_count = mp.cpu_count()
        logger.info(f"훈련 구간 내에서 {len(jobs)}개 전략 조합으로 최적화를 시작합니다. (병렬 코어: {core_count}개)")

        # 각 잡에 필요한 인자들을 튜플 리스트로 준비
        job_args = [(job, self.initial_balance, self.historical_data, self.config) for job in jobs]

        results = []
        # multiprocessing.Pool을 사용하여 병렬 처리
        with mp.Pool(processes=core_count) as pool:
            # tqdm을 사용하여 진행률 표시
            with tqdm(total=len(jobs), desc="Finding Best Params", ncols=100) as pbar:
                for result in pool.imap_unordered(run_backtest_job, job_args):
                    results.append(result)
                    pbar.update()

        if not results:
            return None

        results_df = pd.DataFrame(results)
        if results_df.empty:
            return None

        # 'final_value'는 이제 'summary' 딕셔너리 안에 있음
        results_df['final_value'] = results_df['summary'].apply(lambda x: x.get('final_value', 0))
        best_result = results_df.sort_values(by='final_value', ascending=False).iloc[0]
        return best_result.to_dict()

class WalkForwardOptimizer:
    """전진 분석을 총괄하는 최상위 클래스"""
    def __init__(self, start_date_str: str, end_date_str: str, initial_balance: float):
        self.start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        self.initial_balance = initial_balance
        os.environ['INITIAL_BALANCE'] = str(initial_balance)
        self.config = TradingConfig()
        self.data_manager = MultiCoinDataManager()
        self.all_historical_data = None
        self.out_of_sample_portfolio_histories = []
        self.out_of_sample_trade_histories = []

    def _get_all_possible_params(self) -> dict:
        """최적화 설정에서 가능한 모든 파라미터 값을 추출"""
        all_params = {
            'ma_short_period': set(), 'ma_long_period': set(),
            'rsi_period': set(), 'rsi_oversold_threshold': set(),
            'bollinger_window': set(), 'bollinger_std_dev': set()
        }
        cfg = self.config.OPTIMIZATION_CONFIG

        # 매수/매도 지표 파라미터 수집
        for indicator_type in ['buy_indicators', 'sell_indicators']:
            for _, params_config in cfg.get(indicator_type, {}).items():
                for p_name, p_vals in params_config.items():
                    if p_name in all_params:
                        all_params[p_name].update(np.arange(p_vals['min'], p_vals['max'] + p_vals['step'], p_vals['step']))
        for k in all_params:
            all_params[k] = sorted(list(all_params[k]))
        return all_params

    def _precompute_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임에 모든 가능한 지표를 미리 계산하여 추가"""
        if data.empty:
            return data

        logger.info("전체 데이터에 대한 모든 기술적 지표의 사전 계산을 시작합니다...")
        all_params = self._get_all_possible_params()

        data_with_indicators = []
        for coin, group in data.groupby('coin'):
            df = group.copy().sort_values('timestamp')
            if all_params['ma_short_period']:
                for p in set(all_params['ma_short_period'] + all_params['ma_long_period']):
                    df.ta.sma(length=p, append=True)
            if all_params['rsi_period']:
                for p in all_params['rsi_period']:
                    df.ta.rsi(length=p, append=True)
            if all_params['bollinger_window']:
                for p in all_params['bollinger_window']:
                    for std in all_params['bollinger_std_dev']:
                        df.ta.bbands(length=p, std=std, append=True)

            data_with_indicators.append(df)

        logger.info("모든 기술적 지표 계산 완료.")
        return pd.concat(data_with_indicators) if data_with_indicators else pd.DataFrame()

    def run(self):
        wfc = self.config.WALK_FORWARD_CONFIG
        coins = [c for c in self.config.TARGET_ALLOCATION if c != 'CASH']
        
        # 데이터 로드 및 인덱스 설정
        self.all_historical_data = self.data_manager.get_historical_data_for_backtest(
            coins, self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d')
        )
        if self.all_historical_data.empty: return
        self.all_historical_data['timestamp'] = pd.to_datetime(self.all_historical_data['index'])
        self.all_historical_data = self.all_historical_data.set_index('timestamp')

        self.all_historical_data = self._precompute_all_indicators(self.all_historical_data.reset_index()).set_index('timestamp')

        current_start = self.start_date
        total_balance = self.initial_balance
        latest_best_strategy = None # 최신 최적 전략을 저장할 변수

        while current_start + relativedelta(months=wfc['training_period_months']) < self.end_date:
            train_start, train_end = current_start, current_start + relativedelta(months=wfc['training_period_months'])
            test_end = train_end + relativedelta(months=wfc['testing_period_months'])
            if test_end > self.end_date: test_end = self.end_date

            logger.info(f"\n{'='*80}\n전진 분석 구간: 훈련 [{train_start.date()}-{train_end.date()}] | 검증 [{train_end.date()}-{test_end.date()}]\n{'='*80}")
            
            train_data = self.all_historical_data[(self.all_historical_data.index >= train_start) & (self.all_historical_data.index < train_end)].copy()
            optimizer = Optimizer(total_balance, self.config, train_data)
            best_strategy = optimizer.run_optimization()

            if best_strategy:
                latest_best_strategy = best_strategy # 찾은 최적 전략을 변수에 저장
                buy_combo_str = ', '.join(best_strategy['buy_indicators'].keys())
                sell_combo_str = ', '.join(best_strategy['sell_indicators'].keys())
                logger.info(f"최적 전략 발견: Buy-({buy_combo_str}) | Sell-({sell_combo_str})")

                test_data = self.all_historical_data[(self.all_historical_data.index >= train_end) & (self.all_historical_data.index < test_end)].copy()
                runner = BacktestRunner(total_balance, test_data, self.config)
                test_result = runner.run(best_strategy)

                if test_result and not test_result['portfolio_history'].empty:
                    self.out_of_sample_portfolio_histories.append(test_result['portfolio_history'])
                    if 'trade_history' in test_result and not test_result['trade_history'].empty:
                        self.out_of_sample_trade_histories.append(test_result['trade_history'])
                    
                    total_balance = test_result['summary']['final_value']
                    logger.info(f"검증 구간 성과: 최종 자산 ${total_balance:,.2f} | 수익률 {test_result['summary']['total_return']:.2f}%")
            else:
                logger.warning("현 구간에서 유효한 전략을 찾지 못했습니다. 다음 구간으로 넘어갑니다.")

            current_start += relativedelta(months=wfc['testing_period_months'])

        if not self.out_of_sample_portfolio_histories:
            logger.error("전진 분석 결과가 없습니다.")
            return

        final_portfolio_history = pd.concat(self.out_of_sample_portfolio_histories).sort_index().drop_duplicates()
        final_trade_history = pd.concat(self.out_of_sample_trade_histories).sort_index().drop_duplicates() if self.out_of_sample_trade_histories else pd.DataFrame()
        
        final_summary = self._calculate_summary_stats(final_portfolio_history, self.initial_balance)
        final_result = {
            'summary': final_summary,
            'portfolio_history': final_portfolio_history,
            'trade_history': final_trade_history
        }
        report_final_results(self.start_date, self.end_date, self.initial_balance, final_result, prefix="WalkForward")
        self._save_strategy_to_file(latest_best_strategy)

    def _calculate_summary_stats(self, portfolio_df: pd.DataFrame, initial_balance: float) -> dict:
        """포트폴리오 이력 데이터프레임으로부터 요약 통계를 계산합니다."""
        if portfolio_df.empty:
            return {'total_return': -100, 'mdd': -100, 'final_value': 0}
        
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value / initial_balance - 1) * 100
        peak = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - peak) / peak.replace(0, np.nan)
        mdd = 0 if peak.iloc[-1] == 0 else drawdown.min() * 100
        return {'total_return': total_return, 'mdd': mdd, 'final_value': final_value}

    def _save_strategy_to_file(self, strategy: dict):
        """찾아낸 최적의 전략 파라미터를 JSON 파일로 저장"""
        if not strategy:
            logger.warning("저장할 최적 전략이 없습니다. 파일을 생성하지 않습니다.")
            return

        params_to_save = {
            'buy_indicators': strategy.get('buy_indicators', {}),
            'sell_indicators': strategy.get('sell_indicators', {}),
            'buy_trigger_threshold': strategy.get('buy_trigger_threshold'),
            'sell_trigger_threshold': strategy.get('sell_trigger_threshold'),
            'signal_weights': strategy.get('signal_weights', {})
        }
        filepath = 'optimized_params.json'
        try:
            # Numpy 타입을 파이썬 기본 타입으로 변환
            params_to_save = convert_numpy_types(params_to_save)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(params_to_save, f, ensure_ascii=False, indent=4)
            logger.info(f"✅ 최적화된 파라미터를 '{filepath}' 파일에 성공적으로 저장했습니다.")
        except Exception as e:
            logger.error(f"❌ 최적 파라미터 저장 중 오류 발생: {e}", exc_info=True)

def precompute_indicators_for_single_run(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """단일 실행을 위해 설정 파일 기반으로 지표를 계산합니다."""
    if data.empty:
        return data

    logger.info("단일 백테스트를 위한 기술적 지표 계산을 시작합니다...")
    all_params = {'sma': set(), 'rsi': set(), 'bbands': []}
    indicator_config = {**config.get('buy_indicators', {}), **config.get('sell_indicators', {})}

    for ind, params in indicator_config.items():
        if ind == 'MA_Cross':
            all_params['sma'].add(params.get('ma_short_period'))
            all_params['sma'].add(params.get('ma_long_period'))
        elif ind == 'RSI':
            all_params['rsi'].add(params.get('rsi_period'))
        elif ind == 'BollingerBand':
            all_params['bbands'].add((params.get('bollinger_window'), params.get('bollinger_std_dev')))
    data_with_indicators = []
    for coin, group in data.groupby('coin'):
        df = group.copy().sort_values('timestamp')
        for p in all_params['sma']:
            if p: df.ta.sma(length=p, append=True)
        for p in all_params['rsi']:
            if p: df.ta.rsi(length=p, append=True)
        for p_win, p_std in all_params['bbands']:
            if p_win and p_std: df.ta.bbands(length=p_win, std=p_std, append=True)
        data_with_indicators.append(df)

    logger.info("기술적 지표 계산 완료.")
    return pd.concat(data_with_indicators) if data_with_indicators else pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="백테스팅 및 최적화 실행 스크립트")
    parser.add_argument('--single', action='store_true', help='전진 분석 없이 설정 파일의 기본값으로 단일 백테스트를 실행합니다.')
    args = parser.parse_args()

    config = TradingConfig()
    log_filename = 'logs/backtest_single.log' if args.single else 'logs/backtest_wfo.log'
    setup_logging(config.LOG_LEVEL, log_filename)
    
    logger.info("🚀 백테스트 시스템 시작")
    START_DATE = "2022-01-01"
    END_DATE = "2023-12-31"
    INITIAL_BALANCE = 10000000.0 # 하드코딩된 값 (추후 .env 등으로 관리 가능)

    try:
        if args.single:
            logger.info("단일 설정 기반 백테스트를 시작합니다.")
            data_manager = MultiCoinDataManager()
            coins = [c for c in config.TARGET_ALLOCATION if c != 'CASH']

            # 1. 데이터 로드 및 지표 계산
            historical_data = data_manager.get_historical_data_for_backtest(
                coins, START_DATE, END_DATE
            )
            historical_data['timestamp'] = pd.to_datetime(historical_data['index'])
            historical_data = historical_data.set_index('timestamp')

            precomputed_data = precompute_indicators_for_single_run(
                historical_data.reset_index(), config.TECHNICAL_ANALYSIS_CONFIG
            ).set_index('timestamp')

            # 2. job_config 생성
            job_config = config.TECHNICAL_ANALYSIS_CONFIG

            # 3. 백테스트 실행
            runner = BacktestRunner(INITIAL_BALANCE, precomputed_data, config)
            result = runner.run(job_config)
            if result:
                report_final_results(datetime.strptime(START_DATE, '%Y-%m-%d'), datetime.strptime(END_DATE, '%Y-%m-%d'), INITIAL_BALANCE, result, prefix="SingleRun")
        else:
            logger.info("전진 분석 최적화를 시작합니다.")
            wfo = WalkForwardOptimizer(START_DATE, END_DATE, INITIAL_BALANCE)
            wfo.run()
    except Exception as e:
        logger.error(f"❌ 백테스트 중 오류 발생: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    # Windows/macOS에서 multiprocessing 사용 시 필요
    mp.freeze_support()
    sys.exit(main())
