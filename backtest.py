# -*- coding: utf-8 -*-
"""
기술적 분석 전략 최적화 실행 스크립트 (Walk-Forward Optimization)
"""
import os
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
import json # json 모듈 추가

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
    
    # 현재 작업 중인 파라미터 조합을 DEBUG 레벨로 로깅
    logger.debug(f"백테스트 실행: {job_config}")
    
    runner = BacktestRunner(initial_balance, historical_data, config)
    result = runner.run(job_config)
    # history는 객체 크기가 크므로 결과 전송 시 제외
    if 'history' in result:
        del result['history']
    return {**job_config, **result}


class BacktestRunner:
    """단일 전략 조합으로 백테스팅을 수행하고 결과를 반환하는 클래스"""
    def __init__(self, initial_balance: float, historical_data: pd.DataFrame, config: TradingConfig):
        self.initial_balance = initial_balance
        self.historical_data = historical_data
        self.config = config
        
    def run(self, job_config: dict) -> dict:
        trading_system = MultiCoinTradingSystem(initial_balance=self.initial_balance)
        if self.historical_data.empty:
            return self._analyze_results([])

        data_by_date = {date: group for date, group in self.historical_data.groupby('date')}
        target_allocations = self.config.TARGET_ALLOCATION

        portfolio_history = []

        for current_date in sorted(data_by_date.keys()):
            current_prices = {row['coin']: row['close'] for _, row in data_by_date[current_date].iterrows()}
            trading_system.portfolio_manager.check_risk_management(current_prices)
            portfolio_value = trading_system.portfolio_manager.get_portfolio_value(current_prices)
            current_allocations = trading_system.portfolio_manager.get_current_allocation(current_prices)

            for coin in [c for c in target_allocations if c != 'CASH']:
                coin_history_until_today = self.historical_data[
                    (self.historical_data['coin'] == coin) & (self.historical_data['date'] <= current_date)
                ]
                if coin_history_until_today.empty: continue
                # job_config에 buy/sell indicator combo가 없으면 MultiCoinTradingSystem에서 오류 발생할 수 있음
                # job_config 구조를 변경했으므로 MultiCoinTradingSystem.analyze_coin_signals도 수정 필요
                if 'indicator_combo' in job_config:
                    # Adapt new job format to old expected format if needed
                    # This example assumes analyze_coin_signals expects this new format
                    pass
                analysis = trading_system.analyze_coin_signals(coin, coin_history_until_today, job_config)
                decision = analysis['decision']
                price = current_prices.get(coin)
                if not price: continue

                if decision['action'] == 'BUY':
                    if current_allocations.get(coin, 0) < target_allocations.get(coin, 0):
                        amount_to_invest = min(
                            (target_allocations.get(coin, 0) - current_allocations.get(coin, 0)) * portfolio_value,
                            trading_system.portfolio_manager.cash * 0.1
                        )
                        if amount_to_invest > 10000:
                           trading_system.portfolio_manager.execute_trade(coin, 'BUY', amount_to_invest / price, price)
                elif decision['action'] == 'SELL':
                    position = trading_system.portfolio_manager.coins.get(coin)
                    if position and position.get('quantity', 0) > 0:
                        quantity_to_sell = position['quantity'] * 0.5
                        trading_system.portfolio_manager.execute_trade(coin, 'SELL', quantity_to_sell, price)

            portfolio_value = trading_system.portfolio_manager.get_portfolio_value(current_prices)
            portfolio_history.append({'date': current_date, 'portfolio_value': portfolio_value})

        return self._analyze_results(portfolio_history)

    def _analyze_results(self, portfolio_history: list) -> dict:
        if not portfolio_history:
            return {'total_return': -100, 'mdd': -100, 'final_value': 0, 'history': None}

        results_df = pd.DataFrame(portfolio_history).set_index('date')
        final_value = results_df['portfolio_value'].iloc[-1]
        total_return = (final_value / self.initial_balance - 1) * 100
        peak = results_df['portfolio_value'].cummax()
        drawdown = (results_df['portfolio_value'] - peak) / peak
        mdd = 0 if peak.iloc[-1] == 0 else drawdown.min() * 100
        return {'total_return': total_return, 'mdd': mdd, 'final_value': final_value, 'history': results_df}


class Optimizer:
    """주어진 기간 내에서 최적의 전략 파라미터를 찾는 클래스"""
    def __init__(self, initial_balance: float, config: TradingConfig, historical_data: pd.DataFrame):
        self.initial_balance = initial_balance
        self.config = config
        # Optimizer는 이제 사전 계산된 데이터를 그대로 받아서 사용
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
                    space = self._generate_param_space(cfg['sell_indicators'[ind_name])
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

        # 중복 제거 로직은 현재 구조에서 덜 필요하지만, 만약을 위해 유지
        final_jobs = []
        temp_set = set()
        for job in jobs:
            job_repr = json.dumps(job, sort_keys=True)
            if job_repr not in temp_set:
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
        # 결과가 없는 경우를 대비하여 비어있는지 확인
        if results_df.empty:
            return None

        best_result = results_df.sort_values(by='final_value', ascending=False).iloc[0]
        return best_result.to_dict()


class WalkForwardOptimizer:
    """전진 분석을 총괄하는 최상위 클래스"""
    def __init__(self, start_date_str: str, end_date_str: str, initial_balance: float):
        self.start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        self.initial_balance = initial_balance
        os.environ['BACKTEST_MODE'] = 'true'
        os.environ['INITIAL_BALANCE'] = str(initial_balance)
        self.config = TradingConfig()
        self.data_manager = MultiCoinDataManager()
        self.all_historical_data = None
        self.out_of_sample_results = []

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
            df = group.copy().sort_values('date')

            # 모든 파라미터 조합에 대해 지표 계산
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
        if not wfc['enabled']:
            logger.warning("전진 분석이 비활성화되어 있습니다.")
            return

        coins = [c for c in self.config.TARGET_ALLOCATION if c != 'CASH']
        self.all_historical_data = self.data_manager.get_historical_data_for_backtest(
            coins, self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d')
        )
        if self.all_historical_data.empty: return
        self.all_historical_data['date'] = pd.to_datetime(self.all_historical_data['index']).dt.date

        # 전체 데이터에 대해 지표를 단 한 번만 계산
        self.all_historical_data = self._precompute_all_indicators(self.all_historical_data)

        current_start = self.start_date
        total_balance = self.initial_balance
        latest_best_strategy = None # 최신 최적 전략을 저장할 변수

        while current_start + relativedelta(months=wfc['training_period_months']) < self.end_date:
            train_start, train_end = current_start, current_start + relativedelta(months=wfc['training_period_months'])
            test_end = train_end + relativedelta(months=wfc['testing_period_months'])
            if test_end > self.end_date: test_end = self.end_date

            logger.info(f"\n{'='*80}\n전진 분석 구간: 훈련 [{train_start.date()}-{train_end.date()} | 검증 [{train_end.date()}-{test_end.date()}\n{'='*80}")

            train_data = self.all_historical_data[(self.all_historical_data['date'] >= train_start.date()) & (self.all_historical_data['date'] < train_end.date()).copy()
            optimizer = Optimizer(total_balance, self.config, train_data)
            best_strategy = optimizer.run_optimization()

            if best_strategy:
                latest_best_strategy = best_strategy # 찾은 최적 전략을 변수에 저장
                buy_combo_str = ', '.join(best_strategy['buy_indicators'].keys())
                sell_combo_str = ', '.join(best_strategy['sell_indicators'.keys())
                logger.info(f"최적 전략 발견: Buy-({buy_combo_str}) | Sell-({sell_combo_str})")

                test_data = self.all_historical_data[(self.all_historical_data['date'] >= train_end.date()) & (self.all_historical_data['date' < test_end.date())].copy()
                runner = BacktestRunner(total_balance, test_data, self.config)
                test_result = runner.run(best_strategy)

                if test_result and test_result['history'] is not None:
                    self.out_of_sample_results.append(test_result['history'])
                    total_balance = test_result['final_value']
                    logger.info(f"검증 구간 성과: 최종 자산 ${total_balance:,.2f} | 수익률 {test_result['total_return']:.2f}%")
            else:
                logger.warning("현 구간에서 유효한 전략을 찾지 못했습니다. 다음 구간으로 넘어갑니다.")

            current_start += relativedelta(months=wfc['testing_period_months'])

        self.report_final_results()
        self._save_strategy_to_file(latest_best_strategy)

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


    def report_final_results(self):
        if not self.out_of_sample_results:
            logger.error("전진 분석 결과가 없습니다.")
            return

        final_history = pd.concat(self.out_of_sample_results).drop_duplicates()
        final_stats = BacktestRunner(self.initial_balance, pd.DataFrame(), self.config)._analyze_results(final_history.reset_index().to_dict('records'))

        logger.info(f"\n{'='*80}\n ** 최종 전진 분석(Walk-Forward) 결과 **\n{'='*80}")
        logger.info(f"전체 기간: {self.start_date.date()} ~ {self.end_date.date()}")
        logger.info(f"초기 자본: ${self.initial_balance:,.2f} | 최종 자산: ${final_stats['final_value':,.2f}")
        logger.info(f"총 수익률: {final_stats['total_return']:.2f}% | 최대 낙폭 (MDD): {final_stats['mdd']:.2f}%")
        logger.info("="*80)

        self.plot_results(final_stats['history'])

    def plot_results(self, history_df):
        if history_df is None or history_df.empty: return
        history_df['peak'] = history_df['portfolio_value'].cummax()
        history_df['drawdown'] = (history_df['portfolio_value'] - history_df['peak']) / history_df['peak'

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle('Walk-Forward Analysis Performance', fontsize=16)

        ax1.plot(history_df.index, history_df['portfolio_value'], label='Portfolio Value', color='blue')
        ax1.set_ylabel('Portfolio Value ($)'); ax1.set_title('Portfolio Value Over Time'); ax1.grid(True)
        ax2.fill_between(history_df.index, history_df['drawdown'] * 100, 0, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)'); ax2.set_title('Drawdown Over Time'); ax2.grid(True)

        plt.xlabel('Date'); plt.tight_layout(rect=[0, 0, 1, 0.96])
        try:
            plt.savefig('walk_forward_performance.png', dpi=300)
            logger.info("성과 그래프가 'walk_forward_performance.png'에 성공적으로 저장되었습니다.")
        except Exception as e:
            logger.error(f"그래프 저장 중 오류 발생: {e}")
        finally:
            plt.close(fig)

def main():
    # 1. TradingConfig를 먼저 로드하여 LOG_LEVEL에 접근
    config = TradingConfig()

    # 2. 중앙 로깅 함수를 호출하여 로거를 설정 (로그 파일 이름 지정)
    setup_logging(config.LOG_LEVEL, 'backtest.log')
    
    logger.info("🚀 전략 최적화 시스템 시작")
    START_DATE = "2022-01-01"
    END_DATE = "2023-12-31"
    INITIAL_BALANCE = 100000.0
    try:
        wfo = WalkForwardOptimizer(START_DATE, END_DATE, INITIAL_BALANCE)
        wfo.run()
    except Exception as e:
        logger.error(f"❌ 최적화 중 오류 발생: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    # Windows/macOS에서 multiprocessing 사용 시 필요
    mp.freeze_support()
    sys.exit(main())

