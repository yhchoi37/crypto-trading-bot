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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from config.settings import TradingConfig
from src.trading_system import MultiCoinTradingSystem
from src.data_manager import MultiCoinDataManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


class BacktestRunner:
    """단일 전략 조합으로 백테스팅을 수행하고 결과를 반환하는 클래스"""
    def __init__(self, initial_balance: float, historical_data: pd.DataFrame, config: TradingConfig):
        self.initial_balance = initial_balance
        self.historical_data = historical_data
        self.config = config
        
    def run(self, job_config: dict) -> dict:
        trading_system = MultiCoinTradingSystem(initial_balance=self.initial_balance)
        portfolio_history = []
        if self.historical_data.empty:
            return self._analyze_results([])

        data_by_date = {date: group for date, group in self.historical_data.groupby('date')}
        target_allocations = self.config.TARGET_ALLOCATION

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
        self.historical_data = historical_data
        self.optimization_results = []

    def _generate_param_space(self, params_config: dict) -> dict:
        param_values = {}
        for name, config in params_config.items():
            param_values[name] = list(np.arange(config['min'], config['max'] + config['step'], config['step']))
        return param_values

    def _generate_jobs(self) -> list:
        jobs = []
        cfg = self.config.OPTIMIZATION_CONFIG
        buy_indicator_names = list(cfg['buy_indicators'].keys())
        sell_indicator_names = list(cfg['sell_indicators'].keys())

        buy_combos = [c for i in range(1, len(buy_indicator_names) + 1) for c in itertools.combinations(buy_indicator_names, i)]
        sell_combos = [c for i in range(1, len(sell_indicator_names) + 1) for c in itertools.combinations(sell_indicator_names, i)]

        for buy_combo in buy_combos:
            for sell_combo in sell_combos:
                param_names, param_value_lists = [], []

                param_configs = {**cfg['buy_indicators'], **cfg['sell_indicators']}
                unique_params = {p_name: p_config for ind_name in buy_combo + sell_combo for p_name, p_config in param_configs[ind_name.items()}

                space = self._generate_param_space(unique_params)
                for p_name, p_values in space.items():
                    param_names.append(p_name)
                    param_value_lists.append(p_values)

                buy_trigger_space = self._generate_param_space({'buy_trigger_threshold': cfg['buy_trigger_threshold']})
                param_names.append('buy_trigger_threshold')
                param_value_lists.append(buy_trigger_space['buy_trigger_threshold'])

                sell_trigger_space = self._generate_param_space({'sell_trigger_threshold': cfg['sell_trigger_threshold']})
                param_names.append('sell_trigger_threshold')
                param_value_lists.append(sell_trigger_space['sell_trigger_threshold'])

                for param_values in itertools.product(*param_value_lists):
                    params = dict(zip(param_names, param_values))
                    params['weights'] = {**cfg['buy_signal_weights'], **cfg['sell_signal_weights'}
                    jobs.append({'buy_indicator_combo': buy_combo, 'sell_indicator_combo': sell_combo, 'params': params})
        return jobs

    def find_best_strategy(self) -> dict:
        jobs = self._generate_jobs()
        if not jobs:
            logger.warning("생성된 최적화 작업이 없습니다.")
            return None

        runner = BacktestRunner(self.initial_balance, self.historical_data, self.config)
        for job in tqdm(jobs, desc="Finding Best Params", leave=False, ncols=100):
            result = runner.run(job)
            self.optimization_results.append({**job, **result})

        if not self.optimization_results:
            return None

        results_df = pd.DataFrame(self.optimization_results)
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
        self.all_historical_data['date'] = pd.to_datetime(self.all_historical_data['index').dt.date

        current_start = self.start_date
        total_balance = self.initial_balance

        while current_start + relativedelta(months=wfc['training_period_months']) < self.end_date:
            train_start, train_end = current_start, current_start + relativedelta(months=wfc['training_period_months'])
            test_end = train_end + relativedelta(months=wfc['testing_period_months'])
            if test_end > self.end_date: test_end = self.end_date

            logger.info(f"\n{'='*80}\n전진 분석 구간: 훈련 [{train_start.date()}-{train_end.date()} | 검증 [{train_end.date()}-{test_end.date()}]\n{'='*80}")

            train_data = self.all_historical_data[(self.all_historical_data['date'] >= train_start.date()) & (self.all_historical_data['date'] < train_end.date())]
            optimizer = Optimizer(total_balance, self.config, train_data)
            best_strategy = optimizer.find_best_strategy()

            if best_strategy:
                logger.info(f"최적 전략 발견: Buy-{best_strategy['buy_indicator_combo']} | Sell-{best_strategy['sell_indicator_combo']}")
                test_data = self.all_historical_data[(self.all_historical_data['date'] >= train_end.date()) & (self.all_historical_data['date'] < test_end.date())]
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

    def report_final_results(self):
        if not self.out_of_sample_results:
            logger.error("전진 분석 결과가 없습니다.")
            return

        final_history = pd.concat(self.out_of_sample_results).drop_duplicates()
        final_stats = BacktestRunner(self.initial_balance, pd.DataFrame(), self.config)._analyze_results(final_history.reset_index().to_dict('records'))

        logger.info(f"\n{'='*80}\n ** 최종 전진 분석(Walk-Forward) 결과 **\n{'='*80}")
        logger.info(f"전체 기간: {self.start_date.date()} ~ {self.end_date.date()}")
        logger.info(f"초기 자본: ${self.initial_balance:,.2f} | 최종 자산: ${final_stats['final_value']:,.2f}")
        logger.info(f"총 수익률: {final_stats['total_return']:.2f}% | 최대 낙폭 (MDD): {final_stats['mdd']:.2f}%")
        logger.info("="*80)

        self.plot_results(final_stats['history'])

    def plot_results(self, history_df):
        if history_df is None or history_df.empty: return
        history_df['peak'] = history_df['portfolio_value'].cummax()
        history_df['drawdown'] = (history_df['portfolio_value'] - history_df['peak']) / history_df['peak']

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


class Optimizer:
    """주어진 기간 내에서 최적의 전략 파라미터를 찾는 클래스"""
    def __init__(self, initial_balance: float, config: TradingConfig, historical_data: pd.DataFrame):
        self.initial_balance = initial_balance
        self.config = config
        self.historical_data = historical_data
        self.optimization_results = []

    def _generate_param_space(self, params_config: dict) -> dict:
        param_values = {}
        for name, config in params_config.items():
            param_values[name] = list(np.arange(config['min'], config['max'] + config['step'], config['step']))
        return param_values

    def _generate_jobs(self) -> list:
        """최적화를 위한 모든 테스트 '잡(Job)' 생성 (매수/매도 조합)"""
        jobs = []
        cfg = self.config.OPTIMIZATION_CONFIG

        buy_indicator_names = list(cfg['buy_indicators'].keys())
        buy_combos = [c for i in range(1, len(buy_indicator_names) + 1) for c in itertools.combinations(buy_indicator_names, i)]

        sell_indicator_names = list(cfg['sell_indicators'].keys())
        sell_combos = [c for i in range(1, len(sell_indicator_names) + 1) for c in itertools.combinations(sell_indicator_names, i)]

        for buy_combo in buy_combos:
            for sell_combo in sell_combos:
                param_names, param_value_lists = [], []

                # 매수 파라미터 추가
                for ind_name in buy_combo:
                    space = self._generate_param_space(cfg['buy_indicators'][ind_name])
                    for p_name, p_values in space.items():
                        param_names.append(p_name)
                        param_value_lists.append(p_values)

                # 매도 파라미터 추가
                for ind_name in sell_combo:
                    space = self._generate_param_space(cfg['sell_indicators'][ind_name])
                    for p_name, p_values in space.items():
                        param_names.append(p_name)
                        param_value_lists.append(p_values)

                # 트리거 값 범위 추가
                buy_trigger_space = self._generate_param_space({'buy_trigger_threshold': cfg['buy_trigger_threshold']})
                param_names.append('buy_trigger_threshold')
                param_value_lists.append(buy_trigger_space['buy_trigger_threshold'])

                sell_trigger_space = self._generate_param_space({'sell_trigger_threshold': cfg['sell_trigger_threshold']})
                param_names.append('sell_trigger_threshold')
                param_value_lists.append(sell_trigger_space['sell_trigger_threshold'])

                # 중복 파라미터 조합 제거를 위해 set 사용 후 다시 list로 변환
                if len(param_names) != len(set(param_names)):
                    # 이 경우는 config 설정 오류이므로 건너뛰거나 경고
                    # 지금은 단순화를 위해 그대로 진행 (고유 파라미터 이름 가정)
                    pass

                for param_values in itertools.product(*param_value_lists):
                    params = dict(zip(param_names, param_values))
                    params['weights'] = {**cfg['buy_signal_weights'], **cfg['sell_signal_weights']}
                    jobs.append({'buy_indicator_combo': buy_combo, 'sell_indicator_combo': sell_combo, 'params': params})
        return jobs

    def find_best_strategy(self) -> dict:
        jobs = self._generate_jobs()
        if not jobs:
            logger.warning("생성된 최적화 작업이 없습니다.")
            return None

        runner = BacktestRunner(self.initial_balance, self.historical_data, self.config)
        for job in tqdm(jobs, desc="Finding Best Params", leave=False, ncols=100):
            result = runner.run(job)
            self.optimization_results.append({**job, **result})

        if not self.optimization_results:
            return None

        results_df = pd.DataFrame(self.optimization_results)
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

        current_start = self.start_date
        total_balance = self.initial_balance

        while current_start + relativedelta(months=wfc['training_period_months']) < self.end_date:
            train_start, train_end = current_start, current_start + relativedelta(months=wfc['training_period_months'])
            test_end = train_end + relativedelta(months=wfc['testing_period_months'])
            if test_end > self.end_date: test_end = self.end_date

            logger.info(f"\n{'='*80}\n전진 분석 구간: 훈련 [{train_start.date()}-{train_end.date()} | 검증 [{train_end.date()}-{test_end.date()}]\n{'='*80}")

            train_data = self.all_historical_data[(self.all_historical_data['date'] >= train_start.date()) & (self.all_historical_data['date' < train_end.date())]
            optimizer = Optimizer(total_balance, self.config, train_data)
            best_strategy = optimizer.find_best_strategy()

            if best_strategy:
                buy_combo_str = ', '.join(best_strategy['buy_indicator_combo'])
                sell_combo_str = ', '.join(best_strategy['sell_indicator_combo'])
                logger.info(f"최적 전략 발견: Buy-({buy_combo_str}) | Sell-({sell_combo_str})")

                test_data = self.all_historical_data[(self.all_historical_data['date'] >= train_end.date()) & (self.all_historical_data['date'] < test_end.date())]
                runner = BacktestRunner(total_balance, test_data, self.config)
                test_result = runner.run(best_strategy)

                if test_result and test_result['history'] is not None:
                    self.out_of_sample_results.append(test_result['history')
                    total_balance = test_result['final_value']
                    logger.info(f"검증 구간 성과: 최종 자산 ${total_balance:,.2f} | 수익률 {test_result['total_return']:.2f}%")
            else:
                logger.warning("현 구간에서 유효한 전략을 찾지 못했습니다. 다음 구간으로 넘어갑니다.")

            current_start += relativedelta(months=wfc['testing_period_months'])

        self.report_final_results()

    def report_final_results(self):
        if not self.out_of_sample_results:
            logger.error("전진 분석 결과가 없습니다.")
            return

        final_history = pd.concat(self.out_of_sample_results).drop_duplicates()
        final_stats = BacktestRunner(self.initial_balance, pd.DataFrame(), self.config)._analyze_results(final_history.reset_index().to_dict('records'))

        logger.info(f"\n{'='*80}\n ** 최종 전진 분석(Walk-Forward) 결과 **\n{'='*80}")
        logger.info(f"전체 기간: {self.start_date.date()} ~ {self.end_date.date()}")
        logger.info(f"초기 자본: ${self.initial_balance:,.2f} | 최종 자산: ${final_stats['final_value']:,.2f}")
        logger.info(f"총 수익률: {final_stats['total_return']:.2f}% | 최대 낙폭 (MDD): {final_stats['mdd']:.2f}%")
        logger.info("="*80)

        self.plot_results(final_stats['history'])

    def plot_results(self, history_df):
        if history_df is None or history_df.empty: return
        history_df['peak'] = history_df['portfolio_value'].cummax()
        history_df['drawdown'] = (history_df['portfolio_value'] - history_df['peak']) / history_df['peak']

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
    sys.exit(main())

