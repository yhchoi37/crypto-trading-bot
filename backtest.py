# -*- coding: utf-8 -*-
"""
기술적 분석 전략 최적화 실행 스크립트 (Grid Search)
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

# 프로젝트 루트를 sys.path에 추가하여 src 모듈 임포트
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
        data_by_date = {date: group for date, group in self.historical_data.groupby('date')}
        target_allocations = self.config.TARGET_ALLOCATION

        for current_date in sorted(data_by_date.keys()):
            current_prices = {row['coin']: row['close'] for _, row in data_by_date[current_date.iterrows()}
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
                        # 보유 수량의 50%를 매도
                        quantity_to_sell = position['quantity'] * 0.5
                        trading_system.portfolio_manager.execute_trade(coin, 'SELL', quantity_to_sell, price)

            portfolio_value = trading_system.portfolio_manager.get_portfolio_value(current_prices)
            portfolio_history.append({'date': current_date, 'portfolio_value': portfolio_value})

        return self._analyze_results(portfolio_history)

    def _analyze_results(self, portfolio_history: list) -> dict:
        if not portfolio_history:
            return {'total_return': -100, 'mdd': -100, 'final_value': 0, 'history': None}

        results_df = pd.DataFrame(portfolio_history).set_index('date')
        final_value = results_df['portfolio_value'.iloc[-1]
        total_return = (final_value / self.initial_balance - 1) * 100
        peak = results_df['portfolio_value'].cummax()
        drawdown = (results_df['portfolio_value'] - peak) / peak
        mdd = drawdown.min() * 100

        return {'total_return': total_return, 'mdd': mdd, 'final_value': final_value, 'history': results_df}


class Optimizer:
    """전략 파라미터 최적화를 총괄하는 클래스"""
    def __init__(self, start_date: str, end_date: str, initial_balance: float):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        
        # 백테스트 모드 활성화
        os.environ['BACKTEST_MODE'] = 'true'
        os.environ['INITIAL_BALANCE'] = str(initial_balance)
        
        self.config = TradingConfig()
        self.data_manager = MultiCoinDataManager()
        self.historical_data = None
        self.optimization_results = []

    def _generate_param_space(self, params_config: dict) -> dict:
        """min, max, step 설정으로부터 파라미터 값 리스트 생성"""
        param_values = {}
        for name, config in params_config.items():
            param_values[name] = list(np.arange(config['min'], config['max'] + config['step'], config['step']))
        return param_values

    def _generate_jobs(self) -> list:
        """최적화를 위한 모든 테스트 '잡(Job)' 생성 (매수/매도 조합)"""
        jobs = []
        cfg = self.config.OPTIMIZATION_CONFIG

        buy_indicator_names = list(cfg['buy_indicators'].keys())
        buy_combos = []
        for i in range(1, len(buy_indicator_names) + 1):
            buy_combos.extend(itertools.combinations(buy_indicator_names, i))

        sell_indicator_names = list(cfg['sell_indicators'].keys())
        sell_combos = []
        for i in range(1, len(sell_indicator_names) + 1):
            sell_combos.extend(itertools.combinations(sell_indicator_names, i))

        for buy_combo in buy_combos:
            for sell_combo in sell_combos:
                param_names = []
                param_value_lists = []

                # 매수/매도 파라미터 및 트리거 값 추가
                param_configs = {**cfg['buy_indicators'], **cfg['sell_indicators']}
                unique_params = {}
                for ind_name in buy_combo + sell_combo:
                    for p_name, p_config in param_configs[ind_name].items():
                        unique_params[p_name] = p_config

                space = self._generate_param_space(unique_params)
                    for p_name, p_values in space.items():
                           param_names.append(p_name)
                           param_value_lists.append(p_values)

                buy_trigger_space = self._generate_param_space({'buy_trigger_threshold': cfg['buy_trigger_threshold']})
                param_names.append('buy_trigger_threshold')
                param_value_lists.append(buy_trigger_space['buy_trigger_threshold')

                sell_trigger_space = self._generate_param_space({'sell_trigger_threshold': cfg['sell_trigger_threshold']})
                param_names.append('sell_trigger_threshold')
                param_value_lists.append(sell_trigger_space['sell_trigger_threshold')

                for param_values in itertools.product(*param_value_lists):
                    params = dict(zip(param_names, param_values))
                    params['weights'] = {**cfg['buy_signal_weights'], **cfg['sell_signal_weights'}
                    jobs.append({
                        'buy_indicator_combo': buy_combo,
                        'sell_indicator_combo': sell_combo,
                        'params': params
                    })
        return jobs

    def run_optimization(self):
        """최적화 프로세스 실행"""
        # 데이터 로드
        coins = [c for c in self.config.TARGET_ALLOCATION if c != 'CASH']
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
        self.historical_data = self.data_manager.get_historical_data_for_backtest(
            coins, self.start_date, self.end_date
        )
        if self.historical_data.empty:
            logger.error("데이터 로드 실패. 최적화를 중단합니다.")
            return

        self.historical_data['date' = pd.to_datetime(self.historical_data['index']).dt.date

        jobs = self._generate_jobs()
        logger.info(f"총 {len(jobs)}개의 전략 조합으로 최적화를 시작합니다.")

        runner = BacktestRunner(self.initial_balance, self.historical_data, self.config)

        # tqdm으로 진행률 표시
        for job in tqdm(jobs, desc="Optimizing Strategies"):
            result = runner.run(job)
            self.optimization_results.append({**job, **result})

        self.report_results()

    def report_results(self):
        """최적화 결과 리포트 및 시각화"""
        if not self.optimization_results:
            logger.warning("분석할 최적화 결과가 없습니다.")
            return

        results_df = pd.DataFrame(self.optimization_results)
        # params와 combo를 문자열로 변환
        results_df['params_str'] = results_df['params'].astype(str)
        results_df['buy_combo_str'] = results_df['buy_indicator_combo'].astype(str)
        results_df['sell_combo_str'] = results_df['sell_indicator_combo'].astype(str)

        results_df = results_df.sort_values(by='final_value', ascending=False).reset_index(drop=True)

        # 결과를 CSV 파일로 저장
        results_df.drop(columns=['history']).to_csv('optimization_results.csv', index=False, encoding='utf-8-sig')
        logger.info("전체 최적화 결과가 'optimization_results.csv' 파일로 저장되었습니다.")

        print("\n" + "="*150)
        print(" ** 전략 최적화 결과 리포트 (상위 10개) **")
        print("="*150)
        print(f"{'Rank':<5} {'Return (%)':<12} {'MDD (%)':<12} {'Final Value ($)':<18} {'Buy Combo':<25} {'Sell Combo':<25} {'Parameters'}")
        print("-"*150)

        for i, row in results_df.head(10).iterrows():
            print(
                f"{i+1:<5} "
                f"{row['total_return']:<12.2f} "
                f"{row['mdd']:<12.2f} "
                f"{row['final_value']:<18,.2f} "
                f"{row['buy_combo_str']:<25} "
                f"{row['sell_combo_str':<25} "
                f"{row['params_str'}"
            )

        print("="*150)

        if results_df.empty:
            logger.warning("유효한 최적화 결과를 찾지 못했습니다.")
            return

        best_result = results_df.iloc[0]

        logger.info(f"🏆 최적 매수 전략: {best_result['buy_combo_str']}")
        logger.info(f"🏆 최적 매도 전략: {best_result['sell_combo_str']}")
        logger.info(f"   - 파라미터: {best_result['params_str']}")
        # 최적 전략으로 그래프 생성
        logger.info("최적 전략의 성과 그래프를 'best_strategy_performance.png' 파일로 저장합니다...")
        self.plot_best_strategy(best_result)

    def plot_best_strategy(self, best_result):
        """최고의 전략 성과를 그래프로 저장"""
        history_df = best_result.get('history')
        if history_df is None or history_df.empty:
            logger.warning("그래프를 그릴 포트폴리오 히스토리 데이터가 없습니다.")
            return

        # MDD 계산
        history_df['peak'] = history_df['portfolio_value'].cummax()
        history_df['drawdown'] = (history_df['portfolio_value'] - history_df['peak']) / history_df['peak']

        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        fig.suptitle('Best Strategy Performance', fontsize=16)

        # 포트폴리오 가치 그래프
        ax1.plot(history_df.index, history_df['portfolio_value', label='Portfolio Value', color='blue')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Portfolio Value Over Time')
        ax1.grid(True)
        ax1.legend()

        # Drawdown 그래프
        ax2.fill_between(history_df.index, history_df['drawdown'] * 100, 0, color='red', alpha=0.3)
        ax2.plot(history_df.index, history_df['drawdown'] * 100, label='Drawdown', color='red')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Drawdown Over Time')
        ax2.grid(True)
        ax2.legend()

        plt.xlabel('Date')

        plt.tight_layout(rect=[0, 0, 1, 0.96)
    try:
            plt.savefig('best_strategy_performance.png', dpi=300)
            logger.info("성과 그래프가 'best_strategy_performance.png'에 성공적으로 저장되었습니다.")
    except Exception as e:
            logger.error(f"그래프 저장 중 오류 발생: {e}")
        finally:
            plt.close(fig)


def main():
    """최적화 실행"""
    logger.info("🚀 전략 최적화 시스템 시작")
    
    # 최적화 설정
    START_DATE = "2023-01-01"
    END_DATE = "2023-12-31"
    INITIAL_BALANCE = 100000.0

    try:
        optimizer = Optimizer(START_DATE, END_DATE, INITIAL_BALANCE)
        optimizer.run_optimization()
    except Exception as e:
        logger.error(f"❌ 최적화 중 오류 발생: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    # 백테스팅 중 matplotlib 백엔드 설정 (GUI가 없는 환경 대비)
    import matplotlib
    matplotlib.use('Agg')
    
    sys.exit(main())

