# -*- coding: utf-8 -*-
"""
메인 실행 스크립트 (실시간 거래)
"""
import os
import sys
import time
import logging
import argparse
from datetime import datetime
import pyupbit

from config.settings import TradingConfig
from src.trading_system import MultiCoinTradingSystem
from src.notifications import NotificationManager
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

# 최상단 로깅 설정 제거
class TradingBot:
    """메인 트레이딩 봇 클래스"""
    def __init__(self):
        self.config = TradingConfig()
        self.is_running = False
        self.trading_system = None
        self.notification_manager = None

    def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("🚀 트레이딩 시스템 초기화 중...")
            # 디렉토리 생성
            os.makedirs('logs', exist_ok=True)
            os.makedirs('data_cache', exist_ok=True)
            os.makedirs('backups', exist_ok=True)
            # 트레이딩 시스템 초기화
            self.trading_system = MultiCoinTradingSystem(
                initial_balance=self.config.INITIAL_BALANCE,
                config=self.config
            )
            # 포트폴리오 배분 설정
            self.trading_system.setup_portfolio_allocation(
                self.config.TARGET_ALLOCATION
            )
            # 알림 시스템 초기화
            self.notification_manager = NotificationManager(
                telegram_token=self.config.TELEGRAM_BOT_TOKEN,
                telegram_chat_id=self.config.TELEGRAM_CHAT_ID,
                email_settings=self.config.EMAIL_SETTINGS
            )
            self.setup_schedule()
            logger.info("✅ 시스템 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"❌ 시스템 초기화 실패: {e}", exc_info=True)
            return False

    def setup_schedule(self):
        """거래 스케줄 설정"""
        # 1시간마다 거래 사이클 실행
        schedule.every().hour.do(self.run_trading_cycle)
        # 매일 09:00 리밸런싱
        schedule.every().day.at("09:00").do(self.run_rebalancing)
        # 매일 18:00 일일 보고서 전송
        schedule.every().day.at("18:00").do(self.send_daily_report)
        # 매일 00:00 데이터 백업
        schedule.every().day.at("00:00").do(self.backup_data)
        logger.info("📅 거래 스케줄 설정 완료")

    def run_trading_cycle(self):
        """거래 사이클 실행"""
        try:
            logger.info("🔄 거래 사이클 시작")
            result = self.trading_system.run_trading_cycle()
            if result['active_signals']:
                message = f"🚨 거래 신호 알림\n활성 신호: {len(result['active_signals'])}개\n"
                for signal in result['active_signals'][:3]:
                    message += f"{signal['coin']}: {signal['decision']['action']} (강도: {signal['decision']['strength']:.2f})\n"
                self.notification_manager.send_alert(message, "TRADING_SIGNAL")
            logger.info(f"✅ 거래 사이클 완료 - 포트폴리오 가치: ${result['portfolio_value']:,.2f}")
        except Exception as e:
            logger.error(f"❌ 거래 사이클 실패: {e}", exc_info=True)
        if self.notification_manager:
                self.notification_manager.send_alert(f"거래 사이클 오류: {e}", "ERROR")

    def run_rebalancing(self):
        """포트폴리오 리밸런싱"""
        try:
            logger.info("⚖️ 포트폴리오 리밸런싱 실행")
            prices = self.trading_system.data_manager.get_coin_prices()
            self.trading_system.perform_rebalancing(prices)
            portfolio_value = self.trading_system.portfolio_manager.get_portfolio_value(prices)
            message = f"⚖️ 리밸런싱 완료\n포트폴리오 가치: ${portfolio_value:,.2f}"
            self.notification_manager.send_alert(message, "REBALANCING")
        except Exception as e:
            logger.error(f"❌ 리밸런싱 실패: {e}", exc_info=True)

    def send_daily_report(self):
        """일일 성과 리포트 전송"""
        try:
            prices = self.trading_system.data_manager.get_coin_prices()
            summary = self.trading_system.portfolio_manager.get_portfolio_summary(prices)
            metrics = summary['metrics']
            report = (f"📊 일일 리포트 - {datetime.now().strftime('%Y-%m-%d')}\n"
                      f"💼 포트폴리오 현황:\n"
                      f"- 총 가치: ${metrics['total_value']:,.2f}\n"
                      f"- 총 수익률: {metrics['total_return']:+.2f}%\n"
                      f"- 현금 잔고: ${metrics['cash_balance']:,.2f}\n"
                      f"📈 오늘 거래 수: {metrics['trades_today']}\n"
                      f"🏷️ 보유 포지션 수: {metrics['total_positions']}")
            self.notification_manager.send_alert(report, "DAILY_REPORT")
            logger.info("📧 일일 리포트 전송 완료")
        except Exception as e:
            logger.error(f"❌ 일일 리포트 생성 실패: {e}", exc_info=True)

    def backup_data(self):
        """데이터 백업"""
        try:
            filename = self.trading_system.portfolio_manager.export_trade_history(
                f"backups/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            logger.info(f"💾 데이터 백업 완료: {filename}")
        except Exception as e:
            logger.error(f"❌ 데이터 백업 실패: {e}", exc_info=True)

    def start(self):
        """봇 시작"""
        if not self.initialize():
            return False
        self.is_running = True
        
        mode = "모의 거래" if self.config.SIMULATION_MODE else "실거래"
        logger.info(f"🎯 트레이딩 봇 시작! ({mode} 모드)")
        
        self.notification_manager.send_alert(
            f"🚀 트레이딩 봇이 시작되었습니다! ({mode} 모드)\n"
            f"초기 자본: ${self.config.INITIAL_BALANCE:,.2f}",
            "BOT_START"
        )
        try:
            # 시작 시 한 번 즉시 실행
            self.run_trading_cycle()
            while self.is_running:
                schedule.run_pending()
                time.sleep(1) # CPU 사용량을 줄이기 위해 짧은 sleep 추가
        except KeyboardInterrupt:
            logger.info("⏹️ 사용자에 의해 중지되었습니다")
        except Exception as e:
            logger.critical(f"❌ 시스템 심각한 오류: {e}", exc_info=True)
            if self.notification_manager:
                self.notification_manager.send_alert(f"시스템 오류로 봇이 중지되었습니다: {e}", "ERROR")
        finally:
            self.stop()

    def stop(self):
        """봇 중지"""
        if self.is_running:
            self.is_running = False
            if self.notification_manager:
                self.notification_manager.send_alert("⏹️ 트레이딩 봇이 중지되었습니다", "BOT_STOP")
            logger.info("🛑 트레이딩 봇이 중지되었습니다")
            
def get_real_balance(config: TradingConfig) -> float:
    """거래소에서 실제 잔고 조회"""
    if config.SIMULATION_MODE:
        # 시뮬레이션 모드: 가상 초기 자본
        initial_balance = 10_000_000
        logger.info(f"💰 시뮬레이션 초기 자본: ₩{initial_balance:,}")
        return initial_balance
    else:
        # 실거래 모드: API로 실제 잔고 조회
        try:
            upbit = pyupbit.Upbit(config.UPBIT_ACCESS_KEY, config.UPBIT_SECRET_KEY)
            balances = upbit.get_balances()
            
            # KRW 잔고 확인
            krw_balance = 0
            for balance in balances:
                if balance['currency'] == 'KRW':
                    krw_balance = float(balance['balance'])
                    break
            
            logger.info(f"💰 실거래 현재 KRW 잔고: ₩{krw_balance:,}")
            
            if krw_balance < 10000:
                logger.warning("⚠️ 잔고가 10,000원 미만입니다!")
            
            return krw_balance
            
        except Exception as e:
            logger.error(f"❌ 잔고 조회 실패: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="암호화폐 자동매매 시스템")
    parser.add_argument('--mode', type=str, default=None,
                       choices=['simulation', 'live'],
                       help='실행 모드 (기본값: .env SIMULATION_MODE 사용)')
    args = parser.parse_args()

    """메인 함수"""
    # 1. TradingConfig 인스턴스를 먼저 생성
    # 설정 로드 (명령줄 인자로 오버라이드 가능)
    config = TradingConfig(force_mode=args.mode)

    # 2. 공통 로깅 함수를 호출하여 로거를 설정
    queue_listener = setup_logging(config.LOG_LEVEL, 'logs/trading.log', use_multiprocessing=False)
  
    logger.info("="*80)
    logger.info("🚀 암호화폐 자동매매 시스템 시작")
    logger.info(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    # 실거래 모드 경고
    if not config.is_paper_trading():
        logger.warning("⚠️ " * 20)
        logger.warning("⚠️  실거래 모드로 실행 중입니다!")
        logger.warning("⚠️  실제 자금이 투입됩니다!")
        logger.warning("⚠️ " * 20)
        
        # 5초 대기 (실수 방지)
        for i in range(5, 0, -1):
            logger.warning(f"⏳ {i}초 후 시작... (Ctrl+C로 중단 가능)")
            time.sleep(1)

    bot = TradingBot()
    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("⏹️ 사용자에 의해 중지됨")
    except Exception as e:
        logger.critical(f"❌ 봇 실행 중 치명적인 오류 발생: {e}", exc_info=True)
        # 봇이 초기화되었다면 stop 메서드 호출
        if bot and bot.is_running:
            bot.stop()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())

