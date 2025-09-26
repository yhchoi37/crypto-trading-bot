# -*- coding: utf-8 -*-
"""
다중 코인 + 소셜미디어 센티멘트 분석 자동거래 시스템
Multi-Coin Social Media Sentiment Trading Bot

Author: AI Trading System
Version: 2.0.0
"""
import os
import sys
import time
import schedule
from datetime import datetime
from config.settings import TradingConfig
from src.trading_system import MultiCoinTradingSystem
from src.notifications import NotificationManager
from src.logging_config import setup_logging # 공통 로깅 함수 임포트
import logging

# 최상단 로깅 설정 제거
class TradingBot:
    """메인 트레이딩 봇 클래스"""
    def __init__(self):
        self.config = TradingConfig()
        self.is_running = False
        self.trading_system = None
        self.notification_manager = None
        # self.logger를 main에서 설정된 로거를 사용하도록 변경
        self.logger = logging.getLogger(__name__)

    # setup_logging 메서드 전체 제거
    def initialize(self):
        """시스템 초기화"""
        try:
            self.logger.info("🚀 트레이딩 시스템 초기화 중...")
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
            self.logger.info("✅ 시스템 초기화 완료")
            return True
        except Exception as e:
            self.logger.error(f"❌ 시스템 초기화 실패: {e}", exc_info=True)
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
        self.logger.info("📅 거래 스케줄 설정 완료")

    def run_trading_cycle(self):
        """거래 사이클 실행"""
        try:
            self.logger.info("🔄 거래 사이클 시작")
            result = self.trading_system.run_trading_cycle()
            if result['active_signals']:
                message = f"🚨 거래 신호 알림\n활성 신호: {len(result['active_signals'])}개\n"
                for signal in result['active_signals'][:3]:
                    message += f"{signal['coin']}: {signal['decision']['action']} (강도: {signal['decision']['strength']:.2f})\n"
                self.notification_manager.send_alert(message, "TRADING_SIGNAL")
            self.logger.info(f"✅ 거래 사이클 완료 - 포트폴리오 가치: ${result['portfolio_value']:,.2f}")
        except Exception as e:
            self.logger.error(f"❌ 거래 사이클 실패: {e}", exc_info=True)
        if self.notification_manager:
                self.notification_manager.send_alert(f"거래 사이클 오류: {e}", "ERROR")

    def run_rebalancing(self):
        """포트폴리오 리밸런싱"""
        try:
            self.logger.info("⚖️ 포트폴리오 리밸런싱 실행")
            prices = self.trading_system.data_manager.get_coin_prices()
            self.trading_system.perform_rebalancing(prices)
            portfolio_value = self.trading_system.portfolio_manager.get_portfolio_value(prices)
            message = f"⚖️ 리밸런싱 완료\n포트폴리오 가치: ${portfolio_value:,.2f}"
            self.notification_manager.send_alert(message, "REBALANCING")
        except Exception as e:
            self.logger.error(f"❌ 리밸런싱 실패: {e}", exc_info=True)

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
            self.logger.info("📧 일일 리포트 전송 완료")
        except Exception as e:
            self.logger.error(f"❌ 일일 리포트 생성 실패: {e}", exc_info=True)

    def backup_data(self):
        """데이터 백업"""
        try:
            filename = self.trading_system.portfolio_manager.export_trade_history(
                f"backups/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            self.logger.info(f"💾 데이터 백업 완료: {filename}")
        except Exception as e:
            self.logger.error(f"❌ 데이터 백업 실패: {e}", exc_info=True)

    def start(self):
        """봇 시작"""
        if not self.initialize():
            return False
        self.is_running = True
        
        mode = "모의 거래" if self.config.SIMULATION_MODE else "실거래"
        self.logger.info(f"🎯 트레이딩 봇 시작! ({mode} 모드)")
        
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
            self.logger.info("⏹️ 사용자에 의해 중지되었습니다")
        except Exception as e:
            self.logger.critical(f"❌ 시스템 심각한 오류: {e}", exc_info=True)
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
            self.logger.info("🛑 트레이딩 봇이 중지되었습니다")

def main():
    """메인 함수"""
    # 1. TradingConfig 인스턴스를 먼저 생성
    config = TradingConfig()

    # 2. 공통 로깅 함수를 호출하여 로거를 설정
    setup_logging(config.LOG_LEVEL, 'trading.log')

    # 이제부터 로깅 사용 가능
    logger = logging.getLogger(__name__)

    print("🚀 다중 코인 + 소셜미디어 센티멘트 분석 트레이딩 봇")
    print("=" * 60)

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

