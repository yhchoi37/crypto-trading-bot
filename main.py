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
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    """메인 트레이딩 봇 클래스"""
    def __init__(self):
        self.config = TradingConfig()
        self.is_running = False
        self.trading_system = None
        self.notification_manager = None
        self.logger = None
        self.setup_logging()

    def setup_logging(self):
        numeric_level = getattr(logging, self.config.LOG_LEVEL, logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"로그 레벨 설정: {self.config.LOG_LEVEL}")

    def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("🚀 트레이딩 시스템 초기화 중...")
            # 디렉토리 생성
            os.makedirs('logs', exist_ok=True)
            os.makedirs('data', exist_ok=True)
            os.makedirs('backups', exist_ok=True)
            # 트레이딩 시스템 초기화
            self.trading_system = MultiCoinTradingSystem(
                initial_balance=self.config.INITIAL_BALANCE
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
            logger.error(f"❌ 시스템 초기화 실패: {e}")
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
            logger.error(f"❌ 거래 사이클 실패: {e}")
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
            logger.error(f"❌ 리밸런싱 실패: {e}")

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
            logger.error(f"❌ 일일 리포트 생성 실패: {e}")

    def backup_data(self):
        """데이터 백업"""
        try:
            filename = self.trading_system.portfolio_manager.export_trade_history(
                f"backups/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            logger.info(f"💾 데이터 백업 완료: {filename}")
        except Exception as e:
            logger.error(f"❌ 데이터 백업 실패: {e}")

    def start(self):
        """봇 시작"""
        if not self.initialize():
            return False
        self.is_running = True
        logger.info("🎯 트레이딩 봇 시작!")
        self.notification_manager.send_alert(
            f"🚀 트레이딩 봇이 시작되었습니다!\n초기 자본: ${self.config.INITIAL_BALANCE:,.2f}",
            "BOT_START"
        )
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(self.config.DATA_COLLECTION_INTERVAL)
        except KeyboardInterrupt:
            logger.info("⏹️ 사용자에 의해 중지되었습니다")
            self.stop()
        except Exception as e:
            logger.error(f"❌ 시스템 오류: {e}")
            self.notification_manager.send_alert(f"시스템 오류로 봇이 중지되었습니다: {e}", "ERROR")
            self.stop()

    def stop(self):
        """봇 중지"""
        self.is_running = False
        if self.notification_manager:
            self.notification_manager.send_alert("⏹️ 트레이딩 봇이 중지되었습니다", "BOT_STOP")
        logger.info("🛑 트레이딩 봇이 중지되었습니다")

def main():
    """메인 함수"""
    print("🚀 다중 코인 + 소셜미디어 센티멘트 분석 트레이딩 봇")
    print("=" * 60)
    bot = TradingBot()
    try:
        bot.start()
    except Exception as e:
        logger.error(f"❌ 봇 실행 중 오류: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())

