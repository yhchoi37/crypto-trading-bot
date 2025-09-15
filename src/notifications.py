# -*- coding: utf-8 -*-
"""
알림 시스템 (텔레그램/이메일)
"""
import requests
import logging
import smtplib
from email.mime.text import MIMEText
from config.settings import TradingConfig

logger = logging.getLogger(__name__)

class NotificationManager:
    """텔레그램, 이메일 알림 발신"""
    def __init__(self):
        self.config = TradingConfig()
        self.telegram_token = self.config.TELEGRAM_BOT_TOKEN
        self.telegram_chat_id = self.config.TELEGRAM_CHAT_ID
        self.email_settings = self.config.EMAIL_SETTINGS

    def send_alert(self, message: str, subject="알림"):
        """텔레그램 및 이메일로 알림 전송"""
        self._send_telegram(message)
        self._send_email(subject, message)

    def _send_telegram(self, text):
        """텔레그램으로 메시지 전송"""
        if not (self.telegram_token and self.telegram_chat_id):
            logger.warning("텔레그램 API 정보 미설정")
            return
        url = f'https://api.telegram.org/bot{self.telegram_token}/sendMessage'
        data = {"chat_id": self.telegram_chat_id, "text": text}
        try:
            requests.post(url, data=data, timeout=10)
        except Exception as e:
            logger.error(f"Telegram 전송 실패: {e}")

    def _send_email(self, subject, text):
        """이메일로 메시지 전송"""
        s = self.email_settings
        if not (s and s.get('smtp_server') and s.get('email') and s.get('password')):
            logger.warning("이메일 설정 미완료")
            return
        msg = MIMEText(text)
        msg['Subject'] = subject
        msg['From'] = s['email']
        msg['To'] = s.get('recipient', s['email'])
        try:
            server = smtplib.SMTP(s['smtp_server'], s.get('smtp_port', 587))
            server.starttls()
            server.login(s['email'], s['password'])
            server.sendmail(s['email'], [msg['To']], msg.as_string())
            server.quit()
        except Exception as e:
            logger.error(f"이메일 전송 에러: {e}")
