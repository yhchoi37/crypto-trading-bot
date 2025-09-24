# -*- coding: utf-8 -*-
"""
프로젝트 공통 로깅 설정 모듈
"""
import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler # 핸들러 임포트

def setup_logging(log_level: str, log_filename: str):
    """
    루트 로거를 설정하여 프로젝트 전반의 로깅을 구성합니다.
    
    :param log_level: 설정할 로그 레벨 (e.g., 'DEBUG', 'INFO')
    :param log_filename: 로그를 저장할 파일 이름
    """
    # 로그 디렉터리 생성
    os.makedirs('logs', exist_ok=True)
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 루트 로거 가져오기
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # 기존 핸들러 제거 (중복 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 파일 핸들러를 TimedRotatingFileHandler로 변경
    # when='D': 매일(Day) 파일을 교체
    # backupCount=30: 30개의 백업 파일 유지
    file_handler = TimedRotatingFileHandler(
        os.path.join('logs', log_filename),
        when='D',
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 콘솔(Stream) 핸들러 추가
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    
    logging.info(f"로거가 '{log_level}' 레벨로 설정되었습니다. 로그 파일: 'logs/{log_filename}'")
    logging.debug("디버그 레벨 로깅이 활성화되었습니다.")
