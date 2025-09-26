# -*- coding: utf-8 -*-
"""
프로젝트 공통 로깅 설정 모듈
"""
import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler # 핸들러 임포트

def setup_logging(level='INFO', log_to_file=None):
    """중앙 로깅 설정 함수"""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 프로세스 ID를 포함하는 새로운 포맷
    log_format = logging.Formatter(
        '%(asctime)s - [%(process)d] - %(name)s - %(levelname)s - %(message)s'
    )

    # 루트 로거 설정
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # 기존 핸들러 제거 (중복 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 콘솔(Stream) 핸들러 추가
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)
    
    # 파일 핸들러 추가 (log_to_file이 지정된 경우)
    if log_to_file:
        # 로그 파일의 디렉터리가 존재하지 않으면 생성
        log_dir = os.path.dirname(log_to_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 파일 핸들러를 TimedRotatingFileHandler로 변경
        # when='D': 매일(Day) 파일을 교체
        # backupCount=30: 30개의 백업 파일 유지
        file_handler = TimedRotatingFileHandler(
            log_to_file,
            when='D',
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    logging.info(f"로거가 '{level.upper()}' 레벨로 설정되었습니다.")
    if log_to_file:
        logging.info(f"로그 파일: '{log_to_file}'")
    logging.debug("디버그 레벨 로깅이 활성화되었습니다.")

