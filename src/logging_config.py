# -*- coding: utf-8 -*-
"""
프로젝트 공통 로깅 설정 모듈
"""
import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler, QueueHandler, QueueListener
import multiprocessing
from tqdm import tqdm

class TqdmLoggingHandler(logging.StreamHandler):
    """tqdm 진행률 표시줄과 호환되는 로깅 핸들러"""
    def emit(self, record):
        try:
            msg = self.format(record)
            # tqdm.write()를 사용하여 로그 메시지를 출력하면
            # 진행률 표시줄이 깨지는 것을 방지할 수 있습니다.
            tqdm.write(msg, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

def setup_logging(level='INFO', log_to_file=None, use_multiprocessing=False):
    """
    중앙 로깅 설정 함수 (리팩토링 버전).
    코드 중복을 줄이고 가독성을 향상시키면서 기존 기능은 모두 보존합니다.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    log_format = logging.Formatter(
        '%(asctime)s - [%(process)d] - %(name)s - %(levelname)s - %(message)s'
    )

    # 1. 핸들러 목록을 준비합니다.
    handlers = []

    # 콘솔 핸들러를 TqdmLoggingHandler로 교체합니다.
    console_handler = TqdmLoggingHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    handlers.append(console_handler)
        
    # log_to_file 인자가 주어졌을 때만 파일 핸들러를 추가합니다.
    if log_to_file:
        log_dir = os.path.dirname(log_to_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = TimedRotatingFileHandler(
            log_to_file, when='D', backupCount=30, encoding='utf-8'
        )
        file_handler.setFormatter(log_format)
        handlers.append(file_handler)

    # 2. 루트 로거를 설정합니다. (기존 핸들러 제거 포함)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    listener = None # 멀티프로세싱 리스너를 반환하기 위한 변수

    # 3. 모드에 따라 핸들러를 연결합니다.
    if use_multiprocessing:
        # 멀티프로세싱: QueueHandler를 루트 로거에 연결하고,
        # 실제 핸들러(콘솔, 파일)는 QueueListener가 관리합니다.
        log_queue = multiprocessing.Queue(-1)
        queue_handler = QueueHandler(log_queue)
        root_logger.addHandler(queue_handler)

        listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
        listener.start()
        logging.info(f"멀티프로세싱 로거가 '{level.upper()}' 레벨로 설정되었습니다.")

    else:
        # 단일 프로세스: 준비된 모든 핸들러를 루트 로거에 직접 연결합니다.
        for handler in handlers:
            root_logger.addHandler(handler)

        logging.info(f"로거가 '{level.upper()}' 레벨로 설정되었습니다.")
        # 이 디버그 메시지는 단일 프로세스 모드에서만 의미가 있으므로 여기에 둡니다.
        logging.debug("디버그 레벨 로깅이 활성화되었습니다.")

    # 파일 로깅 정보는 모드에 관계없이 공통으로 출력합니다.
    if log_to_file:
        logging.info(f"로그 파일: '{log_to_file}'")

    return listener

