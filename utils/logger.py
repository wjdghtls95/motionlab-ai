"""
MotionLab AI - 로깅 유틸리티
민감 정보 마스킹 및 로그 포맷 설정
"""
import logging
import re

from config.settings import get_settings

# 설정 로드
settings = get_settings()

def setup_logger(name: str = "motionlab-ai") -> logging.Logger:
    """
    로거 설정

    Args:
        name: 로거 이름

    Returns:
        logging.Logger: 설정된 로거
    """
    logger = logging.getLogger(name)

    # 이미 핸들러가 있으면 재설정하지 않음
    if logger.handlers:
        return logger

    # 로그 레벨 설정
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # 로그 포맷
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


def mask_sensitive(text: str) -> str:
    """
    민감 정보 마스킹

    Args:
        text: 원본 텍스트

    Returns:
        str: 마스킹된 텍스트

    Examples:
        >>> mask_sensitive("API Key: sk-proj-abc123")
        "API Key: sk-proj-***"

        >>> mask_sensitive("https://example.com?key=secret123")
        "https://example.com?key=***"
    """
    # OpenAI API 키 마스킹
    text = re.sub(
        r'(sk-[a-zA-Z0-9-]{10,})',
        lambda m: m.group(1)[:10] + "***",
        text
    )

    # URL의 query parameter 마스킹
    text = re.sub(
        r'([?&][a-zA-Z_]+)=([^&\s]+)',
        r'\1=***',
        text
    )

    # JWT 토큰 마스킹
    text = re.sub(
        r'(Bearer\s+)([a-zA-Z0-9._-]+)',
        r'\1***',
        text
    )

    return text


# 전역 로거 객체
logger = setup_logger()
