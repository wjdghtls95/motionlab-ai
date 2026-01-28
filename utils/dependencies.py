"""
FastAPI Dependencies
재사용 가능한 의존성 함수
"""
from fastapi import Header, HTTPException
from typing import Annotated

from config.settings import get_settings
from utils.logger import logger, mask_sensitive

settings = get_settings()


async def verify_api_key(
        x_internal_api_key: Annotated[str, Header()] = None
) -> None:
    """
    내부 API 키 검증 (의존성 주입용)

    Args:
        x_internal_api_key: 요청 헤더의 API 키 (X-Internal-API-Key)

    Raises:
        HTTPException: API 키가 유효하지 않을 경우 401
    """
    if not x_internal_api_key:
        logger.warning("API 키가 제공되지 않음")
        raise HTTPException(401, detail="Missing API Key")

    if x_internal_api_key != settings.INTERNAL_API_KEY:
        logger.warning(f"유효하지 않은 API 키: {mask_sensitive(x_internal_api_key)}")
        raise HTTPException(401, detail="Invalid API Key")

    logger.debug("API 키 검증 성공")
