"""
FastAPI Dependencies
재사용 가능한 의존성 함수
"""

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Annotated
from datetime import datetime, timezone

from config.settings import get_settings
from utils.exceptions.errors import AnalyzerError
from utils.logger import logger, mask_sensitive

settings = get_settings()


async def verify_api_key(x_internal_api_key: Annotated[str, Header()] = None) -> None:
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


# Exception Handler 등록 (신규)
def register_exception_handlers(app: FastAPI):
    """
    Exception Handler 등록
    """

    # 1. AnalyzerError (커스텀 도메인 에러)
    @app.exception_handler(AnalyzerError)
    async def handle_analyzer_error(request: Request, exc: AnalyzerError):
        """
        모든 AnalyzerError 처리 (AN_XXX, SYS_XXX)

        자동으로 호출되는 경우:
        - services/analysis_service.py에서 raise VideoNotFoundError(...)
        - core/llm_feedback.py에서 raise LLMTimeoutError()
        - 등등
        """

        logger.warning(f"⚠️ 도메인 에러: {exc.error_code} - {exc.message}")

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "data": None,
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": str(request.url),
            },
        )

    # 2. Pydantic 검증 에러 (422)
    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        """
        Pydantic 검증 실패 시 자동 호출

        예: POST /api/analyze { "motion_id": "abc" }  ← 숫자여야 하는데 문자열
        """

        first_error = exc.errors()[0]
        field = " -> ".join(str(loc) for loc in first_error["loc"])

        logger.warning(f"⚠️ 검증 에러: {field} - {first_error['msg']}")

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "data": None,
                "error": {
                    "code": "VAL_000",
                    "message": f"잘못된 요청 데이터: {field}",
                    "retryable": False,
                    "details": {
                        "field": field,
                        "reason": first_error["msg"],
                        "all_errors": exc.errors(),
                    },
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": str(request.url),
            },
        )

    # 3. HTTPException (FastAPI 기본)
    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException):
        """
        FastAPI HTTPException 처리 (verify_api_key 등에서 발생)
        """

        logger.warning(f"⚠️ HTTP 에러: {exc.status_code} - {exc.detail}")

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "data": None,
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "retryable": False,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": str(request.url),
            },
        )

    # 4. 예상치 못한 에러 (500)
    @app.exception_handler(Exception)
    async def handle_unknown_error(request: Request, exc: Exception):
        """
        모든 예상치 못한 에러 (최후의 방어선)

        예: ZeroDivisionError, AttributeError 등
        """

        logger.error(
            f"❌ 시스템 에러: {type(exc).__name__} - {str(exc)}",
            exc_info=True,  # 스택 트레이스 포함
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "data": None,
                "error": {
                    "code": "SYS_000",
                    "message": "내부 서버 오류가 발생했습니다",
                    "retryable": True,
                    "details": (
                        {"type": type(exc).__name__, "message": str(exc)}
                        if logger.level == 10
                        else {}
                    ),  # DEBUG 레벨일 때만
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": str(request.url),
            },
        )
