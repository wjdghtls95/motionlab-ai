"""
MotionLab AI - FastAPI 서버 진입점
라우터 등록 및 CORS 설정
"""

import uvicorn
import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path

from api import health, analyze, admin
from config.settings import get_settings
from utils.logger import logger, mask_sensitive

settings = get_settings()

# Sentry 초기화 (DSN 설정 시에만 활성화)
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.APP_ENV,
        server_name="motionlab-ai",
    )
    logger.info("✅ Sentry 초기화 완료")


def _cleanup_stale_temp_files(temp_dir: str) -> None:
    """서버 시작 시 이전 실행에서 정리되지 않은 임시 영상 파일을 삭제."""
    temp_path = Path(temp_dir)
    if not temp_path.exists():
        return

    stale_files = list(temp_path.glob("*.mp4"))
    if not stale_files:
        return

    deleted, failed = 0, 0
    for f in stale_files:
        try:
            f.unlink()
            deleted += 1
        except Exception as e:
            failed += 1
            logger.error(f"⚠️ 잔여 임시 파일 삭제 실패: {f} — {e}")

    logger.warning(
        f"🗑️ 잔여 임시 파일 정리: 삭제 {deleted}개 / 실패 {failed}개 "
        f"(이전 실행에서 cleanup 누락된 파일)"
    )
    if failed > 0 and settings.SENTRY_DSN:
        sentry_sdk.capture_message(
            f"Startup: {failed}개 잔여 임시 파일 삭제 실패",
            level="warning",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan 이벤트 핸들러

    - yield 전: 서버 시작 시 실행 (startup)
    - yield 후: 서버 종료 시 실행 (shutdown)
    """
    # ========== Startup ==========
    _cleanup_stale_temp_files(settings.TEMP_VIDEO_DIR)
    logger.info("=" * 60)
    logger.info("MotionLab AI Server 시작")
    logger.info(f"포트: {settings.PORT}")
    logger.info(f"호스트: {settings.HOST}")
    logger.info(f"LLM Noop 모드: {settings.ENABLE_LLM_NOOP}")
    logger.info(f"MediaPipe 모델 복잡도: {settings.MEDIAPIPE_MODEL_COMPLEXITY}")
    logger.info(f"API Key: {mask_sensitive(settings.INTERNAL_API_KEY)}")
    logger.info("=" * 60)

    yield  # 여기서 앱 실행 중

    # ========== Shutdown (필요 시) ==========
    logger.info("=" * 60)
    logger.info("MotionLab AI Server 종료")
    logger.info("=" * 60)


# FastAPI 앱 생성
_is_development = settings.APP_ENV == "development"
app = FastAPI(
    title="MotionLab AI Server",
    description="MediaPipe + GPT-4o-mini 기반 운동 분석 서버",
    version="1.0.0",
    docs_url="/docs" if _is_development else None,  # 프로덕션 Swagger 비공개
    redoc_url="/redoc" if _is_development else None,  # 프로덕션 ReDoc 비공개
    openapi_tags=[
        {"name": "Health", "description": "서버 상태 확인"},
        {"name": "Analysis", "description": "운동 영상 분석"},
    ],
    lifespan=lifespan,  # lifespan 이벤트 핸들러 연결
)

# CORS 설정
# 개발 환경: 와일드카드(로컬 테스트 편의)
# 운영 환경: ALLOWED_ORIGINS 환경변수에 명시된 출처만 허용
_allowed_origins = ["*"] if _is_development else settings.ALLOWED_ORIGINS.split(",")
_allowed_methods = ["*"] if _is_development else ["GET", "POST"]
_allowed_headers = ["*"] if _is_development else ["X-Internal-API-Key", "Content-Type"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=_allowed_methods,
    allow_headers=_allowed_headers,
)

# 라우터 등록
app.include_router(health.router, tags=["Health"])
app.include_router(analyze.router, tags=["Analysis"])
app.include_router(admin.router, tags=["Admin"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
