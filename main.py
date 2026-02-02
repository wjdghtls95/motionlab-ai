"""
MotionLab AI - FastAPI 서버 진입점
라우터 등록 및 CORS 설정
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api import health, analyze
from config.settings import get_settings
from utils.logger import logger, mask_sensitive

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan 이벤트 핸들러

    - yield 전: 서버 시작 시 실행 (startup)
    - yield 후: 서버 종료 시 실행 (shutdown)
    """
    # ========== Startup ==========
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
app = FastAPI(
    title="MotionLab AI Server",
    description="MediaPipe + GPT-4o-mini 기반 운동 분석 서버",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc
    openapi_tags=[
        {"name": "Health", "description": "서버 상태 확인"},
        {"name": "Analysis", "description": "운동 영상 분석"}
    ],
    lifespan=lifespan  # lifespan 이벤트 핸들러 연결
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(health.router, tags=["Health"])
app.include_router(analyze.router, tags=["Analysis"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
