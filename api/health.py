"""
Health Check API
서버 상태 및 설정 정보 반환
"""
from fastapi import APIRouter
import sys

from models.responses.health_response import HealthResponse
from config.settings import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/", response_model=dict)
async def root():
    """루트 엔드포인트"""
    return {
        "service": "MotionLab AI Server",
        "version": "1.0.0",
        "status": "running"
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    헬스체크 엔드포인트

    Returns:
        HealthResponse: 서버 상태 정보
    """
    return HealthResponse(
        status="healthy",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        llm_noop_mode=settings.ENABLE_LLM_NOOP,
        mediapipe_model_complexity=settings.MEDIAPIPE_MODEL_COMPLEXITY,
        max_video_size_mb=settings.MAX_VIDEO_SIZE_MB
    )
