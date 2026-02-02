"""
헬스체크 API

프롬프트 A22 규칙:
- api/: HTTP 요청/응답 처리만
- MediaPipe 로드 여부 확인
"""
from fastapi import APIRouter
from models import HealthResponse
import logging

router = APIRouter(prefix="", tags=["Health"])
logger = logging.getLogger(__name__)

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="서버 상태 확인",
    description="서버 헬스체크 및 MediaPipe 로드 확인"
)
async def health_check():
    """
    헬스체크 엔드포인트

    Returns:
        HealthResponse: {status, version, mediapipe_available}
    """
    # MediaPipe 로드 확인
    mediapipe_ok = True
    try:
        import mediapipe as mp
        mp.solutions.pose  # 로드 테스트
    except Exception as e:
        logger.error(f"MediaPipe 로드 실패: {e}")
        mediapipe_ok = False

    return HealthResponse(
        status="ok",
        version="1.0.0",
        mediapipe_available=mediapipe_ok
    )
