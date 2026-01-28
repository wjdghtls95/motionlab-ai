"""
Analysis API
운동 영상 분석 엔드포인트
"""
from fastapi import APIRouter, Depends, HTTPException

from models.requests.analysis_request import AnalysisRequest
from models.responses.analysis_response import AnalysisResponse
from utils.dependencies import verify_api_key
from utils.logger import logger, mask_sensitive

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    dependencies=[Depends(verify_api_key)],
    summary="운동 영상 분석",
    description="MediaPipe 포즈 추출 + 각도 계산 + LLM 피드백 생성"
)
async def analyze_motion(request: AnalysisRequest) -> AnalysisResponse:
    """
    운동 영상 분석 엔드포인트 (Phase 0: 스켈레톤)

    Args:
        request: 분석 요청 (motion_id, video_url, sport_code)

    Returns:
        AnalysisResponse: 분석 결과

    Note:
        Phase 0에서는 요청만 받고 더미 응답 반환
        Phase 1~3에서 실제 분석 로직 구현 예정
    """
    logger.info(f"[Phase 0] 분석 요청 수신: motion_id={request.motion_id}, sport_code={request.sport_code}")
    logger.debug(f"video_url: {mask_sensitive(request.video_url)}")

    # TODO: Phase 1 - MediaPipe 포즈 추출
    # TODO: Phase 2 - 각도 계산 및 구간 감지
    # TODO: Phase 3 - LLM 피드백 생성

    # Phase 0: 더미 응답
    return AnalysisResponse(
        success=True,
        motion_id=request.motion_id,
        result={
            "message": "Phase 0: 요청 수신 완료 (실제 분석은 Phase 1~3에서 구현)"
        },
        feedback="Phase 0 테스트 응답입니다."
    )
