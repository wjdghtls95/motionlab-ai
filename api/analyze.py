"""
분석 API 라우터

프롬프트 A22 규칙:
- api/: HTTP 요청/응답 처리만
- 비즈니스 로직은 services/로 위임
"""

from fastapi import APIRouter, Depends, HTTPException
from models import AnalysisRequest, AnalysisResponse, ErrorResponse
from models.responses import create_error_response
from services.analysis_service import AnalysisService
from utils.dependencies import verify_api_key
from utils.exceptions import AnalyzerError, get_error_info, ErrorCode
import logging

router = APIRouter(prefix="", tags=["Analysis"])
logger = logging.getLogger(__name__)

# 비즈니스 로직은 서비스로 위임
analysis_service = AnalysisService()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "사용자 과실 에러 (AN_)"},
        500: {"model": ErrorResponse, "description": "시스템 에러 (SYS_)"},
    },
    summary="운동 영상 분석",
    description="MediaPipe 포즈 추출 + 각도 계산 + LLM 피드백 생성",
    dependencies=[Depends(verify_api_key)],  # API 키 검증
)
async def analyze_motion(request: AnalysisRequest):
    """
    운동 영상 분석 엔드포인트
    """
    logger.info(
        f"분석 요청 수신: motion_id={request.motion_id}, sport_type={request.sport_type}"
    )

    try:
        result = await analysis_service.analyze(
            motion_id=request.motion_id,
            video_url=request.video_url,
            sport_type=request.sport_type,
            sub_category=request.sub_category,
        )

        logger.info(f"분석 완료: motion_id={request.motion_id}")
        return result

    except AnalyzerError as e:
        # 커스텀 에러 -> HTTP 예외 변환
        logger.error(
            f"분석 실패: motion_id={request.motion_id}, error_code={e.error_code}",
            exc_info=False,  # 스택 트레이스 안 찍음 (사용자 과실)
        )

        return create_error_response(
            error_code=e.error_code,
            message=e.message,
            retryable=e.retryable,
            details=e.details,
        )

    except Exception as e:
        # 예상치 못한 에러
        logger.error(
            f"알 수 없는 에러: motion_id={request.motion_id}",
            exc_info=True,  # 개발 환경에서만 스택 트레이스
        )

        info = get_error_info(ErrorCode.SYSTEM_ERROR)
        return create_error_response(
            error_code=ErrorCode.SYSTEM_ERROR,
            message=info["message_ko"],
            retryable=info["retryable"],
            details=str(e),
        )
