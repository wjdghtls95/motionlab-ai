"""
에러 코드 정의 + 예외 클래스 통합 관리
"""

from typing import Dict, Any, Optional
from fastapi import status

# ========== 에러 레지스트리 (단일 소스) ==========
ERROR_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Analysis Errors (사용자 과실)
    "AN_001": {
        "name": "NoKeypointsError",
        "message_ko": "영상에서 사람의 움직임을 감지할 수 없습니다",
        "status_code": status.HTTP_400_BAD_REQUEST,
        "retryable": False,
    },
    "AN_002": {
        "name": "VideoTooShortError",
        "message_ko": "영상이 너무 짧습니다 (최소 3초 필요)",
        "status_code": status.HTTP_400_BAD_REQUEST,
        "retryable": False,
    },
    "AN_003": {
        "name": "InsufficientFramesError",
        "message_ko": "분석 가능한 프레임이 부족합니다",
        "status_code": status.HTTP_400_BAD_REQUEST,
        "retryable": False,
    },
    "AN_004": {
        "name": "InvalidMotionError",
        "message_ko": "올바르지 않은 동작입니다",
        "status_code": status.HTTP_400_BAD_REQUEST,
        "retryable": False,
    },
    "AN_005": {
        "name": "UnsupportedSportError",
        "message_ko": "지원하지 않는 종목입니다",
        "status_code": status.HTTP_400_BAD_REQUEST,
        "retryable": False,
    },
    # System Errors
    "SYS_010": {
        "name": "VideoNotFoundError",
        "message_ko": "영상 파일을 찾을 수 없습니다",
        "status_code": status.HTTP_404_NOT_FOUND,
        "retryable": False,
    },
    "SYS_011": {
        "name": "VideoDownloadError",
        "message_ko": "영상 다운로드 실패",
        "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
        "retryable": True,
    },
    "SYS_012": {
        "name": "AIServerError",
        "message_ko": "AI 분석 서버가 응답하지 않습니다",
        "status_code": status.HTTP_504_GATEWAY_TIMEOUT,
        "retryable": True,
    },
    "SYS_013": {
        "name": "VideoProcessingError",
        "message_ko": "영상 처리 중 오류가 발생했습니다",
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "retryable": True,
    },
    "SYS_021": {
        "name": "AnalysisTimeoutError",
        "message_ko": "분석 시간이 초과되었습니다",
        "status_code": status.HTTP_504_GATEWAY_TIMEOUT,
        "retryable": True,
    },
    "SYS_999": {
        "name": "SystemError",
        "message_ko": "알 수 없는 시스템 오류가 발생했습니다",
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "retryable": True,
    },
    # LLM Errors (신규 추가)
    "LLM_001": {
        "name": "LLMTimeoutError",
        "message_ko": "AI 피드백 생성 시간이 초과되었습니다",
        "status_code": status.HTTP_504_GATEWAY_TIMEOUT,
        "retryable": True,
    },
    "LLM_002": {
        "name": "LLMParseError",
        "message_ko": "AI 응답 형식이 올바르지 않습니다",
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "retryable": False,  # 파싱 오류는 재시도해도 동일
    },
    "LLM_003": {
        "name": "LLMGenerationError",
        "message_ko": "AI 피드백 생성 중 오류가 발생했습니다",
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "retryable": True,  # 일반 오류는 재시도 가능
    },
    "LLM_004": {
        "name": "LLMInvalidResponseError",
        "message_ko": "AI 응답이 유효하지 않습니다",
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "retryable": False,
    },
    # Validation Errors
    "VAL_000": {
        "name": "InvalidRequestError",
        "message_ko": "잘못된 요청 데이터입니다",
        "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "retryable": False,
    },
}


# ========== 에러 코드 상수 (타입 안전성/IDE 자동완성) ==========
class ErrorCode:
    """에러 코드 상수 (AN: Analysis, SYS: System, VAL: Validation)"""

    # Analysis Errors
    NO_KEYPOINTS = "AN_001"
    VIDEO_TOO_SHORT = "AN_002"
    INSUFFICIENT_FRAMES = "AN_003"
    INVALID_MOTION = "AN_004"
    UNSUPPORTED_SPORT = "AN_005"

    # System Errors
    VIDEO_NOT_FOUND = "SYS_010"
    VIDEO_DOWNLOAD_ERROR = "SYS_011"
    AI_SERVER_ERROR = "SYS_012"
    VIDEO_PROCESSING_ERROR = "SYS_013"
    ANALYSIS_TIMEOUT = "SYS_021"
    SYSTEM_ERROR = "SYS_999"

    # LLM Errors
    LLM_TIMEOUT = "LLM_001"
    LLM_PARSE_ERROR = "LLM_002"
    LLM_GENERATION_ERROR = "LLM_003"
    LLM_INVALID_RESPONSE = "LLM_004"

    # Validation Errors
    INVALID_REQUEST = "VAL_000"


# ========== 기본 예외 클래스 ==========
class AnalyzerError(Exception):
    """
    기본 분석 에러 클래스

    메타데이터는 ERROR_REGISTRY에서 자동으로 가져옴
    """

    def __init__(
        self,
        error_code: str,
        details: Optional[str] = None,
        status_code: Optional[int] = None,
        retryable: Optional[bool] = None,
    ):
        self.error_code = error_code
        self.details = details

        # ERROR_REGISTRY에서 메타데이터 가져오기
        error_info = ERROR_REGISTRY.get(error_code, {})
        self.message = error_info.get("message_ko", "알 수 없는 오류")
        self.status_code = status_code or error_info.get("status_code", 500)
        self.retryable = (
            retryable if retryable is not None else error_info.get("retryable", False)
        )

        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """
        API 응답용 딕셔너리 변환

        Returns:
            {
                "error_code": "AN_001",
                "message": "영상에서 사람의 움직임을 감지할 수 없습니다",
                "retryable": false,
                "details": "MediaPipe 결과 없음"
            }
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


# ========== 도메인별 예외 클래스 (편의성) ==========
class NoKeypointsError(AnalyzerError):
    """키포인트 미감지 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.NO_KEYPOINTS, details)


class VideoTooShortError(AnalyzerError):
    """영상 길이 부족 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.VIDEO_TOO_SHORT, details)


class InsufficientFramesError(AnalyzerError):
    """프레임 부족 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.INSUFFICIENT_FRAMES, details)


class InvalidMotionError(AnalyzerError):
    """잘못된 동작 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.INVALID_MOTION, details)


class UnsupportedSportError(AnalyzerError):
    """지원하지 않는 종목 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.UNSUPPORTED_SPORT, details)


class VideoNotFoundError(AnalyzerError):
    """영상 파일 미발견 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.VIDEO_NOT_FOUND, details)


class VideoDownloadError(AnalyzerError):
    """영상 다운로드 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.VIDEO_DOWNLOAD_ERROR, details)


class VideoProcessingError(AnalyzerError):
    """영상 처리 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.VIDEO_PROCESSING_ERROR, details)


class AnalysisTimeoutError(AnalyzerError):
    """분석 타임아웃 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.ANALYSIS_TIMEOUT, details)


class InvalidRequestError(AnalyzerError):
    """잘못된 요청 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.INVALID_REQUEST, details)


# LLM 예외 클래스
class LLMTimeoutError(AnalyzerError):
    """LLM 타임아웃 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.LLM_TIMEOUT, details)


class LLMParseError(AnalyzerError):
    """LLM 응답 파싱 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.LLM_PARSE_ERROR, details)


class LLMGenerationError(AnalyzerError):
    """LLM 피드백 생성 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.LLM_GENERATION_ERROR, details)


class LLMInvalidResponseError(AnalyzerError):
    """LLM 응답 검증 에러"""

    def __init__(self, details: Optional[str] = None):
        super().__init__(ErrorCode.LLM_INVALID_RESPONSE, details)


# ========== 편의 함수 ==========
def get_error_info(error_code: str) -> Dict[str, Any]:
    """
    에러 코드로 메타데이터 조회

    Args:
        error_code: 에러 코드 (예: "AN_001")

    Returns:
        {
            "name": "NoKeypointsError",
            "message_ko": "영상에서 사람의 움직임을...",
            "status_code": 400,
            "retryable": false
        }
    """
    return ERROR_REGISTRY.get(
        error_code,
        {
            "name": "UnknownError",
            "message_ko": "알 수 없는 오류",
            "status_code": 500,
            "retryable": False,
        },
    )


def raise_error(error_code: str, details: Optional[str] = None):
    """
    에러 발생 헬퍼 함수

    사용 예:
        raise_error(ErrorCode.NO_KEYPOINTS, "MediaPipe 결과 없음")
    """
    raise AnalyzerError(error_code, details)
