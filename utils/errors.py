"""
에러 코드 및 예외 정의

프롬프트 A7 규칙:
- AN_: 사용자 과실 (재시도 X)
- SYS_: 시스템 장애 (재시도 O)
"""
from typing import Dict, Any

class AnalyzerError(Exception):
    """Analyzer 기본 예외"""
    def __init__(
        self,
        error_code: str,
        message: str,
        retryable: bool = False,
        status_code: int = 500
    ):
        self.error_code = error_code
        self.message = message
        self.retryable = retryable
        self.status_code = status_code
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """에러 응답 딕셔너리 변환"""
        return {
            "success": False,
            "error_code": self.error_code,
            "message": self.message,
            "retryable": self.retryable
        }

# ==================== 사용자 과실 에러 (AN_) ====================
class NoKeypointsError(AnalyzerError):
    """포즈 키포인트 미검출"""
    def __init__(self):
        super().__init__(
            error_code="AN_001",
            message="No keypoints detected in video",
            retryable=False,
            status_code=400
        )

class VideoTooShortError(AnalyzerError):
    """영상 길이 부족"""
    def __init__(self, duration: float):
        super().__init__(
            error_code="AN_002",
            message=f"Video too short: {duration:.1f}s (minimum 1.0s required)",
            retryable=False,
            status_code=400
        )

class UnsupportedSportError(AnalyzerError):
    """지원하지 않는 종목"""
    def __init__(self, sport_type: str):
        super().__init__(
            error_code="AN_003",
            message=f"Unsupported sport_type: {sport_type}",
            retryable=False,
            status_code=400
        )

class VideoNotFoundError(AnalyzerError):
    """영상 파일 없음"""
    def __init__(self, video_url: str):
        super().__init__(
            error_code="AN_004",
            message=f"Video file not found: {video_url[:50]}...",
            retryable=False,
            status_code=404
        )

# ==================== 시스템 장애 에러 (SYS_) ====================
class VideoDownloadError(AnalyzerError):
    """영상 다운로드 실패"""
    def __init__(self, reason: str):
        super().__init__(
            error_code="SYS_011",
            message=f"Video download failed: {reason}",
            retryable=True,
            status_code=500
        )

class LLMTimeoutError(AnalyzerError):
    """LLM 타임아웃"""
    def __init__(self):
        super().__init__(
            error_code="SYS_012",
            message="LLM request timeout (60s exceeded)",
            retryable=True,
            status_code=500
        )

class AnalyzerTimeoutError(AnalyzerError):
    """분석 타임아웃"""
    def __init__(self):
        super().__init__(
            error_code="SYS_021",
            message="Analysis timeout (5 minutes exceeded)",
            retryable=True,
            status_code=500
        )

class UnknownError(AnalyzerError):
    """알 수 없는 에러"""
    def __init__(self, original_error: str):
        super().__init__(
            error_code="SYS_999",
            message=f"Unknown error: {original_error}",
            retryable=True,
            status_code=500
        )
