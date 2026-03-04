"""
MotionLab AI - 전역 상수 정의

분석 파이프라인에서 사용하는 점수, 임계값, 기본값 등을 한 곳에서 관리.
매직 넘버를 없애고, 수정 시 이 파일만 변경하면 됨.
"""


# ========== 규칙 기반 피드백 점수 ==========
class FeedbackScore:
    """NOOP 모드 각도별 점수 기준"""

    IDEAL = 90  # ideal_range 안에 들어올 때
    CAUTION = 70  # 정상 범위이지만 ideal 밖
    CORRECTION = 40  # 정상 범위도 벗어남
    NO_DATA = 50  # 각도 데이터 없을 때
    DEFAULT = 70  # fallback 기본값


class FeedbackThreshold:
    """피드백 임계값"""

    VALIDATION_MARGIN = 20  # ideal_range 밖 허용 마진 (도)
    MAX_IMPROVEMENTS = 3  # 최대 개선점 수
    MIN_ANGLES_REQUIRED = 1  # 최소 필요 각도 수


# ========== 분석 파이프라인 ==========
class PipelineConfig:
    """분석 파이프라인 설정"""

    TOTAL_STEPS = 7  # 파이프라인 전체 단계 수
    MIN_VISIBILITY = 0.5  # MediaPipe 랜드마크 최소 신뢰도
    MIN_VALID_FRAME_RATIO = 0.1  # 최소 유효 프레임 비율 (10%)
    MIN_VIDEO_DURATION = 1.0  # 최소 영상 길이 (초)


# ========== LLM 관련 ==========
class LLMConfig:
    """LLM 호출 관련 상수"""

    REQUIRED_RESPONSE_KEYS = ["feedback", "overall_score", "improvements"]
    NOOP_VERSION = "noop"
    UNKNOWN_VERSION = "unknown"


# ========== 각도 범위 기본값 ==========
class AngleDefaults:
    """각도 관련 기본값"""

    RANGE_MIN = 0  # ideal_range fallback 최소
    RANGE_MAX = 360  # ideal_range fallback 최대
    DEFAULT_WEIGHT = 1.0  # 가중치 기본값
