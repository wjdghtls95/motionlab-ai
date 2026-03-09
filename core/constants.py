"""
MotionLab AI - 전역 상수 정의

분석 파이프라인에서 사용하는 점수, 임계값, 기본값 등을 한 곳에서 관리
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

    TOTAL_STEPS = 8  # 파이프라인 전체 단계 수
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


# ========== 구간 감지 ==========
class PhaseDetection:
    """구간 감지 관련 상수 — 종목 독립적"""

    # 감지 규칙 이름
    RULE_STABILIZATION = "stabilization"
    RULE_ANGLE_INCREASE = "angle_increase"
    RULE_ANGLE_DECREASE = "angle_decrease"
    RULE_ANGLE_MAX = "angle_max"
    RULE_ANGLE_MIN = "angle_min"
    RULE_VELOCITY_THRESHOLD = "velocity_threshold"

    # 기본 파라미터
    DEFAULT_WINDOW = 5
    DEFAULT_THRESHOLD = 5.0
    DEFAULT_STD_THRESHOLD = 2.0

    # 프레임 제한
    MIN_PHASE_FRAMES = 3
    MIN_FRAMES_FOR_DETECTION = 10


# ========== 종목 일치 검증 ==========
class MotionValidation:
    """영상-종목 일치 검증 상수"""

    # 각도 검증
    MIN_VALID_ANGLES_RATIO = 0.5  # 전체 각도 중 50% 이상이 정상 범위 안에 있어야 함
    MIN_ANGLES_IN_RANGE = 3  # 최소 3개 각도가 validation 범위 안에 있어야 함

    # Phase 검증
    MIN_PHASES_REQUIRED = 2  # 최소 2개 이상 구간이 감지되어야 함

    # 각도 이상치 판정
    OUTLIER_MARGIN = 30.0  # angle_validation 범위에서 이만큼 벗어나면 이상치

    FALLBACK_PHASE_NAME = "full_motion"  # 폴백 구간 이름 (검증에서 제외)


# ========== 레벨별 피드백 톤 ==========
LEVEL_TONE = {
    "BEGINNER": {
        "style": "쉽고 친근하게",
        "detail": "전문 용어를 피하고 일상 언어로 설명합니다. 비유를 활용하세요.",
        "encouragement": "작은 성과도 칭찬하며, 한 번에 1~2가지만 개선점으로 제시합니다.",
        "max_improvements": 2,
    },
    "INTERMEDIATE": {
        "style": "구체적이고 실용적으로",
        "detail": "기본 전문 용어를 사용하되 간단히 설명을 덧붙입니다.",
        "encouragement": "강점을 인정하고 다음 단계로 나아갈 방향을 제시합니다.",
        "max_improvements": 3,
    },
    "ADVANCED": {
        "style": "전문적이고 분석적으로",
        "detail": "전문 용어를 자유롭게 사용합니다. 세부 수치와 비교 분석을 포함합니다.",
        "encouragement": "미세 조정 포인트와 경기력 향상 전략을 제시합니다.",
        "max_improvements": 3,
    },
    "PRO": {
        "style": "데이터 중심의 정밀 분석",
        "detail": "투어 선수 기준과 비교 분석합니다. 생체역학적 관점에서 설명합니다.",
        "encouragement": "경쟁력 강화를 위한 최적화 포인트를 제시합니다.",
        "max_improvements": 3,
    },
}
