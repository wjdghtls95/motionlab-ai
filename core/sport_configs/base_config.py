"""
Config 타입 정의 (Type Definitions)
"""

from enum import Enum
from typing import TypedDict, List, Tuple, Optional, Dict


# ========== 사용자 레벨 Enum ==========
class UserLevel(str, Enum):
    """
    사용자 실력 레벨
    - str을 상속받아 FastAPI 쿼리 파라미터 / JSON 직렬화 호환
    - CSV의 weight_beginner ~ weight_pro 컬럼에 대응
    """

    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"
    PRO = "PRO"


# ========== v2 레벨 구조 타입 ==========
class LevelConfig(TypedDict):
    """레벨별 기준값"""

    ideal_range: List[float]  # [min, max]
    weight: float  # 가중치 (0.0 ~ 1.0)


class DataSource(TypedDict, total=False):
    """논문 출처 정보"""

    paper: str
    original_value: str
    mediapipe_value: Optional[float]
    mediapipe_center: Optional[float]
    conversion: str


class AngleValidation(TypedDict):
    """각도 유효 범위 (정상 범위)"""

    min_normal: float
    max_normal: float


# ========== 각도 정의 (v1 / v2 호환) ==========
class AngleDefinition(TypedDict, total=False):
    """
    각도 정의
    - v1: ideal_range 직접 포함
    - v2: levels 딕셔너리 안에 레벨별 ideal_range
    """

    points: Optional[List[str]]  # 3개 랜드마크 (None이면 특수 계산)
    ideal_range: List[float]  # v1 호환 or 레벨 resolve 후 주입
    description: str
    weight: float  # 가중치 (레벨 resolve 후 주입)
    # v2 전용 필드
    levels: Optional[Dict[str, LevelConfig]]
    data_source: Optional[DataSource]
    angle_validation: Optional[AngleValidation]


class PhaseRule(TypedDict):
    """구간 감지 규칙"""

    name: str
    detection_rule: str
    target_angle: Optional[str]
    params: dict


class SportConfig(TypedDict):
    """종목 설정 (레벨 resolve 완료 후)"""

    angles: Dict[str, AngleDefinition]
    phases: List[PhaseRule]
    angle_validation: Optional[Dict[str, AngleValidation]]
