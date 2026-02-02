"""
Config 타입 정의 (Type Definitions)
"""
from typing import TypedDict, List, Tuple, Optional


class AngleDefinition(TypedDict):
    """각도 정의"""
    points: Optional[List[str]]  # 3개 랜드마크 이름 (None이면 특수 계산)
    ideal_range: Tuple[float, float]  # 이상 범위 (min, max)
    description: str  # 설명


class PhaseRule(TypedDict):
    """구간 감지 규칙"""
    name: str  # 구간 이름
    detection_rule: str  # 탐지 규칙
    target_angle: Optional[str]  # 기준 각도
    params: dict  # 추가 파라미터


class SportConfig(TypedDict):
    """종목 설정"""
    angles: dict[str, AngleDefinition]
    phases: List[PhaseRule]
