"""
스포츠 설정 로더 (v1/v2 호환)
- v1: flat ideal_range (레벨 없음)
- v2: meta + levels 구조 (레벨별 ideal_range + weight)
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from utils.logger import logger
from .base_config import UserLevel
from core.constants import AngleDefaults

SPORTS_CONFIG_PATH = Path(__file__).parent / "sports_config.json"

# ========== 캐시 ==========
_RAW_CONFIG: Optional[Dict[str, Any]] = None
_CONFIG_VERSION: str = "unknown"
_CONFIG_FORMAT: str = "v1"


def load_sports_config(force_reload: bool = False) -> Dict[str, Any]:
    """
    sports_config.json 로드 (캐싱)
    - force_reload=True: 캐시 무시하고 다시 읽기 (hot-reload용)
    """
    global _RAW_CONFIG, _CONFIG_VERSION, _CONFIG_FORMAT

    if _RAW_CONFIG is not None and not force_reload:
        return _RAW_CONFIG

    with open(SPORTS_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # ========== v1/v2 구분 ==========
    if "meta" in raw and "sports" in raw:
        _CONFIG_FORMAT = "v2"
        _CONFIG_VERSION = raw["meta"].get("config_version", "2.0.0")
        _RAW_CONFIG = raw["sports"]
        logger.info(
            f"✅ 스포츠 설정 로드 (v2): version={_CONFIG_VERSION}, "
            f"sports={list(_RAW_CONFIG.keys())}"
        )
    else:
        _CONFIG_FORMAT = "v1"
        _CONFIG_VERSION = "1.0.0"
        _RAW_CONFIG = raw
        logger.info(f"✅ 스포츠 설정 로드 (v1): sports={list(_RAW_CONFIG.keys())}")

    return _RAW_CONFIG


def get_config_version() -> str:
    """현재 로드된 config 버전 반환"""
    if _RAW_CONFIG is None:
        load_sports_config()
    return _CONFIG_VERSION


def get_config_format() -> str:
    """v1 또는 v2 반환"""
    if _RAW_CONFIG is None:
        load_sports_config()
    return _CONFIG_FORMAT


def reload_config() -> Dict[str, Any]:
    """설정 강제 리로드 (Redis Pub/Sub 등에서 호출)"""
    logger.info("🔄 스포츠 설정 강제 리로드...")
    return load_sports_config(force_reload=True)


def get_available_sports() -> Dict[str, List[str]]:
    """
    사용 가능한 종목 목록 반환
    Returns: {"GOLF": ["DRIVER", "IRON", ...], "WEIGHT": ["SQUAT", ...]}
    """
    config = load_sports_config()
    result = {}
    for sport_type, sport_data in config.items():
        if isinstance(sport_data, dict) and "sub_categories" in sport_data:
            result[sport_type] = list(sport_data["sub_categories"].keys())
        elif isinstance(sport_data, dict):
            result[sport_type] = list(sport_data.keys())
    return result


def _resolve_angles_for_level(
    angles_raw: Dict[str, Any],
    level: UserLevel,
) -> Dict[str, Any]:
    """
    v2 JSON에서 특정 레벨의 ideal_range와 weight를 꺼내서
    flat 구조로 변환.

    변환 전 (v2):
    {
        "left_arm_angle": {
            "points": [...],
            "levels": {
                "PRO": {"ideal_range": [165.0, 180.0], "weight": 0.17},
                "BEGINNER": {"ideal_range": [157.5, 180.0], "weight": 0.17}
            }
        }
    }

    변환 후 (flat):
    {
        "left_arm_angle": {
            "points": [...],
            "ideal_range": [165.0, 180.0],
            "weight": 0.17
        }
    }
    """
    resolved = {}
    level_key = level.value

    for angle_name, angle_data in angles_raw.items():
        entry = dict(angle_data)

        if "levels" in entry and entry["levels"]:
            levels = entry["levels"]

            if level_key in levels:
                level_config = levels[level_key]
            else:
                level_config = levels.get(
                    UserLevel.INTERMEDIATE.value, next(iter(levels.values()))
                )
                logger.warning(
                    f"⚠️ {angle_name}: 레벨 '{level_key}' 없음, " f"fallback 사용"
                )

            entry["ideal_range"] = level_config.get(
                "ideal_range", [AngleDefaults.RANGE_MIN, AngleDefaults.RANGE_MAX]
            )
            entry["weight"] = level_config.get("weight", AngleDefaults.DEFAULT_WEIGHT)

            del entry["levels"]

        else:
            if "ideal_range" not in entry:
                entry["ideal_range"] = [
                    AngleDefaults.RANGE_MIN,
                    AngleDefaults.RANGE_MAX,
                ]
            if "weight" not in entry:
                entry["weight"] = AngleDefaults.DEFAULT_WEIGHT

        resolved[angle_name] = entry

    return resolved


def _resolve_angle_validation(angles_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    v2 JSON에서 angle_validation을 각도별로 추출.
    v1: sub_category 레벨에 angle_validation이 있음
    v2: 각도 안에 angle_validation이 내장
    """
    validation = {}
    for angle_name, angle_data in angles_raw.items():
        if "angle_validation" in angle_data and angle_data["angle_validation"]:
            validation[angle_name] = angle_data["angle_validation"]
    return validation


def get_sport_config(
    sport_type: str,
    sub_category: str,
    level: UserLevel = UserLevel.INTERMEDIATE,
) -> Dict[str, Any]:
    """
    종목별 설정 가져오기 (레벨 적용)

    Args:
        sport_type: GOLF, WEIGHT
        sub_category: DRIVER, SQUAT 등
        level: 사용자 레벨 (BEGINNER ~ PRO)

    Returns:
        {
            "angles": {레벨 resolve된 각도 설정},
            "phases": [...],
            "angle_validation": {각도별 정상 범위}
        }
    """
    config = load_sports_config()

    # ========== sport_type 검증 ==========
    sport = config.get(sport_type)
    if not sport:
        available = list(config.keys())
        raise ValueError(
            f"지원하지 않는 종목: {sport_type}. " f"사용 가능: {available}"
        )

    # ========== sub_category 검증 ==========
    sub_categories = sport.get("sub_categories", sport)
    sub = sub_categories.get(sub_category)
    if not sub:
        available = list(sub_categories.keys())
        raise ValueError(
            f"지원하지 않는 서브카테고리: {sub_category}. " f"사용 가능: {available}"
        )

    # ========== 각도 resolve ==========
    angles_raw = sub.get("angles", {})

    if _CONFIG_FORMAT == "v2":
        angles_resolved = _resolve_angles_for_level(angles_raw, level)
        angle_validation = _resolve_angle_validation(angles_raw)
    else:
        angles_resolved = angles_raw
        angle_validation = sub.get("angle_validation", {})

    # ========== phases ==========
    phases = sub.get("phases", [])

    logger.info(
        f"📋 Config 로드: {sport_type}/{sub_category}, "
        f"level={level.value}, format={_CONFIG_FORMAT}, "
        f"angles={len(angles_resolved)}, phases={len(phases)}"
    )

    return {
        "angles": angles_resolved,
        "phases": phases,
        "angle_validation": angle_validation,
    }


__all__ = [
    "load_sports_config",
    "get_sport_config",
    "get_config_version",
    "get_config_format",
    "reload_config",
    "get_available_sports",
]
