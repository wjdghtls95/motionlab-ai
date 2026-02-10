"""
스포츠 설정 로더 (JSON 기반)
"""

import json
from pathlib import Path
from typing import Dict, Any
from utils.logger import logger

SPORTS_CONFIG_PATH = Path(__file__).parent / "sports_config.json"

_SPORTS_CONFIG_CACHE: Dict[str, Any] = None


def load_sports_config() -> Dict[str, Any]:
    """sports_config.json 로드 (캐싱)"""
    global _SPORTS_CONFIG_CACHE

    if _SPORTS_CONFIG_CACHE is None:
        with open(SPORTS_CONFIG_PATH, "r", encoding="utf-8") as f:
            _SPORTS_CONFIG_CACHE = json.load(f)
        logger.info("✅ 스포츠 설정 로드 완료")

    return _SPORTS_CONFIG_CACHE


def get_sport_config(sport_type: str, sub_category: str) -> Dict[str, Any]:
    """
    종목별 설정 가져오기

    Args:
        sport_type: GOLF, WEIGHT
        sub_category: DRIVER, SQUAT 등

    Returns:
        {"angles": {...}, "phases": [...], "angle_validation": {...}}
    """
    config = load_sports_config()

    sport = config.get(sport_type)
    if not sport:
        available = list(config.keys())
        raise ValueError(f"지원하지 않는 종목: {sport_type} " f"사용 가능: {available}")

    sub = sport["sub_categories"].get(sub_category)
    if not sub:
        available = list(sport["sub_categories"].keys())
        raise ValueError(
            f"지원하지 않는 서브카테고리: {sub_category}. " f"사용 가능: {available}"
        )

    return {
        "angles": sub["angles"],
        "phases": sub["phases"],
        "angle_validation": sub.get("angle_validation", {}),  # sport → sub
    }


__all__ = ["load_sports_config", "get_sport_config"]
