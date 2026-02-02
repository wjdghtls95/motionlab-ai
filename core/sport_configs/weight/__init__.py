"""
웨이트 트레이닝 설정 (Sub-category별)
"""
from .squat_config import SQUAT_CONFIG
from .deadlift_config import DEADLIFT_CONFIG
from .bench_press_config import BENCH_PRESS_CONFIG

WEIGHT_CONFIGS = {
    "SQUAT": SQUAT_CONFIG,
    "DEADLIFT": DEADLIFT_CONFIG,
    "BENCH_PRESS": BENCH_PRESS_CONFIG,
}

__all__ = ['WEIGHT_CONFIGS', 'SQUAT_CONFIG', 'DEADLIFT_CONFIG', 'BENCH_PRESS_CONFIG']
