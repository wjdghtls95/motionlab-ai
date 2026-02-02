"""
종목별 분석 설정 (Sport Configs)

구조:
- GOLF
  - DRIVER
  - IRON
  - PUTTER
- WEIGHT
  - SQUAT
  - DEADLIFT
  - BENCH_PRESS
"""
from .golf import GOLF_CONFIGS
from .weight import WEIGHT_CONFIGS

SPORT_CONFIGS = {
    "GOLF": GOLF_CONFIGS,
    "WEIGHT": WEIGHT_CONFIGS,
}

__all__ = ['SPORT_CONFIGS', 'GOLF_CONFIGS', 'WEIGHT_CONFIGS']
