"""
골프 설정 (Sub-category별)
"""
from .driver_config import DRIVER_CONFIG
from .iron_config import IRON_CONFIG
from .putter_config import PUTTER_CONFIG

GOLF_CONFIGS = {
    "DRIVER": DRIVER_CONFIG,
    "IRON": IRON_CONFIG,
    "PUTTER": PUTTER_CONFIG,
}

__all__ = ['GOLF_CONFIGS', 'DRIVER_CONFIG', 'IRON_CONFIG', 'PUTTER_CONFIG']
