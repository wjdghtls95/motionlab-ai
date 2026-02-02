"""
MotionLab AI - Environment Configuration

주의: core/sport_configs/는 종목별 분석 규칙 (별도 패키지)
"""
from .settings import Settings, get_settings, settings

__all__ = ['Settings', 'get_settings', 'settings']