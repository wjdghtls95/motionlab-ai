"""
벤치프레스 분석 설정
"""
from ..base_config import SportConfig

BENCH_PRESS_CONFIG: SportConfig = {
    "angles": {
        "left_elbow_flexion": {
            "points": ["left_shoulder", "left_elbow", "left_wrist"],
            "ideal_range": (85.0, 95.0),
            "description": "왼쪽 팔꿈치 굽힘 (바닥 위치)"
        },
        "right_elbow_flexion": {
            "points": ["right_shoulder", "right_elbow", "right_wrist"],
            "ideal_range": (85.0, 95.0),
            "description": "오른쪽 팔꿈치 굽힘"
        },
        "shoulder_abduction": {
            "points": ["left_hip", "left_shoulder", "left_elbow"],
            "ideal_range": (70.0, 85.0),
            "description": "어깨 벌림 (45-75도 권장)"
        },
        "wrist_angle": {
            "points": ["left_elbow", "left_wrist", "left_shoulder"],
            "ideal_range": (170.0, 180.0),
            "description": "손목 중립 (꺾이지 않게)"
        }
    },
    "phases": [
        {
            "name": "start",
            "detection_rule": "angle_range",
            "target_angle": "left_elbow_flexion",
            "params": {"min": 160.0, "max": 180.0}
        },
        {
            "name": "descent",
            "detection_rule": "angle_decrease",
            "target_angle": "left_elbow_flexion",
            "params": {}
        },
        {
            "name": "bottom",
            "detection_rule": "angle_min",
            "target_angle": "left_elbow_flexion",
            "params": {"window_size": 3}
        },
        {
            "name": "press",
            "detection_rule": "angle_increase",
            "target_angle": "left_elbow_flexion",
            "params": {}
        },
        {
            "name": "lockout",
            "detection_rule": "angle_range",
            "target_angle": "left_elbow_flexion",
            "params": {"min": 160.0, "max": 180.0}
        }
    ]
}
