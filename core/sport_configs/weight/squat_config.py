"""
스쿼트 분석 설정
"""
from ..base_config import SportConfig

SQUAT_CONFIG: SportConfig = {
    "angles": {
        "left_knee_flexion": {
            "points": ["left_hip", "left_knee", "left_ankle"],
            "ideal_range": (85.0, 95.0),
            "description": "왼쪽 무릎 굽힘 (평행 이하)"
        },
        "right_knee_flexion": {
            "points": ["right_hip", "right_knee", "right_ankle"],
            "ideal_range": (85.0, 95.0),
            "description": "오른쪽 무릎 굽힘"
        },
        "hip_flexion": {
            "points": ["left_shoulder", "left_hip", "left_knee"],
            "ideal_range": (70.0, 90.0),
            "description": "엉덩이 굽힘"
        },
        "spine_angle": {
            "points": ["nose", "left_shoulder", "left_hip"],
            "ideal_range": (160.0, 180.0),
            "description": "척추 정렬 (곧게 유지)"
        }
    },
    "phases": [
        {
            "name": "start",
            "detection_rule": "angle_range",
            "target_angle": "left_knee_flexion",
            "params": {"min": 160.0, "max": 180.0}
        },
        {
            "name": "descent",
            "detection_rule": "angle_decrease",
            "target_angle": "left_knee_flexion",
            "params": {}
        },
        {
            "name": "bottom",
            "detection_rule": "angle_min",
            "target_angle": "left_knee_flexion",
            "params": {"window_size": 3}
        },
        {
            "name": "ascent",
            "detection_rule": "angle_increase",
            "target_angle": "left_knee_flexion",
            "params": {}
        },
        {
            "name": "finish",
            "detection_rule": "angle_range",
            "target_angle": "left_knee_flexion",
            "params": {"min": 160.0, "max": 180.0}
        }
    ]
}
