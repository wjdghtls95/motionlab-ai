"""
골프 아이언 분석 설정
"""
from ..base_config import SportConfig

IRON_CONFIG: SportConfig = {
    "angles": {
        "left_arm_angle": {
            "points": ["left_shoulder", "left_elbow", "left_wrist"],
            "ideal_range": (160.0, 175.0),
            "description": "왼팔 펴기 (아이언은 약간 여유)"
        },
        "right_arm_angle": {
            "points": ["right_shoulder", "right_elbow", "right_wrist"],
            "ideal_range": (160.0, 175.0),
            "description": "오른팔 펴기"
        },
        "spine_angle": {
            "points": ["left_shoulder", "left_hip", "left_knee"],
            "ideal_range": (30.0, 40.0),
            "description": "척추 기울기 (아이언: 더 숙임)"
        },
        "hip_shoulder_separation": {
            "points": None,
            "ideal_range": (75.0, 85.0),  # 아이언: 약간 작은 회전
            "description": "엉덩이-어깨 회전 분리각"
        },
        "left_knee_angle": {
            "points": ["left_hip", "left_knee", "left_ankle"],
            "ideal_range": (150.0, 165.0),
            "description": "왼쪽 무릎"
        },
        "right_knee_angle": {
            "points": ["right_hip", "right_knee", "right_ankle"],
            "ideal_range": (150.0, 165.0),
            "description": "오른쪽 무릎"
        }
    },
    "phases": [
        {
            "name": "address",
            "detection_rule": "velocity_threshold",
            "target_angle": "left_arm_angle",
            "params": {"window_size": 5, "threshold": 5.0}
        },
        {
            "name": "backswing",
            "detection_rule": "angle_increase",
            "target_angle": "left_arm_angle",
            "params": {"min_increase": 5.0}
        },
        {
            "name": "backswing_top",
            "detection_rule": "angle_max",
            "target_angle": "left_arm_angle",
            "params": {"window_size": 2}
        },
        {
            "name": "downswing",
            "detection_rule": "angle_decrease",
            "target_angle": "left_arm_angle",
            "params": {}
        },
        {
            "name": "impact",
            "detection_rule": "angle_min",
            "target_angle": "left_arm_angle",
            "params": {"window_size": 1}
        },
        {
            "name": "follow_through",
            "detection_rule": "stabilization",
            "target_angle": "left_arm_angle",
            "params": {"window_size": 10, "variance_threshold": 2.0}
        }
    ]
}
