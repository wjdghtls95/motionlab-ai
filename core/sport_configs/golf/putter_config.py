"""
골프 퍼터 분석 설정
"""
from ..base_config import SportConfig

PUTTER_CONFIG: SportConfig = {
    "angles": {
        "spine_angle": {
            "points": ["left_shoulder", "left_hip", "left_knee"],
            "ideal_range": (40.0, 50.0),
            "description": "척추 기울기 (퍼터: 많이 숙임)"
        },
        "left_arm_angle": {
            "points": ["left_shoulder", "left_elbow", "left_wrist"],
            "ideal_range": (170.0, 180.0),
            "description": "왼팔 (퍼터: 거의 펴짐)"
        },
        "right_arm_angle": {
            "points": ["right_shoulder", "right_elbow", "right_wrist"],
            "ideal_range": (170.0, 180.0),
            "description": "오른팔"
        },
        "shoulder_tilt": {
            "points": ["left_shoulder", "right_shoulder", "left_hip"],
            "ideal_range": (15.0, 25.0),
            "description": "어깨 기울기 (퍼터 특화)"
        }
    },
    "phases": [
        {
            "name": "address",
            "detection_rule": "velocity_threshold",
            "target_angle": "left_arm_angle",
            "params": {"window_size": 5, "threshold": 3.0}
        },
        {
            "name": "backswing",
            "detection_rule": "angle_increase",
            "target_angle": "left_arm_angle",
            "params": {"min_increase": 2.0}
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
            "params": {"window_size": 8, "variance_threshold": 1.5}
        }
    ]
}
