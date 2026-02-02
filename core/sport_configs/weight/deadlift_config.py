"""
데드리프트 분석 설정
"""
from ..base_config import SportConfig

DEADLIFT_CONFIG: SportConfig = {
    "angles": {
        "hip_flexion": {
            "points": ["left_shoulder", "left_hip", "left_knee"],
            "ideal_range": (45.0, 60.0),
            "description": "엉덩이 굽힘 (시작 자세)"
        },
        "spine_angle": {
            "points": ["nose", "left_shoulder", "left_hip"],
            "ideal_range": (165.0, 180.0),
            "description": "척추 중립 (절대 굽히지 말 것)"
        },
        "left_knee_angle": {
            "points": ["left_hip", "left_knee", "left_ankle"],
            "ideal_range": (130.0, 150.0),
            "description": "무릎 각도 (약간 굽힘)"
        },
        "shoulder_protraction": {
            "points": ["left_shoulder", "left_elbow", "left_wrist"],
            "ideal_range": (170.0, 180.0),
            "description": "어깨 내밀기 (팔 펴짐)"
        }
    },
    "phases": [
        {
            "name": "start",
            "detection_rule": "angle_range",
            "target_angle": "hip_flexion",
            "params": {"min": 45.0, "max": 60.0}
        },
        {
            "name": "lift_off",
            "detection_rule": "angle_decrease",
            "target_angle": "hip_flexion",
            "params": {}
        },
        {
            "name": "mid_pull",
            "detection_rule": "angle_range",
            "target_angle": "hip_flexion",
            "params": {"min": 20.0, "max": 40.0}
        },
        {
            "name": "lockout",
            "detection_rule": "angle_min",
            "target_angle": "hip_flexion",
            "params": {"window_size": 3}
        },
        {
            "name": "finish",
            "detection_rule": "stabilization",
            "target_angle": "hip_flexion",
            "params": {"window_size": 5, "variance_threshold": 2.0}
        }
    ]
}
