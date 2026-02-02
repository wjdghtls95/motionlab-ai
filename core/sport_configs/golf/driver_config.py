"""
골프 드라이버 분석 설정
"""
from ..base_config import SportConfig

DRIVER_CONFIG: SportConfig = {
    "angles": {
        "left_arm_angle": {
            "points": ["left_shoulder", "left_elbow", "left_wrist"],
            "ideal_range": (165.0, 180.0),
            "description": "왼팔 펴기 (드라이버는 최대한 펴야 함)"
        },
        "right_arm_angle": {
            "points": ["right_shoulder", "right_elbow", "right_wrist"],
            "ideal_range": (165.0, 180.0),
            "description": "오른팔 펴기"
        },
        "spine_angle": {
            "points": ["left_shoulder", "left_hip", "left_knee"],
            "ideal_range": (25.0, 35.0),
            "description": "척추 기울기 (드라이버: 약간 뒤로 기울임)"
        },
        "hip_shoulder_separation": {
            "points": None,  # 특수 계산
            "ideal_range": (85.0, 95.0),  # 드라이버: 큰 회전
            "description": "엉덩이-어깨 회전 분리각 (X-Factor)"
        },
        "left_knee_angle": {
            "points": ["left_hip", "left_knee", "left_ankle"],
            "ideal_range": (155.0, 170.0),
            "description": "왼쪽 무릎 (드라이버: 약간 펴짐)"
        },
        "right_knee_angle": {
            "points": ["right_hip", "right_knee", "right_ankle"],
            "ideal_range": (155.0, 170.0),
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
