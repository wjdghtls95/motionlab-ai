from typing import Any, Dict, List


def to_phase_input(frame_angles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """frame_angles 리스트를 PhaseDetector 입력 형식으로 변환.

    AngleCalculator 반환값:
        [{"frame_idx": 0, "angles": {"left_arm_angle": 168.7, ...}}, ...]

    PhaseDetector 기대값:
        {"left_arm_angle": {"frames": {0: 168.7, 1: 170.2, ...}}, ...}
    """
    result: Dict[str, Any] = {}
    for frame in frame_angles:
        for angle_name, value in frame["angles"].items():
            result.setdefault(angle_name, {"frames": {}})
            result[angle_name]["frames"][frame["frame_idx"]] = value
    return result
