"""
분석 응답 생성 헬퍼

역할:
  - 랜드마크 데이터에서 키포인트 샘플 추출
  - 향후 응답 조립 관련 헬퍼 추가 (build_phase_summary 등)
"""

from typing import Dict, Any, List


def extract_keypoints_sample(landmarks_data: Dict[str, Any]) -> List[Dict]:
    """
    첫 프레임의 첫 번째 키포인트를 샘플로 추출

    Args:
        landmarks_data: mediapipe_analyzer 반환값
            { "frames": [...], "total_frames": N, ... }

    Returns:
        [{"x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.95}]
        또는 빈 리스트
    """
    frames = landmarks_data.get("frames", [])
    if not frames:
        return []

    first_landmark = frames[0].get("landmarks", [])
    if not first_landmark:
        return []

    lm = first_landmark[0]
    return [
        {
            "x": lm.get("x", 0),
            "y": lm.get("y", 0),
            "z": lm.get("z", 0),
            "visibility": lm.get("visibility", 0),
        }
    ]
