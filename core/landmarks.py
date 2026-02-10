"""
core/landmarks.py
MediaPipe 랜드마크 인덱스 유틸리티

역할:
  - MediaPipe PoseLandmark enum에서 랜드마크 이름 -> 인덱스 매핑을 자동 생성
  - 하드코딩 딕셔너리 제거, MediaPipe 버전 업데이트 시 자동 반영

사용처:
  - core/angle_calculator.py -> 각도 계산 시 랜드마크 인덱스 조회
  - core/mediapipe_analyzer.py -> 랜드마크 추출 시 인덱스 참조 (필요 시)
"""

import mediapipe as mp
from typing import Dict, Optional
from functools import lru_cache

# MediaPipe PoseLandmark enum 참조
_POSE_LANDMARK = mp.solutions.pose.PoseLandmark


@lru_cache(maxsize=1)
def get_landmark_indices() -> Dict[str, int]:
    """
    MediaPipe PoseLandmark enum -> { "nose": 0, "left_shoulder": 11, ... }
    lru_cache: 첫 호출에서 한 번 생성, 이후 캐시 반환
    """
    return {landmark.name.lower(): landmark.value for landmark in _POSE_LANDMARK}


def get_landmark_index(name: str) -> Optional[int]:
    """
    랜드마크 이름으로 인덱스 조회

    Args:
        name: config에서 사용하는 이름 (예: "left_shoulder")

    Returns:
        인덱스 (예: 11), 이름이 잘못되면 None
    """
    return get_landmark_indices().get(name)


def get_landmark_count() -> int:
    """전체 랜드마크 수 반환 (MediaPipe Pose = 33개)"""
    return len(get_landmark_indices())
