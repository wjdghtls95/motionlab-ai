"""
각도 계산 모듈 (Angle Calculator)

입력: MediaPipe 랜드마크 (33개 × N프레임)
출력: 골프 스윙 핵심 6가지 각도

수학 공식:
- 3점 벡터 내적으로 각도 계산
- angle = arccos((BA · BC) / (|BA| × |BC|)) × 180/π
"""
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# MediaPipe 랜드마크 인덱스 매핑
LANDMARK_INDICES = {
    # 상체
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,

    # 하체
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


class AngleCalculator:
    """
    골프 스윙 각도 계산기

    주요 기능:
    1. 프레임별 6가지 핵심 각도 계산
    2. 유효하지 않은 프레임 필터링 (visibility < 0.5)
    3. 프레임별/평균 각도 반환
    """

    def __init__(self, angle_config: Dict, min_visibility: float = 0.5):
        """
        Args:
            min_visibility: 최소 가시성 임계값 (기본 0.5)
        """
        self.angle_config = angle_config
        self.min_visibility = min_visibility

    def calculate_angles(self, landmarks_data: List[Dict]) -> Dict:
        """
        전체 프레임의 각도 계산

        Args:
            landmarks_data: MediaPipeAnalyzer의 출력
                [
                    {
                        "frame_index": 10,
                        "timestamp": 0.333,
                        "landmarks": [{"x": 0.5, "y": 0.3, ...}, ...]
                    },
                    ...
                ]

        Returns:
            {
                "frame_angles": [
                    {
                        "frame_index": 10,
                        "angles": {"left_arm": 165.2, ...}
                    },
                    ...
                ],
                "average_angles": {"left_arm": 167.3, ...}
            }
        """
        frame_angles = []
        valid_angles = defaultdict(list)

        for frame_data in landmarks_data:
            frame_index = frame_data["frame_index"]
            landmarks = frame_data["landmarks"]

            # 프레임별 6가지 각도 계산
            angles = self._calculate_frame_angles(landmarks)

            if angles:
                frame_angles.append({
                    "frame_index": frame_index,
                    "angles": angles
                })

                # 평균 계산용 누적
                for key, value in angles.items():
                    if value is not None:
                        valid_angles[key].append(value)

        # 평균 각도 계산
        average_angles = {
            key: round(np.mean(values), 1) if values else None
            for key, values in valid_angles.items()
        }

        logger.info(f"✅ {len(frame_angles)}/{len(landmarks_data)}개 프레임 각도 계산 완료")

        return {
            "frame_angles": frame_angles,
            "average_angles": average_angles
        }

    def _calculate_frame_angles(self, landmarks: List[Dict]) -> Optional[Dict]:
        """
        단일 프레임의 6가지 각도 계산

        Args:
            landmarks: 33개 랜드마크

        Returns:
            {"left_arm_angle": 165.2, ...} 또는 None (실패 시)
        """
        try:
            return {
                "left_arm_angle": self._calculate_arm_angle(landmarks, "left"),
                "right_arm_angle": self._calculate_arm_angle(landmarks, "right"),
                "spine_angle": self._calculate_spine_angle(landmarks),
                "hip_shoulder_separation": self._calculate_hip_shoulder_separation(landmarks),
                "left_knee_angle": self._calculate_knee_angle(landmarks, "left"),
                "right_knee_angle": self._calculate_knee_angle(landmarks, "right")
            }
        except Exception as e:
            logger.warning(f"프레임 각도 계산 실패: {e}")
            return None

    def _calculate_arm_angle(self, landmarks: List[Dict], side: str) -> Optional[float]:
        """
        팔 각도 계산 (어깨-팔꿈치-손목)

        Args:
            landmarks: 33개 랜드마크
            side: "left" 또는 "right"
        """
        shoulder_idx = LANDMARK_INDICES[f"{side}_shoulder"]
        elbow_idx = LANDMARK_INDICES[f"{side}_elbow"]
        wrist_idx = LANDMARK_INDICES[f"{side}_wrist"]

        return self._calculate_angle_3points(
            landmarks[shoulder_idx],
            landmarks[elbow_idx],
            landmarks[wrist_idx]
        )

    def _calculate_spine_angle(self, landmarks: List[Dict]) -> Optional[float]:
        """
        척추 각도 (왼쪽 어깨-엉덩이-무릎)
        """
        return self._calculate_angle_3points(
            landmarks[LANDMARK_INDICES["left_shoulder"]],
            landmarks[LANDMARK_INDICES["left_hip"]],
            landmarks[LANDMARK_INDICES["left_knee"]]
        )

    def _calculate_hip_shoulder_separation(self, landmarks: List[Dict]) -> Optional[float]:
        """
        엉덩이-어깨 분리각 (회전 차이)

        계산 방법:
        1. 엉덩이 회전각 = atan2(right_hip.y - left_hip.y, right_hip.x - left_hip.x)
        2. 어깨 회전각 = atan2(right_shoulder.y - left_shoulder.y, right_shoulder.x - left_shoulder.x)
        3. 분리각 = |어깨 회전각 - 엉덩이 회전각|
        """
        try:
            left_hip = landmarks[LANDMARK_INDICES["left_hip"]]
            right_hip = landmarks[LANDMARK_INDICES["right_hip"]]
            left_shoulder = landmarks[LANDMARK_INDICES["left_shoulder"]]
            right_shoulder = landmarks[LANDMARK_INDICES["right_shoulder"]]

            # visibility 체크
            if any(p["visibility"] < self.min_visibility for p in [left_hip, right_hip, left_shoulder, right_shoulder]):
                return None

            # 엉덩이 회전각
            hip_angle = np.arctan2(
                right_hip["y"] - left_hip["y"],
                right_hip["x"] - left_hip["x"]
            )

            # 어깨 회전각
            shoulder_angle = np.arctan2(
                right_shoulder["y"] - left_shoulder["y"],
                right_shoulder["x"] - left_shoulder["x"]
            )

            # 분리각 (라디안 → 도)
            separation = abs(shoulder_angle - hip_angle) * 180 / np.pi

            return round(separation, 1)

        except Exception as e:
            logger.warning(f"엉덩이-어깨 분리각 계산 실패: {e}")
            return None

    def _calculate_knee_angle(self, landmarks: List[Dict], side: str) -> Optional[float]:
        """
        무릎 각도 (엉덩이-무릎-발목)
        """
        hip_idx = LANDMARK_INDICES[f"{side}_hip"]
        knee_idx = LANDMARK_INDICES[f"{side}_knee"]
        ankle_idx = LANDMARK_INDICES[f"{side}_ankle"]

        return self._calculate_angle_3points(
            landmarks[hip_idx],
            landmarks[knee_idx],
            landmarks[ankle_idx]
        )

    def _calculate_angle_3points(
            self,
            point_a: Dict,
            point_b: Dict,
            point_c: Dict
    ) -> Optional[float]:
        """
        3점으로 각도 계산 (B를 중심으로 A-B-C 각도)

        수학 공식:
        1. 벡터 BA = A - B
        2. 벡터 BC = C - B
        3. cos(θ) = (BA · BC) / (|BA| × |BC|)
        4. θ = arccos(cos(θ)) × 180/π

        Args:
            point_a, point_b, point_c: {"x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.95}

        Returns:
            각도 (도) 또는 None (계산 실패 시)
        """
        try:
            # visibility 체크
            if any(p["visibility"] < self.min_visibility for p in [point_a, point_b, point_c]):
                return None

            # 벡터 생성
            ba = np.array([
                point_a["x"] - point_b["x"],
                point_a["y"] - point_b["y"],
                point_a["z"] - point_b["z"]
            ])

            bc = np.array([
                point_c["x"] - point_b["x"],
                point_c["y"] - point_b["y"],
                point_c["z"] - point_b["z"]
            ])

            # 벡터 크기
            ba_norm = np.linalg.norm(ba)
            bc_norm = np.linalg.norm(bc)

            # 0 벡터 체크 (division by zero 방지)
            if ba_norm == 0 or bc_norm == 0:
                return None

            # 내적 계산
            cos_angle = np.dot(ba, bc) / (ba_norm * bc_norm)

            # arccos 정의역 보장 [-1, 1]
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            # 라디안 → 도
            angle = np.arccos(cos_angle) * 180 / np.pi

            return round(angle, 1)

        except Exception as e:
            logger.warning(f"3점 각도 계산 실패: {e}")
            return None
