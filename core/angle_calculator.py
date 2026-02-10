"""
역할:
  - sports_config.json의 angles 정의를 읽어 동적으로 각도 계산
  - points가 있으면 → _calculate_angle_from_points (3점 각도)
  - points가 null이면 → _special_calculators에서 등록된 함수 호출

의존 관계:
  - core/landmarks.py → get_landmark_index() 로 랜드마크 인덱스 조회
  - sports_config.json → angle_config로 전달됨 (analysis_service.py에서 주입)

호출 흐름:
  analysis_service.py
    → AngleCalculator(angle_config=sport_config["angles"])
    → calculate_angles(landmarks_data)
    → _calculate_frame_angles(landmarks)
    → _calculate_angle_from_points() 또는 _special_calculators[name]()
    → core/landmarks.py :: get_landmark_index()
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Callable

from core.landmarks import get_landmark_index

logger = logging.getLogger(__name__)


class AngleCalculator:
    """스포츠 config 기반 동적 각도 계산기"""

    def __init__(self, angle_config: Dict[str, Any], min_visibility: float = 0.5):
        """
        Args:
            angle_config: sport_config["angles"] — dict 형태
            min_visibility: 랜드마크 최소 가시성 임계값 (0.0 ~ 1.0)
        """
        self.angle_config = angle_config
        self.min_visibility = min_visibility

        # 특수 계산 함수 레지스트리
        # points가 null인 각도는 여기에 등록된 함수로 계산
        # 새 특수 계산 추가 시: 함수 구현 → 여기에 한 줄 등록
        self._special_calculators: Dict[str, Callable] = {
            "hip_shoulder_separation": self._calc_hip_shoulder_separation,
            # 향후 추가 예시:
            # "knee_over_toe": self._calc_knee_over_toe,
            # "elbow_symmetry": self._calc_elbow_symmetry,
        }

    # ========== 공개 메서드 ==========

    def calculate_angles(self, landmarks_data: Dict) -> Dict:
        """
        전체 프레임 각도 계산 + 평균 각도 반환

        Args:
            landmarks_data: mediapipe_analyzer에서 반환된 데이터
                { "frames": [ { "frame_idx": 0, "landmarks": [...] }, ... ] }

        Returns:
            {
                "frame_angles": [
                    { "frame_idx": 0, "angles": { "left_arm_angle": 165.3, ... } },
                    ...
                ],
                "average_angles": { "left_arm_angle": 168.7, ... }
            }
        """
        frames = landmarks_data.get("frames", [])
        if not frames:
            logger.warning("프레임 데이터 없음")
            return {"frame_angles": [], "average_angles": {}}

        frame_angles_list = []
        # 각도별 값 수집 (평균 계산용)
        angle_values: Dict[str, List[float]] = {name: [] for name in self.angle_config}

        for frame in frames:
            frame_idx = frame.get("frame_idx", 0)
            landmarks = frame.get("landmarks", [])

            angles = self._calculate_frame_angles(landmarks)
            if angles:
                frame_angles_list.append({"frame_idx": frame_idx, "angles": angles})
                for name, value in angles.items():
                    angle_values[name].append(value)

        # 평균 각도 계산
        average_angles = {}
        for name, values in angle_values.items():
            if values:
                average_angles[name] = round(float(np.mean(values)), 1)

        logger.info(
            f"각도 계산 완료: {len(frame_angles_list)}/{len(frames)} 프레임 성공"
        )

        return {"frame_angles": frame_angles_list, "average_angles": average_angles}

    # ========== 프레임 단위 계산 ==========

    def _calculate_frame_angles(
        self, landmarks: List[Dict]
    ) -> Optional[Dict[str, float]]:
        """단일 프레임의 모든 각도 계산"""
        try:
            angles = {}

            for angle_name, angle_def in self.angle_config.items():
                value = self._calculate_single_angle(angle_name, angle_def, landmarks)
                if value is not None:
                    angles[angle_name] = value

            return angles if angles else None

        except Exception as e:
            logger.debug(f"프레임 각도 계산 실패: {e}")
            return None

    def _calculate_single_angle(
        self, angle_name: str, angle_def: Dict, landmarks: List[Dict]
    ) -> Optional[float]:
        """
        config 정의 하나에 대해 각도 계산
        - points가 null → 특수 계산
        - points가 3개 → 3점 각도 계산
        """
        points = angle_def.get("points")

        if points is None:
            return self._run_special_calculator(angle_name, landmarks)

        if len(points) == 3:
            return self._calculate_angle_from_points(
                landmarks, points[0], points[1], points[2]
            )

        logger.warning(f"지원하지 않는 points 수: {angle_name} ({len(points)}개)")
        return None

    def _run_special_calculator(
        self, angle_name: str, landmarks: List[Dict]
    ) -> Optional[float]:
        """특수 계산 함수 실행"""
        calc_fn = self._special_calculators.get(angle_name)
        if calc_fn is None:
            logger.warning(f"특수 계산 함수 미등록: {angle_name}")
            return None

        return calc_fn(landmarks)

    # ========== 3점 각도 계산 (공통) ==========

    def _calculate_angle_from_points(
        self,
        landmarks: List[Dict],
        point_a_name: str,
        point_b_name: str,
        point_c_name: str,
    ) -> Optional[float]:
        """3점 각도 계산 — point_b가 꼭짓점"""
        indices = self._resolve_indices(point_a_name, point_b_name, point_c_name)
        if indices is None:
            return None

        idx_a, idx_b, idx_c = indices

        if not self._validate_landmarks(landmarks, [idx_a, idx_b, idx_c]):
            return None

        a = landmarks[idx_a]
        b = landmarks[idx_b]
        c = landmarks[idx_c]

        return self._compute_angle(a, b, c)

    def _resolve_indices(self, *names: str) -> Optional[tuple]:
        """랜드마크 이름들을 인덱스로 변환"""
        indices = []
        for name in names:
            idx = get_landmark_index(name)
            if idx is None:
                logger.debug(f"알 수 없는 랜드마크: {name}")
                return None
            indices.append(idx)
        return tuple(indices)

    def _validate_landmarks(self, landmarks: List[Dict], indices: List[int]) -> bool:
        """인덱스 범위 및 가시성 검증"""
        for idx in indices:
            if idx >= len(landmarks):
                return False
            if landmarks[idx].get("visibility", 0) < self.min_visibility:
                return False
        return True

    @staticmethod
    def _compute_angle(a: Dict, b: Dict, c: Dict) -> float:
        """꼭짓점 b 기준 벡터 각도 계산 (도 단위)"""
        ba = np.array([a["x"] - b["x"], a["y"] - b["y"]])
        bc = np.array([c["x"] - b["x"], c["y"] - b["y"]])

        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))

        return round(float(angle), 1)

    # ========== 특수 계산 함수들 ==========

    def _calc_hip_shoulder_separation(self, landmarks: List[Dict]) -> Optional[float]:
        """엉덩이-어깨 분리각 (X-Factor) — 골프 전용"""
        names = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]

        indices = self._resolve_indices(*names)
        if indices is None:
            return None

        idx_ls, idx_rs, idx_lh, idx_rh = indices

        if not self._validate_landmarks(landmarks, list(indices)):
            return None

        ls = landmarks[idx_ls]
        rs = landmarks[idx_rs]
        lh = landmarks[idx_lh]
        rh = landmarks[idx_rh]

        shoulder_angle = np.degrees(np.arctan2(rs["y"] - ls["y"], rs["x"] - ls["x"]))
        hip_angle = np.degrees(np.arctan2(rh["y"] - lh["y"], rh["x"] - lh["x"]))

        separation = abs(shoulder_angle - hip_angle)
        return round(float(separation), 1)
