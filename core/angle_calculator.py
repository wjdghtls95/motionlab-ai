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
    → calculate_angles(landmarks_data)         # 프레임 각도 + 평균만 반환
    → calculate_phase_scores(frame_angles, detected_phases)  # 페이즈별 점수 계산
    → _calculate_frame_angles(landmarks)
    → _calculate_angle_from_points() 또는 _special_calculators[name]()
    → core/landmarks.py :: get_landmark_index()
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Callable

from core.constants import FeedbackScore, AngleDefaults
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
        전체 프레임 각도 계산 + 평균 각도 반환.

        점수 계산은 calculate_phase_scores()가 담당.
        페이즈 감지 전에 호출되므로 여기서는 프레임 데이터만 반환.

        Returns:
        {
            "frame_angles": [{"frame_idx": N, "angles": {...}}, ...],
            "average_angles": {"left_arm_angle": 168.7, ...}
        }
        """
        frames = landmarks_data.get("frames", [])
        if not frames:
            logger.warning("프레임 데이터 없음")
            return {"frame_angles": [], "average_angles": {}}

        frame_angles_list = []
        angle_values: Dict[str, List[float]] = {name: [] for name in self.angle_config}

        for frame in frames:
            frame_idx = frame.get("frame_index", 0)
            landmarks = frame.get("landmarks", [])

            angles = self._calculate_frame_angles(landmarks)
            if angles:
                frame_angles_list.append({"frame_idx": frame_idx, "angles": angles})
                for name, value in angles.items():
                    angle_values[name].append(value)

        average_angles = {}
        for name, values in angle_values.items():
            if values:
                average_angles[name] = round(float(np.mean(values)), 1)

        logger.info(
            f"각도 계산 완료: {len(frame_angles_list)}/{len(frames)} 프레임 성공"
        )

        return {
            "frame_angles": frame_angles_list,
            "average_angles": average_angles,
        }

    def calculate_phase_scores(
        self,
        frame_angles: List[Dict],
        detected_phases: List[Dict],
    ) -> Dict:
        """
        페이즈별 각도 점수 계산.

        각 angle의 config.phase 필드로 해당 페이즈 구간을 특정하고,
        그 구간 프레임의 평균 각도를 ideal_range와 비교해 점수 결정.
        페이즈 미감지 시 해당 angle 제외 (옵션 B).

        Args:
            frame_angles: calculate_angles()["frame_angles"]
            detected_phases: PhaseDetector.detect_phases() 결과
                [{"name": "backswing_top", "start_frame": N, "end_frame": M}, ...]

        Returns:
        {
            "phase_angles": {"left_arm_angle": 172.3, ...},
            "phase_scores": {"left_arm_angle": 90, ...},
            "overall_score": 87.5,
            "diagnosis": {"left_arm_angle": None, "right_arm_angle": "트레일팔 과다 접힘", ...},
            "undetected_phases": ["impact"]
        }
        """
        phase_map: Dict[str, Dict] = {p["name"]: p for p in detected_phases}
        frame_map: Dict[int, Dict[str, float]] = {
            f["frame_idx"]: f["angles"] for f in frame_angles
        }

        phase_angles: Dict[str, float] = {}
        undetected_phases: List[str] = []

        for angle_name, angle_def in self.angle_config.items():
            target_phase = angle_def.get("phase")
            if not target_phase:
                # phase 필드 없는 angle은 건너뜀 (WEDGE/PUTTER/WEIGHT 2단계 이후 처리)
                continue

            if target_phase not in phase_map:
                if target_phase not in undetected_phases:
                    undetected_phases.append(target_phase)
                continue

            detected = phase_map[target_phase]
            start = detected["start_frame"]
            end = detected["end_frame"]

            values = [
                frame_map[idx][angle_name]
                for idx in range(start, end + 1)
                if idx in frame_map and angle_name in frame_map[idx]
            ]

            if values:
                phase_angles[angle_name] = round(float(np.mean(values)), 1)

        phase_scores = self._calculate_scores(phase_angles)
        overall_score = self._calculate_weighted_score(phase_scores)
        diagnosis = self._determine_diagnosis(phase_angles)

        logger.info(
            f"페이즈별 점수 계산 완료: {len(phase_angles)}개 각도, "
            f"전체점수={overall_score}, "
            f"미감지 페이즈={undetected_phases}"
        )

        return {
            "phase_angles": phase_angles,
            "phase_scores": phase_scores,
            "overall_score": overall_score,
            "diagnosis": diagnosis,
            "undetected_phases": undetected_phases,
        }

    def _calculate_scores(self, angles: Dict[str, float]) -> Dict[str, int]:
        """
        입력된 각도값이 ideal_range 안에 있는지 판단해서 점수 부여.

        입력에 있는 각도만 채점한다 (없는 각도는 결과에 포함하지 않음).

        - ideal_range 안 → IDEAL (90점)
        - angle_validation 안 → CAUTION (70점)
        - 둘 다 밖 → CORRECTION (40점)
        """
        scores = {}

        for angle_name, value in angles.items():
            if value is None:
                continue

            angle_def = self.angle_config.get(angle_name, {})
            ideal = angle_def.get(
                "ideal_range", [AngleDefaults.RANGE_MIN, AngleDefaults.RANGE_MAX]
            )

            if ideal[0] <= value <= ideal[1]:
                scores[angle_name] = FeedbackScore.IDEAL
                continue

            validation = angle_def.get("angle_validation")
            if validation:
                v_min = validation.get("min_normal", AngleDefaults.RANGE_MIN)
                v_max = validation.get("max_normal", AngleDefaults.RANGE_MAX)
                if v_min <= value <= v_max:
                    scores[angle_name] = FeedbackScore.CAUTION
                    continue

            scores[angle_name] = FeedbackScore.CORRECTION

        return scores

    def _determine_diagnosis(
        self, phase_angles: Dict[str, float]
    ) -> Dict[str, Optional[str]]:
        """
        각도값이 ideal_range 기준으로 낮으면 diagnosis_low, 높으면 diagnosis_high.
        ideal_range 안이거나 값 없으면 None.
        """
        diagnosis: Dict[str, Optional[str]] = {}
        for angle_name, angle_def in self.angle_config.items():
            value = phase_angles.get(angle_name)
            if value is None:
                diagnosis[angle_name] = None
                continue

            ideal = angle_def.get(
                "ideal_range", [AngleDefaults.RANGE_MIN, AngleDefaults.RANGE_MAX]
            )
            if value < ideal[0]:
                diagnosis[angle_name] = angle_def.get("diagnosis_low")
            elif value > ideal[1]:
                diagnosis[angle_name] = angle_def.get("diagnosis_high")
            else:
                diagnosis[angle_name] = None

        return diagnosis

    def _calculate_weighted_score(self, angle_scores: Dict[str, int]) -> float:
        """
        가중 평균 점수 계산 — weight 자동 정규화

        config의 weight가 뭐든 합이 1.0이 되도록 자동 조정.
        weight가 없으면 균등 배분.
        """
        if not angle_scores:
            return float(FeedbackScore.DEFAULT)

        total_weight = 0.0
        weighted_sum = 0.0

        for angle_name, score in angle_scores.items():
            angle_def = self.angle_config.get(angle_name, {})
            weight = angle_def.get("weight", AngleDefaults.DEFAULT_WEIGHT)
            total_weight += weight
            weighted_sum += score * weight

        if total_weight <= 0:
            return float(FeedbackScore.DEFAULT)

        # 자동 정규화: total_weight로 나눔
        result = weighted_sum / total_weight
        return round(result, 1)

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
