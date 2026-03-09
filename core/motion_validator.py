"""
종목-영상 일치 검증기 (MotionValidator)

영상에서 추출한 각도/구간이 선택한 종목과 맞는지 검증.
코드에 특정 종목 이름이나 각도 이름 없음 — config 기반 검증.

검증 흐름:
  1. 각도 범위 검증: average_angles가 config의 angle_validation 범위 안에 있는지
  2. 구간 수 검증: 감지된 phase 수가 최소 기준을 충족하는지
  3. 종합 판정: 통과 / 실패 + 이유
"""

import logging
from typing import Dict, List, Any

from core.constants import MotionValidation, AngleDefaults

logger = logging.getLogger(__name__)


class MotionValidator:
    """Config 기반 종목-영상 일치 검증 — 종목 독립적"""

    def validate_motion(
        self,
        average_angles: Dict[str, float],
        angle_configs: Dict[str, Any],
        detected_phases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        전체 검증 실행

        Parameters
        ----------
        average_angles : 각도 이름 → 평균값
                         예) {"left_arm_angle": 165.2, "right_arm_angle": 88.1}
        angle_configs  : sports_config.json의 angles 딕셔너리
                         각 angle 안에 "angle_validation": {"min_normal", "max_normal"} 포함
        detected_phases : PhaseDetector가 반환한 구간 리스트

        Returns
        -------
        {"valid": bool, "reason": str | None, "details": dict}
        """
        details: Dict[str, Any] = {}

        # ── 1단계: 각도 범위 검증 ──
        angle_result = self._validate_angles(average_angles, angle_configs)
        details["angles"] = angle_result

        if not angle_result["passed"]:
            reason = (
                f"각도 범위 불일치: {angle_result['in_range']}/{angle_result['total']} "
                f"(기준 {MotionValidation.MIN_VALID_ANGLES_RATIO * 100:.0f}%)"
            )
            logger.warning(f"검증 실패 — {reason}")
            return {"valid": False, "reason": reason, "details": details}

        # ── 2단계: 구간 수 검증 ──
        phase_result = self._validate_phases(detected_phases)
        details["phases"] = phase_result

        if not phase_result["passed"]:
            reason = (
                f"구간 부족: {phase_result['real_count']}개 "
                f"(최소 {MotionValidation.MIN_PHASES_REQUIRED}개 필요)"
            )
            logger.warning(f"검증 실패 — {reason}")
            return {"valid": False, "reason": reason, "details": details}

        # ── 통과 ──
        logger.info(
            f"✅ 검증 통과 — 각도 {angle_result['in_range']}/{angle_result['total']}, "
            f"구간 {phase_result['real_count']}개"
        )
        return {"valid": True, "reason": None, "details": details}

    # ──────────────────────────────────────
    # 내부 검증 함수
    # ──────────────────────────────────────

    def _validate_angles(
        self,
        average_angles: Dict[str, float],
        angle_configs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        각 각도의 평균값이 해당 angle의 angle_validation 범위 안에 있는지 확인.

        angle_configs 구조 (sports_config.json 그대로):
        {
            "left_arm_angle": {
                "points": [...],
                "angle_validation": {"min_normal": 140.0, "max_normal": 180.0},
                "levels": {...}
            },
            ...
        }

        OUTLIER_MARGIN을 더해서 여유 범위를 적용.
        """
        if not angle_configs:
            logger.info("angle_configs 비어있음 → 각도 검증 스킵")
            return {"passed": True, "in_range": 0, "total": 0, "failed_angles": []}

        in_range = 0
        total = 0
        failed_angles = []
        margin = MotionValidation.OUTLIER_MARGIN

        for angle_name, angle_value in average_angles.items():
            angle_def = angle_configs.get(angle_name)
            if angle_def is None:
                continue

            validation = angle_def.get("angle_validation")
            if not validation:
                continue

            total += 1
            v_min = validation.get("min_normal", AngleDefaults.RANGE_MIN) - margin
            v_max = validation.get("max_normal", AngleDefaults.RANGE_MAX) + margin

            if v_min <= angle_value <= v_max:
                in_range += 1
            else:
                failed_angles.append(
                    {
                        "name": angle_name,
                        "value": round(angle_value, 1),
                        "expected_min": validation.get("min_normal"),
                        "expected_max": validation.get("max_normal"),
                    }
                )

        passed = True
        if total > 0:
            ratio = in_range / total
            passed = (
                ratio >= MotionValidation.MIN_VALID_ANGLES_RATIO
                and in_range >= MotionValidation.MIN_ANGLES_IN_RANGE
            )

        return {
            "passed": passed,
            "in_range": in_range,
            "total": total,
            "failed_angles": failed_angles,
        }

    def _validate_phases(
        self,
        detected_phases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        감지된 구간 중 fallback(full_motion)을 제외한 실제 구간 수가
        MIN_PHASES_REQUIRED 이상인지 확인.
        """
        real_phases = [
            p
            for p in detected_phases
            if p.get("name") != MotionValidation.FALLBACK_PHASE_NAME
        ]
        real_count = len(real_phases)
        passed = real_count >= MotionValidation.MIN_PHASES_REQUIRED

        return {
            "passed": passed,
            "real_count": real_count,
            "total_count": len(detected_phases),
            "phase_names": [p.get("name") for p in real_phases],
        }
