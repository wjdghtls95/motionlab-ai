"""
Config 기반 구간 감지기 (PhaseDetector)

sports_config.json의 phases 배열을 읽어 자동으로 구간을 감지.
코드에 특정 종목 이름이나 각도 이름 없음 — config가 모든 것을 결정.

감지 흐름:
  1. phases config 로드
  2. 각 phase의 detection_rule에 맞는 핸들러 실행
  3. 감지된 keyframe들을 시간순 정렬 → 구간 변환
  4. config가 없으면 full_motion 단일 구간 폴백
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

from core.constants import PhaseDetection, MotionValidation

logger = logging.getLogger(__name__)


class PhaseDetector:
    """Config 기반 구간 감지 — 종목 독립적"""

    def __init__(self, phase_config: List[Dict[str, Any]], fps: int = 24):
        """
        Parameters
        ----------
        phase_config : sports_config.json에서 읽어온 phases 리스트
        fps : 영상 프레임 레이트
        """
        self.phase_config = phase_config or []
        self.fps = fps

        self._rule_handlers = {
            PhaseDetection.RULE_STABILIZATION: self._detect_stabilization,
            PhaseDetection.RULE_ANGLE_INCREASE: self._detect_angle_increase,
            PhaseDetection.RULE_ANGLE_DECREASE: self._detect_angle_decrease,
            PhaseDetection.RULE_ANGLE_MAX: self._detect_angle_max,
            PhaseDetection.RULE_ANGLE_MIN: self._detect_angle_min,
            PhaseDetection.RULE_VELOCITY_THRESHOLD: self._detect_velocity_threshold,
            PhaseDetection.RULE_DIRECTION_CHANGE: self._detect_direction_change,
            PhaseDetection.RULE_PRE_MOTION: self._detect_pre_motion,
        }

    # ──────────────────────────────────────
    # Public
    # ──────────────────────────────────────

    def detect_phases(self, angles_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        구간 감지 실행

        Parameters
        ----------
        angles_data : AngleCalculator가 반환한 프레임별 각도 데이터
                      {"angle_name": {"frames": {0: val, 1: val, ...}, "average": float}, ...}

        Returns
        -------
        [{"name": str, "start_frame": int, "end_frame": int, "duration_ms": int}, ...]
        """
        if not self.phase_config:
            logger.warning("phase_config 비어있음 → 폴백 구간 반환")
            return self._build_fallback(angles_data)

        total_frames = self._get_total_frames(angles_data)

        if total_frames < PhaseDetection.MIN_FRAMES_FOR_DETECTION:
            logger.warning(f"프레임 수 부족 ({total_frames}) → 폴백 구간 반환")
            return self._build_fallback(angles_data)

        # 각 phase config의 규칙을 실행해서 keyframe(시작 프레임) 수집
        keyframes: List[Dict[str, Any]] = []
        detected_keyframes: Dict[str, int] = {}  # name → frame (search_after 참조용)

        for phase_def in self.phase_config:
            name = phase_def.get("name", "unknown")
            rule = phase_def.get("detection_rule", "")
            target_angle = phase_def.get("target_angle")
            params = phase_def.get("params", {})

            # 핸들러 찾기
            handler = self._rule_handlers.get(rule)
            if handler is None:
                logger.warning(f"알 수 없는 규칙 '{rule}' → '{name}' 구간 건너뜀")
                continue

            # target_angle 데이터 추출
            if target_angle is None:
                logger.warning(f"'{name}' 구간에 target_angle 없음 → 건너뜀")
                continue

            angle_series = self._extract_angle_series(angles_data, target_angle)
            if angle_series is None:
                logger.warning(
                    f"'{target_angle}' 각도 데이터 없음 → '{name}' 구간 건너뜀"
                )
                continue

            # search_after / search_start_pct 처리
            search_start = self._resolve_search_start(
                params, detected_keyframes, total_frames
            )
            # search_end_pct 처리
            search_end = self._resolve_search_end(params, total_frames)

            # 규칙 실행
            frame = handler(
                angle_series, params, search_start, search_end, total_frames
            )

            if frame is not None:
                keyframes.append({"name": name, "frame": frame})
                detected_keyframes[name] = frame
                logger.debug(f"구간 '{name}' 감지 → frame {frame}")
            else:
                logger.info(f"구간 '{name}' 감지 실패")

        # keyframe → phase 변환
        if not keyframes:
            logger.warning("감지된 구간 없음 → 폴백 구간 반환")
            return self._build_fallback(angles_data)

        return self._keyframes_to_phases(keyframes, total_frames)

    # ──────────────────────────────────────
    # 감지 규칙 핸들러
    # ──────────────────────────────────────

    def _detect_stabilization(
        self,
        series: np.ndarray,
        params: dict,
        search_start: int,
        search_end: int,
        total_frames: int,
    ) -> Optional[int]:
        """안정 구간 감지: 이동 윈도우 내 std가 threshold 이하인 구간"""
        window = params.get("window", PhaseDetection.DEFAULT_WINDOW)
        std_threshold = params.get(
            "std_threshold", PhaseDetection.DEFAULT_STD_THRESHOLD
        )
        position = params.get("position", "start")  # "start" 또는 "end"

        end = min(search_end, len(series))
        if position == "end":
            for i in range(end - window, search_start, -1):
                if i < 0:
                    break
                segment = series[i : i + window]
                if np.std(segment) <= std_threshold:
                    return i
        else:
            for i in range(search_start, end - window + 1):
                segment = series[i : i + window]
                if np.std(segment) > std_threshold:
                    return max(i - 1, 0)

        return None

    def _detect_angle_increase(
        self,
        series: np.ndarray,
        params: dict,
        search_start: int,
        search_end: int,
        total_frames: int,
    ) -> Optional[int]:
        """각도 증가 시작점 감지"""
        window = params.get("window", PhaseDetection.DEFAULT_WINDOW)
        min_change = params.get("min_change", PhaseDetection.DEFAULT_THRESHOLD)

        end = min(search_end, len(series)) - window
        for i in range(search_start, end):
            diff = series[i + window] - series[i]
            if diff >= min_change:
                return i

        return None

    def _detect_angle_decrease(
        self,
        series: np.ndarray,
        params: dict,
        search_start: int,
        search_end: int,
        total_frames: int,
    ) -> Optional[int]:
        """각도 감소 시작점 감지"""
        window = params.get("window", PhaseDetection.DEFAULT_WINDOW)
        min_change = params.get("min_change", PhaseDetection.DEFAULT_THRESHOLD)

        end = min(search_end, len(series)) - window
        for i in range(search_start, end):
            diff = series[i] - series[i + window]
            if diff >= min_change:
                return i

        return None

    def _detect_angle_max(
        self,
        series: np.ndarray,
        params: dict,
        search_start: int,
        search_end: int,
        total_frames: int,
    ) -> Optional[int]:
        """search_start ~ search_end 구간에서 최대 각도 프레임"""
        end = min(search_end, len(series))
        if search_start >= end:
            return None
        sub = series[search_start:end]
        return int(search_start + np.argmax(sub))

    def _detect_angle_min(
        self,
        series: np.ndarray,
        params: dict,
        search_start: int,
        search_end: int,
        total_frames: int,
    ) -> Optional[int]:
        """search_start ~ search_end 구간에서 최소 각도 프레임"""
        end = min(search_end, len(series))
        if search_start >= end:
            return None
        sub = series[search_start:end]
        return int(search_start + np.argmin(sub))

    def _detect_velocity_threshold(
        self,
        series: np.ndarray,
        params: dict,
        search_start: int,
        search_end: int,
        total_frames: int,
    ) -> Optional[int]:
        """각도 변화 속도(velocity)가 threshold를 넘는 시점"""
        threshold = params.get("threshold", PhaseDetection.DEFAULT_THRESHOLD)
        direction = params.get("direction", "any")  # "positive", "negative", "any"

        velocity = np.diff(series)
        end = min(search_end, len(velocity))

        for i in range(search_start, end):
            v = velocity[i]
            if direction == "positive" and v >= threshold:
                return i
            elif direction == "negative" and v <= -threshold:
                return i
            elif direction == "any" and abs(v) >= threshold:
                return i

        return None

    def _detect_direction_change(
        self,
        series: np.ndarray,
        params: dict,
        search_start: int,
        search_end: int,
        total_frames: int,
    ) -> Optional[int]:
        """각도가 감소하다가 증가로 전환하는 첫 번째 지점 (backswing_top용)

        velocity 부호가 음수→양수로 바뀌는 전환점을 찾는다.
        노이즈 제거를 위해 rolling mean으로 velocity 평활화 후 탐색.
        """
        min_decrease_frames = params.get("min_decrease_frames", 3)
        smooth_window = params.get("smooth_window", 7)

        velocity = np.diff(series)

        # Rolling mean으로 노이즈 제거
        if len(velocity) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(velocity, kernel, mode="same")
        else:
            smoothed = velocity

        end = min(search_end, len(smoothed))
        neg_count = 0

        for i in range(search_start, end):
            v = smoothed[i]
            if v < 0:
                neg_count += 1
            elif v > 0 and neg_count >= min_decrease_frames:
                # 충분히 감소한 뒤 처음 증가로 전환된 지점
                return i
            elif v > 0:
                # 감소가 충분하지 않은 상태에서 양수 → 리셋
                neg_count = 0

        return None

    def _detect_pre_motion(
        self,
        series: np.ndarray,
        params: dict,
        search_start: int,
        search_end: int,
        total_frames: int,
    ) -> Optional[int]:
        """큰 움직임이 시작되기 직전 프레임 (address용)

        |velocity|가 motion_threshold를 처음 초과하는 프레임 - 1을 반환.
        스윙이 시작되기 직전의 마지막 정지 프레임을 찾는다.
        """
        motion_threshold = params.get("motion_threshold", 3.0)

        velocity_abs = np.abs(np.diff(series))
        end = min(search_end, len(velocity_abs))

        for i in range(search_start, end):
            if velocity_abs[i] >= motion_threshold:
                return max(i - 1, 0)

        return None

    # ──────────────────────────────────────
    # 유틸리티
    # ──────────────────────────────────────

    def _extract_angle_series(
        self, angles_data: Dict[str, Any], target_angle: str
    ) -> Optional[np.ndarray]:
        """angles_data에서 특정 각도의 프레임별 값을 numpy 배열로 추출"""
        angle_data = angles_data.get(target_angle)
        if angle_data is None:
            return None

        frames = angle_data.get("frames", {})
        if not frames:
            return None

        max_frame = max(frames.keys())
        series = np.zeros(max_frame + 1)

        for frame_idx, value in frames.items():
            series[frame_idx] = value

        # 0인 프레임 보간 (간단한 선형 보간)
        nonzero = np.nonzero(series)[0]
        if len(nonzero) > 1:
            series = np.interp(np.arange(len(series)), nonzero, series[nonzero])

        return series

    def _resolve_search_start(
        self, params: dict, detected_keyframes: Dict[str, int], total_frames: int
    ) -> int:
        """search_after / search_start_pct 기반 탐색 하한 계산"""
        search_after = params.get("search_after")
        base = (
            detected_keyframes[search_after] + 1
            if (search_after and search_after in detected_keyframes)
            else 0
        )

        search_start_pct = params.get("search_start_pct")
        if search_start_pct is not None:
            pct_frame = int(total_frames * search_start_pct)
            return max(base, pct_frame)
        return base

    def _resolve_search_end(self, params: dict, total_frames: int) -> int:
        """search_end_pct 기반 탐색 상한 계산"""
        search_end_pct = params.get("search_end_pct")
        if search_end_pct is not None:
            return int(total_frames * search_end_pct)
        return total_frames

    def _get_total_frames(self, angles_data: Dict[str, Any]) -> int:
        """angles_data에서 전체 프레임 수 추출"""
        max_frame = 0
        for angle_data in angles_data.values():
            frames = angle_data.get("frames", {})
            if frames:
                max_frame = max(max_frame, max(frames.keys()))
        return max_frame + 1

    def _build_fallback(self, angles_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """폴백: 전체 영상을 하나의 구간으로"""
        total_frames = self._get_total_frames(angles_data)
        end_frame = max(total_frames - 1, 0)

        return [
            {
                "name": MotionValidation.FALLBACK_PHASE_NAME,
                "start_frame": 0,
                "end_frame": end_frame,
                "duration_ms": self._frame_to_ms(end_frame),
            }
        ]

    def _keyframes_to_phases(
        self, keyframes: List[Dict[str, Any]], total_frames: int
    ) -> List[Dict[str, Any]]:
        """감지된 keyframe 리스트를 연속 구간으로 변환"""
        # 프레임 순서로 정렬
        sorted_kf = sorted(keyframes, key=lambda x: x["frame"])

        phases = []
        for i, kf in enumerate(sorted_kf):
            start = kf["frame"]
            if i + 1 < len(sorted_kf):
                end = sorted_kf[i + 1]["frame"] - 1
            else:
                end = total_frames - 1

            # 최소 프레임 수 보장
            if end - start < PhaseDetection.MIN_PHASE_FRAMES:
                end = min(start + PhaseDetection.MIN_PHASE_FRAMES, total_frames - 1)

            phases.append(
                {
                    "name": kf["name"],
                    "start_frame": start,
                    "end_frame": end,
                    "duration_ms": self._frame_to_ms(end - start),
                }
            )

        return phases

    def _frame_to_ms(self, frame_count: int) -> int:
        """프레임 수를 밀리초로 변환"""
        if self.fps <= 0:
            return 0
        return int((frame_count / self.fps) * 1000)
