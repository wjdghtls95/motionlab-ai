"""
구간 감지 (Phase Detection)

골프 스윙을 5가지 구간으로 자동 분류:
1. Address (준비)
2. Backswing (백스윙)
3. Top (정점)
4. Downswing (다운스윙)
5. Follow-through (팔로우스루)
"""
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class PhaseDetector:
    """
    골프 스윙 구간 감지

    알고리즘:ㅇ
    1. 왼팔 각도 변화율 기반 구간 탐지
    2. 척추 각도 보조 지표
    3. GolfDB 규칙 적용
    """

    def __init__(self, phase_config: List[Dict], fps: float = 24.0):
        """
        Args:
            fps: 영상 프레임레이트 (기본 24fps)
        """
        self.phase_config = phase_config
        self.fps = fps

    def detect_phases(self, angles_data: Dict) -> List[Dict]:
        """
        구간 감지 메인 함수

        Args:
            angles_data: AngleCalculator 출력
                {
                    "frame_angles": [
                        {"frame_index": 0, "angles": {...}},
                        ...
                    ],
                    "average_angles": {...}
                }

        Returns:
            [
                {
                    "name": "backswing",
                    "start_frame": 21,
                    "end_frame": 60,
                    "duration_ms": 1625
                },
                ...
            ]
        """
        frame_angles = angles_data["frame_angles"]

        if not frame_angles:
            logger.warning("프레임 각도 데이터 없음")
            return []

        # 왼팔 각도 추출
        left_arm_angles = self._extract_angle_series(frame_angles, "left_arm_angle")

        if not left_arm_angles:
            logger.warning("왼팔 각도 데이터 없음")
            return []

        # 구간 경계점 탐지
        address_end = self._detect_address_end(left_arm_angles)
        top_frame = self._detect_top(left_arm_angles, address_end)
        impact_frame = self._detect_impact(left_arm_angles, top_frame)
        finish_frame = self._detect_finish(left_arm_angles, impact_frame)

        # 구간 생성
        phases = self._create_phases(
            address_end,
            top_frame,
            impact_frame,
            finish_frame,
            len(left_arm_angles)
        )

        logger.info(f"✅ {len(phases)}개 구간 감지 완료")
        return phases

    def _extract_angle_series(
            self,
            frame_angles: List[Dict],
            angle_name: str
    ) -> List[float]:
        """
        특정 각도의 시계열 데이터 추출

        Args:
            frame_angles: 프레임별 각도 리스트
            angle_name: 각도 이름 (예: "left_arm_angle")

        Returns:
            [165.2, 168.1, 170.3, ...]
        """
        angles = []
        for frame_data in frame_angles:
            angle_value = frame_data["angles"].get(angle_name)
            if angle_value is not None:
                angles.append(angle_value)
            else:
                # 이전 값으로 보간 (간단한 방법)
                if angles:
                    angles.append(angles[-1])
                else:
                    angles.append(0.0)

        return angles

    def _detect_address_end(self, left_arm_angles: List[float]) -> int:
        """
        Address 구간 종료 지점 (스윙 시작)

        기준: 왼팔 각도가 5도 이상 증가한 첫 프레임
        """
        window_size = 5  # 5프레임 이동 평균
        threshold = 5.0  # 5도 이상 변화

        for i in range(window_size, len(left_arm_angles) - window_size):
            prev_avg = np.mean(left_arm_angles[i - window_size:i])
            curr_avg = np.mean(left_arm_angles[i:i + window_size])

            if curr_avg - prev_avg > threshold:
                logger.info(f"Address 종료: 프레임 {i}")
                return i

        # 감지 실패 시 첫 10% 지점
        return int(len(left_arm_angles) * 0.1)

    def _detect_top(self, left_arm_angles: List[float], start_frame: int) -> int:
        """
        Top (정점) 감지

        기준: 왼팔 각도 최대값
        """
        # Address 이후 구간에서 최대값 찾기
        search_range = left_arm_angles[start_frame:]

        if not search_range:
            return start_frame

        max_idx = np.argmax(search_range)
        top_frame = start_frame + max_idx

        logger.info(f"Top 감지: 프레임 {top_frame} (각도: {left_arm_angles[top_frame]:.1f}°)")
        return top_frame

    def _detect_impact(self, left_arm_angles: List[float], top_frame: int) -> int:
        """
        Impact (임팩트) 감지

        기준: Top 이후 왼팔 각도가 최소값에 도달
        """
        # Top 이후 구간에서 최소값 찾기
        search_range = left_arm_angles[top_frame:]

        if not search_range:
            return top_frame

        min_idx = np.argmin(search_range)
        impact_frame = top_frame + min_idx

        logger.info(f"Impact 감지: 프레임 {impact_frame} (각도: {left_arm_angles[impact_frame]:.1f}°)")
        return impact_frame

    def _detect_finish(self, left_arm_angles: List[float], impact_frame: int) -> int:
        """
        Finish (팔로우스루 종료) 감지

        기준: Impact 이후 각도 변화율이 안정화
        """
        window_size = 10
        threshold = 2.0  # 2도 이내 변화

        for i in range(impact_frame + window_size, len(left_arm_angles) - window_size):
            angle_variance = np.std(left_arm_angles[i:i + window_size])

            if angle_variance < threshold:
                logger.info(f"Finish 감지: 프레임 {i}")
                return i

        # 감지 실패 시 마지막 프레임
        return len(left_arm_angles) - 1

    def _create_phases(
            self,
            address_end: int,
            top_frame: int,
            impact_frame: int,
            finish_frame: int,
            total_frames: int
    ) -> List[Dict]:
        """
        구간 정보 생성

        Returns:
            [
                {"name": "address", "start_frame": 0, "end_frame": 21, "duration_ms": 875},
                ...
            ]
        """
        phases = [
            {
                "name": "address",
                "start_frame": 0,
                "end_frame": address_end,
                "duration_ms": self._frame_to_ms(address_end)
            },
            {
                "name": "backswing",
                "start_frame": address_end,
                "end_frame": top_frame,
                "duration_ms": self._frame_to_ms(top_frame - address_end)
            },
            {
                "name": "top",
                "start_frame": top_frame - 2,  # Top 전후 2프레임
                "end_frame": top_frame + 2,
                "duration_ms": self._frame_to_ms(4)
            },
            {
                "name": "downswing",
                "start_frame": top_frame,
                "end_frame": impact_frame,
                "duration_ms": self._frame_to_ms(impact_frame - top_frame)
            },
            {
                "name": "follow_through",
                "start_frame": impact_frame,
                "end_frame": finish_frame,
                "duration_ms": self._frame_to_ms(finish_frame - impact_frame)
            }
        ]

        return phases

    def _frame_to_ms(self, frames: int) -> int:
        """
        프레임 수 → 밀리초 변환

        Args:
            frames: 프레임 수

        Returns:
            밀리초
        """
        return int((frames / self.fps) * 1000)
