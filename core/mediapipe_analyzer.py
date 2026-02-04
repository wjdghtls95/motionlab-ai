"""
MediaPipe 포즈 추출 모듈

프롬프트 A13 규칙:
- model_complexity=1 (Full 모델)
- min_detection_confidence=0.5
- min_tracking_confidence=0.5

역할:
- 영상에서 33개 랜드마크 추출
- 프레임별 키포인트 수집
- 신뢰도 낮은 프레임 필터링 (visibility < 0.5)
- 유효 프레임 비율 체크 (최소 10%)

에러 처리:
- NoKeypointsError: 유효 프레임이 10% 미만
- VideoTooShortError: 영상 길이가 1초 미만
"""

import cv2
import logging
import mediapipe as mp
from typing import List, Dict, Any
from utils.exceptions.errors import NoKeypointsError, VideoTooShortError
from config.settings import get_settings

logger = logging.getLogger(__name__)


class MediaPipeAnalyzer:
    """
    MediaPipe Pose 추출기

    33개 랜드마크 인덱스:
    0: nose, 11: left_shoulder, 12: right_shoulder,
    13: left_elbow, 14: right_elbow, 15: left_wrist, 16: right_wrist,
    23: left_hip, 24: right_hip, 25: left_knee, 26: right_knee,
    27: left_ankle, 28: right_ankle

    프롬프트 A13 참조:
    - static_image_mode=False (영상 모드, 프레임 간 추적)
    - model_complexity=1 (0=Lite, 1=Full, 2=Heavy)
    """

    def __init__(self):
        """
        MediaPipe Pose 초기화

        왜 이렇게 했나?
        - settings에서 환경 변수 로드 (테스트/프로덕션 분리 가능)
        - 싱글톤 패턴 (get_settings()는 @lru_cache로 캐싱됨)
        """
        settings = get_settings()

        # MediaPipe Pose 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # 영상 모드 (프레임 간 추적 사용)
            model_complexity=settings.MEDIAPIPE_MODEL_COMPLEXITY,  # 1 (Full)
            min_detection_confidence=settings.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,  # 0.5
            min_tracking_confidence=settings.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,  # 0.5
        )

        logger.info(
            f"MediaPipeAnalyzer 초기화: "
            f"model_complexity={settings.MEDIAPIPE_MODEL_COMPLEXITY}, "
            f"detection_confidence={settings.MEDIAPIPE_MIN_DETECTION_CONFIDENCE}"
        )

    def extract_landmarks(self, video_path: str) -> List[Dict[str, Any]]:
        """
        영상에서 프레임별 랜드마크 추출

        Args:
            video_path: 영상 파일 절대 경로

        Returns:
            List[Dict]: 프레임별 랜드마크
            [
                {
                    "frame_index": 0,
                    "timestamp": 0.0,  # 초 단위
                    "landmarks": [
                        {"x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.95},
                        ...  # 33개
                    ]
                },
                ...
            ]

        Raises:
            NoKeypointsError: 유효 프레임이 10% 미만 (AN_001)
            VideoTooShortError: 영상 길이가 1초 미만 (AN_002)
            ValueError: 영상을 열 수 없음

        왜 이렇게 했나?
        - frame_index: 각도 계산 시 프레임 위치 참조
        - timestamp: Phase 구간 감지에 사용
        - visibility: 신뢰도 낮은 키포인트 필터링
        """
        logger.info(f"📹 MediaPipe 분석 시작: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없음: {video_path}")

        # ========== 영상 메타데이터 ==========
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(
            f"📊 영상 정보: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s"
        )

        # ========== 최소 길이 체크 (1초) ==========
        if duration < 1.0:
            cap.release()
            raise VideoTooShortError(duration)

        # ========== 프레임별 랜드마크 추출 ==========
        all_landmarks = []
        frame_index = 0
        valid_frames = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # RGB 변환 (MediaPipe 요구사항)
                # 왜? MediaPipe는 RGB 형식만 처리 가능 (OpenCV는 BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 포즈 추출 (핵심 로직)
                results = self.pose.process(frame_rgb)

                # 랜드마크가 검출된 경우만 저장
                if results.pose_landmarks:
                    # 33개 랜드마크 수집
                    landmarks = []
                    for lm in results.pose_landmarks.landmark:
                        landmarks.append(
                            {
                                "x": float(lm.x),  # 정규화된 x 좌표 (0.0~1.0)
                                "y": float(lm.y),  # 정규화된 y 좌표 (0.0~1.0)
                                "z": float(lm.z),  # 깊이 (음수=앞, 양수=뒤)
                                "visibility": float(lm.visibility),  # 신뢰도 (0.0~1.0)
                            }
                        )

                    all_landmarks.append(
                        {
                            "frame_index": frame_index,
                            "timestamp": frame_index / fps,  # 초 단위 타임스탬프
                            "landmarks": landmarks,
                        }
                    )
                    valid_frames += 1

                frame_index += 1

                # ========== 진행률 로그 (10% 단위) ==========
                if frame_index % max(1, total_frames // 10) == 0:
                    progress = (frame_index / total_frames) * 100
                    logger.debug(
                        f"🔄 진행률: {progress:.0f}% ({frame_index}/{total_frames} frames)"
                    )

        finally:
            # 리소스 정리 (무조건 실행)
            cap.release()

        # ========== 유효 프레임 비율 체크 ==========
        valid_ratio = valid_frames / total_frames if total_frames > 0 else 0

        logger.info(
            f"✅ MediaPipe 분석 완료: "
            f"{valid_frames}/{total_frames} frames ({valid_ratio:.1%} 유효)"
        )

        # 유효 프레임이 10% 미만이면 에러
        # 왜 10%? 영상 품질이 너무 낮으면 분석 불가능
        if valid_ratio < 0.1:
            raise NoKeypointsError()

        return all_landmarks

    def get_landmark_by_name(
        self, landmarks: List[Dict], name: str
    ) -> Dict[str, float]:
        """
        랜드마크 이름으로 좌표 가져오기

        Args:
            landmarks: 33개 랜드마크 리스트
            name: "left_shoulder", "right_elbow" 등

        Returns:
            {"x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.95}

        왜 이 메서드가 필요한가?
        - Phase 6-2 각도 계산에서 사용
        - 인덱스 대신 의미 있는 이름으로 접근 가능

        사용 예:
        ```python
        left_shoulder = analyzer.get_landmark_by_name(
            frame["landmarks"],
            "left_shoulder"
        )
        ```
        """
        LANDMARK_MAP = {
            "nose": 0,
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
        }

        index = LANDMARK_MAP.get(name)
        if index is None:
            raise ValueError(f"Unknown landmark name: {name}")

        return landmarks[index]

    def __del__(self):
        """
        리소스 정리 (소멸자)

        왜 필요한가?
        - MediaPipe Pose 객체는 메모리/GPU 리소스를 점유
        - 명시적으로 close() 호출 필요
        """
        if hasattr(self, "pose"):
            self.pose.close()
            logger.debug("MediaPipe Pose 리소스 정리 완료")
