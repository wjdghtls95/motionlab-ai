import cv2
import logging
import mediapipe as mp
from typing import List, Dict, Any

from core.landmarks import get_landmark_index
from utils.exceptions.errors import NoKeypointsError, VideoTooShortError
from config.settings import get_settings

logger = logging.getLogger(__name__)


class MediaPipeAnalyzer:
    """
    MediaPipe Pose 추출기

    33개 랜드마크 landmark.py 에서 관리
    """

    def __init__(self):
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

    def extract_landmarks(self, video_path: str) -> Dict[str, Any]:
        """
        영상에서 프레임별 랜드마크 추출

        Args:
            video_path: 영상 파일 절대 경로

        Returns:
            Dict[str, Any]: 프레임 데이터 + 메타 정보
            {
                "frames": [
                    {
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "landmarks": [
                            {"x": 0.5, "y": 0.3, "z": -0.1,
                             "visibility": 0.95},
                            ...  # 33개
                        ]
                    },
                    ...
                ],
                "total_frames": 264,
                "valid_frames": 248,
                "fps": 29.0
            }

        Raises:
            NoKeypointsError: 유효 프레임이 10% 미만 (AN_001)
            VideoTooShortError: 영상 길이가 1초 미만 (AN_002)
            ValueError: 영상을 열 수 없음
        """
        logger.info(f"📹 MediaPipe 분석 시작: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없음: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(
            f"📊 영상 정보: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s"
        )

        if duration < 1.0:
            cap.release()
            raise VideoTooShortError(duration)

        all_landmarks, valid_frames = self._process_frames(cap, fps, total_frames)

        valid_ratio = valid_frames / total_frames if total_frames > 0 else 0
        logger.info(
            f"✅ MediaPipe 분석 완료: "
            f"{valid_frames}/{total_frames} frames "
            f"({valid_ratio:.1%} 유효)"
        )

        if valid_ratio < 0.1:
            raise NoKeypointsError()

        return {
            "frames": all_landmarks,
            "total_frames": total_frames,
            "valid_frames": valid_frames,
            "fps": fps,
        }

    def _process_frames(
        self, cap: cv2.VideoCapture, fps: float, total_frames: int
    ) -> tuple:
        """프레임 순회 및 랜드마크 추출"""
        all_landmarks = []
        frame_index = 0
        valid_frames = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)

                if results.pose_landmarks:
                    landmarks = [
                        {
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                            "visibility": float(lm.visibility),
                        }
                        for lm in results.pose_landmarks.landmark
                    ]

                    all_landmarks.append(
                        {
                            "frame_index": frame_index,
                            "timestamp": frame_index / fps,
                            "landmarks": landmarks,
                        }
                    )
                    valid_frames += 1

                frame_index += 1
                self._log_progress(frame_index, total_frames)

        finally:
            cap.release()

        return all_landmarks, valid_frames

    @staticmethod
    def _log_progress(frame_index: int, total_frames: int):
        """진행률 로그 (10% 단위)"""
        interval = max(1, total_frames // 10)
        if frame_index % interval == 0:
            progress = (frame_index / total_frames) * 100
            logger.debug(
                f"🔄 진행률: {progress:.0f}% ({frame_index}/{total_frames} frames)"
            )

    def get_landmark_by_name(
        self, landmarks: List[Dict], name: str
    ) -> Dict[str, float]:
        """
        랜드마크 이름으로 좌표 가져오기
        """
        index = get_landmark_index(name)

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
