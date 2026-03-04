"""
분석 서비스 (비즈니스 로직 orchestration)
"""

from utils.logger import logger
from typing import Optional

from core import MediaPipeAnalyzer, AngleCalculator, PhaseDetector, LLMFeedback
from core.sport_configs import get_sport_config, get_config_version
from core.sport_configs.base_config import UserLevel
from core.constants import PipelineConfig, LLMConfig
from models import AnalysisResponse, AnalysisResult, PhaseInfo
from utils.decorators import measure_time, log_execution
from utils.response_builder import extract_keypoints_sample
from utils.timer import StepTimer
from .video_service import VideoResource, VideoService


class AnalysisService:
    """분석 파이프라인 orchestration"""

    def __init__(self):
        self.video_service = VideoService()
        self.mediapipe_analyzer = MediaPipeAnalyzer()
        self.llm_feedback = LLMFeedback()

    @log_execution(log_result=False)
    @measure_time(threshold_ms=10000)
    async def analyze(
        self,
        motion_id: int,
        video_url: str,
        sport_type: str,
        sub_category: Optional[str] = None,
        level: UserLevel = UserLevel.INTERMEDIATE,
    ) -> AnalysisResponse:
        """
        운동 영상 분석 메인 로직

        흐름:
        1. 영상 다운로드 (Context Manager로 자동 정리)
        2. 메타데이터 추출 (FPS, 해상도, 길이)
        3. MediaPipe 포즈 추출 (33개 랜드마크)
        4. 스포츠 설정 로드 (JSON)
        5. 각도 계산
        6. 구간 감지
        7. LLM 피드백 생성
        8. 응답 생성
        """

        timer = StepTimer()
        timer.start_total()

        config_version = get_config_version()
        logger.info(
            f"🚀 분석 시작: motion_id={motion_id}, "
            f"sport={sport_type}/{sub_category}, "
            f"level={level.value}, "
            f"config_version={config_version}"
        )

        async with VideoResource(motion_id, video_url) as video_path:
            # ========== 1단계: 영상 다운로드 ==========
            with timer.step(1, PipelineConfig.TOTAL_STEPS, "영상 다운로드"):
                pass

            # ========== 2단계: 메타데이터 추출 ==========
            with timer.step(2, PipelineConfig.TOTAL_STEPS, "메타데이터 추출"):
                metadata = self.video_service.extract_metadata(video_path)
            logger.info(
                f"📊 영상: {metadata['width']}x{metadata['height']}, "
                f"{metadata['fps']:.1f}fps, "
                f"{metadata['duration_seconds']}s"
            )

            # ========== 3단계: MediaPipe 분석 ==========
            with timer.step(3, PipelineConfig.TOTAL_STEPS, "MediaPipe 분석"):
                landmarks_data = self.mediapipe_analyzer.extract_landmarks(video_path)
            logger.info(
                f"   → {landmarks_data['valid_frames']}/"
                f"{landmarks_data['total_frames']}프레임 추출"
            )

            # ========== 4단계: 스포츠 설정 로드 ==========
            with timer.step(4, PipelineConfig.TOTAL_STEPS, "Config 로드"):
                sport_config = get_sport_config(sport_type, sub_category, level=level)

            # ========== 5단계: 각도 계산 ==========
            with timer.step(5, PipelineConfig.TOTAL_STEPS, "각도 계산"):
                angle_calculator = AngleCalculator(
                    angle_config=sport_config["angles"],
                    min_visibility=PipelineConfig.MIN_VISIBILITY,
                )
                angles_data = angle_calculator.calculate_angles(landmarks_data)
            logger.info(f"✅ 평균 각도: {angles_data['average_angles']}")

            # ========== 6단계: 구간 감지 ==========
            with timer.step(6, PipelineConfig.TOTAL_STEPS, "구간 감지"):
                phase_detector = PhaseDetector(
                    phase_config=sport_config["phases"], fps=metadata["fps"]
                )
                phases = phase_detector.detect_phases(angles_data)
            logger.info(f"✅ {len(phases)}개 구간: {[p['name'] for p in phases]}")

            # ========== 7단계: LLM 피드백 생성 ==========
            with timer.step(7, PipelineConfig.TOTAL_STEPS, "LLM 피드백"):
                llm_feedback_result = await self.llm_feedback.generate_feedback(
                    sport_type=sport_type,
                    sub_category=sub_category or "default",
                    angles=angles_data["average_angles"],
                    phases=phases,
                    sport_config=sport_config,
                    level=level,
                )

            timer.summary(motion_id)

            # ========== 8단계: 응답 생성 ==========

            return AnalysisResponse(
                success=True,
                motion_id=motion_id,
                result=AnalysisResult(
                    total_frames=landmarks_data["total_frames"],
                    duration_seconds=metadata["duration_seconds"],
                    angles=angles_data["average_angles"],
                    phases=[PhaseInfo(**phase) for phase in phases],
                    keypoints_sample=extract_keypoints_sample(landmarks_data),
                ),
                feedback=llm_feedback_result.get("feedback", ""),
                overall_score=llm_feedback_result.get("overall_score"),
                improvements=llm_feedback_result.get("improvements", []),
                prompt_version=llm_feedback_result.get(
                    "prompt_version", LLMConfig.UNKNOWN_VERSION
                ),
            )

        # ========== Context Manager 종료 시 자동으로 영상 파일 삭제됨 ==========
