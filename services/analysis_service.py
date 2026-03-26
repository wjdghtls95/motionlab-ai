"""
분석 서비스 (비즈니스 로직 orchestration)
"""

from utils.exceptions import InvalidMotionError
from utils.logger import logger
from typing import Optional

from core import MediaPipeAnalyzer, AngleCalculator, PhaseDetector, LLMFeedback
from core.sport_configs import get_sport_config, get_config_version
from core.sport_configs.base_config import UserLevel
from core.constants import PipelineConfig, LLMConfig
from core.motion_validator import MotionValidator
from models import AnalysisResponse, AnalysisResult, PhaseInfo
from utils.angle_utils import to_phase_input
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
        self.motion_validator = MotionValidator()

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
        1. 영상 다운로드
        2. 메타데이터 추출
        3. MediaPipe 포즈 추출
        4. 스포츠 설정 로드
        5. 각도 계산
        6. 구간 감지
        7. 종목 - 영상 검증
        8. LLM 피드백 생성
        """

        timer = StepTimer()
        timer.start_total()
        total = PipelineConfig.TOTAL_STEPS

        config_version = get_config_version()
        logger.info(
            f"🚀 분석 시작: motion_id={motion_id}, "
            f"sport={sport_type}/{sub_category}, "
            f"level={level.value}, "
            f"config_version={config_version}"
        )

        async with VideoResource(motion_id, video_url) as video_path:
            # ========== 영상 다운로드 ==========
            with timer.next_step(total, "영상 다운로드"):
                pass  # VideoResource context manager가 이미 다운로드 완료

            # ========== 메타데이터 추출 ==========
            with timer.next_step(total, "메타데이터 추출"):
                metadata = self.video_service.extract_metadata(video_path)
            logger.info(
                f"📊 영상: {metadata['width']}x{metadata['height']}, "
                f"{metadata['fps']:.1f}fps, "
                f"{metadata['duration_seconds']}s"
            )

            # ========== MediaPipe 분석 ==========
            with timer.next_step(total, "MediaPipe 분석"):
                landmarks_data = self.mediapipe_analyzer.extract_landmarks(video_path)
            logger.info(
                f"   → {landmarks_data['valid_frames']}/"
                f"{landmarks_data['total_frames']}프레임 추출"
            )

            # ========== 스포츠 설정 로드 ==========
            with timer.next_step(total, "Config 로드"):
                sport_config = get_sport_config(sport_type, sub_category, level=level)

            # ========== 각도 계산 ==========
            with timer.next_step(total, "각도 계산"):
                angle_calculator = AngleCalculator(
                    angle_config=sport_config["angles"],
                    min_visibility=PipelineConfig.MIN_VISIBILITY,
                )
                angles_data = angle_calculator.calculate_angles(landmarks_data)
            logger.info(f"✅ 평균 각도: {angles_data['average_angles']}")

            # ========== 구간 감지 ==========
            with timer.next_step(total, "구간 감지"):
                phase_detector = PhaseDetector(
                    phase_config=sport_config["phases"],
                    fps=metadata["fps"],
                )
                phases = phase_detector.detect_phases(
                    to_phase_input(angles_data["frame_angles"])
                )
            logger.info(f"✅ {len(phases)}개 구간: {[p['name'] for p in phases]}")

            # ========== 종목-영상 검증 ==========
            with timer.next_step(total, "종목 검증"):
                validation_result = self.motion_validator.validate_motion(
                    average_angles=angles_data["average_angles"],
                    angle_configs=sport_config["angles"],
                    detected_phases=phases,
                )

            if not validation_result["valid"]:
                raise InvalidMotionError(
                    details=(
                        f"{sport_type}/{sub_category} 종목과 영상이 일치하지 않습니다. "
                        f"{validation_result['reason']}. "
                        f"해당 종목의 영상을 올려주세요."
                    )
                )

            # ========== LLM 피드백 생성 ==========
            with timer.next_step(total, "LLM 피드백"):
                llm_feedback_result = await self.llm_feedback.generate_feedback(
                    sport_type=sport_type,
                    sub_category=sub_category or "default",
                    angles=angles_data["average_angles"],
                    phases=phases,
                    sport_config=sport_config,
                    level=level,
                    angle_scores=angles_data.get("angle_scores", {}),
                    weighted_score=angles_data.get("weighted_score"),
                )

            timer.summary(motion_id)

            # ========== 응답 생성 ==========
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
