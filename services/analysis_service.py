"""
분석 서비스 (비즈니스 로직 orchestration)
"""
from typing import Optional
from pathlib import Path

from .video_service import VideoService
from core import MediaPipeAnalyzer, AngleCalculator, PhaseDetector, LLMFeedback
from core.sport_configs import SPORT_CONFIGS
from models import (AnalysisResponse, AnalysisResult, PhaseInfo)
from utils.errors import UnsupportedSportError
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)
setting = get_settings()

class AnalysisService:
    """
    분석 파이프라인 orchestration

    흐름:
    1. VideoService로 영상 다운로드
    2. MediaPipe로 포즈 추출
    3. 각도 계산
    4. LLM 피드백 생성
    5. 임시 파일 정리

    왜 이렇게 설계했나?
    - AnalysisService는 "지휘자" 역할
    - VideoService, MediaPipeAnalyzer 등을 조합
    - 각 모듈은 독립적으로 테스트 가능
    """

    def __init__(self):
        self.video_service = VideoService()
        self.mediapipe_analyzer = MediaPipeAnalyzer()
        self.llm_feedback = LLMFeedback()

    async def analyze(
            self,
            motion_id: int,
            video_url: str,
            sport_type: str,
            sub_category: Optional[str] = None
    ) -> AnalysisResponse:
        """
        운동 영상 분석 메인 로직
        """
        video_path = None

        try:
            # ========== 1단계: 영상 다운로드 ==========
            logger.info(f"[1/5] 영상 다운로드 시작: motion_id={motion_id}")
            video_path = self.video_service.get_video_path(motion_id, video_url)

            # ========== 2단계: 메타데이터 추출 ==========
            logger.info(f"[2/5] 메타데이터 추출: motion_id={motion_id}")
            metadata = self.video_service.extract_metadata(video_path)
            logger.info(f"📊 영상: {metadata['width']}x{metadata['height']}, {metadata['fps']:.1f}fps, {metadata['duration_seconds']}s")

            # ========== 3단계: MediaPipe 분석 ==========
            logger.info(f"[3/5] MediaPipe 분석: motion_id={motion_id}")
            landmarks_data = self.mediapipe_analyzer.extract_landmarks(video_path)
            logger.info(f"✅ {len(landmarks_data)}개 프레임에서 포즈 추출 완료")

            # ========== 4단계: 종목 + Sub-category Config 로드 ==========
            logger.info(f"[4/7] Config 로드: {sport_type}/{sub_category or 'default'}")
            sport_config = self._load_sport_config(sport_type, sub_category)

            # ========== 5단계: 각도 계산 ==========
            logger.info(f"[5/7] 각도 계산")
            angle_calculator = AngleCalculator(
                angle_config=sport_config["angles"],
                min_visibility=0.5
            )
            angles_data = angle_calculator.calculate_angles(landmarks_data)
            logger.info(f"✅ 평균 각도: {angles_data['average_angles']}")

            # ========== 6단계: 구간 감지 ==========
            logger.info(f"[6/7] 구간 감지")
            phase_detector = PhaseDetector(
                phase_config=sport_config["phases"],
                fps=metadata["fps"]
            )
            phases = phase_detector.detect_phases(angles_data)
            logger.info(f"✅ {len(phases)}개 구간: {[p['name'] for p in phases]}")

            # ========== 7단계: LLM 피드백 ==========
            logger.info(f"[5/5] LLM 피드백 (TODO): motion_id={motion_id}")
            llm_feedback_result = self.llm_feedback.generate_feedback(
                sport_type=sport_type,
                sub_category=sub_category or "default",
                average_angles=angles_data['average_angles'],
                phases=phases,
                sport_config=sport_config
            )

            # ========== 8단계: 응답 생성 ==========
            result = AnalysisResult(
                total_frames=len(landmarks_data),
                duration_seconds=metadata['duration_seconds'],
                angles=angles_data['average_angles'],
                phases=[PhaseInfo(**phase) for phase in phases],
                keypoints_sample=[
                    {
                        'x': landmarks_data[0]['landmarks'][0]['x'],
                        'y': landmarks_data[0]['landmarks'][0]['y'],
                        'z': landmarks_data[0]['landmarks'][0]['z'],
                        'visibility': landmarks_data[0]['landmarks'][0]['visibility']
                    }
                ] if landmarks_data else []
            )

            return AnalysisResponse(
                success=True,
                motion_id=motion_id,
                result=result,
                feedback=llm_feedback_result.get("feedback", ""),
                overall_score=llm_feedback_result.get("overall_score"),
                improvements=llm_feedback_result.get("improvements", []),
                prompt_version="v1.0"
            )

        finally:
            # ========== 임시 파일 정리 (무조건 실행) ==========
            if video_path and Path(video_path).exists():
                try:
                    Path(video_path).unlink()
                    logger.info(f"🗑️ 임시 파일 삭제: {video_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 임시 파일 삭제 실패: {e}")

    def _load_sport_config(self, sport_type: str, sub_category: Optional[str] = None) -> dict:
        """
        종목 + Sub-category Config 로드

        Args:
            sport_type: 종목 (GOLF, WEIGHT, ...)
            sub_category: 서브 카테고리 (DRIVER, SQUAT, ...)

        Returns:
            {"angles": {...}, "phases": [...]}

        Raises:
            UnsupportedSportError: 지원하지 않는 종목/서브카테고리
        """
        # 종목 확인
        sport_configs = SPORT_CONFIGS.get(sport_type)

        if not sport_configs:
            raise UnsupportedSportError(
                f"지원하지 않는 종목: {sport_type}. "
                f"지원 종목: {list(SPORT_CONFIGS.keys())}"
            )

        # Sub-category 확인
        if sub_category:
            config = sport_configs.get(sub_category)
            if not config:
                raise UnsupportedSportError(
                    f"지원하지 않는 {sport_type} 서브카테고리: {sub_category}. "
                    f"지원 서브카테고리: {list(sport_configs.keys())}"
                )
        else:
            # Sub-category 없으면 첫 번째 Config 사용
            config = list(sport_configs.values())[0]
            logger.warning(f"Sub-category 미지정. 기본값 사용: {list(sport_configs.keys())[0]}")

        logger.info(f"✅ {sport_type}/{sub_category or 'default'} Config 로드 완료")

        return config