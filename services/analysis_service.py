"""
분석 서비스 (비즈니스 로직 orchestration)
"""

from utils.logger import logger
from typing import Optional
import time

from core import MediaPipeAnalyzer, AngleCalculator, PhaseDetector, LLMFeedback
from core.sport_configs import get_sport_config
from models import AnalysisResponse, AnalysisResult, PhaseInfo
from utils.decorators import measure_time, log_execution
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

        Args:
            motion_id: 분석 ID
            video_url: 영상 URL 또는 로컬 경로
            sport_type: 종목 (GOLF, WEIGHT)
            sub_category: 세부 종목 (DRIVER, SQUAT 등)

        Returns:
            분석 결과
        """

        total_start = time.time()  # ← 전체 시작 시간

        # ========== 1단계: 영상 다운로드 ==========
        step_start = time.time()
        logger.info(f"[1/7] 영상 다운로드 시작: motion_id={motion_id}")
        async with VideoResource(motion_id, video_url) as video_path:
            step1_duration = time.time() - step_start
            logger.info(f"⏱️ [1/7] 영상 다운로드 완료: {step1_duration:.2f}초")

            # ========== 2단계: 메타데이터 추출 ==========
            step_start = time.time()
            logger.info(f"[2/7] 메타데이터 추출: motion_id={motion_id}")
            metadata = self.video_service.extract_metadata(video_path)
            step2_duration = time.time() - step_start
            logger.info(f"⏱️ [2/7] 메타데이터 추출 완료: {step2_duration:.2f}초")
            logger.info(
                f"📊 영상: {metadata['width']}x{metadata['height']}, "
                f"{metadata['fps']:.1f}fps, {metadata['duration_seconds']}s"
            )

            # ========== 3단계: MediaPipe 분석 ==========
            step_start = time.time()
            logger.info(f"[3/7] MediaPipe 분석: motion_id={motion_id}")
            landmarks_data = self.mediapipe_analyzer.extract_landmarks(video_path)
            step3_duration = time.time() - step_start
            logger.info(
                f"⏱️ [3/7] MediaPipe 분석 완료: {step3_duration:.2f}초 (총 {len(landmarks_data)}개 프레임)"
            )
            logger.info(
                f"   → 프레임당 평균: {step3_duration / len(landmarks_data):.3f}초"
            )

            # ========== 4단계: 스포츠 설정 로드 ==========
            step_start = time.time()
            logger.info(f"[4/7] Config 로드: {sport_type}/{sub_category or 'default'}")
            sport_config = get_sport_config(sport_type, sub_category)
            step4_duration = time.time() - step_start
            logger.info(f"⏱️ [4/7] Config 로드 완료: {step4_duration:.2f}초")

            # ========== 5단계: 각도 계산 ==========
            step_start = time.time()
            logger.info(f"[5/7] 각도 계산")
            angle_calculator = AngleCalculator(
                angle_config=sport_config["angles"], min_visibility=0.5
            )
            angles_data = angle_calculator.calculate_angles(landmarks_data)
            step5_duration = time.time() - step_start
            logger.info(f"⏱️ [5/7] 각도 계산 완료: {step5_duration:.2f}초")
            logger.info(f"✅ 평균 각도: {angles_data['average_angles']}")

            # ========== 6단계: 구간 감지 ==========
            step_start = time.time()
            logger.info(f"[6/7] 구간 감지")
            phase_detector = PhaseDetector(
                phase_config=sport_config["phases"], fps=metadata["fps"]
            )
            phases = phase_detector.detect_phases(angles_data)
            step6_duration = time.time() - step_start
            logger.info(f"⏱️ [6/7] 구간 감지 완료: {step6_duration:.2f}초")
            logger.info(f"✅ {len(phases)}개 구간: {[p['name'] for p in phases]}")

            # ========== 7단계: LLM 피드백 생성 ==========
            step_start = time.time()
            logger.info(f"[7/7] LLM 피드백 생성: motion_id={motion_id}")
            llm_feedback_result = await self.llm_feedback.generate_feedback(
                sport_type=sport_type,
                sub_category=sub_category or "default",
                angles=angles_data["average_angles"],
                phases=phases,
                sport_config=sport_config,
            )
            step7_duration = time.time() - step_start
            logger.info(f"⏱️ [7/7] LLM 피드백 생성 완료: {step7_duration:.2f}초")

            # ========== 8단계: 응답 생성 ==========

            step_start = time.time()
            result = AnalysisResult(
                total_frames=len(landmarks_data),
                duration_seconds=metadata["duration_seconds"],
                angles=angles_data["average_angles"],
                phases=[PhaseInfo(**phase) for phase in phases],
                keypoints_sample=(
                    [
                        {
                            "x": landmarks_data[0]["landmarks"][0]["x"],
                            "y": landmarks_data[0]["landmarks"][0]["y"],
                            "z": landmarks_data[0]["landmarks"][0]["z"],
                            "visibility": landmarks_data[0]["landmarks"][0][
                                "visibility"
                            ],
                        }
                    ]
                    if landmarks_data
                    else []
                ),
            )
            step8_duration = time.time() - step_start
            logger.info(f"⏱️ [8/8] 응답 생성 완료: {step8_duration:.2f}초")

            # ========== 전체 요약 ==========
            total_duration = time.time() - total_start
            logger.info(f"")
            logger.info(f"📊 === 성능 요약 (motion_id={motion_id}) ===")
            logger.info(
                f"  1. 영상 다운로드:    {step1_duration:>6.2f}초 ({step1_duration/total_duration*100:>5.1f}%)"
            )
            logger.info(
                f"  2. 메타데이터 추출:  {step2_duration:>6.2f}초 ({step2_duration/total_duration*100:>5.1f}%)"
            )
            logger.info(
                f"  3. MediaPipe 분석:   {step3_duration:>6.2f}초 ({step3_duration/total_duration*100:>5.1f}%)"
            )
            logger.info(
                f"  4. Config 로드:      {step4_duration:>6.2f}초 ({step4_duration/total_duration*100:>5.1f}%)"
            )
            logger.info(
                f"  5. 각도 계산:        {step5_duration:>6.2f}초 ({step5_duration/total_duration*100:>5.1f}%)"
            )
            logger.info(
                f"  6. 구간 감지:        {step6_duration:>6.2f}초 ({step6_duration/total_duration*100:>5.1f}%)"
            )
            logger.info(
                f"  7. LLM 피드백:       {step7_duration:>6.2f}초 ({step7_duration/total_duration*100:>5.1f}%)"
            )
            logger.info(
                f"  8. 응답 생성:        {step8_duration:>6.2f}초 ({step8_duration/total_duration*100:>5.1f}%)"
            )
            logger.info(f"  총 소요 시간:        {total_duration:>6.2f}초")
            logger.info(f"")

            return AnalysisResponse(
                success=True,
                motion_id=motion_id,
                result=result,
                feedback=llm_feedback_result.get("feedback", ""),
                overall_score=llm_feedback_result.get("overall_score"),
                improvements=llm_feedback_result.get("improvements", []),
                prompt_version=llm_feedback_result.get("prompt_version", "unknown"),
            )

        # ========== 1단계: 영상 다운로드 (Context Manager) ==========
        # logger.info(f"[1/7] 영상 다운로드 시작: motion_id={motion_id}")
        # async with VideoResource(motion_id, video_url) as video_path:
        #
        #     # ========== 2단계: 메타데이터 추출 ==========
        #     logger.info(f"[2/7] 메타데이터 추출: motion_id={motion_id}")
        #     metadata = self.video_service.extract_metadata(video_path)
        #     logger.info(
        #         f"📊 영상: {metadata['width']}x{metadata['height']}, "
        #         f"{metadata['fps']:.1f}fps, {metadata['duration_seconds']}s"
        #     )
        #
        #     # ========== 3단계: MediaPipe 분석 ==========
        #     logger.info(f"[3/7] MediaPipe 분석: motion_id={motion_id}")
        #     landmarks_data = self.mediapipe_analyzer.extract_landmarks(video_path)
        #     logger.info(f"✅ {len(landmarks_data)}개 프레임에서 포즈 추출 완료")
        #
        #     # ========== 4단계: 스포츠 설정 로드 (JSON) ==========
        #     logger.info(f"[4/7] Config 로드: {sport_type}/{sub_category or 'default'}")
        #     sport_config = get_sport_config(sport_type, sub_category)  # 직접 호출
        #
        #     # ========== 5단계: 각도 계산 ==========
        #     logger.info(f"[5/7] 각도 계산")
        #     angle_calculator = AngleCalculator(
        #         angle_config=sport_config["angles"], min_visibility=0.5
        #     )
        #     angles_data = angle_calculator.calculate_angles(landmarks_data)
        #     logger.info(f"✅ 평균 각도: {angles_data['average_angles']}")
        #
        #     # ========== 6단계: 구간 감지 ==========
        #     logger.info(f"[6/7] 구간 감지")
        #     phase_detector = PhaseDetector(
        #         phase_config=sport_config["phases"], fps=metadata["fps"]
        #     )
        #     phases = phase_detector.detect_phases(angles_data)
        #     logger.info(f"✅ {len(phases)}개 구간: {[p['name'] for p in phases]}")
        #
        #     # ========== 7단계: LLM 피드백 생성 ==========
        #     logger.info(f"[7/7] LLM 피드백 생성: motion_id={motion_id}")
        #     llm_feedback_result = await self.llm_feedback.generate_feedback(
        #         sport_type=sport_type,
        #         sub_category=sub_category or "default",
        #         angles=angles_data["average_angles"],
        #         phases=phases,
        #         sport_config=sport_config,  # ← JSON 설정 전달
        #     )
        #
        #     # ========== 8단계: 응답 생성 ==========
        #     result = AnalysisResult(
        #         total_frames=len(landmarks_data),
        #         duration_seconds=metadata["duration_seconds"],
        #         angles=angles_data["average_angles"],
        #         phases=[PhaseInfo(**phase) for phase in phases],
        #         keypoints_sample=(
        #             [
        #                 {
        #                     "x": landmarks_data[0]["landmarks"][0]["x"],
        #                     "y": landmarks_data[0]["landmarks"][0]["y"],
        #                     "z": landmarks_data[0]["landmarks"][0]["z"],
        #                     "visibility": landmarks_data[0]["landmarks"][0][
        #                         "visibility"
        #                     ],
        #                 }
        #             ]
        #             if landmarks_data
        #             else []
        #         ),
        #     )
        #
        #     return AnalysisResponse(
        #         success=True,
        #         motion_id=motion_id,
        #         result=result,
        #         feedback=llm_feedback_result.get("feedback", ""),
        #         overall_score=llm_feedback_result.get("overall_score"),
        #         improvements=llm_feedback_result.get("improvements", []),
        #         prompt_version=llm_feedback_result.get("prompt_version", "unknown"),
        #     )

        # ========== Context Manager 종료 시 자동으로 영상 파일 삭제됨 ==========
