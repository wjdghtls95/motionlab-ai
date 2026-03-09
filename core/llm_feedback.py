"""
MotionLab AI - LLM 피드백 생성 (YAML 프롬프트 + 자동 버전 관리)
"""

import json
from typing import Dict, Any, List, Optional

from openai import AsyncOpenAI

from config import get_settings, settings
from core.prompts.loader import prompt_loader
from core.sport_configs.base_config import UserLevel
from core.constants import (
    FeedbackScore,
    FeedbackThreshold,
    LLMConfig,
    AngleDefaults,
    LEVEL_TONE,
)
from utils.logger import logger
from utils.exceptions import (
    LLMGenerationError,
    LLMParseError,
    LLMInvalidResponseError,
)


class LLMFeedback:
    """OpenAI GPT-4o-mini를 사용한 피드백 생성"""

    def __init__(self):
        settings = get_settings()
        self.noop_mode = settings.ENABLE_LLM_NOOP

        if not self.noop_mode:
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = settings.LLM_MODEL
            logger.info(f"✅ LLM 클라이언트 초기화: model={self.model}")
        else:
            self.client = None
            self.model = None
            logger.warning("⚠️ LLM NOOP 모드 활성화 (규칙 기반 피드백 사용)")

        logger.info(f"✅ LLMFeedback 초기화: model={self.model}")

    # ========== 공개 메서드 ==========

    async def generate_feedback(
        self,
        sport_type: str,
        sub_category: str,
        angles: Dict[str, float],
        phases: List[Dict[str, Any]],
        sport_config: Dict[str, Any] = None,
        level: UserLevel = UserLevel.INTERMEDIATE,
        angle_scores: Optional[Dict[str, int]] = None,
        weighted_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        LLM 피드백 생성

        Args:
            sport_type: 종목 (GOLF, WEIGHT)
            sub_category: 세부 종목 (DRIVER, SQUAT)
            angles: 평균 각도 {"left_arm_angle": 165.3, ...}
            phases: 감지된 구간 [{"name": "backswing", ...}, ...]
            sport_config: 스포츠 설정 (angles, phases 포함)
            level: 사용자 레벨
            angle_scores: AngleCalculator가 계산한 각도별 점수
            weighted_score: AngleCalculator가 계산한 가중 평균 점수
        """

        if self.noop_mode:
            logger.info(f"🔄 NOOP 모드: 규칙 기반 피드백 생성 (level={level.value})")
            return self._generate_rule_based_feedback(
                sport_type=sport_type,
                sub_category=sub_category,
                angles=angles,
                phases=phases,
                sport_config=sport_config,
                level=level,
                angle_scores=angle_scores,
                weighted_score=weighted_score,
            )

        try:
            messages = self._build_prompt(
                sport_type,
                sub_category,
                angles,
                phases,
                sport_config,
                level,
                weighted_score,
            )

            logger.info(
                f"📤 LLM 호출: {sport_type}/{sub_category}, "
                f"level={level.value}, "
                f"prompt_version={messages['version']}"
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": messages["system"]},
                    {"role": "user", "content": messages["user"]},
                ],
                response_format={"type": "json_object"},
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )

            content = response.choices[0].message.content

            # ========== JSON 파싱 ==========
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"❌ LLM 응답 파싱 실패: {e}")
                raise LLMParseError(
                    details=(f"Invalid JSON: {str(e)}, response: {content[:200]}")
                )

            # ========== 응답 구조 검증 ==========
            missing_keys = [
                k for k in LLMConfig.REQUIRED_RESPONSE_KEYS if k not in result
            ]
            if missing_keys:
                logger.error(f"❌ LLM 응답 검증 실패: missing={missing_keys}")
                raise LLMInvalidResponseError(
                    details=f"Missing required keys: {missing_keys}"
                )

            result["prompt_version"] = messages["version"]

            logger.info(
                f"✅ LLM 응답: score={result.get('overall_score')}, "
                f"version={messages['version']}"
            )
            return result

        except (LLMParseError, LLMInvalidResponseError):
            raise

        except Exception as e:
            logger.error(f"❌ LLM 피드백 생성 실패: {e}")
            raise LLMGenerationError(details=f"Unexpected error: {str(e)}")

    # ========== 프롬프트 구성 ==========

    def _build_prompt(
        self,
        sport_type: str,
        sub_category: str,
        angles: Dict[str, float],
        phases: List[Dict[str, Any]],
        sport_config: Dict[str, Any] = None,
        level: UserLevel = UserLevel.INTERMEDIATE,
        weighted_score: Optional[float] = None,
    ) -> Dict[str, str]:
        """프롬프트 구성 — 레벨 톤 + weighted_score 포함"""
        angle_configs = sport_config.get("angles", {}) if sport_config else {}

        angles_list = []
        for name, value in angles.items():
            angle_data = {
                "name": name,
                "value": value,
            }
            if name in angle_configs:
                angle_data["ideal_range"] = angle_configs[name].get("ideal_range")
                angle_data["description"] = angle_configs[name].get("description", "")
                if "weight" in angle_configs[name]:
                    angle_data["weight"] = angle_configs[name]["weight"]
            angles_list.append(angle_data)

        # 레벨별 톤 조회
        tone = LEVEL_TONE.get(level.value, LEVEL_TONE["INTERMEDIATE"])

        return prompt_loader.load(
            sport_type=sport_type,
            sub_category=sub_category,
            context={
                "angles": angles_list,
                "phases": phases,
                "level": level.value,
                "level_tone": tone,
                "weighted_score": weighted_score,
            },
        )

    # ========== NOOP 모드: 규칙 기반 피드백 ==========

    def _generate_rule_based_feedback(
        self,
        sport_type: str,
        sub_category: str,
        angles: Dict[str, float],
        phases: List[Dict[str, Any]],
        sport_config: Dict[str, Any] = None,
        level: UserLevel = UserLevel.INTERMEDIATE,
        angle_scores: Optional[Dict[str, int]] = None,
        weighted_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        규칙 기반 피드백 (NOOP 모드)

        변경점:
        - AngleCalculator가 계산한 angle_scores, weighted_score 재사용
        - 자체 점수 계산 로직 제거 → 중복 제거
        - 레벨별 개선점 수 제한 (BEGINNER: 2개, 나머지: 3개)
        """
        if not angles:
            return {
                "overall_score": FeedbackScore.NO_DATA,
                "feedback": "각도 데이터가 충분하지 않습니다.",
                "improvements": [
                    {
                        "issue": "각도 데이터 부족",
                        "suggestion": "영상에서 신체가 명확히 보이도록 촬영해주세요",
                    }
                ],
                "prompt_version": LLMConfig.NOOP_VERSION,
            }

        angle_configs = sport_config.get("angles", {}) if sport_config else {}

        # ✅ AngleCalculator 결과 재사용 (중복 제거)
        if angle_scores is None:
            angle_scores = self._fallback_calculate_scores(angles, angle_configs)

        if weighted_score is None:
            weighted_score = float(FeedbackScore.DEFAULT)

        # ========== 피드백 분류 ==========
        good_points = []
        improvements = []

        for angle_name, angle_value in angles.items():
            score = angle_scores.get(angle_name, FeedbackScore.NO_DATA)
            angle_config = angle_configs.get(angle_name, {})
            ideal = angle_config.get(
                "ideal_range", [AngleDefaults.RANGE_MIN, AngleDefaults.RANGE_MAX]
            )
            feedback_msgs = angle_config.get("feedback", {})

            if score == FeedbackScore.IDEAL:
                msg = feedback_msgs.get("good", f"{angle_name} 양호")
                good_points.append(f"{angle_name}: {angle_value:.1f}° — {msg}")

            elif score == FeedbackScore.CAUTION:
                msg = feedback_msgs.get("caution", f"{angle_name} 주의")
                improvements.append(
                    {
                        "issue": f"{angle_name} 주의",
                        "current_value": angle_value,
                        "ideal_range": ideal,
                        "suggestion": msg,
                    }
                )

            elif score == FeedbackScore.CORRECTION:
                msg = feedback_msgs.get("correction", f"{angle_name} 교정 필요")
                improvements.append(
                    {
                        "issue": f"{angle_name} 범위 이탈",
                        "current_value": angle_value,
                        "ideal_range": ideal,
                        "suggestion": msg,
                    }
                )

        # ========== 레벨별 톤 적용 ==========
        tone = LEVEL_TONE.get(level.value, LEVEL_TONE["INTERMEDIATE"])
        max_improvements = tone.get(
            "max_improvements", FeedbackThreshold.MAX_IMPROVEMENTS
        )
        overall_score = int(weighted_score)

        # ========== 피드백 텍스트 ==========
        feedback_parts = []
        if good_points:
            feedback_parts.append(good_points[0])
        if improvements:
            feedback_parts.append(improvements[0]["suggestion"])

        feedback = (
            " | ".join(feedback_parts)
            if feedback_parts
            else f"{sport_type}/{sub_category} 분석 완료 (level={level.value})"
        )

        logger.info(
            f"✅ 규칙 기반 피드백: score={overall_score}, "
            f"level={level.value}, tone={tone['style']}, "
            f"good={len(good_points)}, issues={len(improvements)}"
        )

        return {
            "overall_score": overall_score,
            "feedback": feedback,
            "improvements": (
                improvements[:max_improvements]
                if improvements
                else [
                    {"issue": "전반적으로 양호", "suggestion": "현재 자세를 유지하세요"}
                ]
            ),
            "prompt_version": LLMConfig.NOOP_VERSION,
        }

    # ========== Fallback 점수 계산 ==========

    @staticmethod
    def _fallback_calculate_scores(
        angles: Dict[str, float],
        angle_configs: Dict[str, Any],
    ) -> Dict[str, int]:
        """
        AngleCalculator 결과가 없을 때 fallback 점수 계산

        정상 흐름에서는 호출되지 않음.
        analysis_service가 angle_scores를 전달하지 못한 예외 상황에서만 사용.
        """
        scores = {}
        for angle_name, value in angles.items():
            config = angle_configs.get(angle_name, {})
            ideal = config.get(
                "ideal_range", [AngleDefaults.RANGE_MIN, AngleDefaults.RANGE_MAX]
            )

            if ideal[0] <= value <= ideal[1]:
                scores[angle_name] = FeedbackScore.IDEAL
                continue

            validation = config.get("angle_validation")
            if validation:
                v_min = validation.get("min_normal", AngleDefaults.RANGE_MIN)
                v_max = validation.get("max_normal", AngleDefaults.RANGE_MAX)
                if v_min <= value <= v_max:
                    scores[angle_name] = FeedbackScore.CAUTION
                    continue

            scores[angle_name] = FeedbackScore.CORRECTION

        return scores
