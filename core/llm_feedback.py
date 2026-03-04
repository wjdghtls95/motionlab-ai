"""
MotionLab AI - LLM 피드백 생성 (YAML 프롬프트 + 자동 버전 관리)
"""

import json
from typing import Dict, Any, List

from openai import AsyncOpenAI

from config import get_settings, settings
from core.prompts.loader import prompt_loader
from core.sport_configs.base_config import UserLevel
from core.constants import (
    FeedbackScore,
    FeedbackThreshold,
    LLMConfig,
    AngleDefaults,
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
            self.model = "gpt-4o-mini"
            logger.info(f"✅ LLM 클라이언트 초기화: model={self.model}")
        else:
            self.client = None
            self.model = None
            logger.warning("⚠️ LLM NOOP 모드 활성화 (규칙 기반 피드백 사용)")

        logger.info(f"✅ LLMFeedback 초기화: model={self.model}")

    async def generate_feedback(
        self,
        sport_type: str,
        sub_category: str,
        angles: Dict[str, float],
        phases: List[Dict[str, Any]],
        sport_config: Dict[str, Any] = None,
        level: UserLevel = UserLevel.INTERMEDIATE,
    ) -> Dict[str, Any]:
        """LLM 피드백 생성"""

        if self.noop_mode:
            logger.info(
                f"🔄 NOOP 모드: 규칙 기반 피드백 생성 " f"(level={level.value})"
            )
            return self._generate_rule_based_feedback(
                sport_type=sport_type,
                sub_category=sub_category,
                angles=angles,
                phases=phases,
                sport_config=sport_config,
                level=level,
            )

        try:
            messages = self._build_prompt(
                sport_type,
                sub_category,
                angles,
                phases,
                sport_config,
                level,
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
                    details=(f"Invalid JSON: {str(e)}, " f"response: {content[:200]}")
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

    def _build_prompt(
        self,
        sport_type: str,
        sub_category: str,
        angles: Dict[str, float],
        phases: List[Dict[str, Any]],
        sport_config: Dict[str, Any] = None,
        level: UserLevel = UserLevel.INTERMEDIATE,
    ) -> Dict[str, str]:
        """프롬프트 구성"""
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

        return prompt_loader.load(
            sport_type=sport_type,
            sub_category=sub_category,
            context={
                "angles": angles_list,
                "phases": phases,
                "level": level.value,
            },
        )

    def _generate_rule_based_feedback(
        self,
        sport_type: str,
        sub_category: str,
        angles: Dict[str, float],
        phases: List[Dict[str, Any]],
        sport_config: Dict[str, Any] = None,
        level: UserLevel = UserLevel.INTERMEDIATE,
    ) -> Dict[str, Any]:
        """
        규칙 기반 피드백 (NOOP 모드)
        - v2: angle_validation은 angles 안에 내장
        - v1: sub_category 레벨에 angle_validation 존재
        - ideal_range는 이미 레벨 resolve 완료된 상태
        """
        if not angles:
            return {
                "overall_score": FeedbackScore.NO_DATA,
                "feedback": "각도 데이터가 충분하지 않습니다.",
                "improvements": [
                    {
                        "issue": "각도 데이터 부족",
                        "suggestion": ("영상에서 신체가 명확히 보이도록 촬영해주세요"),
                    }
                ],
                "prompt_version": LLMConfig.NOOP_VERSION,
            }

        # angle_validation 가져오기 (v1/v2 모두 지원)
        validation = sport_config.get("angle_validation", {}) if sport_config else {}
        angle_feedbacks = sport_config.get("angles", {}) if sport_config else {}

        angle_scores = []
        good_points = []
        improvements = []

        for angle_name, angle_value in angles.items():
            angle_config = angle_feedbacks.get(angle_name, {})

            # ideal_range는 이미 레벨 resolve 완료됨
            ideal = angle_config.get(
                "ideal_range", [AngleDefaults.RANGE_MIN, AngleDefaults.RANGE_MAX]
            )

            # angle_validation (정상 범위)
            angle_valid = validation.get(angle_name)
            if not angle_valid:
                # v2: angle_config 안에 있을 수 있음
                angle_valid = angle_config.get("angle_validation")

            if angle_valid:
                min_normal = angle_valid.get("min_normal", AngleDefaults.RANGE_MIN)
                max_normal = angle_valid.get("max_normal", AngleDefaults.RANGE_MAX)
            else:
                # validation 없으면 ideal 기준으로 마진 적용
                min_normal = ideal[0] - FeedbackThreshold.VALIDATION_MARGIN
                max_normal = ideal[1] + FeedbackThreshold.VALIDATION_MARGIN

            feedback_msgs = angle_config.get("feedback", {})

            # ========== 점수 판정 ==========
            if ideal[0] <= angle_value <= ideal[1]:
                # ideal_range 안: 최고 점수
                angle_scores.append(FeedbackScore.IDEAL)
                msg = feedback_msgs.get("good", f"{angle_name} 양호")
                good_points.append(f"{angle_name}: {angle_value:.1f}° — {msg}")

            elif min_normal <= angle_value <= max_normal:
                # 정상 범위이지만 ideal 밖: 주의
                angle_scores.append(FeedbackScore.CAUTION)
                msg = feedback_msgs.get("caution", f"{angle_name} 주의")
                improvements.append(
                    {
                        "issue": f"{angle_name} 주의",
                        "current_value": angle_value,
                        "ideal_range": ideal,
                        "suggestion": msg,
                    }
                )

            else:
                # 정상 범위도 벗어남: 교정 필요
                angle_scores.append(FeedbackScore.CORRECTION)
                msg = feedback_msgs.get("correction", f"{angle_name} 교정 필요")
                improvements.append(
                    {
                        "issue": f"{angle_name} 범위 이탈",
                        "current_value": angle_value,
                        "valid_range": [min_normal, max_normal],
                        "suggestion": msg,
                    }
                )

        # ========== 종합 점수 ==========
        overall_score = (
            sum(angle_scores) // len(angle_scores)
            if angle_scores
            else FeedbackScore.DEFAULT
        )

        # ========== 피드백 텍스트 ==========
        feedback_parts = []
        if good_points:
            feedback_parts.append(good_points[0])
        if improvements:
            feedback_parts.append(improvements[0]["suggestion"])

        feedback = (
            " | ".join(feedback_parts)
            if feedback_parts
            else (f"{sport_type}/{sub_category} 분석 완료 " f"(level={level.value})")
        )

        logger.info(
            f"✅ 규칙 기반 피드백: score={overall_score}, "
            f"level={level.value}, "
            f"good={len(good_points)}, issues={len(improvements)}"
        )

        return {
            "overall_score": overall_score,
            "feedback": feedback,
            "improvements": (
                improvements[: FeedbackThreshold.MAX_IMPROVEMENTS]
                if improvements
                else [
                    {
                        "issue": "전반적으로 양호",
                        "suggestion": "현재 자세를 유지하세요",
                    }
                ]
            ),
            "prompt_version": LLMConfig.NOOP_VERSION,
        }
