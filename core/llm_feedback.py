"""
MotionLab AI - LLM 피드백 생성 (YAML 프롬프트 + 자동 버전 관리)
"""

import json
from typing import Dict, Any, List

from openai import AsyncOpenAI

from config import get_settings
from core.prompts.loader import prompt_loader
from utils.logger import logger
from utils.exceptions import (
    LLMGenerationError,
    LLMParseError,
    LLMInvalidResponseError,
)


class LLMFeedback:
    """OpenAI GPT-4o-mini를 사용한 피드백 생성"""

    def __init__(self):
        """LLMFeedback 초기화"""
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
    ) -> Dict[str, Any]:
        """LLM 피드백 생성"""

        if self.noop_mode:
            logger.info("🔄 NOOP 모드: 규칙 기반 피드백 생성")
            return self._generate_rule_based_feedback(
                sport_type=sport_type,
                sub_category=sub_category,
                angles=angles,
                phases=phases,
                sport_config=sport_config,
            )

        try:
            messages = self._build_prompt(
                sport_type, sub_category, angles, phases, sport_config
            )

            logger.info(
                f"📤 LLM 호출: {sport_type}/{sub_category}, "
                f"prompt_version={messages['version']}"
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": messages["system"]},
                    {"role": "user", "content": messages["user"]},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=1000,
            )

            content = response.choices[0].message.content

            # ========== JSON 파싱 예외 처리 개선 ==========
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"❌ LLM 응답 파싱 실패: {e}")
                raise LLMParseError(
                    details=f"Invalid JSON from LLM: {str(e)}, response: {content[:200]}"
                )

            # ========== 응답 구조 검증 ==========
            required_keys = ["feedback", "overall_score", "improvements"]
            missing_keys = [key for key in required_keys if key not in result]
            if missing_keys:
                logger.error(f"❌ LLM 응답 검증 실패: missing keys={missing_keys}")
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
            raise LLMGenerationError(
                details=f"Unexpected error during LLM generation: {str(e)}"
            )

    def _build_prompt(
        self,
        sport_type: str,
        sub_category: str,
        angles: Dict[str, float],
        phases: List[Dict[str, Any]],
        sport_config: Dict[str, Any] = None,
    ) -> Dict[str, str]:
        # angles에 ideal_range를 포함시켜서 넘기기
        angle_configs = sport_config.get("angles", {}) if sport_config else {}

        angles_list = []
        for name, value in angles.items():
            angle_data = {
                "name": name,
                "value": value,
            }
            # config에서 ideal_range 가져오기
            if name in angle_configs:
                angle_data["ideal_range"] = angle_configs[name].get("ideal_range")
                angle_data["description"] = angle_configs[name].get("description", "")
            angles_list.append(angle_data)

        return prompt_loader.load(
            sport_type=sport_type,
            sub_category=sub_category,
            context={"angles": angles_list, "phases": phases},
        )

    def _generate_rule_based_feedback(
        self,
        sport_type: str,
        sub_category: str,
        angles: Dict[str, float],
        phases: List[Dict[str, Any]],
        sport_config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        규칙 기반 피드백 (JSON의 angle_validation 사용)
        """
        if not angles:
            return {
                "overall_score": 50,
                "feedback": "각도 데이터가 충분하지 않습니다.",
                "improvements": [
                    {
                        "issue": "각도 데이터 부족",
                        "suggestion": "영상에서 신체가 명확히 보이도록 촬영해주세요",
                    }
                ],
                "prompt_version": "noop",
            }

        if not sport_config or "angle_validation" not in sport_config:
            logger.error(
                f"❌ sport_config 또는 angle_validation 누락: {sport_type}/{sub_category}"
            )
            raise ValueError(
                f"sport_config.angle_validation이 필요합니다: {sport_type}/{sub_category}"
            )

        validation = sport_config["angle_validation"]
        angle_feedbacks = sport_config.get("angles", {})

        angle_scores = []
        good_points = []
        improvements = []

        # Dict 순회
        for angle_name, angle_value in angles.items():
            # 해당 각도의 validation 가져오기
            angle_valid = validation.get(angle_name)
            if not angle_valid:
                continue

            min_normal = angle_valid["min_normal"]
            max_normal = angle_valid["max_normal"]

            # 해당 각도의 feedback 메시지 가져오기
            angle_config = angle_feedbacks.get(angle_name, {})
            feedback_msgs = angle_config.get("feedback", {})

            if min_normal <= angle_value <= max_normal:
                # ideal_range 안에 있는지도 추가 확인
                ideal = angle_config.get("ideal_range", [min_normal, max_normal])
                if ideal[0] <= angle_value <= ideal[1]:
                    angle_scores.append(90)
                    msg = feedback_msgs.get("good", f"{angle_name} 양호")
                    good_points.append(f"{angle_name}: {angle_value:.1f}도 — {msg}")
                else:
                    angle_scores.append(70)
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
                angle_scores.append(40)
                msg = feedback_msgs.get("correction", f"{angle_name} 교정 필요")
                improvements.append(
                    {
                        "issue": f"{angle_name} 범위 이탈",
                        "current_value": angle_value,
                        "valid_range": [min_normal, max_normal],
                        "suggestion": msg,
                    }
                )

        overall_score = sum(angle_scores) // len(angle_scores) if angle_scores else 70

        feedback_parts = []
        if good_points:
            feedback_parts.append(good_points[0])
        if improvements:
            feedback_parts.append(improvements[0]["suggestion"])

        feedback = (
            " | ".join(feedback_parts)
            if feedback_parts
            else f"{sport_type}/{sub_category} 분석 완료"
        )

        logger.info(
            f"✅ 규칙 기반 피드백: score={overall_score}, "
            f"good={len(good_points)}, issues={len(improvements)}"
        )

        return {
            "overall_score": overall_score,
            "feedback": feedback,
            "improvements": (
                improvements[:3]
                if improvements
                else [
                    {
                        "issue": "전반적으로 양호",
                        "suggestion": "현재 자세를 유지하세요",
                    }
                ]
            ),
            "prompt_version": "noop",
        }
