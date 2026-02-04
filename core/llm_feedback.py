"""
MotionLab AI - LLM 피드백 생성 (YAML 프롬프트 + 자동 버전 관리)
"""

import json
from typing import Dict, Any, List

from openai import AsyncOpenAI

from config import get_settings
from core.prompts.loader import prompt_loader
from utils.logger import logger
from utils.exceptions import AnalyzerError, ErrorCode


class LLMFeedback:
    """OpenAI GPT-4o-mini를 사용한 피드백 생성"""

    def __init__(self):
        """LLMFeedback 초기화"""
        settings = get_settings()
        self.noop_mode = settings.ENABLE_LLM_NOOP

        if not self.noop_mode:
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
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
        angles: List[Dict[str, Any]],
        phases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """LLM 피드백 생성"""

        if self.noop_mode:
            logger.info("🔄 NOOP 모드: 규칙 기반 피드백 생성")
            return self._generate_rule_based_feedback(
                angles, phases, sport_type, sub_category
            )

        try:
            messages = self._build_prompt(sport_type, sub_category, angles, phases)

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
            result = json.loads(content)
            result["prompt_version"] = messages["version"]

            logger.info(
                f"✅ LLM 응답: score={result.get('overall_score')}, "
                f"version={messages['version']}"
            )

            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ LLM 응답 파싱 실패: {e}")
            raise AnalyzerError(
                error_code=ErrorCode.LLM_TIMEOUT,
                custom_message="LLM 응답 파싱 실패",
                error=str(e),
            )
        except Exception as e:
            logger.error(f"❌ LLM 피드백 생성 실패: {e}")
            raise AnalyzerError(
                error_code=ErrorCode.LLM_TIMEOUT,
                custom_message="LLM 피드백 생성 실패",
                error=str(e),
            )

    def _build_prompt(
        self,
        sport_type: str,
        sub_category: str,
        angles: List[Dict[str, Any]],
        phases: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        프롬프트 생성 (YAML 기반 + Git 버전 자동).

        Before (하드코딩):
            200줄의 if-else 지옥 + 수동 버전 관리

        After (YAML + Git):
            1줄로 해결 + 자동 버전 관리!
        """
        return prompt_loader.load(
            sport_type=sport_type,
            sub_category=sub_category,
            context={"angles": angles, "phases": phases},
        )

    def _generate_rule_based_feedback(
        self,
        angles: List[Dict[str, Any]],
        phases: List[Dict[str, Any]],
        sport_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """규칙 기반 피드백 (JSON의 ideal_range 사용)"""
        if not angles:
            return {
                "overall_score": 50,
                "feedback": "각도 데이터가 충분하지 않습니다.",
                "improvements": ["영상에서 신체가 명확히 보이도록 촬영해주세요"],
                "prompt_version": "noop",
            }

        angle_scores = []
        good_points = []
        improvements = []

        angle_configs = sport_config.get("angles", {})

        for angle in angles:
            angle_name = angle.get("name", "")
            avg = angle.get("average", 0)

            angle_config = angle_configs.get(angle_name, {})
            ideal_range = angle_config.get("ideal_range", [0, 180])
            ideal_min, ideal_max = ideal_range

            if ideal_min <= avg <= ideal_max:
                angle_scores.append(95)
                good_points.append(f"{angle_name}: {avg:.1f}도 (이상적)")
            elif ideal_min - 10 <= avg <= ideal_max + 10:
                angle_scores.append(80)
                improvements.append(
                    f"{angle_name}를 {ideal_min}~{ideal_max}도 범위로 조정 (현재: {avg:.1f}도)"
                )
            else:
                angle_scores.append(65)
                improvements.append(
                    f"{angle_name} 개선 필요 (현재: {avg:.1f}도, 권장: {ideal_min}~{ideal_max}도)"
                )

        overall_score = sum(angle_scores) // len(angle_scores) if angle_scores else 70

        feedback_parts = []
        if good_points:
            feedback_parts.append(f"✅ 강점: {', '.join(good_points[:2])}")
        if improvements:
            feedback_parts.append(f"📌 개선: {improvements[0]}")

        feedback = " | ".join(feedback_parts) if feedback_parts else "분석 완료"

        logger.info(f"✅ 규칙 기반 피드백: score={overall_score}")

        return {
            "overall_score": overall_score,
            "feedback": feedback,
            "improvements": improvements[:3],
            "prompt_version": "noop",
        }
