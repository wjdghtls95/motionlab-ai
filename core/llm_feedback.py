"""
MotionLab AI - LLM 피드백 생성
"""
import json
from typing import Dict, List
from openai import OpenAI
from config import get_settings
from utils.logger import logger


class LLMFeedback:
    """LLM 기반 피드백 생성기"""

    def __init__(self):
        """OpenAI 클라이언트 초기화"""
        settings = get_settings()
        self.enable_noop = settings.ENABLE_LLM_NOOP
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS

        if not self.enable_noop:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            self.client = None
            logger.info("⚠️ LLM Noop 모드 활성화")

    def generate_feedback(
            self,
            sport_type: str,
            sub_category: str,
            average_angles: Dict[str, float],
            phases: List[Dict],
            sport_config: Dict
    ) -> Dict:
        """피드백 생성 메인 함수"""

        if self.enable_noop:
            return self._generate_noop_feedback(sport_type, sub_category, average_angles)

        prompt = self._build_prompt(sport_type, sub_category, average_angles, phases, sport_config)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 전문 운동 코치입니다. 동작 분석 데이터를 보고 명확하고 실용적인 피드백을 제공합니다."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )

            feedback_text = response.choices[0].message.content
            feedback = json.loads(feedback_text)

            logger.info(f"✅ LLM 피드백 생성 완료: {sport_type}/{sub_category}")

            return feedback

        except Exception as e:
            logger.error(f"❌ LLM 피드백 생성 실패: {e}")
            return self._generate_error_feedback(str(e))

    def _build_prompt(
            self,
            sport_type: str,
            sub_category: str,
            average_angles: Dict[str, float],
            phases: List[Dict],
            sport_config: Dict
    ) -> str:
        """프롬프트 생성"""

        angle_configs = sport_config.get("angles", {})

        angle_analysis = []
        for angle_name, angle_value in average_angles.items():
            if angle_value is None:
                continue

            angle_config = angle_configs.get(angle_name, {})
            ideal_range = angle_config.get("ideal_range")
            description = angle_config.get("description", angle_name)

            angle_info = {
                "name": description,
                "current_value": angle_value,
                "ideal_range": list(ideal_range) if ideal_range else None
            }
            angle_analysis.append(angle_info)

        phase_summary = [
            {"name": p["name"], "duration_ms": p["duration_ms"]}
            for p in phases
        ]

        prompt = f"""
# 동작 분석 데이터

종목: {sport_type} - {sub_category}

측정된 각도 (평균):
{json.dumps(angle_analysis, indent=2, ensure_ascii=False)}

구간 데이터:
{json.dumps(phase_summary, indent=2, ensure_ascii=False)}

---

위 데이터를 분석하여 다음 형식의 JSON을 생성해주세요:

{{
  "overall_score": 85,
  "feedback": "전반적으로 좋은 자세입니다.",
  "improvements": [
    {{
      "issue": "왼팔 각도가 이상 범위보다 낮습니다",
      "current_value": 158.3,
      "ideal_range": [165.0, 180.0],
      "suggestion": "백스윙 시 왼팔을 더 펴주세요"
    }}
  ]
}}

요구사항:
1. overall_score: 0-100점
2. feedback: 2-3문장 (한글)
3. improvements: 최대 3개 (이상 범위 밖만)
"""

        return prompt

    def _generate_noop_feedback(
            self,
            sport_type: str,
            sub_category: str,
            average_angles: Dict[str, float]
    ) -> Dict:
        """Noop 모드 더미 피드백"""
        return {
            "overall_score": 85,
            "feedback": f"[Noop 모드] {sport_type}/{sub_category} 분석 완료",
            "improvements": [
                {
                    "issue": "왼팔 각도 확인 필요",
                    "current_value": average_angles.get("left_arm_angle", 0),
                    "ideal_range": [165.0, 180.0],
                    "suggestion": "백스윙 시 왼팔을 더 펴주세요 (Noop)"
                }
            ]
        }

    def _generate_error_feedback(self, error_message: str) -> Dict:
        """에러 폴백"""
        return {
            "overall_score": 0,
            "feedback": f"피드백 생성 오류: {error_message}",
            "improvements": []
        }