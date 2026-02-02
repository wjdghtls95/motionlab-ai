"""
헬스체크 응답 모델
"""
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    """
    GET /health 응답
    """
    status: str = Field("ok", description="서버 상태", examples=["ok"])
    version: str = Field("1.0.0", description="서버 버전", examples=["1.0.0"])
    mediapipe_available: bool = Field(
        True,
        description="MediaPipe 로드 가능 여부",
        examples=[True]
    )
