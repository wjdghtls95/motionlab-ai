"""
Health Check Response Model
"""
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """헬스체크 응답"""

    status: str = Field(..., examples=["healthy"])
    python_version: str = Field(..., examples=["3.12.7"])
    llm_noop_mode: bool = Field(..., examples=[True])
    mediapipe_model_complexity: int = Field(..., examples=[1])
    max_video_size_mb: int = Field(..., examples=[500])
