"""
분석 요청 모델 (Backend Worker → Analyzer)
"""

from pydantic import BaseModel, Field
from typing import Optional
from core.sport_configs.base_config import UserLevel


class AnalysisRequest(BaseModel):
    """
    POST /analyze 요청 바디

    - motion_id: Backend Motion 테이블 ID
    - video_url: Presigned URL 또는 로컬 절대 경로
    - sport_type: GOLF, WEIGHT 등
    - sub_category: DRIVER, SQUAT 등 (선택)
    - level: 사용자 레벨 (BEGINNER ~ PRO, 기본 INTERMEDIATE)
    """

    motion_id: int = Field(..., description="Backend Motion ID", examples=[15])

    video_url: str = Field(
        ...,
        description="S3/R2 Presigned URL 또는 로컬 절대 경로",
        examples=["https://r2.example.com/video.mp4?signature=abc"],
    )

    sport_type: str = Field(
        ..., description="스포츠 종목 (GOLF, WEIGHT)", examples=["GOLF"]
    )

    sub_category: Optional[str] = Field(
        None, description="서브 카테고리 (DRIVER, SQUAT 등)", examples=["DRIVER"]
    )

    level: UserLevel = Field(
        default=UserLevel.INTERMEDIATE,
        description="사용자 레벨 (BEGINNER, INTERMEDIATE, ADVANCED, PRO)",
        examples=["INTERMEDIATE"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "motion_id": 15,
                    "video_url": "https://r2.example.com/video.mp4",
                    "sport_type": "GOLF",
                    "sub_category": "DRIVER",
                    "level": "INTERMEDIATE",
                }
            ]
        }
    }
