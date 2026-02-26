"""
분석 요청 모델 (Backend Worker → Analyzer)
"""
from pydantic import BaseModel, Field
from typing import Optional


class AnalysisRequest(BaseModel):
    """
    POST /analyze 요청 바디

    프롬프트 A2 규칙:
    - motion_id: Backend Motion 테이블 ID
    - video_url: Presigned URL 또는 로컬 절대 경로
    - sport_type: GOLF, WEIGHT, TENNIS 등
    - sub_category: DRIVER, SQUAT, SERVE 등 (선택)
    """
    motion_id: int = Field(
        ...,
        description="Backend Motion ID",
        examples=[15]
    )

    video_url: str = Field(
        ...,
        description="S3 Presigned URL 또는 로컬 절대 경로",
        examples=["https://r2.example.com/video.mp4?signature=abc"]
    )

    sport_type: str = Field(
        ...,
        description="스포츠 종목 (GOLF, WEIGHT, TENNIS)",
        examples=["GOLF"]
    )

    sub_category: Optional[str] = Field(
        None,
        description="서브 카테고리 (DRIVER, IRON, SQUAT 등)",
        examples=["DRIVER"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "motion_id": 15,
                    "video_url": "https://r2.example.com/video.mp4",
                    "sport_type": "GOLF",
                    "sub_category": "DRIVER"
                }
            ]
        }
    }
