"""
MotionLab AI - 분석 요청 모델
Backend에서 전달되는 분석 요청 데이터 구조
"""
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """
    분석 요청 데이터 (A2 규칙)

    Attributes:
        motion_id: Motion 고유 ID (Backend DB의 Primary Key)
        video_url: 분석할 영상 URL (S3 Presigned URL 또는 로컬 경로)
        sport_code: 스포츠 종목 코드 (예: "GOLF", "BASEBALL")
    """

    motion_id: int = Field(
        ...,
        description="Motion 고유 ID",
        examples=[15],
        gt=0
    )

    video_url: str = Field(
        ...,
        description="영상 URL (S3 Presigned URL 또는 로컬 절대 경로)",
        examples=["https://r2.example.com/motions/u1/GOLF/video.mp4"],
        min_length=10,
        max_length=1000
    )

    sport_code: str = Field(
        ...,
        description="스포츠 종목 코드",
        examples=["GOLF"],
        pattern=r"^[A-Z]+$",
        min_length=2,
        max_length=20
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "motion_id": 15,
                    "video_url": "https://r2.example.com/motions/u1/GOLF/video.mp4",
                    "sport_code": "GOLF"
                }
            ]
        }
    }
