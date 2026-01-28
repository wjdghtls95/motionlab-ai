"""
MotionLab AI - 분석 응답 모델
Backend로 반환되는 분석 결과 데이터 구조
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class AnalysisResponse(BaseModel):
    """
    분석 결과 응답 (A5, A6 규칙)

    Attributes:
        success: 분석 성공 여부
        motion_id: Motion 고유 ID
        result: 분석 결과 데이터 (성공 시)
        feedback: LLM 생성 피드백 (성공 시)
        error_code: 에러 코드 (실패 시)
        message: 에러 메시지 (실패 시)
        retryable: 재시도 가능 여부 (실패 시)
    """

    success: bool = Field(
        ...,
        description="분석 성공 여부"
    )

    motion_id: int = Field(
        ...,
        description="Motion 고유 ID",
        examples=[15]
    )

    result: Optional[Dict[str, Any]] = Field(
        None,
        description="분석 결과 (성공 시)",
        examples=[{
            "total_frames": 150,
            "duration_seconds": 5.0,
            "angles": {
                "left_elbow": 145.2,
                "right_elbow": 142.8,
                "spine": 85.5
            },
            "phases": [
                {"phase": "address", "start_frame": 0, "end_frame": 30},
                {"phase": "backswing", "start_frame": 31, "end_frame": 60}
            ],
            "keypoints_sample": [
                {"frame": 0, "nose": {"x": 0.5, "y": 0.3, "z": 0.0}}
            ]
        }]
    )

    feedback: Optional[str] = Field(
        None,
        description="LLM 생성 피드백 (성공 시)",
        examples=["당신의 스윙은 전반적으로 안정적입니다."],
        max_length=1000
    )

    error_code: Optional[str] = Field(
        None,
        description="에러 코드 (실패 시)",
        examples=["AN_001"],
        pattern=r"^[A-Z_]+[0-9]{3}$"
    )

    message: Optional[str] = Field(
        None,
        description="에러 메시지 (실패 시)",
        examples=["No keypoints detected in video"]
    )

    retryable: Optional[bool] = Field(
        None,
        description="재시도 가능 여부 (실패 시)",
        examples=[False]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "motion_id": 15,
                    "result": {
                        "total_frames": 150,
                        "duration_seconds": 5.0,
                        "angles": {"left_elbow": 145.2}
                    },
                    "feedback": "당신의 스윙은 안정적입니다."
                },
                {
                    "success": False,
                    "motion_id": 15,
                    "error_code": "AN_001",
                    "message": "No keypoints detected in video",
                    "retryable": False
                }
            ]
        }
    }
