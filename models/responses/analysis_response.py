"""
분석 응답 모델 (Analyzer → Backend Worker)
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class PhaseInfo(BaseModel):
    """스윙 구간 정보"""
    name: str = Field(..., description="구간 이름", examples=["backswing"])
    start_frame: int = Field(..., description="시작 프레임", examples=[10])
    end_frame: int = Field(..., description="종료 프레임", examples=[45])
    duration_ms: int = Field(..., description="구간 길이(ms)", examples=[1500])

class AnalysisResult(BaseModel):
    """분석 결과 상세"""
    total_frames: int = Field(..., description="총 프레임 수", examples=[150])
    duration_seconds: float = Field(..., description="영상 길이(초)", examples=[5.0])

    angles: Dict[str, float] = Field(
        ...,
        description="계산된 각도 (평균값)",
        examples=[{"left_elbow": 145.2, "spine_angle": 32.1}]
    )

    phases: List[PhaseInfo] = Field(
        ...,
        description="스윙 구간 정보",
        examples=[[
            {"name": "backswing", "start_frame": 10, "end_frame": 45, "duration_ms": 1500}
        ]]
    )

    keypoints_sample: List[Dict[str, Any]] = Field(
        ...,
        description="샘플 프레임의 랜드마크 (디버깅용)",
        examples=[[{"x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.95}]]
    )

class AnalysisResponse(BaseModel):
    """
    POST /analyze 성공 응답

    프롬프트 A5 규칙:
    - success: true 고정
    - motion_id: 요청한 Motion ID
    - result: 분석 결과 상세
    - feedback: LLM 피드백 (한국어)
    - prompt_version: 사용한 프롬프트 버전 (추적용)
    """
    success: bool = Field(True, description="성공 여부")
    motion_id: int = Field(..., description="Motion ID", examples=[15])
    result: AnalysisResult = Field(..., description="분석 결과")
    feedback: str = Field(
        ...,
        description="LLM 피드백 (한국어)",
        examples=["당신의 스윙은 백스윙 각도가 적절하며..."]
    )
    overall_score: Optional[int] = Field(None, description="종합 점수 (0-100)")
    improvements: List[Dict[str, Any]] = Field(default_factory=list, description="개선사항")
    prompt_version: str = Field(
        "v1.0",
        description="사용한 프롬프트 버전",
        examples=["v1.0"]
    )

class ErrorResponse(BaseModel):
    """
    POST /analyze 에러 응답

    프롬프트 A6 규칙:
    - success: false
    - error_code: AN_ (사용자 과실) / SYS_ (시스템 장애)
    - message: 에러 메시지
    - retryable: 재시도 가능 여부
    """
    success: bool = Field(False, description="실패 여부")
    error_code: str = Field(
        ...,
        description="에러 코드 (AN_001, SYS_011 등)",
        examples=["AN_001"]
    )
    message: str = Field(
        ...,
        description="에러 메시지",
        examples=["No keypoints detected in video"]
    )
    retryable: bool = Field(
        ...,
        description="재시도 가능 여부",
        examples=[False]
    )
