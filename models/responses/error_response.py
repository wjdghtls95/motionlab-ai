from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from utils.exceptions.errors import get_error_info


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
        ..., description="에러 코드 (AN_001, SYS_011 등)", examples=["AN_001"]
    )
    message: str = Field(
        ..., description="에러 메시지", examples=["No keypoints detected in video"]
    )
    retryable: bool = Field(..., description="재시도 가능 여부", examples=[False])
    details: str | None = Field(
        None,
        description="추가 디버깅 정보",
        examples=["MediaPipe 결과 없음"],
    )


# 응답 생성 함수


def create_error_response(
    error_code: str,
    message: str,
    retryable: bool,
    details: str | None = None,
    status_code: int | None = None,
) -> JSONResponse:
    """
    표준화된 에러 JSONResponse 생성.

    Backend(NestJS)의 mapAxiosToJobError가 data.error_code를 기준으로
    매핑하므로, 응답 body의 키 이름을 변경하면 안 됨
    """
    info = get_error_info(error_code)

    return JSONResponse(
        status_code=status_code or info["status_code"],
        content={
            "success": False,
            "error_code": error_code,
            "message": message,
            "retryable": retryable,
            "details": details,
        },
    )
