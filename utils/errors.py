"""
에러 코드 및 예외 정의
"""
from typing import Dict, Any

# 에러 코드 정의
ERROR_CODES: Dict[str, Dict[str, Any]] = {
    # 사용자 과실 (재시도 X)
    "AN_001": {
        "message": "No keypoints detected in video",
        "retryable": False
    },
    "AN_002": {
        "message": "Video too short (< 1 second)",
        "retryable": False
    },
    "AN_003": {
        "message": "Unsupported sport code",
        "retryable": False
    },
    "AN_004": {
        "message": "Video file not found",
        "retryable": False
    },

    # 시스템 장애 (재시도 O)
    "SYS_011": {
        "message": "Video download failed",
        "retryable": True
    },
    "SYS_012": {
        "message": "LLM timeout",
        "retryable": True
    },
    "SYS_021": {
        "message": "Analyzer timeout",
        "retryable": True
    },
    "SYS_999": {
        "message": "Unknown error",
        "retryable": True
    }
}


def get_error_detail(error_code: str, custom_message: str = None) -> Dict[str, Any]:
    """
    에러 코드로부터 에러 상세 정보 반환

    Args:
        error_code: 에러 코드 (예: "AN_001")
        custom_message: 커스텀 메시지 (선택)

    Returns:
        dict: {error_code, message, retryable}
    """
    error_info = ERROR_CODES.get(error_code, ERROR_CODES["SYS_999"])

    return {
        "error_code": error_code,
        "message": custom_message or error_info["message"],
        "retryable": error_info["retryable"]
    }
