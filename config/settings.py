"""
MotionLab AI - 환경 설정 관리
.env 파일의 환경 변수를 로드하고 검증
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경 변수 설정 클래스"""

    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # 인증
    INTERNAL_API_KEY: str

    # OpenAI
    OPENAI_API_KEY: str = ""
    ENABLE_LLM_NOOP: bool = True

    # 비디오 처리
    MAX_VIDEO_SIZE_MB: int = 500
    TEMP_VIDEO_DIR: str = "./temp_videos"

    # MediaPipe 설정
    MEDIAPIPE_MODEL_COMPLEXITY: int = 1
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5

    # 로깅
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    설정 객체를 캐싱하여 반환

    Returns:
        Settings: 환경 설정 객체
    """
    return Settings()


# 전역 설정 객체 (FastAPI 의존성 주입용)
settings = get_settings()
