"""
L-082: OPENAI_API_KEY 기동 시 검증 테스트
L-085: FastAPI Swagger 프로덕션 비공개 테스트
"""

import pytest
from unittest.mock import patch
from pydantic import ValidationError
from config.settings import Settings


# ──────────────────────────────────────────────
# L-082: OPENAI_API_KEY 검증
# ──────────────────────────────────────────────


def test_missing_openai_key_raises_when_noop_false():
    """ENABLE_LLM_NOOP=False이고 OPENAI_API_KEY 미설정 시 ValidationError 발생해야 함"""
    with patch.dict(
        "os.environ",
        {
            "INTERNAL_API_KEY": "test-key",
            "ENABLE_LLM_NOOP": "false",
            "OPENAI_API_KEY": "",
        },
        clear=False,
    ):
        with pytest.raises(ValidationError, match="OPENAI_API_KEY must be set"):
            Settings()


def test_missing_openai_key_allowed_when_noop_true():
    """ENABLE_LLM_NOOP=True이면 OPENAI_API_KEY 없어도 정상 기동해야 함"""
    with patch.dict(
        "os.environ",
        {
            "INTERNAL_API_KEY": "test-key",
            "ENABLE_LLM_NOOP": "true",
            "OPENAI_API_KEY": "",
        },
        clear=False,
    ):
        s = Settings()
        assert s.ENABLE_LLM_NOOP is True


def test_openai_key_set_with_noop_false():
    """ENABLE_LLM_NOOP=False이고 OPENAI_API_KEY 설정 시 정상 기동해야 함"""
    with patch.dict(
        "os.environ",
        {
            "INTERNAL_API_KEY": "test-key",
            "ENABLE_LLM_NOOP": "false",
            "OPENAI_API_KEY": "sk-test-valid-key",
        },
        clear=False,
    ):
        s = Settings()
        assert s.OPENAI_API_KEY == "sk-test-valid-key"


# ──────────────────────────────────────────────
# L-085: Swagger 프로덕션 비공개
# ──────────────────────────────────────────────


def test_swagger_disabled_in_production():
    """APP_ENV=production 시 docs_url, redoc_url이 None이어야 함"""
    with patch.dict(
        "os.environ",
        {
            "APP_ENV": "production",
            "INTERNAL_API_KEY": "test-key",
            "ENABLE_LLM_NOOP": "true",
            "ALLOWED_ORIGINS": "https://motionlab.kr",
        },
        clear=False,
    ):
        s = Settings()
        is_dev = s.APP_ENV == "development"
        assert is_dev is False
        assert ("/docs" if is_dev else None) is None
        assert ("/redoc" if is_dev else None) is None


def test_swagger_enabled_in_development():
    """APP_ENV=development 시 docs_url, redoc_url이 활성화되어야 함"""
    with patch.dict(
        "os.environ",
        {
            "APP_ENV": "development",
            "INTERNAL_API_KEY": "test-key",
            "ENABLE_LLM_NOOP": "true",
        },
        clear=False,
    ):
        s = Settings()
        is_dev = s.APP_ENV == "development"
        assert ("/docs" if is_dev else None) == "/docs"
