"""
R-006: FastAPI CORS 와일드카드 제한 테스트

APP_ENV에 따라 CORS 허용 출처가 올바르게 적용되는지 검증
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────
# 1. Settings: APP_ENV=development → 와일드카드
# ──────────────────────────────────────────────
def test_settings_development_uses_wildcard():
    """APP_ENV=development 이면 _allowed_origins가 와일드카드여야 함"""
    with patch.dict("os.environ", {"APP_ENV": "development", "INTERNAL_API_KEY": "test-key-32chars-padding-xxxxxxxx"}):
        from config.settings import Settings
        s = Settings()
        is_dev = s.APP_ENV == "development"
        allowed = ["*"] if is_dev else s.ALLOWED_ORIGINS.split(",")
        assert allowed == ["*"]


# ──────────────────────────────────────────────
# 2. Settings: APP_ENV=production → 목록 분리
# ──────────────────────────────────────────────
def test_settings_production_splits_origins():
    """APP_ENV=production 이면 ALLOWED_ORIGINS 쉼표 분리 결과를 사용해야 함"""
    with patch.dict("os.environ", {
        "APP_ENV": "production",
        "INTERNAL_API_KEY": "test-key-32chars-padding-xxxxxxxx",
        "ALLOWED_ORIGINS": "http://localhost:3000,https://api.motionlab.kr",
    }):
        from config.settings import Settings
        s = Settings()
        is_dev = s.APP_ENV == "development"
        allowed = ["*"] if is_dev else s.ALLOWED_ORIGINS.split(",")
        assert allowed == ["http://localhost:3000", "https://api.motionlab.kr"]


# ──────────────────────────────────────────────
# 3. Settings: APP_ENV 기본값은 development
# ──────────────────────────────────────────────
def test_settings_default_app_env_is_development():
    """APP_ENV 미설정 시 기본값은 development여야 함"""
    env = {"INTERNAL_API_KEY": "test-key-32chars-padding-xxxxxxxx"}
    # APP_ENV가 설정돼 있을 수 있으므로 명시적으로 제거
    with patch.dict("os.environ", env, clear=False):
        import os
        os.environ.pop("APP_ENV", None)
        from config.settings import Settings
        s = Settings()
        assert s.APP_ENV == "development"


# ──────────────────────────────────────────────
# 4. Settings: ALLOWED_ORIGINS 기본값 확인
# ──────────────────────────────────────────────
def test_settings_default_allowed_origins():
    """ALLOWED_ORIGINS 기본값은 'http://localhost:3000'이어야 함"""
    with patch.dict("os.environ", {"INTERNAL_API_KEY": "test-key-32chars-padding-xxxxxxxx"}, clear=False):
        import os
        os.environ.pop("ALLOWED_ORIGINS", None)
        from config.settings import Settings
        s = Settings()
        assert s.ALLOWED_ORIGINS == "http://localhost:3000"


# ──────────────────────────────────────────────
# 5. 운영 환경: ALLOWED_ORIGINS 단일 값도 동작
# ──────────────────────────────────────────────
def test_settings_production_single_origin():
    """운영 환경에서 ALLOWED_ORIGINS가 단일 값이면 리스트 1개로 분리되어야 함"""
    with patch.dict("os.environ", {
        "APP_ENV": "production",
        "INTERNAL_API_KEY": "test-key-32chars-padding-xxxxxxxx",
        "ALLOWED_ORIGINS": "https://api.motionlab.kr",
    }):
        from config.settings import Settings
        s = Settings()
        is_dev = s.APP_ENV == "development"
        allowed = ["*"] if is_dev else s.ALLOWED_ORIGINS.split(",")
        assert allowed == ["https://api.motionlab.kr"]


# ──────────────────────────────────────────────
# 헬퍼: APP_ENV에 따라 CORS 파라미터를 계산하는 로직 (main.py와 동일)
# ──────────────────────────────────────────────
def _build_cors_params(app_env: str, allowed_origins_str: str) -> dict:
    """main.py의 CORS 파라미터 계산 로직을 독립적으로 재현"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    is_dev = app_env == "development"
    origins = ["*"] if is_dev else allowed_origins_str.split(",")
    methods = ["*"] if is_dev else ["GET", "POST"]
    headers = ["*"] if is_dev else ["X-Internal-API-Key", "Content-Type"]

    mini_app = FastAPI()
    mini_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=methods,
        allow_headers=headers,
    )

    @mini_app.get("/ping")
    def ping():
        return {"ok": True}

    return mini_app


# ──────────────────────────────────────────────
# 6. E2E: 개발 환경에서 임의 출처 CORS 허용
# ──────────────────────────────────────────────
def test_dev_cors_allows_arbitrary_origin():
    """개발 환경에서는 와일드카드로 임의 출처 요청에 CORS 헤더가 반환되어야 함"""
    app = _build_cors_params("development", "http://localhost:3000")
    client = TestClient(app, raise_server_exceptions=False)
    response = client.options(
        "/ping",
        headers={"Origin": "http://malicious.example.com", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code in (200, 204)
    assert "access-control-allow-origin" in response.headers


# ──────────────────────────────────────────────
# 7. E2E: 운영 환경에서 허용되지 않은 출처는 CORS 헤더 없음
# ──────────────────────────────────────────────
def test_prod_cors_blocks_unlisted_origin():
    """운영 환경에서 ALLOWED_ORIGINS 목록에 없는 출처에는 allow-origin 헤더가 반환되지 않아야 함"""
    app = _build_cors_params("production", "http://localhost:3000")
    client = TestClient(app, raise_server_exceptions=False)
    response = client.options(
        "/ping",
        headers={"Origin": "http://evil.example.com", "Access-Control-Request-Method": "GET"},
    )
    origin_header = response.headers.get("access-control-allow-origin", "")
    assert origin_header != "*"
    assert origin_header != "http://evil.example.com"
