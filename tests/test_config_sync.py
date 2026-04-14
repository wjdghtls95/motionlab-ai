"""
Config 동기화 테스트 (D-002)

local / file / remote 모드 및 /admin/reload-config 엔드포인트 검증
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────

MINIMAL_CONFIG = {
    "meta": {"config_version": "2.0.0"},
    "sports": {
        "GOLF": {
            "sub_categories": {
                "DRIVER": {
                    "angles": {},
                    "phases": [],
                }
            }
        }
    },
}


def _reset_cache():
    """모듈 수준 캐시 초기화 (테스트 격리용)"""
    import core.sport_configs as cs

    cs._RAW_CONFIG = None
    cs._CONFIG_VERSION = "unknown"
    cs._CONFIG_FORMAT = "v1"


def _make_settings(source: str, path: str = "", url: str = ""):
    """Settings 목 객체 생성"""
    s = MagicMock()
    s.CONFIG_SOURCE = source
    s.SPORTS_CONFIG_PATH = path
    s.CONFIG_REMOTE_URL = url
    return s


# ──────────────────────────────────────────────────────────
# 1. local 모드 — 번들된 파일 읽기
# ──────────────────────────────────────────────────────────


def test_local_mode_loads_bundled_file():
    """CONFIG_SOURCE=local: 이미지 번들 파일에서 로드해야 함"""
    _reset_cache()

    with patch("core.sport_configs.get_settings", return_value=_make_settings("local")):
        from core.sport_configs import load_sports_config

        config = load_sports_config(force_reload=True)

    # 번들 파일에는 최소한 GOLF/DRIVER가 있어야 함
    assert "GOLF" in config
    assert "DRIVER" in config["GOLF"]["sub_categories"]


# ──────────────────────────────────────────────────────────
# 2. file 모드 — SPORTS_CONFIG_PATH 경로 읽기
# ──────────────────────────────────────────────────────────


def test_file_mode_reads_from_env_path(tmp_path):
    """CONFIG_SOURCE=file: SPORTS_CONFIG_PATH 경로의 파일을 읽어야 함"""
    config_file = tmp_path / "custom_config.json"
    config_file.write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")

    _reset_cache()

    with patch(
        "core.sport_configs.get_settings",
        return_value=_make_settings("file", path=str(config_file)),
    ):
        from core.sport_configs import load_sports_config

        config = load_sports_config(force_reload=True)

    assert "GOLF" in config


def test_file_mode_uses_local_fallback_when_path_empty():
    """CONFIG_SOURCE=file, SPORTS_CONFIG_PATH 미설정: 번들 파일 fallback"""
    _reset_cache()

    with patch(
        "core.sport_configs.get_settings",
        return_value=_make_settings("file", path=""),
    ):
        from core.sport_configs import load_sports_config

        config = load_sports_config(force_reload=True)

    # 번들 파일에서 읽으므로 GOLF/DRIVER 존재
    assert "GOLF" in config


def test_file_mode_raises_when_path_not_found(tmp_path):
    """CONFIG_SOURCE=file, 파일 없는 경로: FileNotFoundError 발생"""
    missing_path = str(tmp_path / "nonexistent.json")
    _reset_cache()

    with patch(
        "core.sport_configs.get_settings",
        return_value=_make_settings("file", path=missing_path),
    ):
        from core.sport_configs import load_sports_config

        with pytest.raises(FileNotFoundError):
            load_sports_config(force_reload=True)


# ──────────────────────────────────────────────────────────
# 3. remote 모드 — HTTP fetch
# ──────────────────────────────────────────────────────────


def test_remote_mode_fetches_from_url():
    """CONFIG_SOURCE=remote: URL에서 config를 fetch해야 함"""
    _reset_cache()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "meta": {"config_version": "2.1.0"},
        "sports": {"GOLF": {"sub_categories": {"IRON": {"angles": {}, "phases": []}}}},
    }
    mock_response.raise_for_status = MagicMock()

    with (
        patch(
            "core.sport_configs.get_settings",
            return_value=_make_settings(
                "remote", url="http://config.example.com/sports_config.json"
            ),
        ),
        patch("httpx.get", return_value=mock_response),
    ):
        from core.sport_configs import load_sports_config

        config = load_sports_config(force_reload=True)

    assert "GOLF" in config
    assert "IRON" in config["GOLF"]["sub_categories"]


def test_remote_mode_falls_back_to_local_on_failure():
    """CONFIG_SOURCE=remote, fetch 실패: 번들 로컬 파일로 fallback해야 함"""
    _reset_cache()

    with (
        patch(
            "core.sport_configs.get_settings",
            return_value=_make_settings(
                "remote", url="http://unreachable.example.com/config.json"
            ),
        ),
        patch("httpx.get", side_effect=Exception("connection refused")),
    ):
        from core.sport_configs import load_sports_config

        config = load_sports_config(force_reload=True)

    # fallback으로 번들 파일 읽음
    assert "GOLF" in config


# ──────────────────────────────────────────────────────────
# 4. reload_config — 캐시 초기화 후 재로드
# ──────────────────────────────────────────────────────────


def test_reload_config_force_refreshes_cache(tmp_path):
    """reload_config(force_reload=True): 캐시를 버리고 파일을 다시 읽어야 함"""
    v1_file = tmp_path / "config_v1.json"
    v1_data = {**MINIMAL_CONFIG, "meta": {"config_version": "1.0.0"}}
    v1_file.write_text(json.dumps(v1_data), encoding="utf-8")

    _reset_cache()

    settings_mock = _make_settings("file", path=str(v1_file))

    with patch("core.sport_configs.get_settings", return_value=settings_mock):
        from core.sport_configs import load_sports_config, get_config_version

        load_sports_config(force_reload=True)
        assert get_config_version() == "1.0.0"

    # 파일 내용을 v2로 변경
    v2_data = {**MINIMAL_CONFIG, "meta": {"config_version": "2.5.0"}}
    v1_file.write_text(json.dumps(v2_data), encoding="utf-8")

    with patch("core.sport_configs.get_settings", return_value=settings_mock):
        from core.sport_configs import reload_config

        reload_config()
        assert get_config_version() == "2.5.0"


# ──────────────────────────────────────────────────────────
# 5. /admin/reload-config 엔드포인트
# ──────────────────────────────────────────────────────────


def _make_test_app():
    """테스트용 최소 FastAPI 앱 생성"""
    from fastapi import FastAPI
    from api.admin import router

    app = FastAPI()
    app.include_router(router)
    return app


def test_admin_reload_config_requires_api_key():
    """/admin/reload-config: API 키 없이 요청 시 401 반환"""
    client = TestClient(_make_test_app(), raise_server_exceptions=False)
    response = client.post("/admin/reload-config")
    assert response.status_code == 401


def test_admin_reload_config_returns_success_with_valid_key():
    """/admin/reload-config: 유효한 API 키로 요청 시 success=True 반환"""
    _reset_cache()

    # utils.dependencies 모듈의 settings 변수만 패치 (API 키 검증 통과)
    with patch("utils.dependencies.settings") as mock_dep_settings:
        mock_dep_settings.INTERNAL_API_KEY = "test-api-key"

        client = TestClient(_make_test_app(), raise_server_exceptions=False)
        response = client.post(
            "/admin/reload-config",
            headers={"X-Internal-Api-Key": "test-api-key"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "version" in body
    assert "sports" in body


# ──────────────────────────────────────────────────────────
# 6. diagnosis 라벨 — 번들 config에 실제 라벨 존재 확인 (R-096)
# ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("sub_category", ["DRIVER", "IRON"])
def test_bundled_config_has_diagnosis_labels(sub_category):
    """번들 sports_config.json의 DRIVER/IRON 모든 각도에 diagnosis 라벨이 정의돼 있어야 함.

    diagnosis_low / diagnosis_high 중 하나라도 없으면 _determine_diagnosis()가
    None을 반환해 LLM 피드백에 진단명이 빠지는 문제가 생긴다 (R-096).
    """
    _reset_cache()

    with patch("core.sport_configs.get_settings", return_value=_make_settings("local")):
        from core.sport_configs import load_sports_config

        config = load_sports_config(force_reload=True)

    angles = config["GOLF"]["sub_categories"][sub_category]["angles"]
    assert angles, f"{sub_category} angles가 비어있음"

    for angle_name, angle_def in angles.items():
        assert angle_def.get("diagnosis_low"), (
            f"{sub_category}/{angle_name}: diagnosis_low 미정의"
        )
        assert angle_def.get("diagnosis_high"), (
            f"{sub_category}/{angle_name}: diagnosis_high 미정의"
        )
