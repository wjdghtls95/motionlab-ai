# ============================================================
# tests/e2e/test_analyze_scenario.py
# ============================================================
"""
E2E 시나리오 테스트 — 실제 사용 흐름 시뮬레이션
"""

import pytest
from tests.e2e.conftest import e2e_mocks, HEADERS


class TestScenarioHealthThenAnalyze:

    @e2e_mocks
    def test_health_then_analyze(self, client, headers, golf_request):
        health = client.get("/health")
        assert health.status_code == 200

        response = client.post("/analyze", json=golf_request, headers=headers)
        assert response.status_code == 200
        assert response.json()["success"] is True


class TestScenarioSameVideoMultipleLevels:

    @e2e_mocks
    def test_same_video_different_levels(self, client, headers, golf_request):
        results = {}
        for level in ["BEGINNER", "INTERMEDIATE", "ADVANCED", "PRO"]:
            golf_request["level"] = level
            response = client.post("/analyze", json=golf_request, headers=headers)
            assert response.status_code == 200
            results[level] = response.json()

        angles_beginner = results["BEGINNER"]["result"]["angles"]
        for level in ["INTERMEDIATE", "ADVANCED", "PRO"]:
            assert results[level]["result"]["angles"] == angles_beginner


class TestScenarioAuthRetry:

    @e2e_mocks
    def test_auth_fail_then_succeed(self, client, headers, golf_request):
        bad_headers = {"X-Internal-API-Key": "wrong"}
        response1 = client.post("/analyze", json=golf_request, headers=bad_headers)
        assert response1.status_code == 401

        response2 = client.post("/analyze", json=golf_request, headers=headers)
        assert response2.status_code == 200
        assert response2.json()["success"] is True

    @e2e_mocks
    def test_no_key_then_succeed(self, client, headers, golf_request):
        response1 = client.post("/analyze", json=golf_request)
        assert response1.status_code == 401

        response2 = client.post("/analyze", json=golf_request, headers=headers)
        assert response2.status_code == 200


class TestScenarioSequentialAnalysis:

    @e2e_mocks
    def test_sequential_all_levels(self, client, headers, golf_request):
        for level in ["BEGINNER", "INTERMEDIATE", "ADVANCED", "PRO"]:
            golf_request["level"] = level
            response = client.post("/analyze", json=golf_request, headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["motion_id"] == golf_request["motion_id"]


class TestScenarioDuplicateRequest:

    @e2e_mocks
    def test_duplicate_request_same_result(self, client, headers, golf_request):
        response1 = client.post("/analyze", json=golf_request, headers=headers)
        response2 = client.post("/analyze", json=golf_request, headers=headers)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        assert data1["result"]["angles"] == data2["result"]["angles"]
        assert data1["result"]["phases"] == data2["result"]["phases"]


class TestScenarioValidationThenSuccess:

    @e2e_mocks
    def test_fix_and_retry(self, client, headers, golf_request):
        bad_body = {
            "motion_id": 1001,
            "video_url": "https://example.com/video.mp4",
        }
        response1 = client.post("/analyze", json=bad_body, headers=headers)
        assert response1.status_code == 422

        response2 = client.post("/analyze", json=golf_request, headers=headers)
        assert response2.status_code == 200
        assert response2.json()["success"] is True
