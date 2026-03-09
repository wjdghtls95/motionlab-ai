"""
E2E 엔드포인트 테스트 — /analyze API 검증
"""

from tests.e2e.conftest import e2e_mocks


class TestAnalyzeSuccess:

    @e2e_mocks
    def test_full_pipeline(self, client, headers, golf_request):
        response = client.post("/analyze", json=golf_request, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @e2e_mocks
    def test_result_structure(self, client, headers, golf_request):
        response = client.post("/analyze", json=golf_request, headers=headers)
        assert response.status_code == 200
        data = response.json()
        required_keys = {
            "success",
            "motion_id",
            "result",
            "feedback",
            "improvements",
            "prompt_version",
        }
        assert required_keys.issubset(data.keys())

    @e2e_mocks
    def test_result_fields(self, client, headers, golf_request):
        response = client.post("/analyze", json=golf_request, headers=headers)
        assert response.status_code == 200
        data = response.json()
        result = data["result"]
        assert "total_frames" in result
        assert "duration_seconds" in result
        assert "angles" in result
        assert "phases" in result
        assert isinstance(result["phases"], list)
        assert len(result["phases"]) > 0

    @e2e_mocks
    def test_score_range(self, client, headers, golf_request):
        response = client.post("/analyze", json=golf_request, headers=headers)
        data = response.json()
        score = data.get("overall_score")
        if score is not None:
            assert 0 <= score <= 100

    @e2e_mocks
    def test_prompt_version(self, client, headers, golf_request):
        response = client.post("/analyze", json=golf_request, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "prompt_version" in data
        assert isinstance(data["prompt_version"], str)
        assert len(data["prompt_version"]) > 0

    @e2e_mocks
    def test_phases_structure(self, client, headers, golf_request):
        response = client.post("/analyze", json=golf_request, headers=headers)
        assert response.status_code == 200
        data = response.json()
        phases = data["result"]["phases"]
        for phase in phases:
            assert "name" in phase
            assert "start_frame" in phase
            assert "end_frame" in phase
            assert "duration_ms" in phase

    @e2e_mocks
    def test_feedback_is_string(self, client, headers, golf_request):
        response = client.post("/analyze", json=golf_request, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["feedback"], str)

    @e2e_mocks
    def test_improvements_is_list(self, client, headers, golf_request):
        response = client.post("/analyze", json=golf_request, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["improvements"], list)


class TestAnalyzeAuth:

    def test_missing_api_key(self, client, golf_request):
        response = client.post("/analyze", json=golf_request)
        assert response.status_code == 401

    def test_wrong_api_key(self, client, golf_request):
        response = client.post(
            "/analyze", json=golf_request, headers={"X-Internal-API-Key": "wrong_key"}
        )
        assert response.status_code == 401

    def test_empty_api_key(self, client, golf_request):
        response = client.post(
            "/analyze", json=golf_request, headers={"X-Internal-API-Key": ""}
        )
        assert response.status_code == 401


class TestAnalyzeValidation:

    def test_missing_motion_id(self, client, headers):
        body = {
            "video_url": "https://example.com/video.mp4",
            "sport_type": "GOLF",
        }
        response = client.post("/analyze", json=body, headers=headers)
        assert response.status_code == 422

    def test_missing_video_url(self, client, headers):
        body = {
            "motion_id": 1001,
            "sport_type": "GOLF",
        }
        response = client.post("/analyze", json=body, headers=headers)
        assert response.status_code == 422

    def test_missing_sport_type(self, client, headers):
        body = {
            "motion_id": 1001,
            "video_url": "https://example.com/video.mp4",
        }
        response = client.post("/analyze", json=body, headers=headers)
        assert response.status_code == 422

    def test_invalid_level(self, client, headers):
        body = {
            "motion_id": 1001,
            "video_url": "https://example.com/video.mp4",
            "sport_type": "GOLF",
            "level": "INVALID_LEVEL",
        }
        response = client.post("/analyze", json=body, headers=headers)
        assert response.status_code == 422

    def test_wrong_type_motion_id(self, client, headers):
        body = {
            "motion_id": "not_a_number",
            "video_url": "https://example.com/video.mp4",
            "sport_type": "GOLF",
        }
        response = client.post("/analyze", json=body, headers=headers)
        assert response.status_code == 422

    def test_empty_body(self, client, headers):
        response = client.post("/analyze", json={}, headers=headers)
        assert response.status_code == 422

    def test_non_json_body(self, client, headers):
        response = client.post("/analyze", data="not json", headers=headers)
        assert response.status_code == 422


class TestAnalyzeLevel:

    @e2e_mocks
    def test_beginner_level(self, client, headers, golf_request):
        golf_request["level"] = "BEGINNER"
        response = client.post("/analyze", json=golf_request, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @e2e_mocks
    def test_intermediate_level(self, client, headers, golf_request):
        golf_request["level"] = "INTERMEDIATE"
        response = client.post("/analyze", json=golf_request, headers=headers)
        assert response.status_code == 200

    @e2e_mocks
    def test_advanced_level(self, client, headers, golf_request):
        golf_request["level"] = "ADVANCED"
        response = client.post("/analyze", json=golf_request, headers=headers)
        assert response.status_code == 200

    @e2e_mocks
    def test_pro_level(self, client, headers, golf_request):
        golf_request["level"] = "PRO"
        response = client.post("/analyze", json=golf_request, headers=headers)
        assert response.status_code == 200


class TestAnalyzeEdgeCases:

    @e2e_mocks
    def test_empty_video_url_string(self, client, headers):
        body = {
            "motion_id": 9999,
            "video_url": "",
            "sport_type": "GOLF",
        }
        response = client.post("/analyze", json=body, headers=headers)
        assert response.status_code in (200, 400, 422, 500, 503)

    @e2e_mocks
    def test_unknown_sport_type(self, client, headers):
        """존재하지 않는 sport_type — mock이 config도 우회하므로 200 반환"""
        body = {
            "motion_id": 9998,
            "video_url": "https://example.com/unknown.mp4",
            "sport_type": "CURLING",
        }
        response = client.post("/analyze", json=body, headers=headers)
        # e2e_mocks가 get_sport_config를 mock하므로 200 반환
        # 실제 종목 검증은 unit test에서 별도로 테스트
        assert response.status_code == 200

    @e2e_mocks
    def test_negative_motion_id(self, client, headers):
        body = {
            "motion_id": -1,
            "video_url": "https://example.com/video.mp4",
            "sport_type": "GOLF",
        }
        response = client.post("/analyze", json=body, headers=headers)
        assert response.status_code in (200, 400, 422, 500)

    @e2e_mocks
    def test_zero_motion_id(self, client, headers):
        body = {
            "motion_id": 0,
            "video_url": "https://example.com/video.mp4",
            "sport_type": "GOLF",
        }
        response = client.post("/analyze", json=body, headers=headers)
        assert response.status_code in (200, 400, 422, 500)

    @e2e_mocks
    def test_weight_sport_type(self, client, headers, weight_request):
        response = client.post("/analyze", json=weight_request, headers=headers)
        assert response.status_code in (200, 400, 500)


class TestHealthEndpoint:

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
