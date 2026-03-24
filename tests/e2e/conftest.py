# ruff: noqa: E402
"""
E2E 테스트 설정 — 환경변수 오버라이딩 + 파이프라인 Mock

⚠️ E402 (import not at top) 억제 이유:
   os.environ 설정 → get_settings.cache_clear() → app import
   이 순서가 반드시 지켜져야 settings가 테스트용 환경변수를 읽음.
   순서가 바뀌면 INTERNAL_API_KEY가 실제 값으로 캐시되어 401 에러 발생.
"""

import os

# ── 1) 환경변수 설정 (settings import 전에 반드시 먼저) ──
os.environ["INTERNAL_API_KEY"] = "test_key"
os.environ["ENABLE_LLM_NOOP"] = "true"
os.environ["OPENAI_API_KEY"] = "not_used_in_noop"

# ── 2) settings 캐시 초기화 후 app import ──
from config.settings import get_settings

get_settings.cache_clear()

from fastapi.testclient import TestClient
from main import app

import pytest
import copy
import random
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock
from functools import wraps

# ============================================================
# 상수
# ============================================================
TEST_API_KEY = os.environ["INTERNAL_API_KEY"]
HEADERS = {"X-Internal-API-Key": TEST_API_KEY}


# ============================================================
# Mock 데이터 생성 헬퍼
# ============================================================
def mock_metadata():
    return {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "frame_count": 150,
        "duration_seconds": 5.0,
    }


def mock_landmarks_data(num_frames=150, num_landmarks=33):
    rng = random.Random(42)
    frames = []
    for i in range(num_frames):
        landmarks = []
        for j in range(num_landmarks):
            landmarks.append(
                {
                    "x": round(rng.uniform(0.1, 0.9), 4),
                    "y": round(rng.uniform(0.1, 0.9), 4),
                    "z": round(rng.uniform(-0.5, 0.5), 4),
                    "visibility": round(rng.uniform(0.8, 1.0), 4),
                }
            )
        frames.append({"frame_index": i, "landmarks": landmarks})
    return {
        "frames": frames,
        "total_frames": num_frames,
        "valid_frames": num_frames,
    }


def mock_angles_data():
    frame_angles_list = []
    np.random.seed(42)
    for i in range(150):
        frame_angles_list.append(
            {
                "frame_idx": i,
                "angles": {
                    "left_arm_angle": round(float(np.random.uniform(80, 100)), 1),
                    "right_arm_angle": round(float(np.random.uniform(75, 95)), 1),
                    "spine_angle": round(float(np.random.uniform(160, 180)), 1),
                },
            }
        )
    return {
        "frame_angles": frame_angles_list,
        "average_angles": {
            "left_arm_angle": 90.5,
            "right_arm_angle": 85.3,
            "spine_angle": 170.2,
        },
        "angle_scores": {
            "left_arm_angle": 90,
            "right_arm_angle": 70,
            "spine_angle": 90,
        },
        "weighted_score": 83.3,
    }


def mock_phases():
    return [
        {"name": "address", "start_frame": 0, "end_frame": 24, "duration_ms": 800},
        {"name": "backswing", "start_frame": 25, "end_frame": 54, "duration_ms": 1000},
        {"name": "top", "start_frame": 55, "end_frame": 74, "duration_ms": 667},
        {"name": "downswing", "start_frame": 75, "end_frame": 99, "duration_ms": 833},
        {"name": "impact", "start_frame": 100, "end_frame": 119, "duration_ms": 667},
        {
            "name": "follow_through",
            "start_frame": 120,
            "end_frame": 149,
            "duration_ms": 1000,
        },
    ]


def mock_validation_result():
    return {"valid": True, "reason": ""}


def mock_sport_config():
    """get_sport_config 반환값 — 실제 config 로드를 우회"""
    return {
        "angles": {
            "left_arm_angle": {
                "points": ["left_shoulder", "left_elbow", "left_wrist"],
                "ideal_range": [165.0, 180.0],
                "weight": 0.3,
            },
            "right_arm_angle": {
                "points": ["right_shoulder", "right_elbow", "right_wrist"],
                "ideal_range": [165.0, 180.0],
                "weight": 0.3,
            },
            "spine_angle": {
                "points": ["left_shoulder", "left_hip", "left_knee"],
                "ideal_range": [160.0, 180.0],
                "weight": 0.4,
            },
        },
        "phases": [
            {
                "name": "address",
                "detection_rule": "stabilization",
                "target_angle": "spine_angle",
                "params": {"window": 5, "std_threshold": 2.0, "position": "start"},
            },
        ],
        "angle_validation": {},
    }


# ============================================================
# e2e_mocks 데코레이터
# ============================================================


def e2e_mocks(func):
    """파이프라인 전체를 mock하는 데코레이터"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # --- VideoResource mock (async context manager) ---
        mock_video_cm = AsyncMock()
        mock_video_cm.__aenter__ = AsyncMock(return_value="/tmp/mock_video.mp4")
        mock_video_cm.__aexit__ = AsyncMock(return_value=False)
        mock_video_cls = MagicMock(return_value=mock_video_cm)

        # --- AngleCalculator mock ---
        mock_ac_instance = MagicMock()
        mock_ac_instance.calculate_angles.return_value = mock_angles_data()
        mock_ac_cls = MagicMock(return_value=mock_ac_instance)

        # --- PhaseDetector mock ---
        mock_pd_instance = MagicMock()
        mock_pd_instance.detect_phases.return_value = mock_phases()
        mock_pd_cls = MagicMock(return_value=mock_pd_instance)

        with (
            patch("services.analysis_service.VideoResource", mock_video_cls),
            patch(
                "services.analysis_service.VideoService.extract_metadata",
                return_value=mock_metadata(),
            ),
            patch(
                "services.analysis_service.MediaPipeAnalyzer.extract_landmarks",
                return_value=mock_landmarks_data(),
            ),
            patch(
                "services.analysis_service.get_sport_config",
                return_value=mock_sport_config(),
            ),
            patch("services.analysis_service.AngleCalculator", mock_ac_cls),
            patch("services.analysis_service.PhaseDetector", mock_pd_cls),
            patch(
                "services.analysis_service.MotionValidator.validate_motion",
                return_value=mock_validation_result(),
            ),
        ):
            return func(*args, **kwargs)

    return wrapper


# ============================================================
# Pytest Fixtures
# ============================================================


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient — 세션 전체에서 1번만 생성/종료"""
    with TestClient(app) as c:
        yield c


# pytest hook: CI exit code 정확히 반환
_pytest_exit_status = 0


def pytest_sessionfinish(session, exitstatus):
    """테스트 결과 exit code 저장"""
    global _pytest_exit_status
    _pytest_exit_status = exitstatus


def pytest_unconfigure(config):
    """pytest 완전 종료 — 실제 exit code 반환"""
    import os

    os._exit(_pytest_exit_status)


@pytest.fixture
def headers():
    return copy.deepcopy(HEADERS)


@pytest.fixture
def golf_request():
    """골프 분석 요청 — sport_type은 대문자 GOLF"""
    return {
        "motion_id": 1001,
        "video_url": "https://example.com/golf_swing.mp4",
        "sport_type": "GOLF",
        "sub_category": "DRIVER",
        "level": "BEGINNER",
    }


@pytest.fixture
def weight_request():
    return {
        "motion_id": 2001,
        "video_url": "https://example.com/deadlift.mp4",
        "sport_type": "WEIGHT",
        "sub_category": "DEADLIFT",
        "level": "INTERMEDIATE",
    }
