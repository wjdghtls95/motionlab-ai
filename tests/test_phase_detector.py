import pytest
import numpy as np
from core.phase_detector import PhaseDetector
from core.constants import MotionValidation

# ──────────────────────────────────────
# 헬퍼: 합성 각도 데이터 생성
# ──────────────────────────────────────


def _make_angles_data(angle_name: str, values: list) -> dict:
    """AngleCalculator 출력 형태로 생성"""
    frames = {i: v for i, v in enumerate(values)}
    return {
        angle_name: {
            "frames": frames,
            "average": float(np.mean(values)),
        }
    }


def _generate_swing_curve(n_frames: int = 100) -> list:
    """스윙 형태 곡선: 높음 → 낮음 → 높음 → 안정"""
    t = np.linspace(0, 2 * np.pi, n_frames)
    return list(170 - 40 * np.sin(t))


def _generate_squat_curve(n_frames: int = 80) -> list:
    """스쿼트 형태 곡선: 높음 → 낮음 → 높음"""
    t = np.linspace(0, np.pi, n_frames)
    return list(170 - 50 * np.sin(t))


# ──────────────────────────────────────
# Fixtures
# ──────────────────────────────────────


@pytest.fixture
def swing_config():
    """스윙 형태 운동의 phase config"""
    return [
        {
            "name": "setup",
            "detection_rule": "stabilization",
            "target_angle": "main_angle",
            "params": {"window": 5, "std_threshold": 2.0, "position": "start"},
        },
        {
            "name": "motion_start",
            "detection_rule": "angle_decrease",
            "target_angle": "main_angle",
            "params": {"window": 5, "min_change": 5.0},
        },
        {
            "name": "peak",
            "detection_rule": "angle_min",
            "target_angle": "main_angle",
            "params": {"search_after": "motion_start"},
        },
        {
            "name": "return_motion",
            "detection_rule": "angle_increase",
            "target_angle": "main_angle",
            "params": {"window": 5, "min_change": 5.0, "search_after": "peak"},
        },
        {
            "name": "finish",
            "detection_rule": "stabilization",
            "target_angle": "main_angle",
            "params": {"window": 10, "std_threshold": 3.0, "position": "end"},
        },
    ]


@pytest.fixture
def squat_config():
    """스쿼트 형태 운동의 phase config"""
    return [
        {
            "name": "standing",
            "detection_rule": "stabilization",
            "target_angle": "knee_angle",
            "params": {"window": 5, "std_threshold": 2.0, "position": "start"},
        },
        {
            "name": "descent",
            "detection_rule": "angle_decrease",
            "target_angle": "knee_angle",
            "params": {"window": 5, "min_change": 5.0},
        },
        {
            "name": "bottom",
            "detection_rule": "angle_min",
            "target_angle": "knee_angle",
            "params": {"search_after": "descent"},
        },
        {
            "name": "ascent",
            "detection_rule": "angle_increase",
            "target_angle": "knee_angle",
            "params": {"window": 5, "min_change": 5.0, "search_after": "bottom"},
        },
    ]


# ──────────────────────────────────────
# 테스트
# ──────────────────────────────────────


class TestPhaseDetector:

    # 1. 정상 스윙 감지 — 최소 3개 이상 구간
    def test_normal_swing_detection(self, swing_config):
        values = _generate_swing_curve(100)
        angles_data = _make_angles_data("main_angle", values)
        detector = PhaseDetector(swing_config, fps=24)
        phases = detector.detect_phases(angles_data)

        assert len(phases) >= 3
        for p in phases:
            assert p["start_frame"] <= p["end_frame"]
            assert p["duration_ms"] >= 0

    # 2. 빈 config → 폴백
    def test_empty_config_fallback(self):
        values = _generate_swing_curve(100)
        angles_data = _make_angles_data("main_angle", values)
        detector = PhaseDetector([], fps=24)
        phases = detector.detect_phases(angles_data)

        assert len(phases) == 1
        assert phases[0]["name"] == MotionValidation.FALLBACK_PHASE_NAME

    # 3. 프레임 0개 → 폴백
    def test_no_frames(self, swing_config):
        detector = PhaseDetector(swing_config, fps=24)
        phases = detector.detect_phases({})

        assert len(phases) == 1
        assert phases[0]["name"] == MotionValidation.FALLBACK_PHASE_NAME

    # 4. 짧은 영상 (MIN_FRAMES_FOR_DETECTION 미만) → 폴백
    def test_short_video_fallback(self, swing_config):
        values = [170.0] * 5
        angles_data = _make_angles_data("main_angle", values)
        detector = PhaseDetector(swing_config, fps=24)
        phases = detector.detect_phases(angles_data)

        assert len(phases) == 1
        assert phases[0]["name"] == MotionValidation.FALLBACK_PHASE_NAME

    # 5. 알 수 없는 rule → 해당 구간만 건너뜀
    def test_unknown_rule_skipped(self):
        config = [
            {
                "name": "valid_phase",
                "detection_rule": "angle_max",
                "target_angle": "a",
                "params": {},
            },
            {
                "name": "bad_phase",
                "detection_rule": "nonexistent_rule",
                "target_angle": "a",
                "params": {},
            },
        ]
        values = list(range(50))
        angles_data = _make_angles_data("a", values)
        detector = PhaseDetector(config, fps=24)
        phases = detector.detect_phases(angles_data)

        phase_names = [p["name"] for p in phases]
        assert "valid_phase" in phase_names
        assert "bad_phase" not in phase_names

    # 6. target_angle 데이터 없음 → 전부 건너뜀 → 폴백
    def test_missing_angle_data(self, swing_config):
        values = _generate_swing_curve(100)
        angles_data = _make_angles_data("other_angle", values)
        detector = PhaseDetector(swing_config, fps=24)
        phases = detector.detect_phases(angles_data)

        assert len(phases) == 1
        assert phases[0]["name"] == MotionValidation.FALLBACK_PHASE_NAME

    # 7. 스쿼트 config → 코드 수정 없이 동작
    def test_squat_detection(self, squat_config):
        values = _generate_squat_curve(80)
        angles_data = _make_angles_data("knee_angle", values)
        detector = PhaseDetector(squat_config, fps=30)
        phases = detector.detect_phases(angles_data)

        assert len(phases) >= 2
        for p in phases:
            assert p["start_frame"] <= p["end_frame"]

    # 8. search_after 체이닝 → 시간순 정렬 보장
    def test_search_after_ordering(self, swing_config):
        values = _generate_swing_curve(100)
        angles_data = _make_angles_data("main_angle", values)
        detector = PhaseDetector(swing_config, fps=24)
        phases = detector.detect_phases(angles_data)

        for i in range(len(phases) - 1):
            assert phases[i]["start_frame"] <= phases[i + 1]["start_frame"]

    # 9. 평탄한 데이터 → 변화 없으므로 감지 제한적
    def test_flat_data(self, swing_config):
        values = [170.0] * 100
        angles_data = _make_angles_data("main_angle", values)
        detector = PhaseDetector(swing_config, fps=24)
        phases = detector.detect_phases(angles_data)

        # 변화 없으니 대부분 감지 실패 → 폴백이거나 최소 구간
        assert len(phases) >= 1

    # ── 변수 테스트: 실제 앱에서 발생 가능한 케이스 ──

    # 10. 프레임이 비연속 (중간 누락)
    def test_sparse_frames(self, swing_config):
        """MediaPipe가 일부 프레임만 추출한 경우 — 보간 처리 확인"""
        sparse_frames = {
            0: 170.0,
            10: 150.0,
            25: 130.0,
            50: 140.0,
            75: 160.0,
            99: 170.0,
        }
        angles_data = {
            "main_angle": {
                "frames": sparse_frames,
                "average": 155.0,
            }
        }
        detector = PhaseDetector(swing_config, fps=24)
        phases = detector.detect_phases(angles_data)

        # 보간 후 감지 가능 — 에러 없이 결과 반환
        assert len(phases) >= 1
        for p in phases:
            assert "name" in p
            assert "start_frame" in p
            assert "end_frame" in p

    # 11. 각도 값에 NaN 포함
    def test_nan_in_angle_values(self):
        """계산 오류로 NaN이 포함된 경우"""
        config = [
            {
                "name": "test_phase",
                "detection_rule": "angle_max",
                "target_angle": "a",
                "params": {},
            },
        ]
        values = [170.0] * 20 + [float("nan")] * 5 + [170.0] * 25
        angles_data = _make_angles_data("a", values)
        detector = PhaseDetector(config, fps=24)

        # NaN이 있어도 에러 없이 실행되어야 함
        phases = detector.detect_phases(angles_data)
        assert len(phases) >= 1

    # 12. fps가 0인 경우
    def test_zero_fps(self, swing_config):
        """fps=0이면 duration_ms 계산 시 ZeroDivisionError 방지"""
        values = _generate_swing_curve(100)
        angles_data = _make_angles_data("main_angle", values)
        detector = PhaseDetector(swing_config, fps=0)
        phases = detector.detect_phases(angles_data)

        for p in phases:
            assert p["duration_ms"] == 0  # fps=0이면 0ms 반환

    # 13. params 없는 phase config
    def test_missing_params(self):
        """config에 params 키가 아예 없는 경우 — 기본값 사용"""
        config = [
            {"name": "test_phase", "detection_rule": "angle_max", "target_angle": "a"},
            # params 키 자체가 없음
        ]
        values = list(range(50))
        angles_data = _make_angles_data("a", values)
        detector = PhaseDetector(config, fps=24)
        phases = detector.detect_phases(angles_data)

        assert len(phases) >= 1
        assert phases[0]["name"] == "test_phase"

    # 14. target_angle이 None인 경우
    def test_target_angle_none(self):
        """config에 target_angle이 null로 설정된 경우"""
        config = [
            {
                "name": "test_phase",
                "detection_rule": "angle_max",
                "target_angle": None,
                "params": {},
            },
        ]
        values = list(range(50))
        angles_data = _make_angles_data("a", values)
        detector = PhaseDetector(config, fps=24)
        phases = detector.detect_phases(angles_data)

        # target_angle 없으니 건너뜀 → 폴백
        assert len(phases) == 1
        assert phases[0]["name"] == MotionValidation.FALLBACK_PHASE_NAME
