import pytest
import numpy as np
from core.angle_calculator import AngleCalculator, _PHASE_WINDOW_MARGIN, _SMOOTH_WINDOW
from core.constants import FeedbackScore


@pytest.fixture
def sample_angle_config():
    """실제 sports_config.json에서 resolve된 후의 구조"""
    return {
        "angle_a": {
            "points": ["left_shoulder", "left_elbow", "left_wrist"],
            "ideal_range": [160.0, 180.0],
            "weight": 1.0,
            "angle_validation": {"min_normal": 140.0, "max_normal": 180.0},
        },
        "angle_b": {
            "points": ["right_shoulder", "right_elbow", "right_wrist"],
            "ideal_range": [70.0, 110.0],
            "weight": 1.0,
            "angle_validation": {"min_normal": 60.0, "max_normal": 120.0},
        },
        "angle_c": {
            "points": ["left_hip", "left_shoulder", "nose"],
            "ideal_range": [140.0, 165.0],
            "weight": 1.0,
            "angle_validation": {"min_normal": 125.0, "max_normal": 180.0},
        },
    }


class TestAngleScoring:
    # 1. 전부 ideal_range 안 → 전부 90점
    def test_all_ideal(self, sample_angle_config):
        calc = AngleCalculator(sample_angle_config)
        avg = {"angle_a": 170.0, "angle_b": 90.0, "angle_c": 150.0}
        scores = calc._calculate_scores(avg)

        assert scores["angle_a"] == FeedbackScore.IDEAL
        assert scores["angle_b"] == FeedbackScore.IDEAL
        assert scores["angle_c"] == FeedbackScore.IDEAL

    # 2. ideal 밖 + validation 안 → CAUTION
    def test_caution_score(self, sample_angle_config):
        calc = AngleCalculator(sample_angle_config)
        avg = {"angle_a": 145.0}  # ideal [160,180] 밖, validation [140,180] 안
        scores = calc._calculate_scores(avg)

        assert scores["angle_a"] == FeedbackScore.CAUTION

    # 3. 둘 다 밖 → CORRECTION
    def test_correction_score(self, sample_angle_config):
        calc = AngleCalculator(sample_angle_config)
        avg = {"angle_a": 10.0}  # ideal 밖, validation 밖
        scores = calc._calculate_scores(avg)

        assert scores["angle_a"] == FeedbackScore.CORRECTION

    # 4. 입력에 없는 각도는 결과에 포함되지 않음
    def test_missing_angles_not_scored(self, sample_angle_config):
        calc = AngleCalculator(sample_angle_config)
        avg = {"angle_a": 170.0}  # angle_b, angle_c 없음
        scores = calc._calculate_scores(avg)

        assert scores["angle_a"] == FeedbackScore.IDEAL
        assert "angle_b" not in scores
        assert "angle_c" not in scores

    # 5. 가중 점수 — 전부 IDEAL이면 90점
    def test_weighted_all_ideal(self, sample_angle_config):
        calc = AngleCalculator(sample_angle_config)
        scores = {"angle_a": 90, "angle_b": 90, "angle_c": 90}
        result = calc._calculate_weighted_score(scores)

        assert result == 90.0

    # 6. 가중 점수 — 섞여있으면 평균
    def test_weighted_mixed(self, sample_angle_config):
        calc = AngleCalculator(sample_angle_config)
        scores = {"angle_a": 90, "angle_b": 40, "angle_c": 70}
        result = calc._calculate_weighted_score(scores)

        # weight 전부 1.0 → (90+40+70) / 3 = 66.7
        assert result == 66.7

    # 7. 빈 scores → 기본값
    def test_weighted_empty(self, sample_angle_config):
        calc = AngleCalculator(sample_angle_config)
        result = calc._calculate_weighted_score({})

        assert result == float(FeedbackScore.DEFAULT)

    # 8. weight가 다른 경우 자동 정규화
    def test_weighted_different_weights(self):
        config = {
            "angle_a": {"points": None, "ideal_range": [0, 360], "weight": 2.0},
            "angle_b": {"points": None, "ideal_range": [0, 360], "weight": 1.0},
        }
        calc = AngleCalculator(config)
        scores = {"angle_a": 90, "angle_b": 30}
        result = calc._calculate_weighted_score(scores)

        # (90*2 + 30*1) / (2+1) = 210/3 = 70.0
        assert result == 70.0

    # 9. ideal_range 경계값
    def test_boundary_ideal_range(self, sample_angle_config):
        calc = AngleCalculator(sample_angle_config)

        # 정확히 경계 → IDEAL
        scores = calc._calculate_scores({"angle_a": 160.0})
        assert scores["angle_a"] == FeedbackScore.IDEAL

        scores = calc._calculate_scores({"angle_a": 180.0})
        assert scores["angle_a"] == FeedbackScore.IDEAL

    # 10. None 값은 채점에서 제외됨
    def test_none_angle_value_excluded(self, sample_angle_config):
        calc = AngleCalculator(sample_angle_config)
        avg = {"angle_a": None, "angle_b": 90.0, "angle_c": 150.0}
        scores = calc._calculate_scores(avg)

        assert "angle_a" not in scores
        assert scores["angle_b"] == FeedbackScore.IDEAL


@pytest.fixture
def phase_angle_config():
    """페이즈 필드 포함 config — calculate_phase_scores 테스트용"""
    return {
        "angle_a": {
            "points": ["left_shoulder", "left_elbow", "left_wrist"],
            "phase": "backswing_top",
            "ideal_range": [160.0, 180.0],
            "weight": 1.0,
            "diagnosis_low": "오버스윙",
            "diagnosis_high": "언더스윙",
            "angle_validation": {"min_normal": 140.0, "max_normal": 180.0},
        },
        "angle_b": {
            "points": ["right_shoulder", "right_elbow", "right_wrist"],
            "phase": "address",
            "ideal_range": [70.0, 110.0],
            "weight": 1.0,
            "diagnosis_low": "과도한 굽힘",
            "diagnosis_high": "과도한 신전",
            "angle_validation": {"min_normal": 60.0, "max_normal": 120.0},
        },
        "angle_c": {
            "points": ["left_hip", "left_shoulder", "nose"],
            "phase": "impact",
            "ideal_range": [140.0, 165.0],
            "weight": 1.0,
            "diagnosis_low": "과도한 앞 숙임",
            "diagnosis_high": "과도한 뒤 기울기",
            "angle_validation": {"min_normal": 125.0, "max_normal": 180.0},
        },
    }


class TestCalculatePhaseScores:
    """calculate_phase_scores() 단위 테스트"""

    def _make_frame_angles(self, frames: dict) -> list:
        """헬퍼: {frame_idx: {angle_name: value}} → frame_angles 형식 변환"""
        return [{"frame_idx": idx, "angles": angles} for idx, angles in frames.items()]

    def _make_detected_phases(self, phases: dict) -> list:
        """헬퍼: {name: (start, end)} → detected_phases 형식 변환"""
        return [
            {"name": name, "start_frame": start, "end_frame": end, "duration_ms": 0}
            for name, (start, end) in phases.items()
        ]

    # 1. 정상 케이스 — 모든 페이즈 감지, 이상적 각도 → 전체 90점
    def test_all_phases_detected_ideal(self, phase_angle_config):
        calc = AngleCalculator(phase_angle_config)

        # frame 20은 backswing_top(0~1) + margin(5) = range(0,7) 밖에 위치
        # → window 확장에도 평균에 포함되지 않아야 함
        frame_angles = self._make_frame_angles(
            {
                0: {"angle_a": 170.0, "angle_b": 90.0, "angle_c": 150.0},
                1: {"angle_a": 172.0, "angle_b": 92.0, "angle_c": 152.0},
                20: {"angle_a": 0.0, "angle_b": 0.0},
            }
        )
        detected_phases = self._make_detected_phases(
            {
                "backswing_top": (0, 1),
                "address": (0, 1),
                "impact": (0, 1),
            }
        )

        result = calc.calculate_phase_scores(frame_angles, detected_phases)

        assert result["phase_angles"]["angle_a"] == pytest.approx(171.0, abs=0.1)
        assert result["phase_scores"]["angle_a"] == FeedbackScore.IDEAL
        assert result["phase_scores"]["angle_b"] == FeedbackScore.IDEAL
        assert result["phase_scores"]["angle_c"] == FeedbackScore.IDEAL
        assert result["overall_score"] == 90.0
        assert result["diagnosis"]["angle_a"] is None
        assert result["undetected_phases"] == []

    # 2. 페이즈 미감지 → 해당 angle 제외 (옵션 B)
    def test_phase_not_detected_excluded(self, phase_angle_config):
        calc = AngleCalculator(phase_angle_config)

        frame_angles = self._make_frame_angles(
            {
                0: {"angle_a": 170.0, "angle_b": 90.0},
            }
        )
        detected_phases = self._make_detected_phases(
            {
                "backswing_top": (0, 0),
                "address": (0, 0),
                # impact 미감지
            }
        )

        result = calc.calculate_phase_scores(frame_angles, detected_phases)

        assert "angle_c" not in result["phase_angles"]
        assert "angle_c" not in result["phase_scores"]
        assert "impact" in result["undetected_phases"]

    # 3. diagnosis_low — 각도가 ideal_range 아래이면 diagnosis_low 반환
    def test_diagnosis_low(self, phase_angle_config):
        calc = AngleCalculator(phase_angle_config)

        # angle_a ideal_range = [160, 180], 값 150 → diagnosis_low
        frame_angles = self._make_frame_angles(
            {
                0: {"angle_a": 150.0},
            }
        )
        detected_phases = self._make_detected_phases(
            {
                "backswing_top": (0, 0),
            }
        )

        result = calc.calculate_phase_scores(frame_angles, detected_phases)

        assert result["diagnosis"]["angle_a"] == "오버스윙"

    # 4. diagnosis_high — 각도가 ideal_range 위이면 diagnosis_high 반환
    def test_diagnosis_high(self, phase_angle_config):
        calc = AngleCalculator(phase_angle_config)

        # angle_a ideal_range = [160, 180], 값 190 → diagnosis_high
        frame_angles = self._make_frame_angles(
            {
                0: {"angle_a": 190.0},
            }
        )
        detected_phases = self._make_detected_phases(
            {
                "backswing_top": (0, 0),
            }
        )

        result = calc.calculate_phase_scores(frame_angles, detected_phases)

        assert result["diagnosis"]["angle_a"] == "언더스윙"

    # 5. phase 필드 없는 angle은 결과에서 제외됨
    def test_angle_without_phase_field_excluded(self):
        config_no_phase = {
            "angle_x": {
                "points": ["left_shoulder", "left_elbow", "left_wrist"],
                "ideal_range": [160.0, 180.0],
                "weight": 1.0,
                # phase 필드 없음
            }
        }
        calc = AngleCalculator(config_no_phase)
        frame_angles = [{"frame_idx": 0, "angles": {"angle_x": 170.0}}]
        detected_phases = []

        result = calc.calculate_phase_scores(frame_angles, detected_phases)

        assert "angle_x" not in result["phase_angles"]
        assert result["undetected_phases"] == []

    # 6. 모든 페이즈 미감지 → phase_angles 빈 dict, overall_score DEFAULT
    def test_no_phases_detected(self, phase_angle_config):
        calc = AngleCalculator(phase_angle_config)
        frame_angles = [{"frame_idx": 0, "angles": {"angle_a": 170.0}}]
        detected_phases = []

        result = calc.calculate_phase_scores(frame_angles, detected_phases)

        assert result["phase_angles"] == {}
        assert result["phase_scores"] == {}
        assert result["overall_score"] == float(FeedbackScore.DEFAULT)

    # 7. 페이즈 구간 내 여러 프레임 → 평균값 사용
    def test_phase_frames_averaged(self, phase_angle_config):
        calc = AngleCalculator(phase_angle_config)

        frame_angles = self._make_frame_angles(
            {
                2: {"angle_a": 160.0},
                3: {"angle_a": 170.0},
                4: {"angle_a": 180.0},
            }
        )
        detected_phases = self._make_detected_phases(
            {
                "backswing_top": (2, 4),
            }
        )

        result = calc.calculate_phase_scores(frame_angles, detected_phases)

        assert result["phase_angles"]["angle_a"] == pytest.approx(170.0, abs=0.1)

    # 8. phase window ±MARGIN 확장 — 경계 밖 프레임도 포함되는지 확인
    def test_phase_window_margin_includes_boundary_frames(self, phase_angle_config):
        """start-MARGIN ~ end+MARGIN 범위 프레임이 평균에 포함돼야 한다."""
        calc = AngleCalculator(phase_angle_config)

        # 감지된 페이즈: frame 10~12 (3프레임)
        # ±MARGIN 확장 후: frame (10-MARGIN)~(12+MARGIN)
        # 확장 범위 안팎으로 각각 다른 값을 두어 포함 여부 확인
        margin = _PHASE_WINDOW_MARGIN
        inner_frames = {i: {"angle_a": 170.0} for i in range(10, 13)}  # 감지된 구간
        outer_frames = {
            i: {"angle_a": 170.0} for i in range(10 - margin, 10)
        }  # 확장 범위 (start 쪽)
        outer_frames.update(
            {i: {"angle_a": 170.0} for i in range(13, 13 + margin)}
        )  # 확장 범위 (end 쪽)
        outside_frames = {
            10 - margin - 1: {"angle_a": 0.0},
            13 + margin + 1: {"angle_a": 0.0},
        }  # 범위 밖

        all_frames = {**outside_frames, **outer_frames, **inner_frames}
        frame_angles = self._make_frame_angles(all_frames)
        detected_phases = self._make_detected_phases({"backswing_top": (10, 12)})

        result = calc.calculate_phase_scores(frame_angles, detected_phases)

        # 범위 밖 0.0 프레임이 포함됐다면 평균이 170에서 크게 벗어남
        assert result["phase_angles"]["angle_a"] == pytest.approx(170.0, abs=0.5)


class TestSmoothLandmarks:
    """_smooth_landmarks() 단위 테스트"""

    def _make_frames(self, n: int, x_values: list, y_value: float = 0.5) -> list:
        """헬퍼: n개 프레임, 33개 랜드마크, 첫 번째 랜드마크 x를 x_values로 설정"""
        frames = []
        for i in range(n):
            landmarks = [
                {"x": 0.5, "y": y_value, "z": 0.0, "visibility": 1.0} for _ in range(33)
            ]
            if i < len(x_values):
                landmarks[0]["x"] = x_values[i]
            frames.append({"frame_index": i, "landmarks": landmarks})
        return frames

    def test_smoothing_reduces_noise(self):
        """노이즈가 있는 시계열이 스무딩 후 분산이 줄어야 한다."""
        calc = AngleCalculator({})
        # 지그재그 노이즈: 0.3, 0.7, 0.3, 0.7, ...
        noisy_x = [0.3 if i % 2 == 0 else 0.7 for i in range(20)]
        frames = self._make_frames(20, noisy_x)

        smoothed = calc._smooth_landmarks(frames)

        raw_variance = np.var([f["landmarks"][0]["x"] for f in frames])
        smoothed_variance = np.var([f["landmarks"][0]["x"] for f in smoothed])
        assert smoothed_variance < raw_variance

    def test_short_sequence_returns_original(self):
        """프레임 수가 window_length보다 적으면 원본을 그대로 반환한다."""
        calc = AngleCalculator({})
        frames = self._make_frames(_SMOOTH_WINDOW - 1, [0.1, 0.9, 0.1, 0.9])

        result = calc._smooth_landmarks(frames)

        assert result is frames  # 동일 객체 반환 (복사 없음)

    def test_visibility_unchanged(self):
        """스무딩은 x/y 좌표만 변경하고 visibility는 건드리지 않는다."""
        calc = AngleCalculator({})
        frames = self._make_frames(
            10, [0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7]
        )
        original_visibility = frames[0]["landmarks"][0]["visibility"]

        smoothed = calc._smooth_landmarks(frames)

        assert smoothed[0]["landmarks"][0]["visibility"] == original_visibility

    def test_original_frames_not_mutated(self):
        """_smooth_landmarks는 원본 frames를 변경하지 않는다."""
        calc = AngleCalculator({})
        noisy_x = [0.3 if i % 2 == 0 else 0.7 for i in range(10)]
        frames = self._make_frames(10, noisy_x)
        original_x = frames[0]["landmarks"][0]["x"]

        calc._smooth_landmarks(frames)

        assert frames[0]["landmarks"][0]["x"] == original_x
