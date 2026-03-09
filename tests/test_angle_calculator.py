import pytest
from core.angle_calculator import AngleCalculator
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

    # 4. 데이터 없는 각도 → NO_DATA
    def test_no_data_score(self, sample_angle_config):
        calc = AngleCalculator(sample_angle_config)
        avg = {"angle_a": 170.0}  # angle_b, angle_c 없음
        scores = calc._calculate_scores(avg)

        assert scores["angle_a"] == FeedbackScore.IDEAL
        assert scores["angle_b"] == FeedbackScore.NO_DATA
        assert scores["angle_c"] == FeedbackScore.NO_DATA

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

    # 10. None 값 처리
    def test_none_angle_value(self, sample_angle_config):
        calc = AngleCalculator(sample_angle_config)
        avg = {"angle_a": None, "angle_b": 90.0, "angle_c": 150.0}
        scores = calc._calculate_scores(avg)

        assert scores["angle_a"] == FeedbackScore.NO_DATA
        assert scores["angle_b"] == FeedbackScore.IDEAL
