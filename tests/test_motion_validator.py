import pytest
from core.motion_validator import MotionValidator
from core.constants import MotionValidation


@pytest.fixture
def validator():
    return MotionValidator()


@pytest.fixture
def sample_angle_configs():
    """
    실제 sports_config.json의 angles 구조와 동일한 형태.
    angle_validation이 각 angle 안에 내장됨.
    """
    return {
        "angle_a": {
            "points": ["shoulder", "elbow", "wrist"],
            "angle_validation": {"min_normal": 140.0, "max_normal": 180.0},
            "levels": {},
        },
        "angle_b": {
            "points": ["shoulder", "elbow", "wrist"],
            "angle_validation": {"min_normal": 60.0, "max_normal": 120.0},
            "levels": {},
        },
        "angle_c": {
            "points": ["hip", "shoulder", "nose"],
            "angle_validation": {"min_normal": 125.0, "max_normal": 180.0},
            "levels": {},
        },
        "angle_d": {
            "points": None,
            "angle_validation": {"min_normal": 20.0, "max_normal": 70.0},
            "levels": {},
        },
    }


@pytest.fixture
def normal_phases():
    return [
        {"name": "phase_1", "start_frame": 0, "end_frame": 20, "duration_ms": 833},
        {"name": "phase_2", "start_frame": 21, "end_frame": 50, "duration_ms": 1250},
        {"name": "phase_3", "start_frame": 51, "end_frame": 80, "duration_ms": 1250},
    ]


# 1. 전부 통과
def test_all_pass(validator, sample_angle_configs, normal_phases):
    angles = {"angle_a": 165.0, "angle_b": 90.0, "angle_c": 150.0, "angle_d": 45.0}
    result = validator.validate_motion(angles, sample_angle_configs, normal_phases)
    assert result["valid"] is True
    assert result["reason"] is None


# 2. 각도 전부 범위 밖 (OUTLIER_MARGIN 30도 적용해도 밖)
def test_all_angles_out_of_range(validator, sample_angle_configs, normal_phases):
    angles = {"angle_a": 10.0, "angle_b": 300.0, "angle_c": 5.0, "angle_d": 350.0}
    result = validator.validate_motion(angles, sample_angle_configs, normal_phases)
    assert result["valid"] is False
    assert "각도 범위 불일치" in result["reason"]


# 3. full_motion만 있으면 구간 실패
def test_only_fallback_phase(validator, sample_angle_configs):
    angles = {"angle_a": 165.0, "angle_b": 90.0, "angle_c": 150.0, "angle_d": 45.0}
    fallback = [
        {
            "name": MotionValidation.FALLBACK_PHASE_NAME,
            "start_frame": 0,
            "end_frame": 100,
            "duration_ms": 4166,
        }
    ]
    result = validator.validate_motion(angles, sample_angle_configs, fallback)
    assert result["valid"] is False
    assert "구간 부족" in result["reason"]


# 4. 구간 0개
def test_no_phases(validator, sample_angle_configs):
    angles = {"angle_a": 165.0, "angle_b": 90.0, "angle_c": 150.0, "angle_d": 45.0}
    result = validator.validate_motion(angles, sample_angle_configs, [])
    assert result["valid"] is False
    assert "구간 부족" in result["reason"]


# 5. angle_configs 비어있으면 각도 검증 스킵 → 구간만 체크
def test_no_angle_configs(validator, normal_phases):
    angles = {"angle_a": 999.0}
    result = validator.validate_motion(angles, {}, normal_phases)
    assert result["valid"] is True


# 6. angle_validation 없는 angle은 건너뜀
def test_angle_without_validation(validator, normal_phases):
    configs = {
        "angle_a": {
            "points": ["a", "b", "c"],
            "levels": {},
            # angle_validation 없음
        },
    }
    angles = {"angle_a": 999.0}
    result = validator.validate_motion(angles, configs, normal_phases)
    assert result["valid"] is True


# 7. 경계값 테스트 (min_normal ± OUTLIER_MARGIN)
def test_outlier_margin_boundary(validator):
    """min_normal=140, OUTLIER_MARGIN=30 → 실제 허용 하한 110"""
    configs = {
        "angle_a": {
            "points": ["a", "b", "c"],
            "angle_validation": {"min_normal": 140.0, "max_normal": 180.0},
        },
        "angle_b": {
            "points": ["a", "b", "c"],
            "angle_validation": {"min_normal": 60.0, "max_normal": 120.0},
        },
        "angle_c": {
            "points": ["a", "b", "c"],
            "angle_validation": {"min_normal": 100.0, "max_normal": 160.0},
        },
    }
    phases = [
        {"name": "p1", "start_frame": 0, "end_frame": 20, "duration_ms": 833},
        {"name": "p2", "start_frame": 21, "end_frame": 50, "duration_ms": 1250},
    ]

    # 3개 모두 범위 안 → 통과 (MIN_ANGLES_IN_RANGE=3 충족)
    result = validator.validate_motion(
        {"angle_a": 110.0, "angle_b": 30.0, "angle_c": 70.0}, configs, phases
    )
    assert result["valid"] is True

    # angle_a만 범위 밖 → 2/3 통과, but in_range=2 < MIN_ANGLES_IN_RANGE=3 → 실패
    result = validator.validate_motion(
        {"angle_a": 109.0, "angle_b": 90.0, "angle_c": 130.0}, configs, phases
    )
    assert result["valid"] is False


# 8. MIN_VALID_ANGLES_RATIO 테스트 (4개 중 2개 통과 = 50%)
def test_half_ratio_with_enough_count(validator, sample_angle_configs, normal_phases):
    # 4개 중 2개만 통과 → ratio=0.5 BUT in_range=2 < MIN_ANGLES_IN_RANGE=3 → 실패
    angles = {"angle_a": 165.0, "angle_b": 90.0, "angle_c": 0.0, "angle_d": 350.0}
    result = validator.validate_motion(angles, sample_angle_configs, normal_phases)
    assert result["valid"] is False


# 9. 4개 중 3개 통과 → ratio=0.75 >= 0.5, in_range=3 >= 3 → 통과
def test_three_of_four_pass(validator, sample_angle_configs, normal_phases):
    angles = {"angle_a": 165.0, "angle_b": 90.0, "angle_c": 150.0, "angle_d": 350.0}
    result = validator.validate_motion(angles, sample_angle_configs, normal_phases)
    assert result["valid"] is True


# ── 변수 테스트: 실제 앱에서 발생 가능한 케이스 ──


# 10. average_angles에 None 값이 섞여있는 경우
def test_none_value_in_angles(validator, sample_angle_configs, normal_phases):
    """MediaPipe가 일부 각도를 못 구한 경우 — None은 건너뜀"""
    angles = {"angle_a": 165.0, "angle_b": None, "angle_c": 150.0, "angle_d": 45.0}
    result = validator.validate_motion(angles, sample_angle_configs, normal_phases)
    # angle_b는 None이라 건너뜀 → 3개 중 3개 통과 → valid
    assert result["valid"] is True


# 11. average_angles가 빈 dict
def test_empty_average_angles(validator, sample_angle_configs, normal_phases):
    """영상이 너무 짧아서 각도가 하나도 계산 안 된 경우"""
    result = validator.validate_motion({}, sample_angle_configs, normal_phases)
    # total=0 → passed=True (검증할 게 없으니 스킵)
    assert result["valid"] is True


# 12. 음수 각도값
def test_negative_angle_value(validator, normal_phases):
    """계산 오류로 음수 각도가 들어온 경우"""
    configs = {
        "angle_a": {
            "points": ["a", "b", "c"],
            "angle_validation": {"min_normal": 140.0, "max_normal": 180.0},
        },
        "angle_b": {
            "points": ["a", "b", "c"],
            "angle_validation": {"min_normal": 60.0, "max_normal": 120.0},
        },
        "angle_c": {
            "points": ["a", "b", "c"],
            "angle_validation": {"min_normal": 100.0, "max_normal": 160.0},
        },
    }
    angles = {"angle_a": -10.0, "angle_b": -5.0, "angle_c": -20.0}
    result = validator.validate_motion(angles, configs, normal_phases)
    # 전부 범위 밖 → 실패
    assert result["valid"] is False


# 13. phases에 name 키가 없는 dict 섞임
def test_phase_without_name_key(validator, sample_angle_configs):
    """PhaseDetector가 비정상 dict를 반환한 경우"""
    angles = {"angle_a": 165.0, "angle_b": 90.0, "angle_c": 150.0, "angle_d": 45.0}
    phases = [
        {"name": "phase_1", "start_frame": 0, "end_frame": 20, "duration_ms": 833},
        {"start_frame": 21, "end_frame": 50, "duration_ms": 1250},  # name 없음
        {"name": "phase_3", "start_frame": 51, "end_frame": 80, "duration_ms": 1250},
    ]
    result = validator.validate_motion(angles, sample_angle_configs, phases)
    # name이 없는 dict → .get("name") = None ≠ "full_motion" → real_phase로 카운트
    # 3개 real_phases → passed
    assert result["valid"] is True


# 14. config에 있는 각도가 average_angles에 없는 경우
def test_config_has_more_angles_than_measured(validator, normal_phases):
    """config에 6개 각도가 정의되어 있지만 영상에서 2개만 측정된 경우"""
    configs = {
        "angle_a": {
            "points": ["a", "b", "c"],
            "angle_validation": {"min_normal": 140.0, "max_normal": 180.0},
        },
        "angle_b": {
            "points": ["a", "b", "c"],
            "angle_validation": {"min_normal": 60.0, "max_normal": 120.0},
        },
        "angle_c": {
            "points": ["a", "b", "c"],
            "angle_validation": {"min_normal": 100.0, "max_normal": 160.0},
        },
        "angle_d": {
            "points": ["a", "b", "c"],
            "angle_validation": {"min_normal": 20.0, "max_normal": 70.0},
        },
    }
    # 4개 중 2개만 측정됨
    angles = {"angle_a": 165.0, "angle_b": 90.0}
    result = validator.validate_motion(angles, configs, normal_phases)
    # total=2, in_range=2, ratio=1.0 >= 0.5 BUT in_range=2 < MIN_ANGLES_IN_RANGE=3 → 실패
    assert result["valid"] is False
