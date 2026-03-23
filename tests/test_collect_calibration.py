"""
collect_calibration.py 단위 테스트 (R-070)

테스트 대상:
  - _build_angles_per_frame: "frame_index" 키 직접 참조
  - _to_phase_detector_input: PhaseDetector 입력 형식 변환
  - save_raw_landmarks: 3프레임 간격 샘플링, JSON 저장
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


# scripts/ 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collect_calibration import (
    _build_angles_per_frame,
    _to_phase_detector_input,
    save_raw_landmarks,
)


# ──────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────


def _make_frame(frame_index: int, angle_value: float) -> dict:
    """테스트용 단일 프레임 딕셔너리 생성."""
    return {
        "frame_index": frame_index,
        "timestamp": frame_index / 30.0,
        "landmarks": [],  # _calculate_frame_angles를 mock하므로 내용 무관
    }


def _make_landmarks_data(total_frames: int, fps: float = 30.0) -> dict:
    """테스트용 landmarks_data 딕셔너리 생성."""
    return {
        "frames": [_make_frame(i, 0) for i in range(total_frames)],
        "total_frames": total_frames,
        "valid_frames": total_frames,
        "fps": fps,
    }


# ──────────────────────────────────────
# _build_angles_per_frame
# ──────────────────────────────────────


class TestBuildAnglesPerFrame:
    def test_uses_frame_index_key(self):
        """'frame_index' 키로 인덱스를 읽어야 한다 ('frame_idx' 아님)."""
        calc = MagicMock()
        calc._calculate_frame_angles.return_value = {"left_arm_angle": 165.0}

        frames = [
            {"frame_index": 5, "landmarks": []},
            {"frame_index": 10, "landmarks": []},
        ]
        result = _build_angles_per_frame(frames, calc)

        assert 5 in result
        assert 10 in result
        assert result[5]["left_arm_angle"] == 165.0

    def test_skips_frame_with_no_angles(self):
        """_calculate_frame_angles가 None을 반환하면 해당 프레임을 건너뛴다."""
        calc = MagicMock()
        calc._calculate_frame_angles.side_effect = [{"left_arm_angle": 170.0}, None]

        frames = [
            {"frame_index": 0, "landmarks": []},
            {"frame_index": 1, "landmarks": []},
        ]
        result = _build_angles_per_frame(frames, calc)

        assert len(result) == 1
        assert 0 in result
        assert 1 not in result

    def test_empty_frames_returns_empty(self):
        """빈 프레임 리스트는 빈 딕셔너리를 반환한다."""
        calc = MagicMock()
        result = _build_angles_per_frame([], calc)
        assert result == {}

    def test_frame_index_default_zero_when_missing(self):
        """'frame_index' 키가 없으면 기본값 0을 사용한다."""
        calc = MagicMock()
        calc._calculate_frame_angles.return_value = {"left_arm_angle": 160.0}

        frames = [{"landmarks": []}]  # frame_index 키 없음
        result = _build_angles_per_frame(frames, calc)

        assert 0 in result


# ──────────────────────────────────────
# _to_phase_detector_input
# ──────────────────────────────────────


class TestToPhaseDetectorInput:
    def _make_angles_per_frame(self, values: list) -> dict:
        """frame_idx → {"left_arm_angle": value} 맵 생성."""
        return {i: {"left_arm_angle": v} for i, v in enumerate(values)}

    def test_output_format(self):
        """PhaseDetector 기대 형식: {angle: {"frames": {...}, "average": float}}."""
        angles_per_frame = self._make_angles_per_frame([170.0, 160.0, 150.0])
        result = _to_phase_detector_input(angles_per_frame, ["left_arm_angle"])

        assert "left_arm_angle" in result
        entry = result["left_arm_angle"]
        assert "frames" in entry
        assert "average" in entry
        assert entry["frames"] == {0: 170.0, 1: 160.0, 2: 150.0}
        assert abs(entry["average"] - 160.0) < 0.1

    def test_missing_angle_excluded(self):
        """요청 각도가 프레임에 없으면 결과에서 제외한다."""
        angles_per_frame = {0: {"left_arm_angle": 165.0}}
        result = _to_phase_detector_input(
            angles_per_frame, ["left_arm_angle", "spine_angle"]
        )

        assert "left_arm_angle" in result
        assert "spine_angle" not in result  # 데이터 없으므로 제외

    def test_empty_input_returns_empty(self):
        """빈 angles_per_frame은 빈 딕셔너리를 반환한다."""
        result = _to_phase_detector_input({}, ["left_arm_angle"])
        assert result == {}

    def test_average_is_mean_of_values(self):
        """average는 해당 각도 값들의 평균이어야 한다."""
        values = [100.0, 200.0, 300.0]
        angles_per_frame = self._make_angles_per_frame(values)
        result = _to_phase_detector_input(angles_per_frame, ["left_arm_angle"])

        assert abs(result["left_arm_angle"]["average"] - 200.0) < 0.01


# ──────────────────────────────────────
# save_raw_landmarks
# ──────────────────────────────────────


class TestSaveRawLandmarks:
    def _make_landmarks(self, total_frames: int = 30) -> dict:
        return _make_landmarks_data(total_frames)

    def test_saves_json_file(self):
        """JSON 파일이 지정 디렉토리에 생성되어야 한다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://www.youtube.com/watch?v=AbCdEfGhIjK"
            landmarks = self._make_landmarks(30)
            save_raw_landmarks(url, landmarks, "GOLF", "DRIVER", tmpdir)

            files = list(Path(tmpdir).rglob("*.json"))
            assert len(files) == 1

    def test_subdirectory_is_sport_sub(self):
        """저장 디렉토리 구조는 {save_dir}/{SPORT}_{SUB}/이어야 한다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://www.youtube.com/watch?v=AbCdEfGhIjK"
            save_raw_landmarks(url, self._make_landmarks(), "GOLF", "DRIVER", tmpdir)

            subdir = Path(tmpdir) / "GOLF_DRIVER"
            assert subdir.is_dir()

    def test_3frame_interval_sampling(self):
        """30프레임 영상에서 3프레임 간격이면 10개 프레임이 저장되어야 한다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://www.youtube.com/watch?v=AbCdEfGhIjK"
            landmarks = self._make_landmarks(30)
            save_raw_landmarks(url, landmarks, "GOLF", "DRIVER", tmpdir)

            json_file = next(Path(tmpdir).rglob("*.json"))
            with open(json_file) as f:
                data = json.load(f)

            assert data["sampled_frames"] == 10  # 30 / 3
            assert data["sample_interval"] == 3

    def test_video_id_extracted_from_url(self):
        """파일명에 YouTube video ID가 포함되어야 한다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://www.youtube.com/watch?v=TestVidId01"
            save_raw_landmarks(url, self._make_landmarks(), "GOLF", "DRIVER", tmpdir)

            files = list(Path(tmpdir).rglob("*.json"))
            assert any("TestVidId01" in f.name for f in files)

    def test_json_contains_required_fields(self):
        """JSON 파일에 필수 메타 필드가 포함되어야 한다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://www.youtube.com/watch?v=AbCdEfGhIjK"
            save_raw_landmarks(
                url, self._make_landmarks(15), "GOLF", "IRON", tmpdir, model_type="full"
            )

            json_file = next(Path(tmpdir).rglob("*.json"))
            with open(json_file) as f:
                data = json.load(f)

            required_keys = {
                "url",
                "sport",
                "sub",
                "model_type",
                "fps",
                "total_frames",
                "sampled_frames",
                "sample_interval",
                "collected_at",
                "frames",
            }
            assert required_keys.issubset(data.keys())
            assert data["sport"] == "GOLF"
            assert data["sub"] == "IRON"
            assert data["model_type"] == "full"

    def test_unknown_video_id_for_non_youtube_url(self):
        """YouTube 형식이 아닌 URL은 video_id를 'unknown'으로 처리한다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://example.com/video.mp4"
            save_raw_landmarks(url, self._make_landmarks(), "GOLF", "DRIVER", tmpdir)

            files = list(Path(tmpdir).rglob("*.json"))
            assert any("unknown" in f.name for f in files)
