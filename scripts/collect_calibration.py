#!/usr/bin/env python3
"""
캘리브레이션 데이터 수집 스크립트 (R-051 / R-055)

YouTube URL에서 프로 선수 영상을 다운로드하여 MediaPipe로 관절 각도를 측정하고
sports_config.json의 기준값(mediapipe_center, std)을 실측값으로 업데이트합니다.

기능:
  - 논문값 sanity check: 실측값이 논문값 대비 ±SANITY_THRESHOLD° 초과 시 경고
  - 샘플 부족 fallback: sample_count < MIN_SAMPLES이면 논문값 유지
  - 메타 자동 기록: measured_at, source_file, paper_reference, paper_delta
  - config_version 자동 bump: 2.x.x → 3.0.0 (최초 실측값 적용 시)

사용법:
    cd motionlab-ai
    source venv/bin/activate
    pip install yt-dlp  # 또는: pip install -r requirements-dev.txt

    python scripts/collect_calibration.py \\
        --url-file ../motionlab-config/calibration_data/urls/golf_driver.md \\
        --sport GOLF --sub DRIVER

    # 결과 확인만 (파일 저장 없음)
    python scripts/collect_calibration.py \\
        --url-file ../motionlab-config/calibration_data/urls/golf_driver.md \\
        --sport GOLF --sub DRIVER --dry-run

버전 전략:
    v1.0/v2.0 CSV 디렉토리 = 논문값 이력 (보존)
    sports_config.json meta.config_version = 데이터 버전 추적
    Git 커밋 = 버전별 전체 스냅샷

저작권 주의:
    CC BY 라이선스 영상만 사용할 것.
    원본 영상은 임시 디렉토리에 다운로드 후 처리가 끝나면 자동 삭제됩니다.
"""

import argparse
import csv as csv_module
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# motionlab-ai 패키지 루트를 경로에 추가 (scripts/ 하위에서 실행 시)
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.angle_calculator import AngleCalculator
from core.mediapipe_analyzer import MediaPipeAnalyzer
from core.phase_detector import PhaseDetector
from core.sport_configs import load_sports_config
from utils.exceptions.errors import NoKeypointsError, VideoTooShortError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────
# 실측값이 논문값과 이 각도 이상 차이나면 경고 (측정 오류 의심)
SANITY_THRESHOLD = 30.0

# 이 수 미만이면 샘플 부족으로 논문값 fallback
MIN_SAMPLES = 5

# Raw 랜드마크 저장 시 샘플링 간격 (3프레임마다 1개 저장)
_LANDMARK_SAMPLE_INTERVAL = 3

# 캘리브레이션 전용 포즈 모델 경로
# 환경변수 POSE_MODEL_PATH 또는 motionlab-config/models/pose_landmarker_heavy.task
POSE_MODEL_PATH = os.getenv(
    "POSE_MODEL_PATH",
    str(
        Path(__file__).parent.parent.parent
        / "motionlab-config"
        / "models"
        / "pose_landmarker_heavy.task"
    ),
)


# ──────────────────────────────────────────────────────────────
# Heavy 포즈 분석기 (캘리브레이션 전용)
# AI 서버 실시간 분석은 MediaPipeAnalyzer (legacy Full 모델) 유지
# ──────────────────────────────────────────────────────────────


class _HeavyPoseAnalyzer:
    """
    MediaPipe Tasks API + pose_landmarker_heavy.task 기반 포즈 분석기.

    오프라인 캘리브레이션 전용 — 정확도 우선.
    출력 형식은 MediaPipeAnalyzer.extract_landmarks()와 동일.
    """

    def extract_landmarks(self, video_path: str) -> Dict[str, Any]:
        """영상에서 프레임별 랜드마크 추출 (Heavy 모델)."""
        base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없음: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        logger.info(f"📹 MediaPipe Heavy 분석 시작: {video_path}")
        logger.info(
            f"📊 영상 정보: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s"
        )

        if duration < 1.0:
            cap.release()
            raise VideoTooShortError(duration)

        all_landmarks, valid_frames = self._process_frames(
            cap, fps, total_frames, options
        )

        valid_ratio = valid_frames / total_frames if total_frames > 0 else 0
        logger.info(
            f"✅ MediaPipe Heavy 분석 완료: "
            f"{valid_frames}/{total_frames} frames ({valid_ratio:.1%} 유효)"
        )

        if valid_ratio < 0.1:
            raise NoKeypointsError()

        return {
            "frames": all_landmarks,
            "total_frames": total_frames,
            "valid_frames": valid_frames,
            "fps": fps,
        }

    def _process_frames(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        total_frames: int,
        options: mp_vision.PoseLandmarkerOptions,
    ) -> Tuple[list, int]:
        """프레임 순회 및 랜드마크 추출."""
        all_landmarks = []
        valid_frames = 0
        frame_index = 0

        try:
            with mp_vision.PoseLandmarker.create_from_options(options) as detector:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=frame_rgb
                    )
                    timestamp_ms = int(frame_index * 1000 / fps)

                    result = detector.detect_for_video(mp_image, timestamp_ms)

                    if result.pose_landmarks:
                        landmarks = [
                            {
                                "x": float(lm.x),
                                "y": float(lm.y),
                                "z": float(lm.z),
                                "visibility": float(lm.visibility),
                            }
                            for lm in result.pose_landmarks[0]
                        ]
                        all_landmarks.append(
                            {
                                "frame_index": frame_index,
                                "timestamp": frame_index / fps,
                                "landmarks": landmarks,
                            }
                        )
                        valid_frames += 1

                    frame_index += 1
                    MediaPipeAnalyzer._log_progress(frame_index, total_frames)
        finally:
            cap.release()

        return all_landmarks, valid_frames


def _create_pose_analyzer():
    """
    캘리브레이션용 포즈 분석기 생성.

    POSE_MODEL_PATH가 존재하면 Heavy 모델 사용, 아니면 Full(legacy) fallback.
    """
    if Path(POSE_MODEL_PATH).exists():
        logger.info(f"🎯 Heavy 모델 사용: {POSE_MODEL_PATH}")
        return _HeavyPoseAnalyzer()
    logger.warning(
        f"⚠️  Heavy 모델 없음 → Full 모델(legacy) fallback: {POSE_MODEL_PATH}"
    )
    return MediaPipeAnalyzer()


# ──────────────────────────────────────────────────────────────
# URL 파일 읽기
# ──────────────────────────────────────────────────────────────


def read_urls_from_file(file_path: str) -> List[str]:
    """
    URL 목록 파일에서 URL만 추출.

    파일 형식:
        # 주석 (무시됨)
        https://youtube.com/watch?v=abc123
        https://youtube.com/watch?v=def456

    마크다운 리스트 형식도 지원:
        ## 섹션 헤더 (무시됨)
        - https://youtube.com/watch?v=abc123
    """
    urls = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 마크다운 리스트 형식: "- https://..." → "https://..."
            if line.startswith("- "):
                line = line[2:].strip()
            if line.startswith("http"):
                urls.append(line)
    return urls


# ──────────────────────────────────────────────────────────────
# 영상 다운로드
# ──────────────────────────────────────────────────────────────


def download_video(url: str, output_path: str) -> bool:
    """
    yt-dlp로 YouTube URL에서 영상을 다운로드.

    Returns:
        True: 다운로드 성공 / False: 실패 (스킵)
    """
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp가 설치되지 않았습니다: pip install yt-dlp")
        return False

    ydl_opts = {
        "outtmpl": output_path,
        "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        logger.error(f"  다운로드 실패: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# 페이즈 감지 + 각도 추출
# ──────────────────────────────────────────────────────────────


def _build_angles_per_frame(
    frames: List[Dict], calc: AngleCalculator
) -> Dict[int, Dict[str, float]]:
    """
    프레임별 각도 맵 구성.

    AngleCalculator의 frame_idx 키 버그를 우회해 "frame_index"를 직접 참조.
    landmarks_data["frames"]의 각 요소 = {"frame_index": int, "landmarks": [...]}
    """
    result: Dict[int, Dict[str, float]] = {}
    for frame in frames:
        frame_idx = frame.get("frame_index", 0)
        landmarks = frame.get("landmarks", [])
        angles = calc._calculate_frame_angles(landmarks)
        if angles:
            result[frame_idx] = angles
    return result


def _to_phase_detector_input(
    angles_per_frame: Dict[int, Dict[str, float]],
    angle_names: List[str],
) -> Dict[str, Any]:
    """
    { frame_idx: {angle_name: value} } → PhaseDetector 입력 형식으로 변환.

    PhaseDetector 기대 형식:
        {"angle_name": {"frames": {frame_idx: value, ...}, "average": float}}
    """
    result: Dict[str, Any] = {}
    for angle_name in angle_names:
        frames: Dict[int, float] = {}
        values: List[float] = []
        for frame_idx, angle_map in angles_per_frame.items():
            if angle_name in angle_map:
                frames[frame_idx] = angle_map[angle_name]
                values.append(angle_map[angle_name])
        if values:
            result[angle_name] = {
                "frames": frames,
                "average": float(np.mean(values)),
            }
    return result


def extract_angles_from_video(
    video_path: str,
    angle_config_flat: Dict,
    phase_config: Optional[List] = None,
    phase_map: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict]]:
    """
    영상에서 관절 각도를 추출.

    phase_config + phase_map이 있으면 PhaseDetector로 스윙 페이즈를 감지한 후
    각 각도의 지정 페이즈 start_frame에서 값을 추출.
    없으면 전체 프레임 평균 사용 (fallback).

    Args:
        phase_config: sports_config의 phases 리스트 (PhaseDetector 생성자 입력)
        phase_map: {"left_arm_angle": "backswing_top", "spine_angle": "address", ...}

    Returns:
        (angles_dict, landmarks_data) — 실패 시 (None, None)
    """
    analyzer = _create_pose_analyzer()
    landmarks_data = analyzer.extract_landmarks(video_path)
    frames = landmarks_data["frames"]
    fps = landmarks_data.get("fps", 30.0)
    calc = AngleCalculator(angle_config=angle_config_flat)

    if not phase_config or not phase_map:
        # 전체 평균 fallback
        result = calc.calculate_angles(landmarks_data)
        average_angles = result.get("average_angles", {})
        if not average_angles:
            logger.warning("  ⚠️  각도 추출 결과 없음 (영상 품질 또는 포즈 감지 실패)")
            return None, None
        return average_angles, landmarks_data

    # ── PhaseDetector로 스윙 페이즈 감지
    angles_per_frame = _build_angles_per_frame(frames, calc)
    if not angles_per_frame:
        logger.warning("  ⚠️  프레임 각도 추출 실패")
        return None, None

    phase_input = _to_phase_detector_input(
        angles_per_frame, list(angle_config_flat.keys())
    )
    detector = PhaseDetector(phase_config, fps=int(fps))
    detected_phases = detector.detect_phases(phase_input)

    # phase_name → start_frame 맵
    phase_start_frames = {p["name"]: p["start_frame"] for p in detected_phases}
    logger.info(f"  📍 PhaseDetector 감지: {phase_start_frames}")

    # ── 각 각도를 지정 페이즈 start_frame에서 추출
    angles: Dict[str, float] = {}
    for angle_name, phase_name in phase_map.items():
        start_frame = phase_start_frames.get(phase_name)
        if start_frame is None:
            # 페이즈 미감지 → 해당 각도 전체 평균 fallback
            all_vals = [
                v[angle_name] for v in angles_per_frame.values() if angle_name in v
            ]
            if all_vals:
                angles[angle_name] = float(np.mean(all_vals))
                logger.warning(
                    f"  ⚠️  '{phase_name}' 페이즈 미감지 → '{angle_name}' 전체 평균 사용"
                )
        else:
            val = angles_per_frame.get(start_frame, {}).get(angle_name)
            if val is None:
                # start_frame에 각도 없으면 인접 프레임 탐색
                frame_keys = sorted(angles_per_frame.keys())
                closest = min(
                    frame_keys, key=lambda k: abs(k - start_frame), default=None
                )  # noqa: B023
                if closest is not None:
                    val = angles_per_frame[closest].get(angle_name)
            if val is not None:
                angles[angle_name] = val

    if not angles:
        logger.warning("  ⚠️  각도 추출 결과 없음 (영상 품질 또는 포즈 감지 실패)")
        return None, None
    return angles, landmarks_data


def save_raw_landmarks(
    url: str,
    landmarks_data: Dict,
    sport: str,
    sub: str,
    save_dir: str,
    model_type: str = "heavy",
) -> None:
    """
    3프레임 간격으로 샘플링한 랜드마크를 JSON으로 저장.

    파일 경로: {save_dir}/{SPORT}_{SUB}/{YYYYMMDD}_{video_id}.json
    """
    import re

    # URL에서 video ID 추출 (YouTube watch?v=XXXX 또는 youtu.be/XXXX)
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    video_id = match.group(1) if match else "unknown"

    today = date.today().strftime("%Y%m%d")
    subdir = os.path.join(save_dir, f"{sport}_{sub}")
    os.makedirs(subdir, exist_ok=True)
    output_path = os.path.join(subdir, f"{today}_{video_id}.json")

    # 3프레임 간격 샘플링
    all_frames = landmarks_data.get("frames", [])
    sampled = [
        frame
        for i, frame in enumerate(all_frames)
        if i % _LANDMARK_SAMPLE_INTERVAL == 0
    ]

    payload = {
        "url": url,
        "sport": sport,
        "sub": sub,
        "model_type": model_type,
        "fps": landmarks_data.get("fps", 0),
        "total_frames": landmarks_data.get("total_frames", 0),
        "sampled_frames": len(sampled),
        "sample_interval": _LANDMARK_SAMPLE_INTERVAL,
        "collected_at": today,
        "frames": sampled,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    size_kb = os.path.getsize(output_path) / 1024
    logger.info(
        f"  💾 랜드마크 저장: {output_path} ({size_kb:.0f}KB, {len(sampled)}프레임)"
    )


def _flatten_angle_config(angle_config_raw: Dict) -> Dict:
    """
    v2 sports_config의 levels 구조를 AngleCalculator용 flat 구조로 변환.

    캘리브레이션 목적 — 점수 계산이 아닌 각도 추출이므로
    levels의 첫 번째 레벨 값을 임시로 사용.
    """
    flat = {}
    for name, data in angle_config_raw.items():
        entry = dict(data)
        if "levels" in entry and entry["levels"]:
            first_level = next(iter(entry["levels"].values()))
            entry["ideal_range"] = first_level.get("ideal_range", [0, 180])
            entry["weight"] = first_level.get("weight", 0.1)
            del entry["levels"]
        flat[name] = entry
    return flat


def _get_mediapipe_reference(
    angle_config_raw: Dict, angle_name: str
) -> Optional[float]:
    """
    MediaPipe 좌표계 기준 참조값(mediapipe_center) 반환.

    sanity check에 사용. 논문 raw 값(original_value)이 아닌
    MediaPipe 공간으로 변환된 값과 비교해야 의미 있음.
    """
    ds = angle_config_raw.get(angle_name, {}).get("data_source", {})
    return ds.get("mediapipe_center")


# ──────────────────────────────────────────────────────────────
# 통계 계산
# ──────────────────────────────────────────────────────────────


def compute_statistics(
    all_angles: List[Dict[str, float]],
) -> Dict[str, Dict]:
    """
    여러 영상에서 수집한 각도 데이터의 mean / std / n 계산.

    Returns:
        {"left_arm_angle": {"mean": 168.3, "std": 4.2, "n": 5}, ...}
    """
    collections: Dict[str, List[float]] = {}
    for angles in all_angles:
        for name, value in angles.items():
            collections.setdefault(name, []).append(value)

    return {
        name: {
            "mean": round(float(np.mean(values)), 1),
            "std": round(float(np.std(values)), 1),
            "n": len(values),
        }
        for name, values in collections.items()
    }


# ──────────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────────


def check_sanity(
    angle_name: str,
    measured_mean: float,
    paper_value: Optional[float],
) -> Tuple[bool, str]:
    """
    실측값과 논문값의 차이가 SANITY_THRESHOLD를 초과하면 경고.

    Returns:
        (is_ok, message)
    """
    if paper_value is None:
        return True, ""

    delta = abs(measured_mean - paper_value)
    if delta > SANITY_THRESHOLD:
        return False, (
            f"  ⚠️  SANITY FAIL [{angle_name}]: "
            f"실측 {measured_mean}° vs 논문 {paper_value}° "
            f"(차이 {delta:.1f}° > {SANITY_THRESHOLD}°) — 영상 품질 또는 각도 정의 확인 필요"
        )
    return True, ""


# ──────────────────────────────────────────────────────────────
# CSV 파이프라인 (R-076)
# ──────────────────────────────────────────────────────────────


def _find_csv(sport: str, sub: str, config_repo: str) -> Optional[str]:
    """v2.0 CSV 경로 반환. 파일이 없으면 None."""
    sport_lower = sport.lower()
    sub_lower = sub.lower()
    csv_path = (
        Path(config_repo)
        / "calibration_data"
        / "v2.0"
        / sport_lower
        / f"{sport_lower}_{sub_lower}_standards.csv"
    )
    return str(csv_path) if csv_path.exists() else None


def _update_csv_measured(
    csv_path: str,
    stats: Dict[str, Dict],
    dry_run: bool,
) -> int:
    """
    CSV의 measured_* 컬럼을 실측값으로 업데이트.

    Returns:
        업데이트된 각도 수
    """
    today = date.today().isoformat()
    measured_cols = ["measured_value", "measured_std", "measured_n", "measured_at"]

    rows: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        fieldnames: List[str] = list(reader.fieldnames or [])
        rows = list(reader)

    # measured 컬럼이 없으면 헤더에 추가
    for col in measured_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    updated = 0
    for row in rows:
        angle_name = row["angle_name"]
        stat = stats.get(angle_name)
        if stat is None:
            continue
        if stat["n"] < MIN_SAMPLES:
            logger.warning(
                f"  ⏭️  {angle_name}: 샘플 부족 (n={stat['n']} < {MIN_SAMPLES}) → CSV 미갱신"
            )
            continue
        row["measured_value"] = f"{stat['mean']:.2f}"
        row["measured_std"] = f"{stat['std']:.2f}"
        row["measured_n"] = str(stat["n"])
        row["measured_at"] = today
        updated += 1
        logger.info(
            f"  ✅ {angle_name}: {stat['mean']:.1f}° ± {stat['std']:.1f}° (n={stat['n']})"
        )

    if dry_run:
        logger.info(f"[dry-run] CSV {updated}개 행 업데이트 예정 — 파일 저장 안 함")
        return updated

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"  💾 CSV 저장: {csv_path} ({updated}개 각도 갱신)")
    return updated


def _run_csv_to_config(
    config_repo: str,
    ai_config_path: str,
    dry_run: bool,
) -> None:
    """
    csv_to_config.py 실행 후 output/latest/sports_config.json을
    motionlab-ai/core/sport_configs/sports_config.json으로 복사.
    """
    csv_to_config_script = Path(config_repo) / "scripts" / "csv_to_config.py"
    output_latest = Path(config_repo) / "output" / "latest" / "sports_config.json"

    if dry_run:
        logger.info(
            f"[dry-run] csv_to_config.py 실행 예정 → {ai_config_path} 복사 예정"
        )
        return

    logger.info("🔄 csv_to_config.py 실행 중...")
    result = subprocess.run(
        [
            sys.executable,
            str(csv_to_config_script),
            "--version",
            "v2.0",
            "--bump",
            "patch",
            "--changelog",
            "collect_calibration.py 실측값 자동 업데이트",
            "--storage",
            "local",
        ],
        cwd=config_repo,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"csv_to_config.py 실패:\n{result.stderr}")
        return

    logger.info("✅ csv_to_config.py 완료")

    if not output_latest.exists():
        logger.error(f"출력 파일 없음: {output_latest}")
        return

    shutil.copy2(str(output_latest), ai_config_path)
    logger.info(f"✅ sports_config.json 복사: {output_latest} → {ai_config_path}")


# sports_config.json 직접 업데이트 (레거시 — v2.0 CSV 없는 종목용)
# ──────────────────────────────────────────────────────────────


def _bump_config_version(current: str) -> str:
    """
    2.x.x → 3.0.0 으로 bump (최초 실측값 적용).
    이미 3.x.x 이상이면 patch 번호만 +1.
    """
    parts = current.split(".")
    if len(parts) == 3 and parts[0] == "2":
        return "3.0.0"
    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{major}.{minor}.{patch + 1}"
    except (ValueError, IndexError):
        return "3.0.0"


def update_sports_config(
    config_path: str,
    sport: str,
    sub: str,
    stats: Dict[str, Dict],
    angle_config_raw: Dict,
    source_file: str,
    phase_map: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> None:
    """
    sports_config.json의 data_source 값을 실측값으로 교체.

    변경 전:
        "data_source": {"conversion": "direct", "mediapipe_center": 172.5, "std": 5.0}

    변경 후:
        "data_source": {
            "conversion": "mediapipe_measured",
            "mediapipe_center": 168.3, "std": 4.2, "sample_count": 12,
            "measured_at": "2026-03-21", "source_file": "urls/golf_driver.md",
            "paper_reference": 172.5, "paper_delta": -4.2
        }

    fallback 규칙:
        sample_count < MIN_SAMPLES → 논문값 유지, 이유 기록
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    angles = config["sports"][sport]["sub_categories"][sub]["angles"]
    today = date.today().isoformat()
    source_name = Path(source_file).name

    updated = 0
    fallback_count = 0
    sanity_failures = []

    logger.info("")
    logger.info("─" * 70)
    logger.info(
        f"  {'각도명':<28} {'논문값':>8}  {'실측값':>8}  {'std':>6}  {'n':>4}  상태"
    )
    logger.info("─" * 70)

    for angle_name, stat in stats.items():
        if angle_name not in angles:
            logger.warning(f"  ⚠️  '{angle_name}' config에 없음 — 스킵")
            continue

        paper_value = _get_mediapipe_reference(angle_config_raw, angle_name)
        ds = angles[angle_name].setdefault("data_source", {})

        # ── 샘플 부족 fallback
        if stat["n"] < MIN_SAMPLES:
            ds["fallback"] = "insufficient_samples"
            ds["fallback_reason"] = (
                f"sample_count={stat['n']} < MIN_SAMPLES={MIN_SAMPLES}"
            )
            status = f"⏭️  fallback (n={stat['n']} < {MIN_SAMPLES})"
            fallback_count += 1
            logger.info(
                f"  {angle_name:<28} {str(paper_value or '?'):>8}  "
                f"{'—':>8}  {'—':>6}  {stat['n']:>4}  {status}"
            )
            continue

        # ── Sanity check
        is_ok, warn_msg = check_sanity(angle_name, stat["mean"], paper_value)
        if not is_ok:
            sanity_failures.append(warn_msg)
            status = "⚠️  sanity fail"
        else:
            status = "✅"

        # ── 업데이트
        paper_delta = (
            round(stat["mean"] - paper_value, 1) if paper_value is not None else None
        )

        update_fields: Dict = {
            "conversion": "mediapipe_measured",
            "mediapipe_center": stat["mean"],
            "std": stat["std"],
            "sample_count": stat["n"],
            "measured_at": today,
            "source_file": source_name,
            "paper_reference": paper_value,
            "paper_delta": paper_delta,
        }
        if phase_map and angle_name in phase_map:
            update_fields["measurement_phase"] = phase_map[angle_name]

        ds.update(update_fields)
        # fallback 키 제거 (이전 실행에서 남아있을 수 있음)
        ds.pop("fallback", None)
        ds.pop("fallback_reason", None)

        logger.info(
            f"  {angle_name:<28} {str(paper_value or '?'):>8}  "
            f"{stat['mean']:>8.1f}  {stat['std']:>6.1f}  {stat['n']:>4}  {status}"
        )
        updated += 1

    logger.info("─" * 70)

    # ── Sanity 경고 출력
    if sanity_failures:
        logger.warning("\n[Sanity Check 실패 목록]")
        for msg in sanity_failures:
            logger.warning(msg)
        logger.warning(
            "\n  → 영상 촬영 각도, 화질, 종목 정의를 재확인한 후 재측정을 권장합니다."
        )

    if fallback_count:
        logger.info(f"\n  📌 {fallback_count}개 각도 논문값 유지 (샘플 부족)")

    if dry_run:
        logger.info(
            f"\n[dry-run] {updated}개 업데이트 예정, {fallback_count}개 fallback — 파일 저장 안 함"
        )
        return

    # ── config_version bump
    old_version = config.get("meta", {}).get("config_version", "2.0.0")
    new_version = _bump_config_version(old_version)
    config.setdefault("meta", {})["config_version"] = new_version
    config["meta"]["generated_at"] = today

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(
        f"\n✅ {config_path} 업데이트 완료 "
        f"({updated}개 각도, version {old_version} → {new_version})"
    )


# ──────────────────────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YouTube URL → MediaPipe 각도 측정 → CSV 저장 → sports_config.json 생성"
    )
    parser.add_argument(
        "--url-file",
        required=True,
        help="URL 목록 파일 경로 (예: ../motionlab-config/calibration_data/urls/golf_driver.md)",
    )
    parser.add_argument("--sport", required=True, help="종목 대문자 (예: GOLF)")
    parser.add_argument("--sub", required=True, help="서브카테고리 대문자 (예: DRIVER)")
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).parent.parent / "core/sport_configs/sports_config.json"
        ),
        help="수정할 sports_config.json 경로 (기본: core/sport_configs/sports_config.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일을 저장하지 않고 결과만 출력",
    )
    parser.add_argument(
        "--landmarks-dir",
        default=None,
        help=(
            "Raw 랜드마크 JSON 저장 디렉토리 "
            "(예: ../motionlab-config/calibration_data/raw_landmarks). "
            "지정하지 않으면 저장 안 함."
        ),
    )
    parser.add_argument(
        "--config-repo",
        default=str(Path(__file__).parent.parent.parent / "motionlab-config"),
        help="motionlab-config 레포 경로 (기본: ../motionlab-config)",
    )
    args = parser.parse_args()

    # ── URL 읽기
    urls = read_urls_from_file(args.url_file)
    if not urls:
        logger.error("처리할 URL이 없습니다. 파일을 확인해 주세요.")
        sys.exit(1)

    logger.info(f"📋 {len(urls)}개 URL  |  종목: {args.sport}/{args.sub}")

    # ── 각도 config 로드
    raw_config = load_sports_config()
    try:
        sub_config = raw_config[args.sport]["sub_categories"][args.sub]
        angle_config_raw = sub_config["angles"]
    except KeyError:
        logger.error(f"종목을 찾을 수 없습니다: {args.sport}/{args.sub}")
        sys.exit(1)

    angle_config_flat = _flatten_angle_config(angle_config_raw)

    # 페이즈 config (PhaseDetector 입력) 및 각도별 페이즈 맵
    phase_config: Optional[List] = sub_config.get("phases") or None
    phase_map: Optional[Dict[str, str]] = {
        name: data["phase"]
        for name, data in angle_config_raw.items()
        if "phase" in data
    } or None

    if phase_config and phase_map:
        logger.info(f"📍 페이즈 맵: { {k: v for k, v in phase_map.items()} }")
    else:
        logger.info("📍 페이즈 config 없음 → 전체 평균 모드")
    logger.info(f"📐 측정 각도: {list(angle_config_flat.keys())}\n")

    # ── 영상별 처리 (임시 디렉토리 자동 정리)
    all_results: List[Dict[str, float]] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, url in enumerate(urls, 1):
            logger.info(f"[{i}/{len(urls)}] {url}")
            video_path = os.path.join(tmp_dir, f"video_{i}.mp4")

            if not download_video(url, video_path):
                logger.warning("  ⏭️  스킵\n")
                continue

            logger.info("  📥 다운로드 완료")
            angles, landmarks_data = extract_angles_from_video(
                video_path, angle_config_flat, phase_config, phase_map
            )

            if angles:
                logger.info(f"  📐 각도: { {k: f'{v}°' for k, v in angles.items()} }")
                all_results.append(angles)

            if landmarks_data and args.landmarks_dir:
                model_type = "heavy" if Path(POSE_MODEL_PATH).exists() else "full"
                save_raw_landmarks(
                    url,
                    landmarks_data,
                    args.sport,
                    args.sub,
                    args.landmarks_dir,
                    model_type,
                )

            logger.info("")

    if not all_results:
        logger.error("처리된 영상이 없습니다. URL 또는 영상 품질을 확인해 주세요.")
        sys.exit(1)

    # ── 통계 계산 및 출력
    logger.info(f"📊 통계 계산 — {len(all_results)}개 영상 기준")
    stats = compute_statistics(all_results)

    # ── CSV 파이프라인 (R-076)
    csv_path = _find_csv(args.sport, args.sub, args.config_repo)
    if csv_path:
        logger.info(f"\n📄 CSV 파이프라인 모드: {csv_path}")
        updated = _update_csv_measured(csv_path, stats, args.dry_run)
        if updated > 0:
            _run_csv_to_config(args.config_repo, args.output, args.dry_run)
        else:
            logger.warning("업데이트된 각도가 없어 csv_to_config.py 실행 건너뜀")
    else:
        logger.warning(
            f"⚠️  v2.0 CSV 없음 ({args.sport}/{args.sub}) "
            f"→ sports_config.json 직접 업데이트 (레거시 모드)\n"
            f"   CSV 생성 후 다시 실행하면 CSV 파이프라인으로 전환됩니다."
        )
        update_sports_config(
            args.output,
            args.sport,
            args.sub,
            stats,
            angle_config_raw,
            args.url_file,
            phase_map,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
