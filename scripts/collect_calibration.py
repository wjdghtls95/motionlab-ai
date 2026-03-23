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
import json
import logging
import os
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

# MediaPipe 랜드마크 인덱스 (33개 pose landmarks)
_IDX_LEFT_SHOULDER = 11
_IDX_RIGHT_SHOULDER = 12
_IDX_LEFT_WRIST = 15
_IDX_RIGHT_WRIST = 16
_IDX_LEFT_HIP = 23
_IDX_RIGHT_HIP = 24
_WRIST_INDICES = [_IDX_LEFT_WRIST, _IDX_RIGHT_WRIST]

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
# 스윙 페이즈 감지
# ──────────────────────────────────────────────────────────────


def _hip_shoulder_separation_frame(landmarks: List[Dict]) -> Optional[float]:
    """단일 프레임의 hip-shoulder separation 각도 계산 (X-Factor)."""
    required = [_IDX_LEFT_SHOULDER, _IDX_RIGHT_SHOULDER, _IDX_LEFT_HIP, _IDX_RIGHT_HIP]
    if len(landmarks) <= max(required):
        return None
    if any(landmarks[i].get("visibility", 0) < 0.5 for i in required):
        return None

    ls = landmarks[_IDX_LEFT_SHOULDER]
    rs = landmarks[_IDX_RIGHT_SHOULDER]
    lh = landmarks[_IDX_LEFT_HIP]
    rh = landmarks[_IDX_RIGHT_HIP]

    shoulder_angle = np.degrees(np.arctan2(rs["y"] - ls["y"], rs["x"] - ls["x"]))
    hip_angle = np.degrees(np.arctan2(rh["y"] - lh["y"], rh["x"] - lh["x"]))
    diff = abs(shoulder_angle - hip_angle)
    # arctan2 범위(-180~180) 차이가 180° 초과 시 보완각으로 변환
    return round(min(diff, 360.0 - diff), 1)


def _calc_landmark_motion(
    frames: List[Dict], center_pos: int, window_size: int, indices: List[int]
) -> float:
    """윈도우 내 특정 랜드마크의 평균 이동 거리 계산 (어드레스 감지용)."""
    half = window_size // 2
    start = max(0, center_pos - half)
    end = min(len(frames) - 1, center_pos + half)

    total_motion = 0.0
    count = 0
    for i in range(start, end):
        prev_lm = frames[i].get("landmarks", [])
        curr_lm = frames[i + 1].get("landmarks", [])
        for idx in indices:
            if (
                idx < len(prev_lm)
                and idx < len(curr_lm)
                and prev_lm[idx].get("visibility", 0) > 0.5
                and curr_lm[idx].get("visibility", 0) > 0.5
            ):
                dx = curr_lm[idx]["x"] - prev_lm[idx]["x"]
                dy = curr_lm[idx]["y"] - prev_lm[idx]["y"]
                total_motion += np.sqrt(dx**2 + dy**2)
                count += 1

    return total_motion / count if count > 0 else float("inf")


def detect_swing_phases(frames: List[Dict], fps: float) -> Dict[str, Optional[int]]:
    """
    스윙 영상에서 핵심 페이즈 프레임 위치를 감지.

    - address: 영상 초반 최소 모션 구간 (셋업 자세)
    - top_of_backswing: hip-shoulder separation 최대 프레임

    Returns:
        {"address": frames_list_index, "top_of_backswing": frames_list_index}
        감지 실패 시 해당 값 None
    """
    n = len(frames)
    if n < 10:
        return {"address": None, "top_of_backswing": None}

    window_size = max(int(fps * 0.5), 10)  # 0.5초 윈도우, 최소 10프레임
    first_30_pct = max(int(n * 0.30), window_size + 1)

    # ── 어드레스: 초반 30% 구간에서 손목 모션 최소 지점
    address_pos: Optional[int] = None
    min_motion = float("inf")
    step = max(1, window_size // 4)

    for center in range(window_size // 2, first_30_pct - window_size // 2, step):
        motion = _calc_landmark_motion(frames, center, window_size, _WRIST_INDICES)
        if motion < min_motion:
            min_motion = motion
            address_pos = center

    if address_pos is None:
        address_pos = min(int(n * 0.05), n - 1)

    # ── 백스윙 탑: 어드레스 이후 50% 구간에서 X-Factor 최대 프레임
    top_pos: Optional[int] = None
    max_sep = -1.0
    search_start = address_pos + window_size
    search_end = min(n, search_start + int(n * 0.5))

    for i in range(search_start, search_end):
        sep = _hip_shoulder_separation_frame(frames[i].get("landmarks", []))
        if sep is not None and sep > max_sep:
            max_sep = sep
            top_pos = i

    logger.info(
        f"  📍 페이즈 감지: address=frame[{address_pos}], "
        f"top_of_backswing=frame[{top_pos}] (X-Factor {max_sep:.1f}°)"
    )
    return {"address": address_pos, "top_of_backswing": top_pos}


# ──────────────────────────────────────────────────────────────
# 각도 추출
# ──────────────────────────────────────────────────────────────


def extract_angles_from_video(
    video_path: str,
    angle_config_flat: Dict,
    phase_map: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, float]]:
    """
    영상에서 관절 각도를 추출.

    phase_map이 있으면 각 각도를 지정된 스윙 페이즈 프레임에서 추출.
    phase_map이 없으면 전체 프레임 평균 사용 (기존 동작).

    Args:
        phase_map: {"left_arm_angle": "top_of_backswing", "spine_angle": "address", ...}

    Returns:
        {"left_arm_angle": 168.3, ...} 또는 None (추출 실패 시)
    """
    analyzer = _create_pose_analyzer()
    landmarks_data = analyzer.extract_landmarks(video_path)
    calc = AngleCalculator(angle_config=angle_config_flat)

    # ── 페이즈 감지 없음: 전체 평균 (기존 동작)
    if not phase_map:
        result = calc.calculate_angles(landmarks_data)
        average_angles = result.get("average_angles", {})
        if not average_angles:
            logger.warning("  ⚠️  각도 추출 결과 없음 (영상 품질 또는 포즈 감지 실패)")
            return None
        return average_angles

    # ── 페이즈별 추출
    frames = landmarks_data["frames"]
    fps = landmarks_data.get("fps", 30.0)
    phases = detect_swing_phases(frames, fps)

    angles: Dict[str, float] = {}
    for angle_name, phase_name in phase_map.items():
        frame_pos = phases.get(phase_name)
        if frame_pos is None or frame_pos >= len(frames):
            # 페이즈 감지 실패 → 전체 평균 fallback
            full_result = calc.calculate_angles(landmarks_data)
            val = full_result.get("average_angles", {}).get(angle_name)
        else:
            # 해당 페이즈 단일 프레임에서 각도 계산
            single = {
                "frames": [frames[frame_pos]],
                "total_frames": 1,
                "valid_frames": 1,
                "fps": fps,
            }
            val = (
                calc.calculate_angles(single).get("average_angles", {}).get(angle_name)
            )

        if val is not None:
            angles[angle_name] = val

    if not angles:
        logger.warning("  ⚠️  각도 추출 결과 없음 (영상 품질 또는 포즈 감지 실패)")
        return None
    return angles


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
# sports_config.json 업데이트
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
        description="YouTube URL → MediaPipe 각도 측정 → sports_config.json 업데이트"
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
        angle_config_raw = raw_config[args.sport]["sub_categories"][args.sub]["angles"]
    except KeyError:
        logger.error(f"종목을 찾을 수 없습니다: {args.sport}/{args.sub}")
        sys.exit(1)

    angle_config_flat = _flatten_angle_config(angle_config_raw)

    # 각도별 페이즈 맵 구성 (sports_config.json의 "phase" 필드)
    phase_map = {
        name: data["phase"]
        for name, data in angle_config_raw.items()
        if "phase" in data
    }
    if phase_map:
        logger.info(f"📍 페이즈 맵: { {k: v for k, v in phase_map.items()} }")
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
            angles = extract_angles_from_video(
                video_path, angle_config_flat, phase_map or None
            )

            if angles:
                logger.info(f"  📐 각도: { {k: f'{v}°' for k, v in angles.items()} }")
                all_results.append(angles)
            logger.info("")

    if not all_results:
        logger.error("처리된 영상이 없습니다. URL 또는 영상 품질을 확인해 주세요.")
        sys.exit(1)

    # ── 통계 계산 및 출력
    logger.info(f"📊 통계 계산 — {len(all_results)}개 영상 기준")
    stats = compute_statistics(all_results)

    # ── sports_config.json 업데이트
    update_sports_config(
        args.output,
        args.sport,
        args.sub,
        stats,
        angle_config_raw,
        args.url_file,
        phase_map or None,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
