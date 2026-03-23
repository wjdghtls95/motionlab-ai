#!/usr/bin/env python3
"""
Full vs Heavy 모델 각도 비교 테스트 (일회성 검증 스크립트)

동일 영상에서 mp.solutions.pose Full 모델과 pose_landmarker_heavy.task를
각각 실행하여 주요 각도(left_arm_angle, spine_angle, right_knee_angle)를 비교.

사용법:
    cd motionlab-ai
    source venv/bin/activate
    python scripts/compare_models.py <video_url_or_path>

    # URL 직접 지정
    python scripts/compare_models.py "https://www.youtube.com/watch?v=JfKSVRV1vTA"
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from core.angle_calculator import AngleCalculator
from core.mediapipe_analyzer import MediaPipeAnalyzer
from core.sport_configs import load_sports_config
from scripts.collect_calibration import (
    _HeavyPoseAnalyzer,
    _flatten_angle_config,
    detect_swing_phases,
    download_video,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 비교할 각도
COMPARE_ANGLES = ["left_arm_angle", "spine_angle", "right_knee_angle"]
DIFF_THRESHOLD_MINOR = 3.0  # 이하: 유의미한 차이 없음
DIFF_THRESHOLD_MAJOR = 5.0  # 이상: Heavy 기준 전체 재수집 필요


def extract_phase_angles(landmarks_data: dict, angle_config_flat: dict) -> dict:
    """페이즈별 각도 추출 (top_of_backswing: left_arm, address: spine/knee)."""
    frames = landmarks_data["frames"]
    fps = landmarks_data.get("fps", 30.0)
    phases = detect_swing_phases(frames, fps)
    calc = AngleCalculator(angle_config=angle_config_flat)

    phase_map = {
        "left_arm_angle": "top_of_backswing",
        "spine_angle": "address",
        "right_knee_angle": "address",
    }

    angles = {}
    for angle_name, phase_name in phase_map.items():
        frame_pos = phases.get(phase_name)
        if frame_pos is not None and frame_pos < len(frames):
            single = {
                "frames": [frames[frame_pos]],
                "total_frames": 1,
                "valid_frames": 1,
                "fps": fps,
            }
            val = (
                calc.calculate_angles(single).get("average_angles", {}).get(angle_name)
            )
        else:
            val = (
                calc.calculate_angles(landmarks_data)
                .get("average_angles", {})
                .get(angle_name)
            )
        if val is not None:
            angles[angle_name] = val

    return angles


def main():
    parser = argparse.ArgumentParser(description="Full vs Heavy 모델 각도 비교")
    parser.add_argument("source", help="YouTube URL 또는 로컬 영상 경로")
    args = parser.parse_args()

    # 각도 설정 로드
    raw_config = load_sports_config()
    angle_config_raw = raw_config["GOLF"]["sub_categories"]["DRIVER"]["angles"]
    angle_config_flat = _flatten_angle_config(angle_config_raw)

    # 영상 확보 (URL이면 다운로드, 로컬 파일이면 그대로)
    source = args.source
    tmp_ctx = None

    if source.startswith("http"):
        tmp_ctx = tempfile.TemporaryDirectory()
        video_path = str(Path(tmp_ctx.name) / "video.mp4")
        logger.info(f"📥 영상 다운로드 중: {source}")
        if not download_video(source, video_path):
            logger.error("다운로드 실패")
            sys.exit(1)
        logger.info("✅ 다운로드 완료")
    else:
        video_path = source

    try:
        # ── Full 모델 (legacy mp.solutions.pose)
        logger.info("\n" + "=" * 60)
        logger.info("🔵 Full 모델 (mp.solutions.pose, model_complexity=1)")
        logger.info("=" * 60)
        full_analyzer = MediaPipeAnalyzer()
        full_data = full_analyzer.extract_landmarks(video_path)
        full_angles = extract_phase_angles(full_data, angle_config_flat)

        # ── Heavy 모델 (Tasks API)
        logger.info("\n" + "=" * 60)
        logger.info("🟠 Heavy 모델 (pose_landmarker_heavy.task)")
        logger.info("=" * 60)
        heavy_analyzer = _HeavyPoseAnalyzer()
        heavy_data = heavy_analyzer.extract_landmarks(video_path)
        heavy_angles = extract_phase_angles(heavy_data, angle_config_flat)

        # ── 결과 비교
        print("\n" + "=" * 70)
        print(f"  {'각도명':<28} {'Full':>8}  {'Heavy':>8}  {'차이':>7}  판정")
        print("=" * 70)

        max_diff = 0.0
        for angle_name in COMPARE_ANGLES:
            full_val = full_angles.get(angle_name)
            heavy_val = heavy_angles.get(angle_name)

            if full_val is None or heavy_val is None:
                verdict = "⚠️  측정값 없음"
                print(f"  {angle_name:<28} {'?':>8}  {'?':>8}  {'?':>7}  {verdict}")
                continue

            diff = abs(heavy_val - full_val)
            max_diff = max(max_diff, diff)

            if diff <= DIFF_THRESHOLD_MINOR:
                verdict = "✅ 차이 없음"
            elif diff < DIFF_THRESHOLD_MAJOR:
                verdict = "🟡 소폭 차이"
            else:
                verdict = "🔴 큰 차이"

            print(
                f"  {angle_name:<28} {full_val:>8.1f}°  {heavy_val:>8.1f}°  "
                f"{diff:>6.1f}°  {verdict}"
            )

        print("=" * 70)

        if max_diff <= DIFF_THRESHOLD_MINOR:
            conclusion = "✅ 유의미한 차이 없음 — Heavy 기본 설정 유지, 기존 데이터 그대로 사용 가능"
        elif max_diff < DIFF_THRESHOLD_MAJOR:
            conclusion = f"🟡 소폭 차이 ({max_diff:.1f}°) — Heavy 기준으로 재수집 권장"
        else:
            conclusion = (
                f"🔴 큰 차이 ({max_diff:.1f}°) — Heavy 기준으로 전체 재수집 필요"
            )

        print(f"\n결론: {conclusion}\n")

    finally:
        if tmp_ctx:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
