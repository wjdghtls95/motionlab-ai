#!/usr/bin/env python3
"""
Full vs Heavy 모델 각도 비교 테스트 (일회성 검증 스크립트)

동일 영상에서 mp.solutions.pose Full 모델과 pose_landmarker_heavy.task를
각각 독립적으로 실행:
  1. PhaseDetector를 각 모델 결과에 독립 실행 → 페이즈 감지 프레임 차이 측정
  2. Full 모델 기준 프레임에서 양 모델의 각도 차이 측정

두 가지 오차를 분리 측정:
  - 페이즈 감지 오차: Full vs Heavy의 페이즈 시작 프레임 차이 (프레임 단위)
  - 각도 계산 오차: 동일 프레임에서의 각도 값 차이 (도 단위)

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
from typing import Dict, List, Optional, Tuple, Any


sys.path.insert(0, str(Path(__file__).parent.parent))

from core.angle_calculator import AngleCalculator
from core.mediapipe_analyzer import MediaPipeAnalyzer
from core.phase_detector import PhaseDetector
from core.sport_configs import load_sports_config
from scripts.collect_calibration import (
    _HeavyPoseAnalyzer,
    _build_angles_per_frame,
    _flatten_angle_config,
    _to_phase_detector_input,
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
FRAME_DIFF_MINOR = 3  # 이하: 페이즈 감지 오차 허용 범위 (프레임)
FRAME_DIFF_MAJOR = 8  # 이상: 페이즈 감지 신뢰도 낮음


def run_pipeline(
    landmarks_data: Dict[str, Any],
    angle_config_flat: Dict,
    phase_config: Optional[List],
    phase_map: Optional[Dict[str, str]],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, int]]:
    """
    랜드마크 데이터 → (angles_per_frame, phase_start_frames) 반환.

    Parameters
    ----------
    landmarks_data : extract_landmarks() 반환값
    angle_config_flat : _flatten_angle_config() 결과
    phase_config : sports_config phases 리스트 (PhaseDetector 입력)
    phase_map : {"angle_name": "phase_name", ...}

    Returns
    -------
    angles_per_frame : {frame_idx: {angle_name: value}}
    phase_start_frames : {phase_name: start_frame}
    """
    frames = landmarks_data["frames"]
    fps = landmarks_data.get("fps", 30.0)
    calc = AngleCalculator(angle_config=angle_config_flat)

    angles_per_frame = _build_angles_per_frame(frames, calc)
    if not angles_per_frame:
        return {}, {}

    if not phase_config:
        return angles_per_frame, {}

    phase_input = _to_phase_detector_input(
        angles_per_frame, list(angle_config_flat.keys())
    )
    detector = PhaseDetector(phase_config, fps=int(fps))
    detected_phases = detector.detect_phases(phase_input)

    phase_start_frames = {p["name"]: p["start_frame"] for p in detected_phases}
    return angles_per_frame, phase_start_frames


def get_angle_at_frame(
    angles_per_frame: Dict[str, Dict[int, float]],
    angle_name: str,
    target_frame: int,
) -> Optional[float]:
    """
    target_frame에서 angle_name 값 추출.
    해당 프레임에 값이 없으면 가장 가까운 프레임 값 사용.
    """
    if not angles_per_frame:
        return None
    frame_val = angles_per_frame.get(target_frame, {}).get(angle_name)
    if frame_val is not None:
        return frame_val

    frame_keys = sorted(angles_per_frame.keys())
    if not frame_keys:
        return None
    closest = min(frame_keys, key=lambda k: abs(k - target_frame))
    return angles_per_frame[closest].get(angle_name)


def main():
    parser = argparse.ArgumentParser(description="Full vs Heavy 모델 각도 비교")
    parser.add_argument("source", help="YouTube URL 또는 로컬 영상 경로")
    args = parser.parse_args()

    # 각도 설정 로드
    raw_config = load_sports_config()
    sub_config = raw_config["GOLF"]["sub_categories"]["DRIVER"]
    angle_config_raw = sub_config["angles"]
    angle_config_flat = _flatten_angle_config(angle_config_raw)

    # PhaseDetector 설정
    phase_config: Optional[List] = sub_config.get("phases") or None
    phase_map: Optional[Dict[str, str]] = {
        name: data["phase"]
        for name, data in angle_config_raw.items()
        if "phase" in data
    } or None

    if phase_config:
        logger.info(f"📍 PhaseDetector 사용 (페이즈 수: {len(phase_config)})")
        logger.info(f"📍 페이즈 맵: {phase_map}")
    else:
        logger.warning("⚠️  phase_config 없음 → 페이즈 감지 비교 불가")

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
        full_angles_per_frame, full_phase_starts = run_pipeline(
            full_data, angle_config_flat, phase_config, phase_map
        )
        logger.info(f"   페이즈 감지 결과: {full_phase_starts}")

        # ── Heavy 모델 (Tasks API)
        logger.info("\n" + "=" * 60)
        logger.info("🟠 Heavy 모델 (pose_landmarker_heavy.task)")
        logger.info("=" * 60)
        heavy_analyzer = _HeavyPoseAnalyzer()
        heavy_data = heavy_analyzer.extract_landmarks(video_path)
        heavy_angles_per_frame, heavy_phase_starts = run_pipeline(
            heavy_data, angle_config_flat, phase_config, phase_map
        )
        logger.info(f"   페이즈 감지 결과: {heavy_phase_starts}")

        fps = full_data.get("fps", 30.0)

        # ══════════════════════════════════════════════════════════════════════
        # 결과 1: 페이즈 감지 프레임 정렬 비교
        # ══════════════════════════════════════════════════════════════════════
        if phase_config and (full_phase_starts or heavy_phase_starts):
            all_phase_names = sorted(
                set(full_phase_starts.keys()) | set(heavy_phase_starts.keys())
            )

            print("\n" + "=" * 72)
            print("  [1] 페이즈 감지 프레임 정렬 비교")
            print("=" * 72)
            print(
                f"  {'페이즈':<22} {'Full 프레임':>12}  {'Heavy 프레임':>12}  {'차이(프레임)':>12}  {'차이(ms)':>9}  판정"
            )
            print("-" * 72)

            max_frame_diff = 0
            for phase_name in all_phase_names:
                f_frame = full_phase_starts.get(phase_name)
                h_frame = heavy_phase_starts.get(phase_name)

                if f_frame is None or h_frame is None:
                    verdict = "⚠️  한쪽 미감지"
                    print(
                        f"  {phase_name:<22} {str(f_frame or '?'):>12}  {str(h_frame or '?'):>12}  "
                        f"{'?':>12}  {'?':>9}  {verdict}"
                    )
                    continue

                frame_diff = abs(f_frame - h_frame)
                ms_diff = int(frame_diff / fps * 1000)
                max_frame_diff = max(max_frame_diff, frame_diff)

                if frame_diff <= FRAME_DIFF_MINOR:
                    verdict = "✅ 정렬 양호"
                elif frame_diff < FRAME_DIFF_MAJOR:
                    verdict = "🟡 소폭 차이"
                else:
                    verdict = "🔴 큰 차이"

                print(
                    f"  {phase_name:<22} {f_frame:>12}  {h_frame:>12}  "
                    f"{frame_diff:>12}  {ms_diff:>8}ms  {verdict}"
                )

            print("=" * 72)
            if max_frame_diff <= FRAME_DIFF_MINOR:
                print(
                    f"  페이즈 정렬 결론: ✅ 두 모델의 페이즈 감지 프레임 일치 (최대 차이 {max_frame_diff}프레임)"
                )
            elif max_frame_diff < FRAME_DIFF_MAJOR:
                print(
                    f"  페이즈 정렬 결론: 🟡 소폭 차이 ({max_frame_diff}프레임) — 페이즈 경계 오차 허용 범위 내"
                )
            else:
                print(
                    f"  페이즈 정렬 결론: 🔴 큰 차이 ({max_frame_diff}프레임) — 모델 간 페이즈 감지 신뢰도 낮음"
                )

        # ══════════════════════════════════════════════════════════════════════
        # 결과 2: 동일 기준 프레임에서 각도 차이 비교 (Full 페이즈 기준)
        # ══════════════════════════════════════════════════════════════════════
        print("\n" + "=" * 72)
        print("  [2] 각도 계산 오차 비교 (Full 페이즈 기준 프레임 고정)")
        print("=" * 72)
        print(
            f"  {'각도명':<28} {'기준프레임':>10}  {'Full':>8}  {'Heavy':>8}  {'차이':>7}  판정"
        )
        print("-" * 72)

        max_angle_diff = 0.0
        for angle_name in COMPARE_ANGLES:
            # phase_map에서 이 각도의 참조 페이즈 찾기
            ref_phase = (phase_map or {}).get(angle_name)
            ref_frame = full_phase_starts.get(ref_phase) if ref_phase else None

            # 기준 프레임이 없으면 전체 평균 사용
            if ref_frame is None:
                all_full_vals = [
                    v[angle_name]
                    for v in full_angles_per_frame.values()
                    if angle_name in v
                ]
                all_heavy_vals = [
                    v[angle_name]
                    for v in heavy_angles_per_frame.values()
                    if angle_name in v
                ]
                full_val = (
                    float(sum(all_full_vals) / len(all_full_vals))
                    if all_full_vals
                    else None
                )
                heavy_val = (
                    float(sum(all_heavy_vals) / len(all_heavy_vals))
                    if all_heavy_vals
                    else None
                )
                frame_label = "전체평균"
            else:
                full_val = get_angle_at_frame(
                    full_angles_per_frame, angle_name, ref_frame
                )
                heavy_val = get_angle_at_frame(
                    heavy_angles_per_frame, angle_name, ref_frame
                )
                frame_label = str(ref_frame)

            if full_val is None or heavy_val is None:
                print(
                    f"  {angle_name:<28} {frame_label:>10}  {'?':>8}  {'?':>8}  {'?':>7}  ⚠️  측정값 없음"
                )
                continue

            diff = abs(heavy_val - full_val)
            max_angle_diff = max(max_angle_diff, diff)

            if diff <= DIFF_THRESHOLD_MINOR:
                verdict = "✅ 차이 없음"
            elif diff < DIFF_THRESHOLD_MAJOR:
                verdict = "🟡 소폭 차이"
            else:
                verdict = "🔴 큰 차이"

            print(
                f"  {angle_name:<28} {frame_label:>10}  {full_val:>8.1f}°  {heavy_val:>8.1f}°  "
                f"{diff:>6.1f}°  {verdict}"
            )

        print("=" * 72)

        if max_angle_diff <= DIFF_THRESHOLD_MINOR:
            angle_conclusion = "✅ 유의미한 각도 차이 없음 — Full 기준 캘리브레이션 데이터 그대로 사용 가능"
        elif max_angle_diff < DIFF_THRESHOLD_MAJOR:
            angle_conclusion = f"🟡 소폭 각도 차이 ({max_angle_diff:.1f}°) — Heavy 기준으로 재수집 권장"
        else:
            angle_conclusion = f"🔴 큰 각도 차이 ({max_angle_diff:.1f}°) — Heavy 기준으로 전체 재수집 필요"

        print(f"\n  각도 결론: {angle_conclusion}\n")

    finally:
        if tmp_ctx:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
