"""
MotionLab AI - 영상 서비스
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yt_dlp
import cv2

from utils.exceptions import (
    VideoDownloadError,
    VideoNotFoundError,
    VideoProcessingError,
)

logger = logging.getLogger(__name__)


class VideoResource:
    """영상 리소스 Context Manager"""

    def __init__(
        self,
        motion_id: int,
        video_url: str,
        output_dir: str = "./temp_videos",
        max_retries: int = 3,
    ):
        self.motion_id = motion_id
        self.video_url = video_url
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.video_path: Optional[str] = None
        self.is_local_file = False

    async def __aenter__(self) -> str:
        """영상 다운로드"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.video_path = await self._download_video()
            logger.info(f"✅ 영상 다운로드 완료: {self.video_path}")
            return self.video_path
        except Exception as e:
            logger.error(f"❌ 영상 다운로드 실패: {e}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """임시 파일 정리 (로컬 파일은 보존)"""
        if self.video_path:
            if self.is_local_file:
                logger.info(f"✅ Local file preserved: {self.video_path}")
            else:
                await self._safe_cleanup()
        return False

    async def _download_video(self) -> str:
        """영상 다운로드"""
        output_path = str(self.output_dir / f"{self.motion_id}.mp4")

        if self.video_url.startswith(("/", "./")) or os.path.exists(self.video_url):
            if os.path.exists(self.video_url):
                logger.info(f"✅ 로컬 파일 사용: {self.video_url}")
                self.is_local_file = True  # 플래그 설정

                return self.video_url
            else:
                raise VideoNotFoundError(
                    details=f"motion_id={self.motion_id}, path={self.video_url}"
                )

        self.is_local_file = False  # 다운로드 파일 표시

        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
        }

        for attempt in range(self.max_retries):
            try:
                logger.info(f"[{attempt + 1}/{self.max_retries}] 영상 다운로드 중...")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([self.video_url])

                if os.path.exists(output_path):
                    return output_path

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise VideoDownloadError(
                        details=f"motion_id={self.motion_id}, url={self.video_url}, error={str(e)}"
                    )
                time.sleep(1)

        raise VideoDownloadError(
            details=f"motion_id={self.motion_id}, retries={self.max_retries}"
        )

    async def _safe_cleanup(self):
        """파일 삭제"""
        if not self.video_path:
            return

        for attempt in range(self.max_retries):
            try:
                if os.path.exists(self.video_path):
                    os.remove(self.video_path)
                    logger.info(f"🗑️ 파일 삭제 완료: {self.video_path}")
                    return
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(0.5)
                else:
                    logger.error(f"⚠️ 파일 삭제 실패: {self.video_path}")


class VideoService:
    """영상 메타데이터 추출"""

    @staticmethod
    def extract_metadata(video_path: str) -> Dict[str, Any]:
        """메타데이터 추출"""
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise VideoProcessingError(
                    details=f"영상 파일을 열 수 없습니다: {video_path}"
                )

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_seconds": round(duration, 2),
            }

        except (VideoNotFoundError, VideoDownloadError, VideoProcessingError):
            raise
        except Exception as e:
            raise VideoProcessingError(details=f"메타데이터 추출 실패: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
