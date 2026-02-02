import os
import cv2
import requests
from pathlib import Path
from typing import Optional, Dict
from urllib.parse import urlparse
from config.settings import get_settings


class VideoService:
    """영상 다운로드 및 메타데이터 추출 서비스"""

    def __init__(self):
        settings = get_settings()
        self.temp_dir = Path(settings.TEMP_VIDEO_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = settings.MAX_VIDEO_SIZE_MB

    def get_video_path(self, motion_id: int, video_url: str) -> str:
        """
        영상 URL을 로컬 경로로 변환
        - 로컬 경로: 그대로 반환
        - Presigned URL: 다운로드 후 임시 경로 반환
        """
        if self._is_local_path(video_url):
            return self._handle_local_path(video_url)
        else:
            return self._download_from_url(motion_id, video_url)

    def extract_metadata(self, video_path: str) -> Dict:
        """영상 메타데이터 추출"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        metadata = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
            "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2)
        }

        cap.release()
        return metadata

    def cleanup(self, video_path: str) -> None:
        """임시 파일 삭제"""
        if video_path.startswith(str(self.temp_dir)) and os.path.exists(video_path):
            os.remove(video_path)
            print(f"✅ 임시 파일 삭제: {video_path}")

    def _is_local_path(self, video_url: str) -> bool:
        """로컬 경로 여부 확인"""
        parsed = urlparse(video_url)
        return parsed.scheme in ('', 'file') or video_url.startswith('/')

    def _download_from_url(self, motion_id: int, video_url: str) -> str:
        """Presigned URL에서 영상 다운로드"""
        # 1) HEAD 요청으로 파일 크기 확인
        head_response = requests.head(video_url, timeout=10)
        content_length = int(head_response.headers.get('Content-Length', 0))
        size_mb = content_length / (1024 * 1024)

        if size_mb > self.max_size_mb:
            raise ValueError(
                f"Video size ({size_mb:.2f}MB) exceeds limit ({self.max_size_mb}MB)"
            )

        # 2) 임시 파일 경로 생성
        temp_path = self.temp_dir / f"motion_{motion_id}.mp4"

        # 3) 다운로드 (1MB 단위 청크)
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()

        downloaded_size = 0
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # 실시간 용량 체크
                    if downloaded_size > self.max_size_mb * 1024 * 1024:
                        f.close()
                        os.remove(temp_path)
                        raise ValueError("Video size exceeded during download")

        print(f"✅ 영상 다운로드 완료: {temp_path} ({size_mb:.2f}MB)")
        return str(temp_path)

    def _handle_local_path(self, video_url: str) -> str:
        """로컬 경로 검증 및 절대 경로 변환"""
        path = video_url.replace('file://', '')
        abs_path = Path(path).resolve()

        if not abs_path.exists():
            raise FileNotFoundError(
                f"영상 파일을 찾을 수 없습니다.\n"
                f"입력: {video_url}\n"
                f"절대 경로: {abs_path}"
            )

        print(f"📁 로컬 파일 사용: {abs_path}")

        return str(abs_path)