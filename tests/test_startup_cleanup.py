"""
서버 시작 시 잔여 임시 파일 정리 테스트 (main._cleanup_stale_temp_files)
"""

import os
import tempfile
from unittest.mock import patch
from main import _cleanup_stale_temp_files


# 1. 잔여 mp4 파일이 있으면 서버 시작 시 모두 삭제된다
def test_cleanup_deletes_stale_mp4_files():
    # 방지: 이전 실행의 임시 파일이 디스크에 축적되어 용량 초과
    with tempfile.TemporaryDirectory() as tmpdir:
        mp4_files = [os.path.join(tmpdir, f"{i}.mp4") for i in range(3)]
        for f in mp4_files:
            open(f, "w").close()

        with patch("main.settings.SENTRY_DSN", ""):
            _cleanup_stale_temp_files(tmpdir)

        remaining = list(os.scandir(tmpdir))
        assert len(remaining) == 0


# 2. mp4 파일이 없으면 아무것도 하지 않는다
def test_cleanup_skips_when_no_files():
    # 방지: 빈 디렉토리에서 불필요한 작업 수행
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("main.settings.SENTRY_DSN", ""):
            _cleanup_stale_temp_files(tmpdir)  # 에러 없이 실행되어야 함


# 3. temp 디렉토리가 없으면 에러 없이 스킵한다
def test_cleanup_skips_nonexistent_dir():
    # 방지: 첫 번째 서버 시작 시 temp 디렉토리 없어 서버 기동 실패
    _cleanup_stale_temp_files("/nonexistent/path/that/does/not/exist")


# 4. 삭제 실패 시 Sentry capture_message가 호출된다 (DSN 있을 때)
def test_cleanup_failure_sends_sentry_when_dsn_set():
    # 방지: 삭제 실패가 운영팀에 전달 안 돼 디스크 풀 인지 못함
    with tempfile.TemporaryDirectory() as tmpdir:
        mp4_file = os.path.join(tmpdir, "1.mp4")
        open(mp4_file, "w").close()

        with (
            patch("main.settings.SENTRY_DSN", "https://fake@sentry.io/1"),
            patch("pathlib.Path.unlink", side_effect=OSError("Permission denied")),
            patch("main.sentry_sdk.capture_message") as mock_sentry,
        ):
            _cleanup_stale_temp_files(tmpdir)

        mock_sentry.assert_called_once()
        call_args = mock_sentry.call_args
        assert call_args[1]["level"] == "warning"


# 5. DSN 없을 때 Sentry capture_message가 호출되지 않는다
def test_cleanup_failure_no_sentry_when_dsn_empty():
    # 방지: DSN 없는 개발 환경에서 Sentry 초기화 에러 발생
    with tempfile.TemporaryDirectory() as tmpdir:
        mp4_file = os.path.join(tmpdir, "1.mp4")
        open(mp4_file, "w").close()

        with (
            patch("main.settings.SENTRY_DSN", ""),
            patch("pathlib.Path.unlink", side_effect=OSError("Permission denied")),
            patch("main.sentry_sdk.capture_message") as mock_sentry,
        ):
            _cleanup_stale_temp_files(tmpdir)

        mock_sentry.assert_not_called()
