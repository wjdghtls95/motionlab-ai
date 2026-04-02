import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from services.video_service import VideoService, VideoResource


@pytest.fixture
def video_service():
    return VideoService()


# 1. 존재하지 않는 파일 → 에러
def test_metadata_invalid_path(video_service):
    """존재하지 않는 영상 파일 → VideoProcessingError"""
    from utils.exceptions.errors import VideoProcessingError

    with pytest.raises(VideoProcessingError):
        video_service.extract_metadata("/nonexistent/path/video.mp4")


# 2. 정상 메타데이터 추출 (cv2 mock)
def test_metadata_extraction_mocked(video_service):
    """cv2.VideoCapture를 mock해서 실제 파일 없이 메타데이터 추출 테스트"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        3: 1920.0,  # CAP_PROP_FRAME_WIDTH
        4: 1080.0,  # CAP_PROP_FRAME_HEIGHT
        5: 30.0,  # CAP_PROP_FPS
        7: 150.0,  # CAP_PROP_FRAME_COUNT
    }.get(prop, 0.0)

    with patch("services.video_service.cv2.VideoCapture", return_value=mock_cap):
        metadata = video_service.extract_metadata("/fake/video.mp4")

    assert metadata["width"] == 1920
    assert metadata["height"] == 1080
    assert metadata["fps"] == 30.0
    assert metadata["duration_seconds"] == 5.0


# 3. VideoService 인스턴스 정상 생성
def test_video_service_init(video_service):
    assert video_service is not None


# 4. fps가 0인 영상
def test_metadata_zero_fps(video_service):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        3: 1920.0,
        4: 1080.0,
        5: 0.0,
        7: 150.0,
    }.get(prop, 0.0)

    with patch("services.video_service.cv2.VideoCapture", return_value=mock_cap):
        metadata = video_service.extract_metadata("/fake/video.mp4")

    assert metadata["fps"] == 0.0
    assert metadata["duration_seconds"] >= 0


# 5. 빈 영상 (frame_count = 0)
def test_metadata_empty_video(video_service):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        3: 0.0,
        4: 0.0,
        5: 30.0,
        7: 0.0,
    }.get(prop, 0.0)

    with patch("services.video_service.cv2.VideoCapture", return_value=mock_cap):
        metadata = video_service.extract_metadata("/fake/video.mp4")

    assert metadata["duration_seconds"] == 0


# ── VideoResource._safe_cleanup 테스트 ──────────────────────────────────────


# 6. 파일이 존재하면 정상 삭제되고 Sentry 호출 없음
def test_safe_cleanup_success_no_sentry():
    # 방지: 정상 삭제 시 불필요한 Sentry 알림 전송
    resource = VideoResource(motion_id=1, video_url="http://example.com/video.mp4")
    resource.video_path = "/tmp/1.mp4"

    with (
        patch("os.path.exists", return_value=True),
        patch("os.remove") as mock_remove,
        patch("services.video_service.sentry_sdk.capture_message") as mock_sentry,
    ):
        asyncio.run(resource._safe_cleanup())

    mock_remove.assert_called_once_with("/tmp/1.mp4")
    mock_sentry.assert_not_called()


# 7. 재시도 모두 실패 시 Sentry capture_message 호출됨
def test_safe_cleanup_failure_sends_sentry():
    # 방지: cleanup 실패가 Sentry에 기록 안 돼 디스크 풀 상황 인지 못함
    resource = VideoResource(motion_id=42, video_url="http://example.com/video.mp4")
    resource.video_path = "/tmp/42.mp4"

    with (
        patch("os.path.exists", return_value=True),
        patch("os.remove", side_effect=OSError("Permission denied")),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("services.video_service.sentry_sdk.capture_message") as mock_sentry,
    ):
        asyncio.run(resource._safe_cleanup())

    mock_sentry.assert_called_once()
    call_args = mock_sentry.call_args
    assert "42" in call_args[0][0]  # motion_id 포함
    assert call_args[1]["level"] == "error"


# 8. video_path가 None이면 cleanup 시도 없이 조기 반환
def test_safe_cleanup_no_path_skips():
    # 방지: 다운로드 실패 후 None path로 cleanup 시 AttributeError
    resource = VideoResource(motion_id=1, video_url="http://example.com/video.mp4")
    resource.video_path = None

    with patch("os.remove") as mock_remove:
        asyncio.run(resource._safe_cleanup())

    mock_remove.assert_not_called()
