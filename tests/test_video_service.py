import pytest
from unittest.mock import patch, MagicMock
from services.video_service import VideoService


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
