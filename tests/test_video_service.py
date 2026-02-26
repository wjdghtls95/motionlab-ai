from services.video_service import VideoService


def test_local_path_handling():
    """로컬 경로 처리 테스트"""
    service = VideoService()

    # 로컬 파일 경로 (실제 파일 필요)
    local_path = "/app/uploads/test.mp4"
    result = service.get_video_path(1, local_path)

    assert result == local_path


def test_metadata_extraction():
    """메타데이터 추출 테스트"""
    service = VideoService()
    video_path = "/path/to/test/video.mp4"

    metadata = service.extract_metadata(video_path)

    assert metadata['width'] > 0
    assert metadata['height'] > 0
    assert metadata['fps'] > 0
