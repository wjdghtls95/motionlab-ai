"""
관리자 API 라우터

현재 제공 엔드포인트:
- POST /admin/reload-config  — sports_config.json 강제 리로드 (인증 필요)
"""

from fastapi import APIRouter, Depends
from core.sport_configs import reload_config, get_config_version, get_available_sports
from utils.dependencies import verify_api_key
from config.settings import get_settings

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post(
    "/reload-config",
    summary="Config 강제 리로드",
    description="sports_config.json을 강제로 다시 읽는다. X-Internal-API-Key 헤더 필요.",
    dependencies=[Depends(verify_api_key)],
)
async def reload_sport_config() -> dict:
    """
    CONFIG_SOURCE 설정에 따라 config를 재로드하고 결과를 반환한다.

    Returns:
        success:  처리 성공 여부
        source:   현재 config 소스 (local | file | remote)
        version:  로드된 config 버전
        sports:   사용 가능한 종목 목록
    """
    reload_config()

    return {
        "success": True,
        "source": get_settings().CONFIG_SOURCE,
        "version": get_config_version(),
        "sports": get_available_sports(),
    }
