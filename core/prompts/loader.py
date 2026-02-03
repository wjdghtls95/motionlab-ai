"""
MotionLab AI - 프롬프트 로더
YAML 템플릿을 Jinja2로 렌더링
"""

import subprocess
from pathlib import Path
from typing import Dict, Any

import yaml
from jinja2 import Template

from utils.logger import logger


class PromptLoader:
    """
    YAML 프롬프트 템플릿 로더 (Git 버전 자동 추적)

    - 매번 파일을 읽어서 항상 최신 프롬프트 반영
    - Git 커밋 해시로 버전 자동 관리 (수동 업데이트 불필요)
    - Docker Volume 덕분에 서버 재시작 없이 업데이트 가능
    """

    def __init__(self, template_dir: str = "core/prompts/templates"):
        """
        Args:
            template_dir: YAML 템플릿 디렉토리 경로
        """
        self.template_dir = Path(template_dir)

        if not self.template_dir.exists():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")

        # Git 루트 디렉토리 캐싱 (초기화 시 1회만)
        self.git_root = self._find_git_root()

        # 공통 템플릿 (1회만)
        self.shared = self._load_shared()

        logger.info(f"✅ PromptLoader 초기화: {self.template_dir}")

    def load(
        self, sport_type: str, sub_category: str, context: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        YAML 파일을 읽어서 Jinja2 렌더링

        Args:
            sport_type: 종목 (GOLF, WEIGHT)
            sub_category: 세부 종목 (DRIVER, SQUAT)
            context: 렌더링에 사용할 변수들
                - angles: 각도 데이터
                - phases: 구간 데이터
                - standards: 이상 범위 등

        Returns:
            {"system": "...", "user": "..."}

        Raises:
            FileNotFoundError: 템플릿 파일이 없을 때
        """
        # 파일명 생성 (예: driver.yaml)
        file_path = (
            self.template_dir / sport_type.lower() / f"{sub_category.lower()}.yaml"
        )

        if not file_path.exists():
            logger.error(f"❌ 템플릿 파일 없음: {file_path}")

            available = [f.name for f in self.template_dir.glob("*.yaml")]
            raise FileNotFoundError(
                f"Template not found: {file_path}. " f"Available: {available}"
            )

        # YAML 파일 읽기 (항상 최신 반영)
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Git 커밋 해시 자동 추출 (버전 관리)
        version_short = self._get_git_commit_hash(file_path, short=True)
        version_full = self._get_git_commit_hash(file_path, short=False)

        logger.debug(
            f"✅ 프롬프트 로드: {sport_type}/{sub_category}, "
            f"version={version_short}"
        )

        # Jinja2 렌더링
        context["shared"] = self.shared
        system_msg = Template(data["system"]).render(**context)
        user_msg = Template(data["user"]).render(**context)

        return {
            "system": system_msg.strip(),
            "user": user_msg.strip(),
            "version": version_short,
            "version_full": version_full,
        }

    def _find_git_root(self) -> Path:
        """
        Git 루트 디렉토리 찾기

        Returns:
            Git 루트 디렉토리 Path (못 찾으면 현재 디렉토리)
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=1,
                check=True,
            )
            git_root = Path(result.stdout.strip())
            logger.debug(f"🔍 Git 루트: {git_root}")
            return git_root
        except Exception as e:
            logger.warning(f"⚠️ Git 루트를 찾을 수 없음: {e}")
            return Path.cwd()

    def _get_git_commit_hash(self, file_path: Path, short: bool = True) -> str:
        """
        파일의 최신 Git 커밋 해시 반환.

        Args:
            file_path: YAML 파일 경로
            short: True면 짧은 해시 (7자), False면 전체 해시

        Returns:
            "abc123d" (short=True) or "abc123def456..." (short=False)
            Git 없거나 실패 시 "unknown"
        """
        try:
            format_str = "%h" if short else "%H"

            try:
                relative_path = file_path.absolute().relative_to(self.git_root)
            except ValueError:
                logger.warning(f"⚠️ {file_path}가 Git 저장소 밖에 있음")
                return "unknown"

            result = subprocess.run(
                ["git", "log", "-1", f"--format={format_str}", str(relative_path)],
                capture_output=True,
                text=True,
                timeout=1,
                cwd=self.git_root,
            )

            hash_value = result.stdout.strip()

            if hash_value:
                logger.debug(f"🔍 Git 해시 추출: {file_path.name} → {hash_value}")
                return hash_value
            else:
                logger.warning(f"⚠️ Git 해시 없음: {file_path.name}")
                return "unknown"

        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️ Git 명령 타임아웃: {file_path.name}")
            return "unknown"
        except FileNotFoundError:
            logger.warning(f"⚠️ Git 설치 안 됨")
            return "unknown"
        except Exception as e:
            logger.warning(f"⚠️ Git 해시 추출 실패: {e}")
            return "unknown"

    def _load_shared(self) -> Dict[str, str]:
        """
        공통 지침 로드 (_shared/base.yaml)

        Returns:
            Dict: 공통 지침 (scoring_guide, feedback_guide 등)
        """
        shared_path = self.template_dir / "_shared" / "base.yaml"

        if not shared_path.exists():
            logger.warning(f"⚠️ 공통 템플릿 없음: {shared_path}")
            return {}

        with open(shared_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        logger.info(f"✅ 공통 템플릿 로드 완료: {len(data)} 항목")

        return data

    def _get_available_templates(self) -> Dict[str, list]:
        """
        사용 가능한 템플릿 목록 조회

        Returns:
            {"GOLF": ["driver", "iron"], "WEIGHT": ["squat"]}
        """
        available = {}

        for sport_dir in self.template_dir.iterdir():
            if sport_dir.is_dir() and not sport_dir.name.startswith("_"):
                sport_name = sport_dir.name.upper()
                templates = [f.stem for f in sport_dir.glob("*.yaml")]
                available[sport_name] = templates

        return available


# 싱글톤 인스턴스
prompt_loader = PromptLoader()
