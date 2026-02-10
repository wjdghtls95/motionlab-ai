"""
파이프라인 단계별 시간 측정 유틸리티
"""

import time
from utils.logger import logger
from contextlib import contextmanager


class StepTimer:
    """파이프라인 단계별 시간 측정"""

    def __init__(self):
        self.durations: dict[str, float] = {}
        self.total_start: float = 0
        self._step_count: int = 0

    def start_total(self):
        self.total_start = time.time()

    def summary(self, motion_id: int) -> float:
        """전체 성능 요약 로그"""
        total = time.time() - self.total_start

        logger.info("")
        logger.info(f"📊 === 성능 요약 (motion_id={motion_id}) ===")
        for i, (name, duration) in enumerate(self.durations.items(), 1):
            pct = duration / total * 100 if total > 0 else 0
            logger.info(f"  {i}. {name:<16} {duration:>6.2f}초 ({pct:>5.1f}%)")
        logger.info(f"  총 소요 시간:        {total:>6.2f}초")
        logger.info("")

        return total

    @property
    def total_steps(self) -> int:
        """등록된 전체 단계 수"""
        return self._step_count

    @contextmanager
    def step(self, step_num: int, total: int, title: str):
        """
        단계별 시간 측정 (번호 + 이름)

        사용법:
            with timer.step(1, 7, "영상 다운로드"):
                result = download()
        """
        label = f"[{step_num}/{total}]"
        logger.info(f"{label} {title}")
        start = time.time()

        yield

        duration = time.time() - start
        self.durations[title] = duration
        logger.info(f"⏱️ {label} {title} 완료: {duration:.2f}초")

    @contextmanager
    def measure_func(self, name: str):
        """
        단순 시간 측정 (번호 없이)

        사용법:
            with timer.measure("비디오 인코딩"):
                encode(video)
        """
        logger.info(f"{name}")
        start = time.time()

        yield

        duration = time.time() - start
        self.durations[name] = duration
        logger.info(f"⏱️ {name} 완료: {duration:.2f}초")
