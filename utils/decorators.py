"""
MotionLab AI - Smart Decorators
A1~A29 규칙 준수 + Sync/Async 지원
"""

import functools
import asyncio
import time
from typing import Type, Tuple, Optional, Callable

from utils.logger import logger
from utils.exceptions import AnalyzerError, ErrorCode

# ========================================
# 1. Smart Retry (retryable 속성 인식)
# ========================================


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    지능형 재시도 데코레이터

    특징:
    - AnalyzerError의 retryable 속성 자동 감지
    - AN_ 에러 (retryable=False) 발생 시 즉시 중단
    - 지수 백오프 (1초 → 2초 → 4초)

    Example:
        from openai import OpenAIError

        @retry(max_attempts=3, exceptions=(OpenAIError,))
        async def call_openai():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    # ⭐ 핵심: retryable 속성 체크
                    is_retryable = getattr(e, "retryable", True)

                    if not is_retryable:
                        logger.warning(
                            f"⛔ [Retry Aborted] {func.__name__}: "
                            f"{e} (retryable=False)"
                        )
                        raise e

                    # 마지막 시도면 에러 던짐
                    if attempt == max_attempts:
                        logger.error(
                            f"❌ [Retry Failed] {func.__name__} failed "
                            f"after {max_attempts} attempts: {e}"
                        )
                        raise e

                    logger.warning(
                        f"⚠️ [Retry {attempt}/{max_attempts}] {func.__name__}: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    is_retryable = getattr(e, "retryable", True)

                    if not is_retryable:
                        logger.warning(
                            f"⛔ [Retry Aborted] {func.__name__}: "
                            f"{e} (retryable=False)"
                        )
                        raise e

                    if attempt == max_attempts:
                        logger.error(
                            f"❌ [Retry Failed] {func.__name__} failed "
                            f"after {max_attempts} attempts: {e}"
                        )
                        raise e

                    logger.warning(
                        f"⚠️ [Retry {attempt}/{max_attempts}] {func.__name__}: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff

        # async/sync 함수 자동 감지
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# ========================================
# 2. Adaptive Logging (threshold 기반)
# ========================================


def measure_time(threshold_ms: int = 1000):
    """
    스마트 성능 측정 데코레이터

    특징:
    - threshold 초과 시 WARNING (CloudWatch 비용 절감)
    - threshold 미만 시 DEBUG (로그 노이즈 감소)

    Example:
        @measure_time(threshold_ms=3000)  # 3초 초과 시 경고
        async def analyze_video():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            func_name = func.__name__

            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000

                # ⭐ 핵심: 임계값 기반 로그 레벨 분기
                if elapsed_ms > threshold_ms:
                    logger.warning(
                        f"🐢 [Slow] {func_name} took {elapsed_ms:.2f}ms "
                        f"(threshold: {threshold_ms}ms)"
                    )
                else:
                    logger.debug(f"⚡ {func_name} took {elapsed_ms:.2f}ms")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            func_name = func.__name__

            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000

                if elapsed_ms > threshold_ms:
                    logger.warning(
                        f"🐢 [Slow] {func_name} took {elapsed_ms:.2f}ms "
                        f"(threshold: {threshold_ms}ms)"
                    )
                else:
                    logger.debug(f"⚡ {func_name} took {elapsed_ms:.2f}ms")

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# ========================================
# 3. Async Timeout (안전한 비동기 타임아웃)
# ========================================


def timeout(seconds: int):
    """
    비동기 함수 실행 시간 제한

    특징:
    - asyncio.wait_for 사용 (Event Loop 차단 없음)
    - TimeoutError 발생 시 AnalyzerError(SYS_021)로 변환

    Example:
        @timeout(60)  # 1분 제한
        async def download_video(url: str):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                # ⭐ 핵심: asyncio.wait_for 사용
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

            except asyncio.TimeoutError:
                logger.error(f"⏰ [Timeout] {func.__name__} exceeded {seconds}s")

                # A21: ANALYZER_TIMEOUT 에러로 변환
                raise AnalyzerError(
                    ErrorCode.ANALYZER_TIMEOUT,
                    custom_message=f"{func.__name__} 실행 시간 초과 ({seconds}초)",
                )

        # sync 함수는 timeout 지원 안 함
        if not asyncio.iscoroutinefunction(func):
            logger.warning(f"⚠️ @timeout은 async 함수만 지원합니다: {func.__name__}")
            return func

        return async_wrapper

    return decorator


# ========================================
# 4. Execution Logging (실행 추적)
# ========================================


def log_execution(log_result: bool = False):
    """
    함수 실행/종료 로깅

    Example:
        @log_execution(log_result=True)
        async def analyze_motion(motion_id: int):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.info(f"▶️ [START] {func_name}")

            try:
                result = await func(*args, **kwargs)

                if log_result:
                    # 결과가 너무 길면 잘라서 출력
                    res_str = str(result)
                    display_res = (
                        res_str[:100] + "..." if len(res_str) > 100 else res_str
                    )
                    logger.info(f"✅ [END] {func_name} - Result: {display_res}")
                else:
                    logger.info(f"✅ [END] {func_name}")

                return result

            except Exception as e:
                logger.error(f"🔥 [ERROR] {func_name}: {e}")
                raise e

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.info(f"▶️ [START] {func_name}")

            try:
                result = func(*args, **kwargs)

                if log_result:
                    res_str = str(result)
                    display_res = (
                        res_str[:100] + "..." if len(res_str) > 100 else res_str
                    )
                    logger.info(f"✅ [END] {func_name} - Result: {display_res}")
                else:
                    logger.info(f"✅ [END] {func_name}")

                return result

            except Exception as e:
                logger.error(f"🔥 [ERROR] {func_name}: {e}")
                raise e

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
