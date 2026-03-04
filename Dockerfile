# ============================================
# motionlab-ai Dockerfile
# Multi-stage build: 빌드 캐시 최적화
# ============================================

# --- Stage 1: Base ---
FROM python:3.11-slim AS base

# 시스템 패키지 (OpenCV, MediaPipe 의존성)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Stage 2: Dependencies ---
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 3: Production ---
FROM deps AS production

# 앱 코드 복사
COPY . .

# 임시 비디오 디렉토리 생성
RUN mkdir -p /app/temp_videos

# 비 root 유저 실행 (보안)
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
