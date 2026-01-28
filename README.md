# MotionLab AI Server

FastAPI 기반 **AI 분석 서버** (MediaPipe + GPT-4o-mini)

## 🎯 프로젝트 목표

- MediaPipe를 활용한 실시간 포즈 추출
- 각도 계산 및 스윙 구간 감지
- GPT-4o-mini 기반 피드백 생성

## 🛠️ 기술 스택

- **Framework**: FastAPI 0.109.0
- **Vision**: MediaPipe 0.10.14, OpenCV 4.9.0
- **LLM**: OpenAI GPT-4o-mini
- **Python**: 3.12.7

## 📦 설치 및 실행

### 1. 가상환경 생성

```bash
python3.12 -m venv venv
source venv/bin/activate 
