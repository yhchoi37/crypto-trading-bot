# Crypto Trading Bot

## 🤖 프로젝트 개요

이 프로젝트는 다중 암호화폐(10종)를 지원하는 완전 자동 거래 시스템입니다. X(구 Twitter)와 Reddit의 실시간 소셜 미디어 감성 분석 데이터를 통합하여 시장을 예측하고, 이를 바탕으로 최적의 매수/매도 전략을 실행합니다.

---

## ✨ 주요 기능

-   **10개 주요 암호화폐 지원**: BTC, ETH, XRP, ADA, DOGE, SOL, DOT, LINK, LTC, MATIC
-   **실시간 소셜 미디어 분석**: X(구 Twitter)와 Reddit의 데이터를 실시간으로 수집 및 분석
-   **다중 알고리즘 조합**: 규칙 기반 및 머신러닝(Transformer) 알고리즘 결합
-   **지능형 포트폴리오 관리**: 설정된 목표에 따라 자동으로 포트폴리오 리밸런싱 수행
-   **고급 위험 관리**: 자동 손절(Stop-loss) 및 익절(Take-profit) 기능
-   **실시간 알림 시스템**: 텔레그램 및 이메일을 통한 거래 신호 및 시스템 상태 알림
-   **안정적인 운영 환경**: Docker 컨테이너 환경을 완벽히 지원하며, CI/CD 연동이 용이

---

## 🛠️ 설치 및 실행 방법

### 1. 사전 요구사항

-   Python 3.11 이상 버전 설치를 권장합니다.
-   Git이 설치되어 있어야 합니다.

### 2. 저장소 복제 (Clone)

```bash
git clone https://github.com/your-repo/crypto-trading-bot.git
cd crypto-trading-bot
```

### 3. 가상환경 생성 및 활성화

```bash
python -m venv trading_env
# Linux / macOS
source trading_env/bin/activate

# Windows (PowerShell)
trading_env\Scripts\Activate.ps1

# Windows (cmd.exe)
trading_env\Scripts\activate.bat
```

### 4. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

### 5. 환경변수 설정

`.env.template` 파일을 복사하여 `.env` 파일을 생성한 후, 개인 API 키를 입력합니다.


```bash
copy .env.template .env  # Windows
# 또는 Linux/macOS
cp .env.template .env

# 편집: Windows PowerShell
notepad .env

# 편집: Linux/macOS
nano .env  # 또는 사용하는 텍스트 에디터로 파일 열기
```

> **중요**: `.env` 파일은 민감한 정보를 담고 있으므로 Git에 절대 올리면 안 됩니다. `.gitignore` 파일에 `.env`가 포함되어 있는지 확인하세요.

### 6. 시스템 실행
```bash
python main.py
```

---

## 기여 및 문의

- Pull Request, Issue 환영
- 라이센스: MIT# crypto-trading-bot
