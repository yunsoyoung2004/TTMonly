# 🧠 TTMchatbot (Basic Version)

TTMchatbot은 **Transtheoretical Model (TTM)** 기반으로 설계된 다단계 상담 챗봇 시스템입니다.  
사용자의 대화 흐름에 따라 `empathy → mi → cbt1 → cbt2 → cbt3` 단계를 순차적으로 진행하며, 각 단계는 독립된 LLM을 통해 응답을 생성합니다.

이 레포지토리는 **페르소나 드리프트 감지 기능 없이**, TTM 단계별 대화 흐름만 구현된 **기본 버전**입니다.

---

## 💡 주요 특징

- 단계별 LLM 호출을 통한 다단계 상담 흐름 구성
- FastAPI 기반 서버 및 스트리밍 응답
- Hugging Face 모델 자동 다운로드
- 단계 정보 포함한 상태(state) 객체 관리

---

## 🗂️ 프로젝트 구조

```
TTMchatbot/
├── agents/             # 단계별 응답 처리 함수 (stream_XXX_reply)
├── main.py             # FastAPI 엔트리포인트
├── utils/              # 드리프트 관련 유틸 (기본 버전에서는 미사용)
├── requirements.txt    # 의존성 패키지
├── Dockerfile          # 도커 배포 파일
└── ...
```

---

## 🚀 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. Hugging Face 인증 토큰 등록

```bash
export HUGGINGFACE_TOKEN=hf_...
```

### 3. FastAPI 서버 실행

```bash
python main.py
```

---

## ✅ API 엔드포인트

| Method | Endpoint         | 설명                        |
|--------|------------------|-----------------------------|
| GET    | `/`              | 서버 준비 상태 확인         |
| GET    | `/status`        | 각 단계별 모델 준비 확인    |
| POST   | `/chat/stream`   | 단계별 챗봇 응답 스트리밍    |

---

## 🧪 요청 예시

```json
{
  "state": {
    "session_id": "user123",
    "stage": "cbt1",
    "question": "요즘 기분이 어떤가요?",
    "response": "",
    "history": [],
    "turn": 0,
    "intro_shown": false
  }
}
```

응답은 스트리밍 형태로 전송되며, 마지막에 다음 턴 정보가 포함된 `---END_STAGE---` JSON 블록이 함께 전달됩니다.

---

## 🧠 단계 구성

```
1. empathy → 2. mi → 3. cbt1 → 4. cbt2 → 5. cbt3
```

사용자가 직접 단계를 지정해야 하며, 자동 전이 및 drift 탐지는 포함되지 않습니다.

---

## 👩‍💻 개발자

- **이름**: 윤소영 (SoYoung Yun)  
- 📧 **개인 이메일**: yunsoyoung2004@gmail.com  
- 📧 **학교 이메일**: thdud041113@g.skku.edu  
- 🔗 **GitHub**: [yunsoyoung2004](https://github.com/yunsoyoung2004)

---

> ✨ 이 레포지토리는 TTM 기반 챗봇의 **기본 구조와 흐름 구현**에 집중한 버전입니다.  
> 확장형(페르소나 드리프트 탐지, 자동 전이 기능 포함)은 별도 레포지토리에서 관리됩니다.
