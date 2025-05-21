from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List
import json, os, asyncio, time, re

from agents.empathy_agent import stream_empathy_reply
from agents.mi_agent import stream_mi_reply
from agents.cbt1_agent import stream_cbt1_reply
from agents.cbt2_agent import stream_cbt2_reply
from agents.cbt3_agent import stream_cbt3_reply

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 상태 모델 정의 (간결화)
class AgentState(BaseModel):
    stage: Literal["empathy", "mi", "cbt1", "cbt2", "cbt3", "end"]
    question: str
    response: str
    history: List[str]
    turn: int = 0
    topic_index: int = 0

# ✅ 인트로 메시지
AGENT_INTROS = {
    "empathy": "안녕하세요. 저는 감정 지원을 도와드리는 공감 상담가입니다. 편하게 느끼는 감정에 대해 말씀해 주세요.",
    "mi": "안녕하세요. 저는 변화 동기를 함께 탐색하는 동기강화 상담가입니다. 마음속 갈등이나 망설임이 있다면 함께 이야기해 볼까요?",
    "cbt1": "안녕하세요. 저는 자동사고를 탐색하는 인지 상담가입니다. 최근에 불편했던 생각이나 감정에 대해 말씀해 주세요.",
    "cbt2": "안녕하세요. 저는 사고를 더 건강하게 바꾸는 인지 재구성 상담가입니다. 함께 다른 시각에서 생각을 정리해볼게요.",
    "cbt3": "안녕하세요. 저는 실천 계획을 도와드리는 행동 상담가입니다. 앞으로 어떤 행동을 시도해볼 수 있을지 함께 정해볼게요.",
}

model_ready = False
model_paths = {}

@app.on_event("startup")
async def set_model_paths():
    global model_ready, model_paths
    try:
        model_paths = {
            "empathy": "경로/empathy.gguf",
            "mi": "경로/mi.gguf",
            "cbt1": "경로/cbt1.gguf",
            "cbt2": "경로/cbt2.gguf",
            "cbt3": "경로/cbt3.gguf",
        }
        model_ready = True
        print("✅ 모델 경로 등록 완료", flush=True)
    except Exception as e:
        print(f"❌ 모델 경로 등록 실패: {e}", flush=True)

@app.get("/")
def root():
    return JSONResponse({"message": "✅ TTM 챗봇 서버 실행 중"})

@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        data = await request.json()
        state = AgentState(**data.get("state", {}))
    except Exception:
        return StreamingResponse(iter([
            r"⚠️ 입력 상태가 잘못되었습니다.\n",
            b"\n---END_STAGE---\n" + json.dumps({
                "next_stage": "empathy",
                "response": "입력 상태가 잘못되었습니다.",
                "turn": 0,
                "history": [],
                "topic_index": 0
            }, ensure_ascii=False).encode("utf-8")
        ]), media_type="text/plain")

    async def async_gen():
        if not model_ready:
            yield r"⚠️ 모델이 아직 준비되지 않았습니다.\n"
            return

        # ✅ 첫 턴이면 인트로 먼저 출력
        if state.turn == 0 and state.stage in AGENT_INTROS:
            intro = AGENT_INTROS[state.stage]
            yield (intro + "\n").encode("utf-8")
            yield b"\n---END_STAGE---\n" + json.dumps({
                "next_stage": state.stage,
                "response": intro,
                "turn": 0,
                "history": state.history + [intro],
                "topic_index": state.topic_index
            }, ensure_ascii=False).encode("utf-8")
            return

        # ✅ 응답 수집
        full_text = ""
        async def collect_stream(generator):
            nonlocal full_text
            async for chunk in generator:
                try:
                    full_text += chunk.decode("utf-8")
                except:
                    continue
                yield chunk

        agent_streams = {
            "empathy": lambda: stream_empathy_reply(state.question, model_paths["empathy"], state.turn),
            "mi": lambda: stream_mi_reply(state, model_paths["mi"]),
            "cbt1": lambda: stream_cbt1_reply(state, model_paths["cbt1"]),
            "cbt2": lambda: stream_cbt2_reply(state, model_paths["cbt2"]),
            "cbt3": lambda: stream_cbt3_reply(state, model_paths["cbt3"]),
        }

        if state.stage not in agent_streams:
            yield r"⚠️ 지원되지 않는 단계입니다.\n"
            return

        try:
            async for chunk in collect_stream(agent_streams[state.stage]()):
                yield chunk
        except Exception as e:
            yield f"\n⚠️ 응답 중 오류 발생: {e}".encode("utf-8")

        # ✅ 최종 상태 반환
        match = re.search(r'---END_STAGE---\n({.*})', full_text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                next_stage = result.get("next_stage", state.stage)
                state.turn = result.get("turn", 0)
                state.history = result.get("history", [])
                state.response = result.get("response", "")
                state.topic_index = result.get("topic_index", 0)
            except:
                next_stage = state.stage
        else:
            next_stage = state.stage

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "response": state.response.strip() or "응답이 생성되지 않았습니다.",
            "turn": state.turn,
            "history": state.history,
            "topic_index": state.topic_index
        }, ensure_ascii=False).encode("utf-8")

    return StreamingResponse(async_gen(), media_type="text/plain")
