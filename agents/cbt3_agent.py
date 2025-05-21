import os, json, multiprocessing, difflib, re, asyncio
from typing import AsyncGenerator, Literal, List
from pydantic import BaseModel, Field
from llama_cpp import Llama

# ✅ 모델 캐시 및 전역 질문 세트
LLM_CBT3_INSTANCE = {}
GLOBAL_CBT3_QUESTIONS: List[str] = []

# ✅ 모델 로딩
def load_cbt3_model(model_path: str) -> Llama:
    global LLM_CBT3_INSTANCE
    if model_path not in LLM_CBT3_INSTANCE:
        print("🚀 CBT3 모델 최초 로딩 중...", flush=True)
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_CBT3_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=NUM_THREADS,
            n_batch=4,
            max_tokens=128,
            temperature=0.65,
            top_p=0.9,
            presence_penalty=1.0,
            frequency_penalty=0.8,
            repeat_penalty=1.1,
            n_gpu_layers=0,
            low_vram=True,
            use_mlock=False,
            verbose=False,
            chat_format="llama-3",
            stop=["<|im_end|>", "\n\n"]
        )
        print(f"✅ CBT3 모델 로딩 완료: {model_path}", flush=True)
    return LLM_CBT3_INSTANCE[model_path]

# ✅ 질문 생성
def generate_preset_questions(llm: Llama) -> List[str]:
    global GLOBAL_CBT3_QUESTIONS
    if GLOBAL_CBT3_QUESTIONS:
        return GLOBAL_CBT3_QUESTIONS
    prompt = (
        "너는 따뜻하고 논리적인 CBT 상담자야. 사용자가 실천할 수 있도록 이끌 수 있는 열린 질문 5개를 제안해줘. "
        "질문은 짧고 명확해야 해. 다음 주제를 활용해도 좋아: 방해 요인, 감정 변화, 습관 형성, 환경 조정, 피드백 실천."
    )
    result = llm.create_completion(prompt=prompt, max_tokens=256)
    text = result["choices"][0]["text"]
    questions = re.findall(r"[^.\n!?]*\?", text)
    GLOBAL_CBT3_QUESTIONS = [q.strip() for q in questions if q.strip()][:5]
    return GLOBAL_CBT3_QUESTIONS

# ✅ 상태 모델
class AgentState(BaseModel):
    stage: Literal["cbt3", "end"]
    question: str
    response: str
    history: List[str]
    turn: int
    preset_questions: List[str] = Field(default_factory=list)

# ✅ 스트리밍 응답 함수
async def stream_cbt3_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    try:
        llm = load_cbt3_model(model_path)

        # ✅ 질문 세트 로딩
        if not state.preset_questions:
            print("⚙️ CBT3 질문 세트 로딩 중...", flush=True)
            global_questions = await asyncio.to_thread(generate_preset_questions, llm)
            state.preset_questions = global_questions.copy()
            print("✅ CBT3 질문 세트 복사 완료")

        # ✅ 현재 턴에서 질문 선택
        if state.turn < len(state.preset_questions):
            reply = state.preset_questions[state.turn]
        else:
            reply = "지금까지의 대화를 바탕으로 좋은 실천 계획이 세워졌어요."

        # ✅ 스트리밍 응답
        first_token_sent = False
        for ch in reply:
            if not first_token_sent:
                yield b"\n"
                first_token_sent = True
            yield ch.encode("utf-8")
            await asyncio.sleep(0.015)

        # ✅ 상태 전이 계산
        next_turn = state.turn + 1
        next_stage = "end" if next_turn >= 5 else "cbt3"

        # ✅ 종료 메시지 처리
        if next_stage == "end":
            end_msg = "\n\n🎯 실천 계획을 잘 정리해주셨어요. 이제 오늘 대화를 마무리할게요."
            for ch in end_msg:
                yield ch.encode("utf-8")
                await asyncio.sleep(0.015)

        # ✅ 다음 상태 전송
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "turn": next_turn if next_stage != "end" else 0,
            "response": reply,
            "history": state.history + [state.question, reply],
            "preset_questions": state.preset_questions
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        print(f"⚠️ CBT3 오류: {e}", flush=True)
        fallback = "죄송해요. 지금은 잠시 오류가 발생했어요. 다시 이야기해 주시겠어요?"
        for ch in fallback:
            yield ch.encode("utf-8")
            await asyncio.sleep(0.02)
