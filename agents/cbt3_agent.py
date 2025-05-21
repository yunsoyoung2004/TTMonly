import os, json, multiprocessing
from typing import AsyncGenerator, Literal, List
from pydantic import BaseModel, Field
from llama_cpp import Llama

# ✅ CBT3 모델 캐시
LLM_CBT3_INSTANCE = {}

def load_cbt3_model(model_path: str) -> Llama:
    global LLM_CBT3_INSTANCE
    if model_path not in LLM_CBT3_INSTANCE:
        print("🚀 CBT3 모델 최초 로딩 중...", flush=True)
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_CBT3_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_ctx=384,  # 🔽 더 작게 줄여서 빠르게
            n_threads=NUM_THREADS,
            n_batch=8,  # 🔼 병렬 토큰 처리
            max_tokens=64,  # 🔽 생성량 제한
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

# ✅ 상태 모델 정의
class AgentState(BaseModel):
    stage: Literal["cbt3", "end"]
    question: str
    response: str
    history: List[str]
    turn: int
    preset_questions: List[str] = Field(default_factory=list)

# ✅ 질문 세트 생성
def generate_preset_questions(llm: Llama) -> List[str]:
    prompt = (
        "너는 따뜻하고 논리적인 CBT 상담자야. 다음 주제에 대해 실천을 유도하는 짧고 직접적인 질문 5개를 번호 없이 쉼표로 나열해줘. "
        "주제: 방해 요인, 감정 변화, 습관 형성, 환경 조정, 피드백 실천."
    )
    result = llm.create_completion(prompt=prompt, max_tokens=128)
    text = result["choices"][0]["text"]
    return [q.strip() for q in text.split(",") if "?" in q][:5]

# ✅ 스트리밍 응답 함수
async def stream_cbt3_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    try:
        llm = load_cbt3_model(model_path)

        if not state.preset_questions:
            state.preset_questions = generate_preset_questions(llm)
            state.turn = 0
            print("✅ CBT3 질문 세트 생성됨")

        # ✅ 현재 질문
        reply = (
            state.preset_questions[state.turn]
            if state.turn < len(state.preset_questions)
            else "지금까지의 대화를 바탕으로 좋은 실천 계획이 세워졌어요."
        )

        # ✅ 상태 전이
        next_turn = state.turn + 1
        next_stage = "end" if next_turn >= 5 else "cbt3"
        next_turn = 0 if next_stage == "end" else next_turn

        if next_stage == "end":
            reply += "\n\n🎯 실천 계획을 잘 정리해주셨어요. 이제 오늘 대화를 마무리할게요."

        yield b"\n" + reply.encode("utf-8")

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "turn": next_turn,
            "response": reply,
            "history": state.history + [state.question, reply],
            "preset_questions": state.preset_questions
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        print(f"⚠️ CBT3 오류: {e}", flush=True)
        fallback = "죄송해요. 지금은 잠시 오류가 발생했어요. 다시 이야기해 주시겠어요?"
        yield fallback.encode("utf-8")
