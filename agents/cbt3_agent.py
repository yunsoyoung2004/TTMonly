import os, json, multiprocessing, difflib, re
from typing import AsyncGenerator, Literal, List
from pydantic import BaseModel
from llama_cpp import Llama

# ✅ 모델 캐시
LLM_CBT3_INSTANCE = {}

def load_cbt3_model(model_path: str) -> Llama:
    global LLM_CBT3_INSTANCE
    if model_path not in LLM_CBT3_INSTANCE:
        print("🚀 CBT3 모델 최초 로딩 중...", flush=True)
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_CBT3_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_ctx=1024,
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

# ✅ 상태 모델 정의
class AgentState(BaseModel):
    stage: Literal["cbt3", "end"]
    question: str
    response: str
    history: List[str]
    turn: int
    preset_questions: List[str] = []

# ✅ 질문 세트 생성
def generate_preset_questions(llm: Llama) -> List[str]:
    prompt = (
        "너는 따뜻하고 논리적인 CBT 상담자야. 사용자가 실천할 수 있도록 이끌 수 있는 열린 질문 5개를 제안해줘. "
        "질문은 짧고 명확해야 해. 다음 주제를 활용해도 좋아: 방해 요인, 감정 변화, 습관 형성, 환경 조정, 피드백 실천."
    )
    result = llm.create_completion(prompt=prompt, max_tokens=256)
    text = result["choices"][0]["text"]
    questions = re.findall(r"[^.\n!?]*\?", text)
    return [q.strip() for q in questions if q.strip()][:5]

# ✅ 스트리밍 응답 함수
async def stream_cbt3_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    # 무의미한 입력 방지
    if len(user_input) < 2 or re.fullmatch(r"[ㅋㅎ]+", user_input):
        fallback = "좀 더 구체적으로 말씀해 주시겠어요?"
        yield fallback.encode("utf-8")
        return

    try:
        llm = load_cbt3_model(model_path)

        # 질문 세트가 없다면 최초 생성
        if not state.preset_questions:
            state.preset_questions = generate_preset_questions(llm)
            state.turn = 0

        # 현재 턴의 질문
        if state.turn < len(state.preset_questions):
            reply = state.preset_questions[state.turn]
        else:
            reply = "지금까지의 대화를 바탕으로 좋은 실천 계획이 세워졌어요."

        # 전환 처리
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
            "history": state.history + [user_input, reply],
            "preset_questions": state.preset_questions
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        print(f"⚠️ CBT3 오류: {e}", flush=True)
        fallback = "죄송해요. 지금은 잠시 오류가 발생했어요. 다시 이야기해 주시겠어요?"
        yield fallback.encode("utf-8")
