import os, json, multiprocessing
from typing import AsyncGenerator, Literal, List, Optional
from pydantic import BaseModel
from llama_cpp import Llama

# ✅ 모델 캐시
LLM_CBT1_INSTANCE = {}

def load_cbt1_model(model_path: str) -> Llama:
    global LLM_CBT1_INSTANCE
    if model_path not in LLM_CBT1_INSTANCE:
        print("🚀 CBT1 모델 로딩 중...", flush=True)
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_CBT1_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_threads=NUM_THREADS,
            n_batch=8,
            max_tokens=128,
            temperature=0.75,
            top_p=0.9,
            presence_penalty=1.0,
            frequency_penalty=0.8,
            n_gpu_layers=0,
            low_vram=True,
            use_mlock=False,
            verbose=False,
            chat_format="llama-3",
            stop=["<|im_end|>"]
        )
        print("✅ CBT1 모델 로딩 완료:", model_path)
    return LLM_CBT1_INSTANCE[model_path]

# ✅ 상태 정의
class AgentState(BaseModel):
    stage: Literal["cbt1", "cbt2"]
    question: str
    response: str
    history: List[str]
    turn: int
    intro_shown: bool
    pending_response: Optional[str] = None

# ✅ 프롬프트 기반 CBT1 응답
async def stream_cbt1_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()
    history = state.history or []

    if not user_input:
        fallback = "떠오른 생각이나 감정이 있다면 부담 없이 이야기해 주세요."
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "cbt1",
            "turn": state.turn,
            "response": fallback,
            "intro_shown": state.intro_shown,
            "history": history
        }, ensure_ascii=False).encode("utf-8")
        return

    try:
        llm = load_cbt1_model(model_path)

        # ✅ 질문 다양화 강조된 system prompt
        system_prompt = (
            "너는 따뜻하고 이성적인 소크라테스 상담자야. 사용자의 자동사고를 탐색해야 해.\n"
            "- 매번 새로운 시각으로 질문을 던져야 해.\n"
            "- 질문은 1~2문장, 존댓말로 마무리해.\n"
            "- 감정, 근거, 결과, 대안사고, 생각의 패턴을 다양하게 유도해.\n"
            "- 예시: "
            "'그 생각이 들었을 때 어떤 감정이 가장 컸나요?', "
            "'그 생각이 사실이라고 느낀 이유는 무엇이었나요?', "
            "'비슷한 상황에서 늘 이런 생각이 드시나요?', "
            "'그 생각을 계속 믿으면 어떤 결과가 생길까요?', "
            "'다른 시각에서 보면 어떤 해석이 가능할까요?', "
            "'친한 친구가 같은 말을 했다면 뭐라고 답했을 것 같나요?'"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # ✅ history를 중복 없이 쌓기
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                messages.append({"role": "user", "content": history[i]})
                messages.append({"role": "assistant", "content": history[i + 1]})

        messages.append({"role": "user", "content": user_input})

        full_response = ""
        first_token_sent = False

        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                full_response += token
                if not first_token_sent:
                    yield b"\n"
                    first_token_sent = True
                yield token.encode("utf-8")

        reply = full_response.strip()
        next_turn = state.turn + 1
        next_stage = "cbt2" if next_turn >= 5 else "cbt1"

        if next_stage == "cbt2":
            reply += "\n\n📘 사고 탐색이 잘 마무리되었어요. 이제 생각을 재구성해보는 CBT2 단계로 넘어갈게요."

        # ✅ history 중복 방지
        updated_history = history.copy()
        if not (len(updated_history) >= 2 and updated_history[-2] == user_input and updated_history[-1] == reply):
            updated_history.extend([user_input, reply])

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "turn": 0 if next_stage == "cbt2" else next_turn,
            "response": reply,
            "question": "",
            "intro_shown": state.intro_shown,
            "history": updated_history
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        err = f"⚠️ 오류 발생: {e}"
        print(err, flush=True)
        yield err.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "cbt1",
            "turn": state.turn,
            "response": "죄송합니다. 오류가 발생했어요. 다시 말씀해 주시겠어요?",
            "intro_shown": state.intro_shown,
            "history": history
        }, ensure_ascii=False).encode("utf-8")
