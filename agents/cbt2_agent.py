import os, json, multiprocessing, difflib
from typing import AsyncGenerator, List
from pydantic import BaseModel
from llama_cpp import Llama

# ✅ CBT2 모델 캐시
LLM_CBT2_INSTANCE = {}

def load_cbt2_model(model_path: str) -> Llama:
    global LLM_CBT2_INSTANCE
    if model_path not in LLM_CBT2_INSTANCE:
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_CBT2_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_threads=NUM_THREADS,
            n_batch=4,
            max_tokens=128,
            temperature=0.6,
            top_p=0.85,
            repeat_penalty=1.1,
            n_gpu_layers=0,
            low_vram=True,
            use_mlock=False,
            verbose=False,
            chat_format="llama-3",
            stop=["<|im_end|>", "\n\n"]
        )
    return LLM_CBT2_INSTANCE[model_path]

# ✅ 상태 모델 정의
class AgentState(BaseModel):
    question: str
    response: str
    history: List[str]
    turn: int
    topic_index: int = 0

# ✅ 시스템 프롬프트
def get_cbt2_prompt() -> str:
    return (
        "너는 인지 재구조화를 도와주는 전문 CBT 상담자야.\n"
        "사용자와 함께 자동사고를 탐색하고, 그 생각을 다양한 관점에서 바라볼 수 있게 도와줘.\n"
        "- 반드시 하나의 질문만 하세요.\n"
        "- 매번 표현을 다르게 바꾸고, 질문 구조를 반복하지 마세요.\n"
        "- 다음 주제를 순환하며 질문하세요:\n"
        "  1. 감정\n"
        "  2. 사실/증거\n"
        "  3. 반복된 패턴\n"
        "  4. 장기적 영향\n"
        "  5. 대안적 해석\n"
        "  6. 타인의 시각\n"
        "  7. 가치/신념\n"
        "  8. 긍정적 가능성\n"
        "  9. 과거 경험\n"
        "  10. 개인의 강점\n"
        "- 정중하고 부드러운 어조로 마무리하세요."
    )

# ✅ CBT2 응답 함수
async def stream_cbt2_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()
    if not user_input:
        fallback = "조금 더 구체적으로 이야기해주실 수 있을까요?"
        yield fallback.encode("utf-8")
        return

    try:
        llm = load_cbt2_model(model_path)

        # ✅ 메시지 준비
        messages = [{"role": "system", "content": get_cbt2_prompt()}]
        for i in range(max(0, len(state.history) - 10), len(state.history), 2):
            if i + 1 < len(state.history):
                messages.append({"role": "user", "content": state.history[i]})
                messages.append({"role": "assistant", "content": state.history[i + 1]})

        topic_hint = f"[주제 {state.topic_index + 1}]"
        messages.append({"role": "user", "content": f"{topic_hint} {user_input}"})

        full_response, first_token_sent = "", False
        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if token:
                full_response += token
                if not first_token_sent:
                    yield b"\n"
                    first_token_sent = True
                yield token.encode("utf-8")

        reply = full_response.strip() or "괜찮습니다. 천천히 정리해서 말씀해주셔도 괜찮아요."

        # ✅ 반복 회피
        for past in state.history[-10:]:
            if isinstance(past, str) and difflib.SequenceMatcher(None, reply[:40], past[:40]).ratio() > 0.8:
                reply += " 이번에는 다른 각도로 접근해봤어요."
                break

        # ✅ 출력 JSON
        next_turn = state.turn + 1
        next_topic = (state.topic_index + 1) % 10
        next_stage = "cbt3" if next_turn >= 5 else "cbt2"

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "response": reply,
            "history": state.history + [user_input, reply],
            "turn": 0 if next_stage == "cbt3" else next_turn,
            "topic_index": next_topic
        }, ensure_ascii=False).encode("utf-8")

    except Exception:
        fallback = "죄송해요. 다시 한 번 이야기해주실 수 있을까요?"
        yield fallback.encode("utf-8")
