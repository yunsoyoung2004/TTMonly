import os, json, multiprocessing
from typing import AsyncGenerator, Literal, List
from pydantic import BaseModel
from llama_cpp import Llama

LLM_MI_INSTANCE = {}

def load_mi_model(model_path: str) -> Llama:
    global LLM_MI_INSTANCE
    if model_path not in LLM_MI_INSTANCE:
        try:
            print("🚀 MI 모델 로딩 중...", flush=True)
            LLM_MI_INSTANCE[model_path] = Llama(
                model_path=model_path,
                n_ctx=512,
                n_threads=max(1, multiprocessing.cpu_count() - 1),
                n_batch=4,
                max_tokens=128,
                temperature=0.7,
                top_p=0.85,
                top_k=40,
                repeat_penalty=1.1,
                frequency_penalty=0.7,
                presence_penalty=0.5,
                n_gpu_layers=0,
                low_vram=True,
                use_mlock=False,
                verbose=False,
                chat_format="llama-3",
                stop=["<|im_end|>", "\n\n"]
            )
            print("✅ MI 모델 로드 완료", flush=True)
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}", flush=True)
            raise RuntimeError("MI 모델 로딩 실패")
    return LLM_MI_INSTANCE[model_path]

class AgentState(BaseModel):
    question: str
    response: str
    history: List[str]
    intro_shown: bool = True

def get_mi_prompt() -> str:
    return (
        "당신은 공감적이고 지지적인 상담자입니다.\n"
        "- 감정을 판단 없이 수용하고, 변화 동기를 탐색하세요.\n"
        "- 위로나 충고보다는 공감과 질문으로 대화하세요.\n"
        "- 말투는 존댓말, 응답은 1~2문장으로 짧고 다양하게.\n"
        "- 예: '그때 어떤 감정이 가장 크게 느껴졌나요?', '지금 이 상황에서 가장 힘든 부분은 무엇인가요?'\n"
        "- 예: '그렇다면 지금부터 어떤 행동을 해볼 수 있을까요?', '작은 실천부터 함께 생각해볼까요?'"
    )

async def stream_mi_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    # ✅ 인트로 출력 (초기 진입 시)
    if not state.intro_shown:
        intro = "우선 지금 이 자리에 와주셔서 감사합니다. 어떤 이야기를 나누고 싶으신가요?"
        yield intro.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "mi",
            "response": intro,
            "history": state.history + [intro],
            "intro_shown": True
        }, ensure_ascii=False).encode("utf-8")
        return

    if not user_input or len(user_input) < 2:
        fallback = "조금 더 구체적으로 말씀해주실 수 있을까요?"
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "mi",
            "response": fallback,
            "history": state.history + [user_input, fallback],
            "intro_shown": True
        }, ensure_ascii=False).encode("utf-8")
        return

    try:
        llm = load_mi_model(model_path)
        messages = [{"role": "system", "content": get_mi_prompt()}]
        for i in range(max(0, len(state.history) - 10), len(state.history), 2):
            messages.append({"role": "user", "content": state.history[i]})
            if i + 1 < len(state.history):
                messages.append({"role": "assistant", "content": state.history[i + 1]})
        messages.append({"role": "user", "content": user_input})

        full_response, first_token_sent = "", False
        for chunk in llm.create_chat_completion(messages=messages, stream=True):
            token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if token:
                full_response += token
                if not first_token_sent:
                    yield b"\n"
                    first_token_sent = True
                yield token.encode("utf-8")

        reply = full_response.strip() or "괜찮아요. 마음을 천천히 들려주셔도 괜찮습니다."

        # ✅ 5턴 이상이면 CBT1으로 전환
        turn_count = len(state.history) // 2
        next_stage = "cbt1" if turn_count + 1 >= 5 else "mi"

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "response": reply,
            "history": state.history + [user_input, reply],
            "intro_shown": True
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        print(f"⚠️ 오류 발생: {e}", flush=True)
        fallback = "죄송합니다. 잠시 문제가 발생했어요. 다시 한 번 말씀해 주시겠어요?"
        yield fallback.encode("utf-8")
        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": "mi",
            "response": fallback,
            "history": state.history + [user_input],
            "intro_shown": True
        }, ensure_ascii=False).encode("utf-8")
