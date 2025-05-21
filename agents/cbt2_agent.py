import os, json, multiprocessing, difflib, random, re
from typing import AsyncGenerator, List
from pydantic import BaseModel
from llama_cpp import Llama

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
            temperature=0.65,
            top_p=0.9,
            repeat_penalty=1.1,
            n_gpu_layers=0,
            low_vram=True,
            use_mlock=False,
            verbose=False,
            chat_format="llama-3",
            stop=["<|im_end|>", "\n\n"]
        )
    return LLM_CBT2_INSTANCE[model_path]

class AgentState(BaseModel):
    question: str
    response: str
    history: List[str]
    turn: int

def get_cbt2_prompt() -> str:
    return (
        "너는 인지 재구조화를 도와주는 전문 CBT 상담자야.\n"
        "사용자와 함께 자동사고를 탐색하고, 그 생각을 다양한 관점에서 바라볼 수 있게 도와줘.\n"
        "- 반드시 하나의 질문만 하세요.\n"
        "- 매번 표현을 바꾸고, 질문 구조를 반복하지 마세요.\n"
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
        "- 정중하고 따뜻한 어조로 마무리하세요.\n"
        "- 질문 문장의 어투와 구조는 매번 달라야 합니다."
    )

def is_similar_to_past_response(reply: str, history: List[str]) -> bool:
    recent_responses = [h for i, h in enumerate(history[-10:]) if i % 2 == 1]
    for past in recent_responses:
        ratio = difflib.SequenceMatcher(None, reply[:50], past[:50]).ratio()
        if ratio > 0.8:
            return True
    return False

async def stream_cbt2_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()
    if len(user_input) < 2 or re.fullmatch(r"[ㅋㅎㅠㅜ]+", user_input):
        yield "조금 더 구체적으로 이야기해주실 수 있을까요?".encode("utf-8")
        return

    try:
        llm = load_cbt2_model(model_path)
        messages = [{"role": "system", "content": get_cbt2_prompt()}]
        for i in range(max(0, len(state.history) - 10), len(state.history), 2):
            if i + 1 < len(state.history):
                messages.append({"role": "user", "content": state.history[i]})
                messages.append({"role": "assistant", "content": state.history[i + 1]})
        messages.append({"role": "user", "content": user_input})

        # ✅ 최대 2회 재시도
        reply = ""
        max_attempts = 2
        attempt = 0
        while attempt < max_attempts:
            full_response, first_token_sent = "", False
            for chunk in llm.create_chat_completion(messages=messages, stream=True):
                token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if token:
                    full_response += token
                    if not first_token_sent:
                        yield b"\n"
                        first_token_sent = True
                    yield token.encode("utf-8")

            reply = full_response.strip()
            if reply and not is_similar_to_past_response(reply, state.history):
                break
            attempt += 1
            reply += random.choice([
                " 이번엔 다른 관점으로 여쭤볼게요.",
                " 표현을 조금 다르게 해봤어요.",
                " 새로운 방식으로 질문드릴게요."
            ])

        next_turn = state.turn + 1
        if next_turn >= 5:
            next_stage = "cbt3"
            next_turn = 0
            updated_history = []
        else:
            next_stage = "cbt2"
            updated_history = state.history + [user_input, reply]

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "response": reply,
            "history": updated_history,
            "turn": next_turn
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        print(f"⚠️ CBT2 오류: {e}", flush=True)
        yield "죄송해요. 다시 한 번 이야기해주실 수 있을까요?".encode("utf-8")
