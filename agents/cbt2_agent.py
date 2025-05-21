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
        "사용자의 감정과 자동사고를 경청하고, 그 생각을 탐색할 수 있는 하나의 질문만 해.\n"
        "- 질문은 따뜻하고 진심 어린 어조로, 다양한 표현과 주제를 순환하면서 던지세요.\n"
        "- 반드시 하나의 질문만 해주세요.\n"
        "- 사용자의 말을 반복하거나 흉내 내지 마세요.\n"
        "- 같은 문장을 반복하지 마세요."
    )

def is_similar_to_past_response(reply: str, history: List[str]) -> bool:
    recent_responses = [h for i, h in enumerate(history[-10:]) if i % 2 == 1]
    for past in recent_responses:
        ratio = difflib.SequenceMatcher(None, reply[:50], past[:50]).ratio()
        if ratio > 0.8:
            return True
    return False

def contains_user_echo(reply: str, user_input: str) -> bool:
    norm = lambda s: re.sub(r'\s+', '', s.lower())
    return norm(user_input) in norm(reply)

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

        attempt, max_attempts = 0, 3
        final_response = ""
        while attempt < max_attempts:
            full_question = ""
            for chunk in llm.create_chat_completion(messages=messages, stream=True):
                token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if token:
                    full_question += token

            full_question = full_question.strip().rstrip("?.!。！？") + "?"

            if not is_similar_to_past_response(full_question, state.history) and not contains_user_echo(full_question, user_input):
                final_response = full_question
                break
            attempt += 1

        # fallback: 최종적으로도 실패한 경우
        if final_response == "":
            final_response = "혹시 그 생각에 대해 다르게 바라볼 수 있는 시각이 있을까요?"

        next_turn = state.turn + 1
        if next_turn >= 5:
            next_stage = "cbt3"
            next_turn = 0
            updated_history = []
        else:
            next_stage = "cbt2"
            updated_history = state.history + [user_input, final_response]

        yield b"\n" + final_response.encode("utf-8")

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "response": final_response,
            "history": updated_history,
            "turn": next_turn
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        print(f"⚠️ CBT2 오류: {e}", flush=True)
        yield "죄송해요. 다시 한 번 이야기해주실 수 있을까요?".encode("utf-8")
