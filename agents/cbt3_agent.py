import os, json, multiprocessing, difflib, re
from typing import AsyncGenerator, Literal, List
from pydantic import BaseModel
from llama_cpp import Llama

LLM_CBT3_INSTANCE = {}

def load_cbt3_model(model_path: str) -> Llama:
    global LLM_CBT3_INSTANCE
    if model_path not in LLM_CBT3_INSTANCE:
        print("🚀 CBT3 모델 최초 로딩 중...", flush=True)
        NUM_THREADS = max(1, multiprocessing.cpu_count() - 1)
        LLM_CBT3_INSTANCE[model_path] = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=NUM_THREADS,
            n_batch=8,
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
            stop=["<|im_end|>"]
        )
        print(f"✅ CBT3 모델 로딩 완료: {model_path}", flush=True)
    return LLM_CBT3_INSTANCE[model_path]

class AgentState(BaseModel):
    stage: Literal["cbt3", "end"]
    question: str
    response: str
    history: List[str]
    turn: int

async def stream_cbt3_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    if len(user_input) < 2 or re.fullmatch(r"[ㅋㅎ]+", user_input):
        fallback = "조금 더 구체적으로 말씀해주실 수 있을까요?"
        yield fallback.encode("utf-8")
        return

    try:
        llm = load_cbt3_model(model_path)

        system_prompt = (
            "너는 따뜻하고 논리적인 CBT 상담자야.\n"
            "너의 목표는 사용자의 상황에 맞는 **실행 가능한 행동 한 가지**를 유도하는 질문을 제시하는 거야.\n"
            "- 반드시 질문은 하나만 해. 하나 이상 하면 안 돼.\n"
            "- 총 응답은 2~3문장 이내여야 해.\n"
            "- 항상 열린 질문으로 마무리하고, 질문 앞에 설명이 오면 안 돼.\n"
            "- 다음 주제 중 한 가지를 선택해 질문해: 방해 요인, 감정 변화, 습관 형성, 환경 조정, 피드백 실천 등.\n"
            "- 같은 질문 구조나 어미, 말투를 반복하지 마. 매 응답은 다르게.\n"
            "- 예시 (금지된 형태): '무엇이 도움이 될까요? 어떤 계획이 좋을까요?'\n"
            "- 예시 (허용된 형태): '어떤 방식으로 시작할 수 있을까요?'"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for i in range(max(0, len(state.history) - 10), len(state.history), 2):
            if i + 1 < len(state.history):
                messages.append({"role": "user", "content": state.history[i]})
                messages.append({"role": "assistant", "content": state.history[i + 1]})
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

        reply = full_response.strip() or "괜찮아요. 지금 떠오르는 작은 아이디어라도 함께 나눠볼 수 있어요."

        # ✅ 질문이 여러 개일 경우 첫 질문만 유지
        questions = re.findall(r"[^.!?]*\?", reply)
        if len(questions) > 1:
            reply = questions[0].strip()

        # ✅ 유사 응답 회피
        for past in state.history[-10:]:
            if isinstance(past, str):
                similarity = difflib.SequenceMatcher(None, reply[:30], past[:30]).ratio()
                if similarity > 0.85:
                    reply += " 이번엔 다른 각도에서 접근해봤어요."
                    break

        next_turn = state.turn + 1
        next_stage = "end" if next_turn >= 5 else "cbt3"
        next_turn = 0 if next_stage == "end" else next_turn

        if next_stage == "end":
            reply += "\n\n🎯 실천 계획을 잘 정리해주셨어요. 이제 오늘 대화를 마무리할게요."

        yield b"\n---END_STAGE---\n" + json.dumps({
            "next_stage": next_stage,
            "turn": next_turn,
            "response": reply,
            "history": state.history + [user_input, reply]
        }, ensure_ascii=False).encode("utf-8")

    except Exception as e:
        print(f"⚠️ CBT3 오류: {e}", flush=True)
        fallback = "죄송해요. 지금은 잠시 오류가 발생했어요. 다시 이야기해 주시겠어요?"
        yield fallback.encode("utf-8")

