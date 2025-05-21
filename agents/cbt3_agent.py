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
            n_ctx=2048,  # ✅ context window 확장
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

# ✅ 상태 정의
class AgentState(BaseModel):
    stage: Literal["cbt3", "end"]
    question: str
    response: str
    history: List[str]
    turn: int

# ✅ CBT3 응답 함수
async def stream_cbt3_reply(state: AgentState, model_path: str) -> AsyncGenerator[bytes, None]:
    user_input = state.question.strip()

    # ✅ 무의미한 입력 걸러내기
    if len(user_input) < 2 or re.fullmatch(r"[ㅋㅎ]+", user_input):
        fallback = "조금 더 구체적으로 말씀해주실 수 있을까요?"
        yield fallback.encode("utf-8")
        return

    try:
        llm = load_cbt3_model(model_path)

        # ✅ 시스템 프롬프트
        system_prompt = (
            "너는 따뜻하고 논리적인 CBT 상담자야.\n"
            "- 사용자의 목표나 상황에 맞춰 실천 행동을 도와주는 역할이야.\n"
            "- 반드시 **하나의 질문만** 포함하고, 전체는 2~3문장 구성으로 말해줘.\n"
            "- 단정적인 말투 대신 열린 질문으로 유도해.\n"
            "- 방해 요소, 감정 변화, 피드백, 환경 설정, 습관 형성 등 다양한 주제를 활용해.\n"
            "- 같은 문장 구조, 말투, 표현 반복 금지.\n"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for i in range(max(0, len(state.history) - 10), len(state.history), 2):
            if i + 1 < len(state.history):
                messages.append({"role": "user", "content": state.history[i]})
                messages.append({"role": "assistant", "content": state.history[i + 1]})
        messages.append({"role": "user", "content": user_input})

        # ✅ 스트리밍 생성
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
