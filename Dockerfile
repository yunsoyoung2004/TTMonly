FROM gcr.io/ttmchatbotbot/ttmchatbot:latest

WORKDIR /app

# requirements 설치
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install llama-cpp-python==0.3.8 --no-cache-dir --config-settings=cmake.define.LLAMA_CUBLAS=OFF

# ✅ 변경된 소스코드 반영
COPY main.py ./main.py
COPY agents/ ./agents
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]