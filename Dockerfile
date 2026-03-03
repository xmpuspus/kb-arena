FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY kb_arena/ kb_arena/
COPY cypher/ cypher/
COPY datasets/ datasets/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "kb_arena.chatbot.api:app", "--host", "0.0.0.0", "--port", "8000"]
