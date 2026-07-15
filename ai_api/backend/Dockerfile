# Build từ thư mục ai_api (KHÔNG phải ai_api/backend):
#   docker build -f backend/Dockerfile -t rag-api-production .
FROM python:3.10-slim

WORKDIR /app/

RUN apt-get update && apt-get install -y libpq-dev gcc curl && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY llm_client.py /app/llm_client.py
COPY backend/main.py backend/database.py backend/redis_client.py \
     backend/semantic_cache.py backend/background_tasks.py \
     backend/gunicorn.conf.py /app/backend/

ENV PYTHONPATH=/app
WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "main:app"]
