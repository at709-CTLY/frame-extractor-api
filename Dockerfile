FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MAX_UPLOAD_MB=300 \
    CORS_ALLOW_ORIGINS=*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py ./

EXPOSE 8000
# Use $PORT if provided by the host (Render/Railway/Cloud Run), else 8000
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 120"]
