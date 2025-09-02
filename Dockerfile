FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MAX_UPLOAD_MB=300 \
    CORS_ALLOW_ORIGINS=*

# FFmpeg handles all MP4/MOV codecs reliably
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# app
COPY main.py ./

EXPOSE 8000
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 120"]
