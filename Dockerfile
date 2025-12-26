FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir yt-dlp

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY backend /app/backend
# Optionally copy only necessary data files
# COPY data/cookies/youtube_cookies.txt /app/data/cookies/

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]