FROM python:3.13.18-slim

# pyttsx3 speaks through espeak-ng on Linux (audio overview)
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUTF8=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
