FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub

COPY src/ src/
COPY web/ web/
COPY scripts/ scripts/

ARG HF_TOKEN
ARG HF_REPO_ID=YOUR_USERNAME/esl-model
ENV HF_TOKEN=$HF_TOKEN
ENV HF_REPO_ID=$HF_REPO_ID
RUN python scripts/download_model.py

EXPOSE 8000
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]