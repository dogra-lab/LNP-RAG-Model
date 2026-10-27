FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

# Tesseract for the OCR fallback on scanned PDFs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only torch keeps the image ~2 GB smaller; use a CUDA base image for GPUs.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
    && pip install -r requirements.txt

COPY lnp_rag ./lnp_rag
COPY config ./config
COPY scripts ./scripts
COPY ui ./ui

ENV PAPERS_DIR=/papers DATA_DIR=/data API_PUBLIC_URL=http://localhost:8000
EXPOSE 8000 8501

CMD ["uvicorn", "lnp_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
