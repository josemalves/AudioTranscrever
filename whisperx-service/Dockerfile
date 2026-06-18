FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Force CPU-only torch via constraints file + the CPU wheel index.
COPY constraints.txt requirements.txt ./
RUN pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -c constraints.txt \
    -r requirements.txt

# Some wheels (ctranslate2 4.4.0) ship .so files marked as needing an executable
# stack, which hardened kernels (recent WSL2) refuse to load. Patch in place.
COPY fix_execstack.py /tmp/fix_execstack.py
RUN python /tmp/fix_execstack.py /usr/local/lib/python3.11/site-packages \
    && rm /tmp/fix_execstack.py

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
