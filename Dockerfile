ARG PYTHON_BASE_IMAGE=python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}

ARG INSTALL_CUDA=0

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PAPER2ANY_RUNTIME_TMPDIR=/app/outputs/system/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    inkscape \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libsndfile1 \
    libxext6 \
    libxrender1 \
    libreoffice \
    poppler-utils \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && wget -q https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6.1-3/wkhtmltox_0.12.6.1-3.bookworm_amd64.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends ./wkhtmltox_0.12.6.1-3.bookworm_amd64.deb \
    && rm wkhtmltox_0.12.6.1-3.bookworm_amd64.deb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-base.txt requirements-paper.txt requirements-paper-backup.txt requirements-cu12.txt ./

RUN pip install --upgrade pip && \
    pip install -r requirements-paper.txt && \
    if [ "$INSTALL_CUDA" = "1" ]; then pip install -r requirements-cu12.txt; fi

COPY . .

RUN pip install -e . && \
    mkdir -p /app/outputs/system/tmp /app/models /app/logs /app/data /app/database /app/raw_data_store /app/rebuttal_sessions

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
