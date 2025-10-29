FROM python:3.11-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /app
ENTRYPOINT ["python"]
