FROM python:3.11-slim

# ---------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Limit numerical-library CPU usage.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

WORKDIR /app

# ---------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    --disable-pip-version-check \
    -r requirements.txt

# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

COPY main.py .

# Model is bundled inside the image.
COPY models ./models

# Cloud Run provides PORT automatically.
ENV PORT=8080

# ---------------------------------------------------------
# Start API
# ---------------------------------------------------------

CMD exec uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers 1
