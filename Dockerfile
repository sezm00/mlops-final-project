# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system deps needed for some ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="Member 3 — MLOps Team"
LABEL description="Telco Churn Prediction API"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY monitoring/ ./monitoring/
COPY configs/ ./configs/

# Create non-root user for security
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app

USER appuser

# Expose API port
EXPOSE 8000

# Health check (Docker native)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Environment defaults (override at runtime)
ENV MLFLOW_TRACKING_URI="http://mlflow:5000"
ENV MLFLOW_MODEL_NAME="churn-model"
ENV MLFLOW_MODEL_STAGE="Production"
ENV MLFLOW_MODEL_PATH="model.pkl"

# Start FastAPI with uvicorn
CMD ["uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
