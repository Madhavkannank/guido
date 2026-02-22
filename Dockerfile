# ─── UI Builder stage ─────────────────────────────────────────────────────────
FROM node:20-slim AS ui-builder

WORKDIR /ui

COPY ui/package.json ui/package-lock.json* ./
RUN npm ci --prefer-offline

COPY ui/ .

# Empty VITE_API_BASE → relative URLs → works with any domain / IP
ARG VITE_API_BASE=
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

# ─── Python Builder stage ─────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY . .

# Copy pre-built React UI from the ui-builder stage
COPY --from=ui-builder /ui/dist /app/ui/dist

# Create necessary directories
RUN mkdir -p data/raw data/processed models_artifacts results mlruns

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# FastAPI serves both API (/api/*) and the built React UI (/)
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start Uvicorn
# Start Uvicorn — uses $PORT if set (Render injects it), falls back to 8000
CMD ["sh", "-c", "python -m uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level info"]
