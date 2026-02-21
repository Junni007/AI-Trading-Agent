# Build Stage
FROM python:3.10-slim as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final Stage
FROM python:3.10-slim

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/appuser/.local/bin:$PATH

# Copy installed packages from builder to appuser's home
COPY --from=builder /root/.local /home/appuser/.local
RUN chown -R appuser:appuser /home/appuser/.local

# Copy source code (use .dockerignore to exclude heavy files)
COPY --chown=appuser:appuser src/ src/

# Copy checkpoint if it exists (optional — build won't fail if missing)
COPY --chown=appuser:appuser checkpoints/best_ppo_light.np[z] checkpoints/

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check — verify API is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
