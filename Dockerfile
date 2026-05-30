# =============================================================================
# Dockerfile
# PURPOSE : Package the entire app into a portable container image.
# BUILD   : docker build -t spam-classifier .
# RUN     : docker run -p 8000:8000 spam-classifier
# REQUIRES: model.pkl must exist before you run docker build
#           (run: python train.py  FIRST)
# WEEK    : Weeks 3–4 — Docker
# =============================================================================

# -----------------------------------------------------------------------------
# Base image
# -----------------------------------------------------------------------------
# python:3.11-slim is a minimal Debian image with Python pre-installed.
# It is ~50MB vs ~900MB for the full python:3.11 image.
# Always pin a specific version (3.11-slim, not just 3-slim) so builds
# are reproducible — "latest" can change and break your pipeline.
FROM python:3.11-slim

# -----------------------------------------------------------------------------
# Set working directory
# -----------------------------------------------------------------------------
# All subsequent commands (COPY, RUN, CMD) operate relative to /app.
# This is the directory inside the container where your code will live.
WORKDIR /app

# -----------------------------------------------------------------------------
# Install dependencies BEFORE copying source code
# -----------------------------------------------------------------------------
# Docker builds in layers. Each instruction is cached separately.
# By copying requirements.txt FIRST and installing before copying source code,
# the expensive pip install layer is only re-run when requirements.txt changes —
# not every time you edit main.py. This makes rebuilds much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Copy source code and trained model
# -----------------------------------------------------------------------------
# model.pkl is copied from your local machine INTO the image.
# This means the trained model is baked into the image — no retraining needed.
# Anyone who runs this image gets the exact same model.
COPY main.py .
COPY model.pkl .

# -----------------------------------------------------------------------------
# Expose port 8000
# -----------------------------------------------------------------------------
# This is documentation only — it does NOT publish the port.
# You still need -p 8000:8000 when running: docker run -p 8000:8000 ...
EXPOSE 8000

# -----------------------------------------------------------------------------
# Start the API server
# -----------------------------------------------------------------------------
# --host 0.0.0.0 is REQUIRED inside Docker.
# Without it, uvicorn binds to 127.0.0.1 (localhost inside the container)
# and the port is unreachable from outside the container.
# 0.0.0.0 means "listen on all network interfaces" — including the one
# Docker maps to your host machine's port.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
