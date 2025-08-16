# Multi-stage Docker build for Arabic Dialect Sentiment Analysis
# Stage 1: Backend build
FROM python:3.9-slim as backend

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p logs results

# Stage 2: Frontend build
FROM node:16-alpine as frontend

# Set work directory
WORKDIR /app

# Copy package files
COPY webapp/frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy frontend source code
COPY webapp/frontend/src/ ./src/
COPY webapp/frontend/public/ ./public/

# Build the frontend
RUN npm run build

# Stage 3: Final image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy Python dependencies from backend stage
COPY --from=backend /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=backend /usr/local/bin /usr/local/bin

# Copy backend source code
COPY --from=backend /app/src ./src
COPY --from=backend /app/configs ./configs
COPY --from=backend /app/models ./models
COPY --from=backend /app/data ./data

# Copy frontend build from frontend stage
COPY --from=frontend /app/build ./webapp/frontend/build

# Copy nginx configuration
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Copy startup script
COPY docker/start.sh ./start.sh
RUN chmod +x ./start.sh

# Create necessary directories
RUN mkdir -p logs results

# Expose ports
EXPOSE 8000 80

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["./start.sh"]
