# Stage 1: Build the React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Build the FastAPI backend
FROM python:3.10-slim-bookworm
WORKDIR /app

# Suppress the "Running pip as the 'root' user" warning
ENV PIP_ROOT_USER_ACTION=ignore

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Copy the built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Render uses the PORT environment variable.
# FastAPI will run on this port.
EXPOSE 8000

# Start the FastAPI server
# We use ${PORT:-8000} to respect Render's dynamic port assignment
CMD ["sh", "-c", "python -m uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
