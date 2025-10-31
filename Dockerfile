# Use Python 3.10 (DeepFace + TF 2.15 are stable here)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV + Pillow
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy rest of the project
COPY . .

# Expose port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
