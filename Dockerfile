# Stage 1: Base image menggunakan python slim
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Salin file yang dibutuhkan
COPY requirements.txt .

# Install torch CPU-only terlebih dahulu agar tidak narik dependensi CUDA
RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install dependensi lain
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua source code ke container
COPY . .

# Jalankan server dengan gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
