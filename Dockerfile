FROM python:3.10-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Salin file requirements
COPY requirements.txt .

# Install numpy < 2
RUN pip install --no-cache-dir numpy==1.26.4

# Install torch CPU-only dengan versi yang mendukung transformers terbaru
RUN pip install --no-cache-dir torch==2.2.1+cpu torchvision==0.17.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install dependensi lain
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode ke image
COPY . .

# Jalankan server
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
