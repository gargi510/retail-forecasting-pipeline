FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# Step 1: install core numerical stack FIRST
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.16.3 \
    pandas==2.2.0

# Step 2: install everything else WITHOUT re-resolving deps
RUN pip install --no-cache-dir --no-deps -r requirements.txt

COPY . .

CMD ["python", "run_pipeline.py"]
