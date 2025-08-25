# Use an official Python 3.10 image
FROM python:3.10-slim

# Prevent Python from writing pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies required for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose FastAPI port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
