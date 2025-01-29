FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model files and code
COPY . .
RUN mkdir -p templates
COPY templates/* templates/

# Expose the port
EXPOSE 5000

# Run the server
CMD ["python", "server.py"] 