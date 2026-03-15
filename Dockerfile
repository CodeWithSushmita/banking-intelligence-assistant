FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY agents/ ./agents/
COPY data/ ./data/
COPY vectorstore/ ./vectorstore/
COPY app.py .
COPY ingest.py .
COPY .env.example .

# Expose Streamlit port
EXPOSE 7860

# Run ingest first if vectorstore is empty, then launch app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]