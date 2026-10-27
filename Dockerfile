# ---------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------
# Runs the Streamlit app inside a container.
# Ollama must run as a SEPARATE container (see docker-compose.yml).
#
# Build:   docker build -t research-assistant .
# Run:     docker run -p 8501:8501 research-assistant
# ---------------------------------------------------------------

    FROM python:3.11-slim

    WORKDIR /app

    # Install system dependencies needed by PyMuPDF and OCR
    RUN apt-get update && apt-get install -y \
        libmupdf-dev \
        tesseract-ocr \
        && rm -rf /var/lib/apt/lists/*

    # Install Python dependencies first (layer caching)
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy source code
    COPY . .

    # Create data directories
    RUN mkdir -p data/pdfs data/faiss_index

    # Streamlit default port
    EXPOSE 8501

    # Streamlit config: disable file watcher in production
    ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

    CMD ["streamlit", "run", "app.py", \
         "--server.port=8501", \
         "--server.address=0.0.0.0", \
         "--server.headless=true"]
