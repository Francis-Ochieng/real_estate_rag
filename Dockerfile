# Use slim python base
FROM python:3.11-slim

# Avoid writing .pyc and buffer issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (faiss requires build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy code
COPY . /app

# Install dependencies
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
