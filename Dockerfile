FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV, etc.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the startup script is executable
RUN chmod +x start.sh

# Expose Streamlit port (FastAPI runs internally on 8000)
EXPOSE 7860

# Run the unified startup script
CMD ["./start.sh"]
