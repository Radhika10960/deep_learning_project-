#!/bin/bash
# Start the FastAPI backend in the background
echo "🚀 Starting FastAPI Backend on port 8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait a moment for backend to initialize
sleep 2

# Start the Streamlit frontend in the foreground
echo "📊 Starting Streamlit Frontend on port 7860..."
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0
