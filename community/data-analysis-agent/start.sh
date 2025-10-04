#!/bin/bash

# Startup script for Data Analysis Agent (Next.js version)

echo "🚀 Starting Data Analysis Agent..."

# Check if NVIDIA_API_KEY is set
if [ -z "$NVIDIA_API_KEY" ]; then
    echo "⚠️  WARNING: NVIDIA_API_KEY environment variable is not set!"
    echo "Please set it with: export NVIDIA_API_KEY='your_key_here'"
    exit 1
fi

# Start backend in background
echo "📡 Starting FastAPI backend on port 8000..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend
echo "🎨 Starting Next.js frontend on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Both servers are starting!"
echo "📊 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
