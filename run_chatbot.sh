#!/bin/bash

# run_chatbot.sh
# Launch script for Secure RAG Chatbot with Image Annotation

set -e

echo "==================================="
echo "Secure RAG Chatbot Launcher"
echo "==================================="
echo ""

# Check if virtual environment exists
if [ ! -d "../docling_env" ]; then
    echo "❌ Error: Virtual environment not found at ../docling_env"
    echo "Please create the environment first:"
    echo "  python3 -m venv ../docling_env"
    echo "  source ../docling_env/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source ../docling_env/bin/activate

# Check for Azure credentials
if [ -z "$CONTENT_SAFETY_ENDPOINT" ] || [ -z "$CONTENT_SAFETY_KEY" ]; then
    echo ""
    echo "⚠️  Warning: Azure Content Safety credentials not set"
    echo ""
    echo "Please set environment variables:"
    echo "  export CONTENT_SAFETY_ENDPOINT='your-endpoint-url'"
    echo "  export CONTENT_SAFETY_KEY='your-api-key'"
    echo ""
    echo "Or create .env file with:"
    echo "  CONTENT_SAFETY_ENDPOINT=your-endpoint-url"
    echo "  CONTENT_SAFETY_KEY=your-api-key"
    echo ""
    read -p "Continue without Content Safety? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Ollama status
echo "✓ Checking Ollama service..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Error: Ollama not found"
    echo "Please install Ollama from https://ollama.ai"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Warning: Ollama service may not be running"
    echo "Starting Ollama..."
    # Try to start Ollama in background if available
    if command -v systemctl &> /dev/null; then
        sudo systemctl start ollama 2>/dev/null || true
    fi
fi

# Check required models
echo "✓ Checking Ollama models..."
REQUIRED_MODELS=("llama3.1:8b" "llama3.2-vision:11b-q8_0")
MISSING_MODELS=()

for model in "${REQUIRED_MODELS[@]}"; do
    if ! ollama list | grep -q "$model"; then
        MISSING_MODELS+=("$model")
    fi
done

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Missing Ollama models:"
    for model in "${MISSING_MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "Download them with:"
    for model in "${MISSING_MODELS[@]}"; do
        echo "  ollama pull $model"
    done
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check GPU availability
echo "✓ Checking GPU status..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected")
    echo "  GPU: $GPU_INFO"
else
    echo "  ⚠️  nvidia-smi not found, running without GPU acceleration"
fi

# Suppress tokenizer warning
export TOKENIZERS_PARALLELISM=false

# Create necessary directories
echo "✓ Creating directories..."
mkdir -p uploaded_pdfs
mkdir -p chroma_db

# Get network interfaces for display
HOST_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "==================================="
echo "Starting Streamlit Application"
echo "==================================="
echo ""
echo "Access URLs:"
echo "  Local:   http://localhost:8501"
echo "  Network: http://${HOST_IP}:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit
streamlit run secure_chatbot_with_images.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.maxUploadSize 200 \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none
