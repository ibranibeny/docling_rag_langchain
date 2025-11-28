# Secure RAG Chatbot with Image Annotation

A production-ready RAG (Retrieval-Augmented Generation) chatbot with advanced image annotation capabilities, Azure Content Safety integration, and conversation memory.

## 🌟 Features

### Document Processing
- **PDF Upload & Processing**: Upload custom PDFs or use default documents
- **Image Detection**: Automatically detects all images/figures in documents
- **VLM Image Annotation**: Uses Llama 3.2 Vision (Q8 quantized) for detailed 500-1000 word image descriptions
- **Smart Chunking**: HybridChunker with tokenization-aware splitting (1024 tokens)
- **Context Enrichment**: Merges image annotations into document for comprehensive search

### AI & ML
- **Embeddings**: BAAI/bge-base-en-v1.5 (768 dimensions) on CPU
- **Text Generation**: Llama 3.1 8B via Ollama
- **Vision Model**: Llama 3.2 Vision 11B Q8 (optimized for 8GB GPU)
- **Re-ranking**: CrossEncoder ms-marco-MiniLM-L-6-v2 (10→5 results)
- **Vector Store**: ChromaDB with automatic collection management

### Security
- **Azure Content Safety**: Input/output filtering for hate, violence, sexual, self-harm
- **Jailbreak Detection**: Pattern-based prompt injection prevention
- **Rate Limiting**: Automatic blocking after 3 violations
- **Indonesian Language Support**: Deteksi prompt injection dalam Bahasa Indonesia

### User Experience
- **Conversation History**: Remembers last 3 exchanges (6 messages)
- **Streaming Responses**: Real-time answer generation
- **Progress Tracking**: Step-by-step processing feedback
- **Chat Statistics**: Document and conversation metrics

## 📋 Prerequisites

- **Python**: 3.10+
- **GPU**: NVIDIA GPU with 8GB VRAM (e.g., A10) for VLM
- **Ollama**: Installed and running
- **Azure Account**: For Content Safety API

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Navigate to project directory
cd /home/ubuntu/docling-rag/final-chatbot

# Activate virtual environment
source ../docling_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Ollama Models Setup

```bash
# Pull required models
ollama pull llama3.1:8b
ollama pull llama3.2-vision:11b-q8_0

# Verify models
ollama list
```

### 3. Azure Content Safety Setup

```bash
# Set environment variables
export CONTENT_SAFETY_ENDPOINT='your-endpoint-url'
export CONTENT_SAFETY_KEY='your-api-key'

# Or create .env file
cat > .env << EOF
CONTENT_SAFETY_ENDPOINT=your-endpoint-url
CONTENT_SAFETY_KEY=your-api-key
EOF
```

### 4. Run the Application

```bash
# Using run script
chmod +x run_chatbot.sh
./run_chatbot.sh

# Or directly with Streamlit
streamlit run secure_chatbot_with_images.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.maxUploadSize 200
```

Access the application at: `http://localhost:8501`

## 📁 Project Structure

```
final-chatbot/
├── secure_chatbot_with_images.py  # Main application
├── requirements.txt                # Python dependencies
├── run_chatbot.sh                 # Launch script
├── README.md                      # This file
├── CODE_GUIDE.md                  # Code architecture guide
├── FLOW_DIAGRAM.md                # System flow visualization
├── uploaded_pdfs/                 # Uploaded PDF storage (created automatically)
└── chroma_db/                     # Vector database (created automatically)
```

## 🔧 Configuration

### GPU Memory Optimization

The system is optimized for 8GB GPU:
- Embeddings run on CPU to save GPU memory
- Vision model uses Q8 quantization (fits in ~7GB)
- Automatic CUDA memory management

### Model Configuration

Edit in `secure_chatbot_with_images.py`:

```python
# Text LLM
self.llm = Ollama(model="llama3.1:8b", temperature=0.7)

# Vision LLM
self.vision_llm = Ollama(
    model="llama3.2-vision:11b-q8_0", 
    temperature=0.7,
    num_predict=2048
)

# Embeddings
self.embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cpu'}
)
```

### Chunking Configuration

```python
self.chunker = HybridChunker(
    tokenizer="BAAI/bge-base-en-v1.5",
    max_tokens=1024  # Adjust for longer/shorter chunks
)
```

## 📖 Usage Guide

### Upload and Process Document

1. Click "Upload your PDF document" in sidebar
2. Select PDF file (max 200MB)
3. Click "🚀 Process Document"
4. Wait for processing steps:
   - PDF conversion with Docling
   - Image detection and annotation
   - Document chunking
   - Vector store creation

### Ask Questions

1. Type question in chat input
2. System performs:
   - Jailbreak detection
   - Content safety check
   - Semantic search (retrieves 10 chunks)
   - Re-ranking (selects top 5)
   - Answer generation with streaming
   - Output validation

### View Image Annotations

- Expand "🖼️ View X Annotated Images" in main area
- See detailed VLM analysis for each figure

### Reset or Switch Documents

- **Reset Chat**: Clear messages, keep document
- **Clear All & Start Over**: Full reset, upload new document

## 🛠️ Troubleshooting

### CUDA Out of Memory

```bash
# Clear GPU memory
nvidia-smi --query-gpu=index --format=csv,noheader | xargs -I {} nvidia-smi -i {} --gpu-reset

# Or kill Python processes
pkill -9 python
```

### ChromaDB Issues

```bash
# Clear vector database
rm -rf chroma_db/

# Reprocess document
```

### Ollama Connection Error

```bash
# Check Ollama status
systemctl status ollama

# Restart Ollama
sudo systemctl restart ollama

# Verify models
ollama list
```

### Tokenizer Warning

Ignore this warning - it's a false alarm from HybridChunker:
```
Token indices sequence length is longer than the specified maximum...
```

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 16GB
- **GPU**: 8GB VRAM (NVIDIA)
- **Disk**: 10GB free space
- **Network**: Stable internet for Azure API

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 32GB
- **GPU**: 16GB+ VRAM
- **Disk**: 50GB SSD
- **Network**: Low-latency connection

## 🔒 Security Best Practices

1. **Never commit credentials** to version control
2. **Use environment variables** for API keys
3. **Rotate API keys** regularly
4. **Monitor API usage** to detect abuse
5. **Review blocked content logs** periodically

## 📝 Example Questions

```
General:
- What is this document about?
- Summarize the main findings
- What are the key conclusions?

Image-Specific:
- What does Figure 1 show?
- Describe the diagram on page 3
- Explain the workflow illustrated in the images

Technical:
- What AI models are mentioned?
- What is the system architecture?
- How does the process work?
```

## 🤝 Contributing

For contributions, please:
1. Fork the repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request with description

## 📄 License

MIT License - See repository for details

## 🙏 Acknowledgments

- **Docling**: Document processing framework
- **LangChain**: RAG orchestration
- **Ollama**: Local LLM inference
- **ChromaDB**: Vector database
- **Azure**: Content Safety API
- **Streamlit**: UI framework

## 📞 Support

For issues or questions:
- Check [CODE_GUIDE.md](CODE_GUIDE.md) for architecture details
- Review [FLOW_DIAGRAM.md](FLOW_DIAGRAM.md) for system flow
- Check GitHub Issues for known problems

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Maintained By**: Your Team
