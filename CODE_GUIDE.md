# Code Architecture Guide

## 📚 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Class Structure](#class-structure)
4. [Data Flow](#data-flow)
5. [Key Components](#key-components)
6. [Code Walkthrough](#code-walkthrough)

---

## Overview

This application is a **Secure RAG (Retrieval-Augmented Generation) Chatbot** with three core capabilities:

1. **Document Processing**: Converts PDFs to searchable chunks with Docling
2. **Image Annotation**: Uses Vision LLM to create detailed image descriptions
3. **Secure Q&A**: Answers questions with Azure Content Safety filtering

**Technology Stack:**
- **Frontend**: Streamlit (lines 676-956)
- **Backend Logic**: Python classes (lines 44-674)
- **Document Processing**: Docling 2.63.0
- **Vector DB**: ChromaDB 1.3.5
- **LLMs**: Ollama (llama3.1:8b, llama3.2-vision:11b)
- **Security**: Azure Content Safety API

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI LAYER                       │
│  (secure_chatbot_with_images.py: lines 676-956)            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Sidebar    │  │  Main Area   │  │  Chat Input  │     │
│  │  - Upload    │  │  - Messages  │  │  - User Q's  │     │
│  │  - Process   │  │  - Images    │  │  - Streaming │     │
│  │  - Stats     │  │  - Progress  │  │  - History   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION CLASSES                       │
│  (secure_chatbot_with_images.py: lines 44-674)             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         ContentSafetyGuard (lines 44-130)            │  │
│  │  - check_content(): Azure API validation            │  │
│  │  - detect_jailbreak(): Pattern-based detection      │  │
│  │  - Severity thresholds: 0, 2, 4 for different types │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          ImageAnnotator (lines 133-357)              │  │
│  │  - describe_image_with_vlm(): Vision LLM inference  │  │
│  │  - annotate_images_in_document(): Batch processing  │  │
│  │  - 7-section structured prompt (500-1000 words)     │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │    SecureChatbotRAGWithImages (lines 360-674)        │  │
│  │  - process_document(): PDF → Chunks → VectorDB      │  │
│  │  - stream_response(): Question → Answer + Safety    │  │
│  │  - HybridChunker, BGE embeddings, CrossEncoder      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   EXTERNAL SERVICES                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Ollama     │  │  ChromaDB    │  │    Azure     │     │
│  │  - Text LLM  │  │  - Vectors   │  │  - Content   │     │
│  │  - Vision    │  │  - Search    │  │    Safety    │     │
│  │    LLM       │  │  - Persist   │  │  - API       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Class Structure

### 1. ContentSafetyGuard (Lines 44-130)

**Purpose**: Pre-validation and post-validation of all text content

**Key Methods:**

```python
def __init__(self, endpoint: str, key: str):
    """
    Initialize Azure Content Safety client
    - Creates AnalyzeTextOptions with language="id" (Indonesian support)
    - Sets up ContentSafetyClient with AzureKeyCredential
    """

def detect_jailbreak(self, prompt: str) -> tuple[bool, str]:
    """
    Pattern-based jailbreak detection
    - Checks 15 patterns (English + Indonesian)
    - Examples: "ignore previous", "abaikan instruksi"
    - Returns: (is_jailbreak, matched_pattern)
    """

def check_content(self, text: str) -> tuple[bool, dict]:
    """
    Azure Content Safety API validation
    - Categories: Hate, Sexual, Violence, SelfHarm
    - Thresholds: [0, 2, 4, 4] for each category
    - Returns: (is_safe, analysis_dict)
    """
```

**Data Flow:**
```
User Input → detect_jailbreak() → check_content() → Safe/Blocked
Generated Answer → check_content() → Safe/Blocked
```

**Configuration:**
- Hate: 0 tolerance (block all)
- Sexual: 2 (block medium+)
- Violence: 4 (block severe only)
- SelfHarm: 4 (block severe only)

---

### 2. ImageAnnotator (Lines 133-357)

**Purpose**: Extract images from PDFs and generate detailed descriptions with Vision LLM

**Key Methods:**

```python
def __init__(self, vision_model: str = "llama3.2-vision:11b-q8_0"):
    """
    Initialize Ollama vision client
    - Model: llama3.2-vision:11b-q8_0 (Q8 quantized for 8GB GPU)
    - Temperature: 0.7 (creative but focused)
    - num_predict: 2048 (long detailed responses)
    """

def describe_image_with_vlm(self, image_path: str) -> str:
    """
    Generate 500-1000 word structured analysis
    - Loads image as base64
    - 7-section prompt:
      1. Overview & Content (what is shown)
      2. Visual Elements (colors, shapes, layout)
      3. Text Content (labels, annotations)
      4. Technical Details (data, measurements)
      5. Context & Purpose (why it exists)
      6. Key Insights (important findings)
      7. Relevance (how it connects to document)
    - Returns: Markdown-formatted description
    """

def annotate_images_in_document(self, doc: DoclingDocument) -> dict:
    """
    Batch process all images in document
    - Extracts PictureItems from doc.pictures
    - Saves each image to temp file
    - Calls describe_image_with_vlm() for each
    - Returns: {index: {uri, description}} dictionary
    - Handles errors gracefully (continues on failure)
    """
```

**7-Section Prompt Structure:**
```
1. OVERVIEW: High-level summary
2. VISUAL ELEMENTS: Colors, shapes, composition
3. TEXT CONTENT: Labels, legends, annotations  
4. TECHNICAL DETAILS: Data points, measurements
5. CONTEXT: Document relevance
6. KEY INSIGHTS: Important findings
7. RELEVANCE: Relationship to main content
```

**Image Processing Flow:**
```
DoclingDocument → Extract Pictures → Save to File → VLM Inference → 
Structured Description → Merge into Document Text
```

---

### 3. SecureChatbotRAGWithImages (Lines 360-674)

**Purpose**: Main RAG orchestration with document processing and Q&A

**Initialization (Lines 360-409):**

```python
def __init__(self):
    # LLM Setup
    self.llm = Ollama(model="llama3.1:8b", temperature=0.7)
    
    # Embeddings (CPU to save GPU for VLM)
    self.embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    
    # Re-ranker (10 results → 5 best)
    self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Chunker (tokenizer-aligned, 1024 tokens)
    self.chunker = HybridChunker(
        tokenizer="BAAI/bge-base-en-v1.5",
        max_tokens=1024
    )
    
    # Docling pipeline (images at 2x scale)
    self.pipeline_options = PdfPipelineOptions(
        images_scale=2.0,
        generate_picture_images=True
    )
    
    # Content safety & image annotation
    self.content_guard = ContentSafetyGuard(endpoint, key)
    self.image_annotator = ImageAnnotator()
```

**Key Methods:**

#### process_document() (Lines 411-523)

**Purpose**: Convert PDF to searchable vector database

```python
def process_document(self, pdf_path: str) -> dict:
    """
    Full document processing pipeline
    
    Steps:
    1. Delete existing ChromaDB collection (fresh start)
    2. Convert PDF with Docling
    3. Detect and annotate images with VLM
    4. Merge annotations into document text
    5. Chunk document with HybridChunker
    6. Create ChromaDB vector store
    7. Set up RAG chain
    
    Returns: {
        'num_chunks': int,
        'num_images': int,
        'annotations': {index: {uri, description}}
    }
    """
```

**Detailed Flow:**

```
┌─────────────────┐
│   Upload PDF    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Delete Old ChromaDB Collection      │ ← Fresh start
│ (if exists)                          │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Docling Conversion                   │
│ - PdfPipelineOptions                 │
│ - images_scale=2.0                   │
│ - generate_picture_images=True       │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Image Detection                      │
│ - Extract doc.pictures               │
│ - Count PictureItems                 │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ VLM Image Annotation                 │
│ - For each image:                    │
│   * Save to temp file                │
│   * Call Vision LLM                  │
│   * Generate 500-1000 word desc      │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Merge Annotations                    │
│ - Insert descriptions near images    │
│ - Format: "IMAGE X DESCRIPTION: ..." │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ HybridChunker.chunk()                │
│ - Tokenizer: bge-base-en-v1.5        │
│ - max_tokens: 1024                   │
│ - Returns: List[BaseChunk]           │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Contextualize Chunks                 │
│ - HybridChunker.serialize()          │
│ - Adds document context              │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Create ChromaDB VectorStore          │
│ - Embed with BGE-base (768 dims)     │
│ - Store in ./chroma_db/              │
│ - Collection: auto-generated name    │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Setup RAG Chain                      │
│ - Retriever: similarity_search       │
│ - ChatPromptTemplate with history    │
│ - Ollama LLM: llama3.1:8b            │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Return Statistics                    │
│ - num_chunks, num_images             │
│ - annotations dict                   │
└─────────────────────────────────────┘
```

#### stream_response() (Lines 525-674)

**Purpose**: Answer questions with safety checks and re-ranking

```python
def stream_response(self, question: str, chat_history: List[Tuple[str, str]]):
    """
    Secure Q&A with streaming
    
    Steps:
    1. Jailbreak detection on question
    2. Content safety check on question
    3. Retrieve 10 candidate chunks
    4. Re-rank to top 5 with CrossEncoder
    5. Format chat history (last 3 exchanges)
    6. Generate answer with streaming
    7. Content safety check on answer
    8. Yield tokens or error messages
    
    Yields: str tokens for streaming display
    """
```

**Detailed Flow:**

```
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Jailbreak Detection                  │
│ - Pattern matching (15 patterns)     │
│ - English + Indonesian               │
│ - Blocks: ignore instructions, etc   │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Azure Content Safety (Input)         │
│ - Categories: Hate/Sexual/Violence   │
│ - Thresholds: [0, 2, 4, 4]           │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Vector Search                        │
│ - Similarity search: k=10            │
│ - BGE-base embeddings                │
│ - ChromaDB retrieval                 │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Re-ranking                           │
│ - CrossEncoder scoring               │
│ - Sort by relevance                  │
│ - Select top 5 chunks                │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Format Chat History                  │
│ - Last 3 exchanges (6 messages)      │
│ - Format: "Human: ...\nAI: ..."      │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Prompt Template                      │
│ - System: "helpful AI assistant"     │
│ - Includes: {chat_history}           │
│ - Includes: {context} (5 chunks)     │
│ - Includes: {question}               │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ LLM Streaming                        │
│ - Ollama llama3.1:8b                 │
│ - Temperature: 0.7                   │
│ - Yields tokens in real-time         │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Azure Content Safety (Output)        │
│ - Same thresholds as input           │
│ - Blocks unsafe generations          │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Display Answer                       │
│ - Streaming in chat UI               │
│ - Or error message if blocked        │
└─────────────────────────────────────┘
```

---

## Data Flow

### Complete User Journey

```
┌──────────────────────────────────────────────────────────────┐
│                      1. DOCUMENT UPLOAD                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  User uploads PDF → Streamlit saves to uploaded_pdfs/        │
│  Triggers: process_document() button click                   │
│                                                               │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                   2. DOCUMENT PROCESSING                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Docling Conversion (5-20 seconds)                    │    │
│  │ - PDF → DoclingDocument object                       │    │
│  │ - Extract text, tables, figures                      │    │
│  │ - images_scale=2.0 for quality                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Image Annotation (2-5 seconds per image)            │    │
│  │ - Extract doc.pictures                               │    │
│  │ - Vision LLM inference                               │    │
│  │ - Generate 500-1000 word descriptions                │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Text Merging (instant)                               │    │
│  │ - Insert image descriptions into document text       │    │
│  │ - Format: "IMAGE X DESCRIPTION: [detail]"            │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Chunking (2-5 seconds)                               │    │
│  │ - HybridChunker with tokenizer alignment             │    │
│  │ - max_tokens=1024 per chunk                          │    │
│  │ - Contextualize with document metadata               │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Vector Embedding (5-10 seconds for 100 chunks)       │    │
│  │ - BGE-base-en-v1.5 (768 dimensions)                  │    │
│  │ - CPU inference (saves GPU for VLM)                  │    │
│  │ - Store in ChromaDB                                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                      3. USER QUESTION                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  User types question → Streamlit chat_input                  │
│  Triggers: stream_response() call                            │
│                                                               │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                    4. SAFETY VALIDATION                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Jailbreak Detection (instant)                        │    │
│  │ - Pattern matching against 15 signatures             │    │
│  │ - Blocks prompt injection attempts                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Azure Content Safety (0.5-1 second)                  │    │
│  │ - API call to Azure                                  │    │
│  │ - Check: Hate(0), Sexual(2), Violence(4), Harm(4)   │    │
│  │ - Block if exceeds threshold                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                   5. RETRIEVAL & RANKING                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Vector Search (0.1-0.5 seconds)                      │    │
│  │ - Embed question with BGE-base                       │    │
│  │ - Similarity search in ChromaDB                      │    │
│  │ - Retrieve k=10 candidate chunks                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Re-ranking (0.5-1 second)                            │    │
│  │ - CrossEncoder scores question-chunk pairs           │    │
│  │ - Sort by relevance score                            │    │
│  │ - Select top 5 chunks                                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                    6. ANSWER GENERATION                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Prompt Construction (instant)                        │    │
│  │ - System: "helpful AI assistant"                     │    │
│  │ - chat_history: Last 3 exchanges                     │    │
│  │ - context: Top 5 chunks concatenated                 │    │
│  │ - question: User's current question                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ LLM Streaming (3-10 seconds)                         │    │
│  │ - Ollama llama3.1:8b inference                       │    │
│  │ - Temperature: 0.7                                   │    │
│  │ - Yields tokens in real-time                         │    │
│  │ - Display immediately in UI                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Output Validation (0.5-1 second)                     │    │
│  │ - Azure Content Safety check                         │    │
│  │ - Same thresholds as input                           │    │
│  │ - Block unsafe generations                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                     7. DISPLAY RESULT                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  - Answer appears in chat with streaming effect              │
│  - Or error message if content blocked                       │
│  - Chat history updated (last 6 messages kept)               │
│  - Ready for next question                                   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Components

### Prompt Template (Lines 604-618)

```python
system_message = """You are a helpful AI assistant. Answer the user's 
question based on the provided context and conversation history. If the 
context doesn't contain enough information, say so clearly. Be precise 
and cite specific details from the context when possible."""

template = """
{system_message}

Previous conversation:
{chat_history}

Context from the document:
{context}

Question: {question}

Answer:"""
```

**Template Variables:**
- `system_message`: Behavioral instructions
- `chat_history`: Last 3 exchanges (formatted as "Human: ...\nAI: ...")
- `context`: Top 5 re-ranked chunks (concatenated with "\n\n")
- `question`: Current user question

### Re-ranking Logic (Lines 558-581)

```python
# Retrieve 10 candidates
docs = self.vectorstore.similarity_search(question, k=10)

# Score each with CrossEncoder
pairs = [[question, doc.page_content] for doc in docs]
scores = self.reranker.predict(pairs)

# Combine and sort
doc_score_pairs = list(zip(docs, scores))
doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

# Take top 5
top_docs = [doc for doc, _ in doc_score_pairs[:5]]
```

**Why Re-ranking?**
- Vector search uses cosine similarity (semantic)
- Re-ranker uses cross-attention (contextual relevance)
- Improves precision by 10-20% in benchmarks

### Chat History Management (Lines 586-593)

```python
# Take last 3 exchanges (6 messages)
recent_history = chat_history[-3:] if len(chat_history) > 3 else chat_history

# Format for prompt
history_str = "\n".join([
    f"Human: {h}\nAI: {a}" 
    for h, a in recent_history
])
```

**Memory Strategy:**
- Keeps last 3 Q&A pairs (6 messages total)
- Prevents context overflow
- Maintains conversational coherence
- Each exchange ~100-300 tokens

### Image Description Prompt (Lines 193-227)

```python
prompt = f"""Analyze this image in detail and provide a comprehensive 
description of 500-1000 words covering:

1. OVERVIEW & CONTENT: What is shown in the image? Describe the main 
subject, scene, or data being presented.

2. VISUAL ELEMENTS: Describe colors, shapes, patterns, layout, 
composition, and visual hierarchy.

3. TEXT CONTENT: Transcribe any visible text, labels, legends, 
annotations, axis labels, or textual elements.

4. TECHNICAL DETAILS: Identify any charts, graphs, diagrams, or 
technical visualizations. Explain data points, measurements, scales, 
or quantitative information.

5. CONTEXT & PURPOSE: Explain the likely purpose of this image within 
a document. What information does it convey?

6. KEY INSIGHTS: Highlight important findings, trends, comparisons, or 
notable features visible in the image.

7. RELEVANCE: How does this image relate to the overall document theme? 
What role does it play?

Be thorough, specific, and accurate in your description."""
```

**Why 7 Sections?**
- Structured output for consistent quality
- Covers both visual and semantic aspects
- Helps LLM understand image-document relationships
- 500-1000 words ensures richness

---

## Code Walkthrough

### Entry Point (Line 956)

```python
if __name__ == "__main__":
    main()
```

Calls `main()` at line 676.

### Main Function (Lines 676-956)

**Structure:**
```python
def main():
    # Streamlit page config (lines 677-680)
    st.set_page_config(title, icon, layout, initial_sidebar_state)
    
    # Environment setup (lines 682-689)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Session state initialization (lines 691-703)
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.chatbot = None
        st.session_state.messages = []
        st.session_state.processing_stats = None
        st.session_state.annotations = {}
        st.session_state.violation_count = 0
    
    # Sidebar UI (lines 705-847)
    with st.sidebar:
        st.title("📄 Document Upload")
        uploaded_file = st.file_uploader(...)
        
        if st.button("🚀 Process Document"):
            # Document processing workflow (lines 738-817)
            ...
        
        # Display stats (lines 819-847)
        ...
    
    # Main area UI (lines 849-954)
    st.title("💬 Secure Chatbot with Image Annotation")
    
    # Image gallery (lines 851-869)
    if st.session_state.annotations:
        with st.expander("🖼️ View X Annotated Images"):
            ...
    
    # Chat messages display (lines 871-878)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input and response (lines 880-954)
    if prompt := st.chat_input("Ask a question..."):
        # Validation and streaming (lines 884-950)
        ...
```

### Document Processing Workflow (Lines 738-817)

```python
# Save uploaded file
file_path = save_uploaded_file(uploaded_file)

# Progress tracking
progress_bar = st.progress(0)
status_text = st.empty()

# Step 1: Initialize chatbot
status_text.text("Initializing chatbot...")
chatbot = SecureChatbotRAGWithImages()
progress_bar.progress(20)

# Step 2: Process document
status_text.text("Processing document...")
stats = chatbot.process_document(file_path)
progress_bar.progress(80)

# Step 3: Update state
status_text.text("Finalizing...")
st.session_state.chatbot = chatbot
st.session_state.initialized = True
st.session_state.processing_stats = stats
st.session_state.annotations = stats['annotations']
st.session_state.messages = []  # Clear chat
progress_bar.progress(100)

# Success
st.success("Document processed successfully!")
st.rerun()
```

### Chat Response Workflow (Lines 884-950)

```python
# Display user message
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.markdown(prompt)

# Generate assistant response
with st.chat_message("assistant"):
    message_placeholder = st.empty()
    full_response = ""
    
    # Stream response
    for token in st.session_state.chatbot.stream_response(
        prompt, 
        [(m["content"], st.session_state.messages[i+1]["content"]) 
         for i, m in enumerate(st.session_state.messages[:-1]) 
         if m["role"] == "user" and i+1 < len(st.session_state.messages)]
    ):
        full_response += token
        message_placeholder.markdown(full_response + "▌")
    
    message_placeholder.markdown(full_response)
    
    # Check for violations
    if "CONTENT POLICY VIOLATION" in full_response:
        st.session_state.violation_count += 1
        if st.session_state.violation_count >= 3:
            st.error("Too many violations. Please restart.")
            st.stop()

# Save assistant message
st.session_state.messages.append({"role": "assistant", "content": full_response})
```

---

## Configuration Reference

### Model Parameters

| Component | Model | Parameters |
|-----------|-------|------------|
| Text LLM | llama3.1:8b | temperature=0.7 |
| Vision LLM | llama3.2-vision:11b-q8_0 | temperature=0.7, num_predict=2048 |
| Embeddings | BAAI/bge-base-en-v1.5 | device='cpu', dims=768 |
| Re-ranker | ms-marco-MiniLM-L-6-v2 | default |

### Content Safety Thresholds

| Category | Threshold | Meaning |
|----------|-----------|---------|
| Hate | 0 | Block all |
| Sexual | 2 | Block medium+ |
| Violence | 4 | Block severe |
| SelfHarm | 4 | Block severe |

### Chunking Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| tokenizer | bge-base-en-v1.5 | Aligned with embeddings |
| max_tokens | 1024 | Optimal for retrieval |
| method | HybridChunker | Smart text splitting |

### Retrieval Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Initial k | 10 | Cast wide net |
| Re-ranked k | 5 | Focus on quality |
| History | 3 exchanges | Balance context/token budget |

---

## Performance Tips

### GPU Memory Optimization
- Run embeddings on CPU: `model_kwargs={'device': 'cpu'}`
- Use Q8 quantized VLM: `llama3.2-vision:11b-q8_0`
- Avoid parallel LLM calls

### Speed Optimization
- Batch image annotations
- Cache embeddings (ChromaDB persistence)
- Use smaller chunks for faster search

### Quality Optimization
- Increase max_tokens for longer chunks (more context)
- Adjust re-ranking k for precision/recall tradeoff
- Tune LLM temperature (0.5=focused, 0.9=creative)

---

**Next Steps:**
- Read [FLOW_DIAGRAM.md](FLOW_DIAGRAM.md) for visual architecture
- Check [README.md](README.md) for setup instructions
- Explore code comments for implementation details
