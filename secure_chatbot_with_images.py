"""
Secure RAG Chatbot with Image Annotation and Azure Content Safety
Features:
- PDF upload capability
- Image detection and flagging
- Image annotation and enrichment with Llama3 8B
- Merge enriched images into document before vectorization
- Streaming chat with content safety protection
"""

import streamlit as st
import os
import sys
import base64
import io
from pathlib import Path
from datetime import datetime
from PIL import Image
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.exceptions import HttpResponseError

# Import for OCR capabilities
try:
    import fitz  # PyMuPDF
    from rapidocr_onnxruntime import RapidOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    fitz = None
    RapidOCR = None

# Suppress tokenizer parallelism warning and sequence length warning
# See: https://github.com/huggingface/transformers/issues/5486
# See: https://github.com/docling-project/docling-core/issues/119#issuecomment-2577418826
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, ImageRefMode, PictureItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from sentence_transformers import CrossEncoder


class ContentSafetyGuard:
    """Azure Content Safety integration for input/output filtering"""
    
    def __init__(self):
        self.endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT")
        self.key = os.environ.get("CONTENT_SAFETY_KEY")
        
        if not self.endpoint or not self.key:
            raise ValueError("Azure Content Safety credentials not set")
        
        self.client = ContentSafetyClient(self.endpoint, AzureKeyCredential(self.key))
        
        self.thresholds = {
            "hate": 2,
            "sexual": 2,
            "violence": 2,
            "self_harm": 2
        }
    
    def check_content(self, text: str, strict_mode: bool = False):
        """Check text for harmful content"""
        try:
            request = AnalyzeTextOptions(text=text)
            response = self.client.analyze_text(request)
            
            results = {}
            blocked = False
            reasons = []
            
            for item in response.categories_analysis:
                category_name = item.category.lower() if isinstance(item.category, str) else item.category.value.lower()
                severity = item.severity
                
                results[category_name] = {
                    "severity": severity,
                    "threshold": self.thresholds.get(category_name, 2)
                }
                
                threshold = self.thresholds.get(category_name, 2)
                if strict_mode:
                    threshold = max(1, threshold - 1)
                
                if severity >= threshold:
                    blocked = True
                    reasons.append(f"{category_name.upper()} (severity: {severity})")
            
            return {
                "safe": not blocked,
                "blocked": blocked,
                "reason": ", ".join(reasons) if reasons else "Safe",
                "categories": results
            }
            
        except HttpResponseError as e:
            return {
                "safe": False,
                "blocked": True,
                "reason": f"Error: {str(e)}",
                "categories": {}
            }
    
    def detect_jailbreak(self, text: str):
        """Detect prompt injection attempts"""
        jailbreak_patterns = [
            "ignore previous instructions",
            "ignore all previous",
            "forget everything",
            "you are now",
            "pretend you are",
            "act as if",
            "abaikan instruksi sebelumnya",
            "lupakan semua",
            "kamu sekarang adalah",
            "berpura-pura menjadi",
            "anggap kamu adalah",
            "system prompt",
            "override instructions"
        ]
        
        text_lower = text.lower()
        detected = [p for p in jailbreak_patterns if p in text_lower]
        
        return {
            "is_jailbreak": len(detected) > 0,
            "confidence": "high" if len(detected) > 0 else "low",
            "patterns": detected
        }


class OCRExtractor:
    """Extract high-resolution images from PDF and apply OCR"""
    
    def __init__(self, zoom_factor: float = 3.0, output_dir: str = "extracted_pdf_images"):
        """
        Initialize OCR extractor
        
        Args:
            zoom_factor: Resolution multiplier (3.0 = 3x zoom for high quality)
            output_dir: Directory to save extracted images
        """
        self.zoom_factor = zoom_factor
        self.output_dir = Path(output_dir)
        self.ocr_engine = RapidOCR() if OCR_AVAILABLE and RapidOCR else None
        self.extracted_images = []
        self.ocr_text_by_page = {}
    
    def extract_images_from_pdf(self, pdf_path: str, progress_callback=None):
        """Extract all pages as high-resolution PNG images"""
        if not OCR_AVAILABLE or not fitz:
            if progress_callback:
                progress_callback("⚠️ PyMuPDF not available, skipping image extraction")
            return []
        
        self.output_dir.mkdir(exist_ok=True)
        
        doc = fitz.open(pdf_path)
        self.extracted_images = []
        
        if progress_callback:
            progress_callback(f"📷 Extracting {len(doc)} pages as high-res images (zoom={self.zoom_factor}x)...")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Convert page to high-resolution image
            mat = fitz.Matrix(self.zoom_factor, self.zoom_factor)
            pix = page.get_pixmap(matrix=mat)
            
            # Save image
            img_path = self.output_dir / f"page_{page_num + 1}.png"
            pix.save(str(img_path))
            
            self.extracted_images.append({
                "page_num": page_num + 1,
                "path": str(img_path),
                "width": pix.width,
                "height": pix.height
            })
            
            if progress_callback and (page_num + 1) % 5 == 0:
                progress_callback(f"   📄 Extracted {page_num + 1}/{len(doc)} pages...")
        
        doc.close()
        
        if progress_callback:
            progress_callback(f"✅ Extracted {len(self.extracted_images)} page images")
        
        return self.extracted_images
    
    def apply_ocr_to_images(self, progress_callback=None):
        """Apply RapidOCR to all extracted images"""
        if not self.ocr_engine:
            if progress_callback:
                progress_callback("⚠️ RapidOCR not available, skipping OCR")
            return {}
        
        if progress_callback:
            progress_callback(f"🔍 Applying OCR to {len(self.extracted_images)} images...")
        
        self.ocr_text_by_page = {}
        total_chars = 0
        
        for img_info in self.extracted_images:
            page_num = img_info["page_num"]
            img_path = img_info["path"]
            
            try:
                result, elapse = self.ocr_engine(img_path)
                
                if result:
                    # Extract text from OCR result
                    page_text = []
                    for line in result:
                        text = line[1]  # line format: [bbox, text, confidence]
                        page_text.append(text)
                    
                    page_content = "\n".join(page_text)
                    self.ocr_text_by_page[page_num] = page_content
                    total_chars += len(page_content)
                    
                    if progress_callback and page_num % 5 == 0:
                        progress_callback(f"   📝 OCR processed page {page_num}/{len(self.extracted_images)}...")
                else:
                    self.ocr_text_by_page[page_num] = "[No text detected]"
                    
            except Exception as e:
                self.ocr_text_by_page[page_num] = f"[OCR Error: {str(e)[:100]}]"
        
        if progress_callback:
            progress_callback(f"✅ OCR complete: {total_chars:,} characters extracted")
        
        return self.ocr_text_by_page
    
    def save_ocr_text(self, output_file: str = "ocr_extracted_text.txt", progress_callback=None):
        """Save all OCR text to a file"""
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# OCR Extracted Text\n")
            f.write(f"Total Pages: {len(self.ocr_text_by_page)}\n")
            f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "=" * 60 + "\n\n")
            
            for page_num in sorted(self.ocr_text_by_page.keys()):
                f.write(f"## Page {page_num}\n\n")
                f.write(self.ocr_text_by_page[page_num])
                f.write("\n\n" + "-" * 60 + "\n\n")
        
        if progress_callback:
            progress_callback(f"💾 OCR text saved to: {output_path.absolute()}")
        
        return str(output_path.absolute())
    
    def get_combined_text(self):
        """Get all OCR text combined as a single string"""
        combined = []
        for page_num in sorted(self.ocr_text_by_page.keys()):
            combined.append(f"[Page {page_num}]\n{self.ocr_text_by_page[page_num]}")
        return "\n\n".join(combined)
    
    def get_ocr_chunks(self):
        """Get OCR text as structured chunks for vector DB indexing"""
        chunks = []
        for page_num in sorted(self.ocr_text_by_page.keys()):
            text = self.ocr_text_by_page[page_num]
            # Skip empty or error pages
            if text and not text.startswith("[No text detected]") and not text.startswith("[OCR Error"):
                chunks.append({
                    "page": page_num,
                    "text": text,
                    "source": "ocr"
                })
        return chunks
    
    @staticmethod
    def is_image_based_pdf(doc_text: str, threshold: int = 500) -> bool:
        """Check if PDF is image-based (minimal text) or text-based
        
        Args:
            doc_text: Text extracted by Docling
            threshold: Minimum characters to consider text-based (default 500)
        
        Returns:
            bool: True if image-based (needs OCR), False if text-based
        """
        # Remove markdown formatting and whitespace
        clean_text = doc_text.replace('#', '').replace('*', '').replace('-', '')
        clean_text = ''.join(clean_text.split())
        
        # Check if text content is minimal
        char_count = len(clean_text)
        return char_count < threshold


class ImageAnnotator:
    """Handle image detection, annotation, and enrichment using Vision Language Model"""
    
    def __init__(self, vision_llm):
        """
        Initialize with a Vision Language Model (VLM)
        Recommended models for 8GB GPU (A10):
        - llama3.2-vision:11b-q8_0 (Q8 quantized - fits 8GB VRAM)
        - llama3.2-vision:11b-q4_0 (Q4 quantized - smaller, faster)
        - llava:7b (alternative, smaller model)
        - bakllava (lightweight option for very limited memory)
        """
        self.vision_llm = vision_llm
        self.annotated_images = []
    
    def describe_image_with_vlm(self, picture_item: PictureItem, doc: DoclingDocument, context: str = ""):
        """
        Generate description for image using Vision Language Model (VLM)
        This method extracts the actual image from Docling and sends it to the VLM
        
        Args:
            picture_item: PictureItem from Docling
            doc: DoclingDocument containing the image
            context: Surrounding text context
        
        Returns:
            str: Enhanced description from VLM analyzing the actual image
        """
        import tempfile
        import os
        import base64
        
        try:
            caption = picture_item.caption_text(doc) if hasattr(picture_item, 'caption_text') else "No caption"
            
            # Get image from PictureItem
            # Docling stores image reference in the picture item
            image_path = None
            
            # Method 1: Try to get PIL Image if available
            if hasattr(picture_item, 'image') and picture_item.image is not None:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    picture_item.image.save(tmp.name, format='PNG')
                    image_path = tmp.name
            
            # Method 2: Try to get from document export
            elif hasattr(picture_item, 'get_image'):
                pil_image = picture_item.get_image(doc)
                if pil_image:
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        pil_image.save(tmp.name, format='PNG')
                        image_path = tmp.name
            
            # Method 3: Export document and extract images
            if not image_path:
                # Use Docling's image export capability
                try:
                    # Export with embedded images
                    doc_with_images = doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
                    # This is complex, fallback to caption-based
                    image_path = None
                except:
                    pass
            
            if image_path and os.path.exists(image_path):
                # Prompt optimized for detailed, comprehensive analysis
                # Provides example format for better output
                prompt = f"""You are a technical document analyst. Analyze this figure/diagram in extreme detail.

CAPTION: {caption}

DOCUMENT CONTEXT: {context[:500] if context else 'Technical document about document processing'}

TASK: Provide a COMPREHENSIVE technical description following this structure:

=== FIGURE DESCRIPTION ===

**Type & Purpose:**
Identify the type (flowchart, architecture diagram, process flow, data flow, system diagram, etc.) and explain what it illustrates.

**Visual Layout & Structure:**
Describe the overall organization - is it linear (left-to-right, top-to-bottom), hierarchical, cyclical, or modular? How many main sections or stages are visible?

**Detailed Components:**
List and describe EVERY major element, box, stage, component, or module shown:
- First stage/component: [name] - [what it does] - [what it contains]
- Second stage/component: [name] - [what it does] - [what it contains]
- Continue for ALL visible components...

**Connections & Flow:**
Describe how components connect: arrows, lines, data flow, process flow, transformations between stages.

**Text & Labels:**
List ALL visible text, labels, annotations, titles, or captions within the figure itself (not just the figure caption).

**Technical Details:**
Identify technical terms, technologies, algorithms, models, methods, or specific processes mentioned or illustrated.

**Key Insights:**
What are the main concepts? What problem does this solve? What workflow or architecture does it demonstrate?

Write 500-1000 words. Be exhaustive and precise. Imagine explaining this to someone who cannot see the image."""
                
                # Use Ollama's vision capability with image file
                response = self.vision_llm.invoke(
                    prompt,
                    images=[image_path]  # Pass actual image to VLM
                )
                
                # Cleanup temp file
                os.unlink(image_path)
                
                return response.strip()
            else:
                # Fallback: Use caption and context for inference
                fallback_prompt = f"""Based on the caption and context, infer what this figure shows.

Caption: {caption}
Context: {context[:300]}

Provide a technical description of what this figure likely contains."""
                
                response = self.vision_llm.invoke(fallback_prompt)
                return f"[Inferred from caption] {response.strip()}"
                
        except Exception as e:
            # Fallback to caption-based description
            caption = picture_item.caption_text(doc) if hasattr(picture_item, 'caption_text') else "Figure"
            import traceback
            error_detail = traceback.format_exc()
            return f"[Figure: {caption}] - VLM analysis error: {str(e)[:100]}"
    
    def annotate_images_in_document(self, doc: DoclingDocument, progress_callback=None):
        """
        Find all images in document, extract them, and annotate with VLM
        
        Args:
            doc: DoclingDocument with images
            progress_callback: Optional function to report progress
        
        Returns:
            list: Enriched image annotations with VLM analysis
        """
        self.annotated_images = []
        
        # Find all picture items in document
        pictures = []
        for item, _level in doc.iterate_items():
            if isinstance(item, PictureItem):
                pictures.append(item)
        
        if progress_callback:
            progress_callback(f"🖼️ Found {len(pictures)} images in document")
        
        # Process each image with VLM
        for idx, picture in enumerate(pictures):
            if progress_callback:
                progress_callback(f"🤖 VLM analyzing image {idx+1}/{len(pictures)}...")
            
            # Extract image metadata
            caption = picture.caption_text(doc) if hasattr(picture, 'caption_text') else f"Figure {idx+1}"
            page = picture.prov[0].page_no if (picture.prov and len(picture.prov) > 0) else "Unknown"
            
            image_info = {
                "index": idx,
                "caption": caption,
                "page": page,
                "label": picture.label if hasattr(picture, 'label') else "",
                "self_ref": picture.self_ref if hasattr(picture, 'self_ref') else ""
            }
            
            # Get surrounding context (text before and after image)
            context = self._get_image_context(doc, picture)
            
            # Generate enriched description using Vision LLM with actual image
            enriched_description = self.describe_image_with_vlm(picture, doc, context)
            
            # Create annotation with VLM analysis
            annotation = {
                "image_info": image_info,
                "context": context[:200],
                "enriched_description": enriched_description,
                "flagged": True,
                "annotation_text": f"[IMAGE {idx+1} - Page {page}] {caption}\n\nVLM Analysis: {enriched_description}"
            }
            
            self.annotated_images.append(annotation)
        
        if progress_callback:
            progress_callback(f"✅ Completed VLM annotation of {len(self.annotated_images)} images")
        
        return self.annotated_images
    
    def _get_image_context(self, doc: DoclingDocument, picture: PictureItem, window: int = 500):
        """Get text context around an image from the same page"""
        try:
            page_num = picture.prov[0].page_no if (picture.prov and len(picture.prov) > 0) else None
            if page_num is not None:
                page_text = []
                for item, _level in doc.iterate_items():
                    if hasattr(item, 'prov') and item.prov and len(item.prov) > 0 and item.prov[0].page_no == page_num:
                        if hasattr(item, 'text') and item.text:
                            page_text.append(item.text.strip())
                
                # Join and limit context
                context = " ".join(page_text)
                return context[:window] if len(context) > window else context
        except Exception as e:
            pass
        return ""
    
    def merge_annotations_into_text(self, text: str, annotations: list):
        """
        Merge image annotations into document text
        This enriches the text with image descriptions for better RAG
        """
        enriched_text = text
        
        for annotation in annotations:
            # Insert annotation at appropriate location
            enriched_text += f"\n\n{annotation['annotation_text']}\n"
        
        return enriched_text


class SecureChatbotRAGWithImages:
    """Secure RAG Chatbot with Image Processing and Content Safety"""
    
    def __init__(self, collection_name: str = "secure_chatbot_images"):
        self.collection_name = collection_name
        self.vectorstore = None
        self.current_pdf_path = None
        self.annotated_images = []
        
        # Initialize Content Safety
        self.guard = ContentSafetyGuard()
        
        # Rate limiting
        self.blocked_count = 0
        self.max_blocked = 3
        
        # Initialize embeddings with BGE-base model (768 dimensions - fits 8GB GPU)
        # BGE-large (1024 dims) causes CUDA OOM on 8GB GPU
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'}  # Use CPU for embeddings to save GPU memory for VLM
        )
        
        # Initialize chunker with larger window for detailed image annotations
        # Align tokenizer with embedding model
        self.chunker = HybridChunker(
            tokenizer="BAAI/bge-base-en-v1.5",
            max_tokens=1024  # Support longer chunks for detailed image descriptions
        )
        
        # Initialize re-ranker
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize LLM for text generation
        self.llm = Ollama(model="llama3.1:8b", temperature=0.7)
        
        # Initialize Vision LLM for image annotation (Llama 3.2 Vision Q8 quantized)
        # Using Q8 quantization to fit in 8GB GPU RAM (A10)
        # Higher temperature for more detailed, creative descriptions
        self.vision_llm = Ollama(
            model="llama3.2-vision:11b-q8_0", 
            temperature=0.7,
            num_predict=2048  # Max tokens for longer descriptions
        )
        
        # Initialize image annotator with vision model
        self.image_annotator = ImageAnnotator(self.vision_llm)
        
        # Initialize OCR extractor (3x zoom for high resolution)
        self.ocr_extractor = OCRExtractor(zoom_factor=3.0, output_dir="./extracted_pdf_images")
        
        # Prompt template with conversation history
        self.prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer questions based on the provided context and conversation history.
The context may include detailed descriptions of images, figures, diagrams, and charts from the document.
Be clear, thorough, and accurate. If you cannot find the answer, say so.

Conversation History:
{chat_history}

Context from Document:
{context}

Current Question: {question}

Provide a comprehensive answer based on the context and conversation history:""")
    
    def process_document(self, pdf_path: str, progress_callback=None):
        """
        Process PDF with image annotation and enrichment
        
        Steps:
        1. Convert PDF with Docling
        2. Detect and flag all images
        3. Annotate images with Llama3 8B
        4. Merge enriched image descriptions into document
        5. Chunk the enriched document
        6. Create vector store
        """
        self.current_pdf_path = pdf_path
        
        if progress_callback:
            progress_callback("📄 Converting PDF with Docling...")
        
        # Configure Docling to extract images with high quality
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0  # Higher resolution for better VLM analysis
        pipeline_options.generate_picture_images = True  # Extract images as PIL objects
        
        # Important: Set image mode to EMBEDDED to get actual image data
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
        # Convert document
        result = converter.convert(pdf_path)
        doc: DoclingDocument = result.document
        
        if progress_callback:
            progress_callback(f"✅ Document converted: {doc.name}")
        
        # Step 1: Check PDF content type
        if progress_callback:
            progress_callback("🔍 Analyzing PDF content type...")
        
        # Export to get initial text
        initial_text = doc.export_to_markdown()
        is_image_pdf = OCRExtractor.is_image_based_pdf(initial_text)
        
        if is_image_pdf:
            if progress_callback:
                progress_callback("📋 Detected: IMAGE-BASED PDF (minimal text)")
                progress_callback("🔄 OCR akan digunakan untuk ekstraksi teks")
        else:
            if progress_callback:
                progress_callback("📋 Detected: TEXT-BASED PDF (dengan text/table/image)")
                progress_callback("✅ Docling + VLM akan digunakan (OCR tidak diperlukan)")
        
        # Step 2: Apply OCR only for image-based PDFs
        use_ocr = False
        if is_image_pdf and OCR_AVAILABLE:
            if progress_callback:
                progress_callback("📷 Extracting high-resolution page images...")
            
            self.ocr_extractor.extract_images_from_pdf(pdf_path, progress_callback)
            
            if progress_callback:
                progress_callback("🔍 Applying RapidOCR to extracted images...")
            
            self.ocr_extractor.apply_ocr_to_images(progress_callback)
            
            # Save OCR text to file
            ocr_output_file = "./ocr_extracted_text.txt"
            self.ocr_extractor.save_ocr_text(ocr_output_file, progress_callback)
            use_ocr = True
        elif is_image_pdf and not OCR_AVAILABLE:
            if progress_callback:
                progress_callback("⚠️ PDF is image-based but OCR not available")
                progress_callback("💡 Install PyMuPDF and rapidocr-onnxruntime for better extraction")
        
        # Step 3: Annotate images/tables/charts with VLM (Docling's primary feature)
            progress_callback("🖼️ Detecting and annotating images/tables/charts with VLM...")
        
        self.annotated_images = self.image_annotator.annotate_images_in_document(
            doc, 
            progress_callback=progress_callback
        )
        
        # Step 4: Export document to markdown (includes text/tables/images)
        if progress_callback:
            progress_callback("📝 Exporting document content (text/tables/images)...")
        
        doc_text = doc.export_to_markdown()
        
        # Step 5: Merge image annotations into text
        if progress_callback:
            progress_callback("🔗 Merging image annotations into document...")
        
        enriched_text = self.image_annotator.merge_annotations_into_text(
            doc_text, 
            self.annotated_images
        )
        
        # Step 6: Chunk the enriched document using HybridChunker
        if progress_callback:
            progress_callback("✂️ Chunking document with HybridChunker...")
        
        # Use HybridChunker to intelligently chunk the document
        chunk_iter = self.chunker.chunk(doc)
        chunks_list = list(chunk_iter)
        
        # Create documents from chunks + add image annotations as separate chunks
        documents = []
        
        # Add regular text chunks using HybridChunker's contextualization
        for i, chunk in enumerate(chunks_list):
            # Use HybridChunker's contextualize method for optimal context
            contextualized_content = self.chunker.contextualize(chunk)
            
            documents.append(
                Document(
                    page_content=contextualized_content,
                    metadata={
                        "source": pdf_path,
                        "chunk_id": i,
                        "type": "text",
                        "headings": str(chunk.meta.headings) if hasattr(chunk.meta, 'headings') else "",
                        "num_doc_items": len(chunk.meta.doc_items) if hasattr(chunk.meta, 'doc_items') else 0
                    }
                )
            )
        
        # Add image annotations as separate searchable chunks
        # These are merged into the vector store for comprehensive search
        for img_annotation in self.annotated_images:
            documents.append(
                Document(
                    page_content=img_annotation['annotation_text'],
                    metadata={
                        "source": pdf_path,
                        "chunk_id": f"image_{img_annotation['image_info']['index']}",
                        "type": "image_annotation",
                        "page": str(img_annotation['image_info']['page']),
                        "caption": img_annotation['image_info']['caption']
                    }
                )
            )
        
        # Add OCR chunks ONLY if PDF was image-based and OCR was used
        # For text-based PDFs, Docling already extracted everything properly
        if use_ocr and self.ocr_extractor.ocr_text_by_page:
            ocr_chunks = self.ocr_extractor.get_ocr_chunks()
            if progress_callback:
                progress_callback(f"📋 Adding {len(ocr_chunks)} OCR chunks to vector store (image-based PDF)...")
            
            for ocr_chunk in ocr_chunks:
                # Split large OCR pages into smaller chunks if needed (max 1000 chars per chunk)
                ocr_text = ocr_chunk['text']
                page_num = ocr_chunk['page']
                
                if len(ocr_text) > 1000:
                    # Split into smaller chunks
                    chunk_size = 1000
                    for i in range(0, len(ocr_text), chunk_size):
                        chunk_text = ocr_text[i:i+chunk_size]
                        documents.append(
                            Document(
                                page_content=f"[OCR Page {page_num} - Part {i//chunk_size + 1}]\n\n{chunk_text}",
                                metadata={
                                    "source": pdf_path,
                                    "chunk_id": f"ocr_page_{page_num}_part_{i//chunk_size}",
                                    "type": "ocr_text",
                                    "page": str(page_num),
                                    "ocr_part": i//chunk_size
                                }
                            )
                        )
                else:
                    documents.append(
                        Document(
                            page_content=f"[OCR Page {page_num}]\n\n{ocr_text}",
                            metadata={
                                "source": pdf_path,
                                "chunk_id": f"ocr_page_{page_num}",
                                "type": "ocr_text",
                                "page": str(page_num)
                            }
                        )
                    )
        
        documents = filter_complex_metadata(documents)
        
        ocr_chunks_count = len(self.ocr_extractor.get_ocr_chunks()) if use_ocr else 0
        if progress_callback:
            if ocr_chunks_count > 0:
                progress_callback(f"📊 Created {len(documents)} chunks ({len(chunks_list)} text + {len(self.annotated_images)} images + {ocr_chunks_count} OCR chunks)")
            else:
                progress_callback(f"📊 Created {len(documents)} chunks ({len(chunks_list)} text + {len(self.annotated_images)} images - Docling extraction)")
        
        # Step 7: Create or update vector store
        if progress_callback:
            progress_callback("🔢 Creating vector store...")
        
        # Check if collection exists, delete if needed for fresh start
        persist_dir = "./chroma_db"
        try:
            # Try to get existing collection
            existing_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_dir
            )
            # Delete existing collection for fresh indexing
            existing_store.delete_collection()
            if progress_callback:
                progress_callback("🗑️ Cleared existing collection")
        except:
            # Collection doesn't exist, which is fine
            pass
        
        # Create new vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=persist_dir
        )
        
        if progress_callback:
            progress_callback("✅ Vector store ready!")
        
        ocr_chunks_count = len(self.ocr_extractor.get_ocr_chunks()) if use_ocr else 0
        
        return {
            "total_chunks": len(documents),
            "text_chunks": len(chunks_list),
            "image_chunks": len(self.annotated_images),
            "images_found": len(self.annotated_images),
            "ocr_chunks": ocr_chunks_count,
            "ocr_pages": len(self.ocr_extractor.ocr_text_by_page) if use_ocr else 0,
            "ocr_images_saved": len(self.ocr_extractor.extracted_images) if use_ocr else 0,
            "ocr_available": OCR_AVAILABLE,
            "ocr_used": use_ocr,
            "pdf_type": "image-based" if is_image_pdf else "text-based"
        }
    
    def rerank_results(self, query: str, documents, top_k: int = 5):
        """Re-rank documents using cross-encoder"""
        if not documents:
            return []
        
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker.predict(pairs)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:top_k]
    
    def stream_response(self, question: str, k: int = 10, top_k_rerank: int = 5, chat_history: str = ""):
        """
        Generate streaming response with content safety checks and conversation history
        
        Args:
            question: User's current question
            k: Number of documents to retrieve initially
            top_k_rerank: Number of documents after re-ranking
            chat_history: Previous conversation context
        
        Yields: dict with 'type' and 'content'
        """
        # Step 1: Check input for jailbreak
        jailbreak_check = self.guard.detect_jailbreak(question)
        if jailbreak_check["is_jailbreak"]:
            self.blocked_count += 1
            yield {
                "type": "error",
                "content": f"⚠️ Terdeteksi percobaan prompt injection/jailbreak\nPola: {', '.join(jailbreak_check['patterns'])}"
            }
            return
        
        # Step 2: Check input content safety
        yield {"type": "status", "content": "🛡️ Memeriksa keamanan input..."}
        safety_check = self.guard.check_content(question, strict_mode=False)
        
        if safety_check["blocked"]:
            self.blocked_count += 1
            yield {
                "type": "error",
                "content": f"⚠️ Input diblokir karena: {safety_check['reason']}"
            }
            
            if self.blocked_count >= self.max_blocked:
                yield {
                    "type": "error",
                    "content": "\n🚫 Terlalu banyak pelanggaran. Sesi diakhiri."
                }
            return
        
        # Step 3: Retrieve documents (including image annotations)
        yield {"type": "status", "content": "🔍 Mencari informasi relevan (termasuk gambar)..."}
        initial_results = self.vectorstore.similarity_search(question, k=k)
        
        # Step 4: Re-rank
        yield {"type": "status", "content": "📊 Menyortir hasil..."}
        reranked_results = self.rerank_results(question, initial_results, top_k=top_k_rerank)
        top_docs = [doc for doc, score in reranked_results]
        
        # Check if any image annotations or OCR text in top results
        image_docs = [doc for doc in top_docs if doc.metadata.get("type") == "image_annotation"]
        ocr_docs = [doc for doc in top_docs if doc.metadata.get("type") == "ocr_text"]
        
        if image_docs:
            yield {"type": "status", "content": f"🖼️ Ditemukan {len(image_docs)} gambar relevan dalam hasil"}
        if ocr_docs:
            yield {"type": "status", "content": f"📋 Ditemukan {len(ocr_docs)} chunk OCR relevan dalam hasil"}
        
        # Step 5: Generate response with streaming
        yield {"type": "status", "content": "💭 Menghasilkan jawaban..."}
        
        context = "\n\n".join([doc.page_content for doc in top_docs])
        prompt_text = self.prompt.format(
            context=context, 
            question=question,
            chat_history=chat_history if chat_history else "No previous conversation."
        )
        
        # Stream from LLM
        full_response = ""
        for chunk in self.llm.stream(prompt_text):
            full_response += chunk
            yield {"type": "chunk", "content": chunk}
        
        # Step 6: Validate output
        yield {"type": "status", "content": "✅ Memvalidasi output..."}
        output_check = self.guard.check_content(full_response, strict_mode=True)
        
        if output_check["blocked"]:
            yield {
                "type": "error",
                "content": f"\n\n⚠️ Output diblokir karena: {output_check['reason']}"
            }
        else:
            yield {"type": "complete", "content": ""}


def init_session_state():
    """Initialize Streamlit session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = SecureChatbotRAGWithImages()
    
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    
    if "processing_stats" not in st.session_state:
        st.session_state.processing_stats = {}


def main():
    st.set_page_config(
        page_title="Secure RAG Chatbot with Images",
        page_icon="🛡️",
        layout="wide"
    )
    
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🛡️ Secure RAG Chatbot")
        st.markdown("**with Image Annotation**")
        st.markdown("---")
        
        st.markdown("### 📄 Upload PDF")
        uploaded_file = st.file_uploader(
            "Upload your PDF document",
            type=["pdf"],
            help="Upload a PDF to analyze. Images will be detected and annotated automatically."
        )
        
        # Process button - allow reprocessing new documents
        process_button_disabled = st.session_state.initialized and uploaded_file is None
        if uploaded_file and st.button("🚀 Process Document", disabled=process_button_disabled):
            # Reset state for new document
            st.session_state.initialized = False
            st.session_state.processing_stats = {}
            st.session_state.messages = []
            st.session_state.chatbot = SecureChatbotRAGWithImages()
            
            # Save uploaded file
            upload_dir = Path("./uploaded_pdfs")
            upload_dir.mkdir(exist_ok=True)
            pdf_path = upload_dir / uploaded_file.name
            
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info(f"📁 Saved: {uploaded_file.name}")
            
            # Process with progress
            progress_placeholder = st.empty()
            
            def progress_callback(message):
                progress_placeholder.info(message)
            
            with st.spinner("Processing document..."):
                try:
                    stats = st.session_state.chatbot.process_document(
                        str(pdf_path),
                        progress_callback=progress_callback
                    )
                    
                    st.session_state.initialized = True
                    st.session_state.processing_stats = stats
                    
                    progress_placeholder.success("✅ Document ready!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Use default PDF option
        if not st.session_state.initialized and not uploaded_file:
            st.markdown("---")
            st.markdown("### 📚 Or use default PDF")
            if st.button("🚀 Load Default (paper.pdf)"):
                pdf_path = "../paper.pdf"
                if not os.path.exists(pdf_path):
                    st.error(f"Default PDF not found: {pdf_path}")
                else:
                    progress_placeholder = st.empty()
                    
                    def progress_callback(message):
                        progress_placeholder.info(message)
                    
                    with st.spinner("Processing document..."):
                        try:
                            stats = st.session_state.chatbot.process_document(
                                pdf_path,
                                progress_callback=progress_callback
                            )
                            
                            st.session_state.initialized = True
                            st.session_state.processing_stats = stats
                            
                            progress_placeholder.success("✅ Document ready!")
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        # Statistics
        if st.session_state.initialized:
            st.markdown("---")
            st.markdown("### 📊 Document Stats")
            stats = st.session_state.processing_stats
            st.metric("Total Chunks", stats.get("total_chunks", 0))
            st.metric("Text Chunks", stats.get("text_chunks", 0))
            st.metric("🖼️ Images Found", stats.get("images_found", 0))
            st.metric("Image Chunks", stats.get("image_chunks", 0))
            
            # Show PDF type detection
            st.markdown("---")
            st.markdown("### 📋 PDF Analysis")
            pdf_type = stats.get("pdf_type", "unknown")
            if pdf_type == "image-based":
                st.warning(f"📸 Type: {pdf_type.upper()}")
                st.caption("PDF berisi gambar scan - OCR digunakan")
            else:
                st.success(f"📝 Type: {pdf_type.upper()}")
                st.caption("PDF berisi text/table/image - Docling + VLM digunakan")
            
            # Show OCR stats only if OCR was actually used
            if stats.get("ocr_used"):
                st.markdown("---")
                st.markdown("### 🔍 OCR Stats (Image-based PDF)")
                st.metric("OCR Chunks Indexed", stats.get("ocr_chunks", 0))
                st.metric("OCR Pages Processed", stats.get("ocr_pages", 0))
                st.metric("OCR Images Saved", stats.get("ocr_images_saved", 0))
                if stats.get("ocr_chunks", 0) > 0:
                    st.success("✅ OCR text indexed in vector database")
                    st.info("📄 OCR text saved to ocr_extracted_text.txt")
                    st.info("📁 Images saved to extracted_pdf_images/")
            elif stats.get("pdf_type") == "text-based":
                st.info("ℹ️ OCR tidak diperlukan - Docling berhasil ekstrak text/table/image")
            
            st.markdown("---")
            st.markdown("### 💬 Chat Stats")
            st.metric("Messages", len(st.session_state.messages))
            st.metric("Blocked Attempts", st.session_state.chatbot.blocked_count)
        
        # Reset button
        if st.session_state.initialized:
            st.markdown("---")
            if st.button("🔄 Reset Chat"):
                st.session_state.messages = []
                st.session_state.chatbot.blocked_count = 0
                st.rerun()
            
            if st.button("🗑️ Clear All & Start Over"):
                st.session_state.messages = []
                st.session_state.initialized = False
                st.session_state.processing_stats = {}
                st.session_state.chatbot = SecureChatbotRAGWithImages()
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🛡️ Security Features")
        st.markdown("""
        - ✅ Prompt injection detection
        - ✅ Content safety filtering
        - ✅ Hate speech prevention
        - ✅ Violence blocking
        - ✅ Output validation
        """)
        
        st.markdown("---")
        st.markdown("### 🖼️ Image Features")
        st.markdown("""
        - 🔍 Automatic image detection
        - 🏷️ Image flagging & annotation
        - 🤖 AI-powered descriptions
        - 📝 Context enrichment
        - 🔗 Merged into vectorization
        """)
        
        if OCR_AVAILABLE:
            st.markdown("---")
            st.markdown("### 🔍 OCR Features (Fallback)")
            st.markdown("""
            - ✅ Auto-detect PDF type
            - 📷 High-res extraction (image PDFs)
            - 🔤 RapidOCR text extraction
            - 💾 Save images & text files
            - 📋 Indexed in vector database
            - 🎯 Only used for image-based PDFs
            """)
            st.caption("OCR hanya digunakan untuk PDF full image. Text-based PDF menggunakan Docling + VLM.")
    
    # Main chat area
    st.title("💬 Secure RAG Chatbot with Image Understanding")
    
    if not st.session_state.initialized:
        st.info("👈 Upload a PDF or use the default document to start")
        
        st.markdown("### ✨ Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📄 Document Processing:**
            - Upload custom PDFs
            - Auto-detect PDF type (text/image)
            - Docling: text/table/image extraction
            - VLM: image/table/chart annotation
            - Context-aware chunking
            - OCR fallback (image-only PDFs)
            """)
        
        with col2:
            st.markdown("""
            **🛡️ Security & Safety:**
            - Azure Content Safety
            - Jailbreak detection
            - Content filtering
            - Rate limiting
            """)
        
        st.markdown("---")
        st.markdown("### 📝 Example Questions")
        st.markdown("""
        - What is shown in the images/figures in this document?
        - Describe the diagrams and their purpose
        - What does Figure 1 illustrate?
        - Summarize the visual content
        - What AI models are used? (references images if relevant)
        """)
        return
    
    # Display image annotations summary
    if st.session_state.chatbot.annotated_images:
        with st.expander(f"🖼️ View {len(st.session_state.chatbot.annotated_images)} Annotated Images"):
            for annotation in st.session_state.chatbot.annotated_images:
                img_info = annotation['image_info']
                st.markdown(f"**Image {img_info['index']+1}:** {img_info['caption']}")
                st.markdown(f"*Page {img_info['page']}*")
                st.markdown(f"**Enriched Description:**")
                st.info(annotation['enriched_description'])
                st.markdown("---")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the document (including images)..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            status_placeholder = st.empty()
            
            full_response = ""
            error_message = ""
            
            try:
                # Build chat history from previous messages
                chat_history = "\n".join([
                    f"{msg['role'].capitalize()}: {msg['content'][:200]}..."
                    for msg in st.session_state.messages[-6:]  # Last 3 exchanges (6 messages)
                ])
                
                for chunk_data in st.session_state.chatbot.stream_response(prompt, chat_history=chat_history):
                    if chunk_data["type"] == "status":
                        status_placeholder.info(chunk_data["content"])
                    
                    elif chunk_data["type"] == "chunk":
                        full_response += chunk_data["content"]
                        response_placeholder.markdown(full_response + "▌")
                    
                    elif chunk_data["type"] == "error":
                        error_message = chunk_data["content"]
                        response_placeholder.error(error_message)
                    
                    elif chunk_data["type"] == "complete":
                        status_placeholder.empty()
                
                # Display final response
                if full_response and not error_message:
                    response_placeholder.markdown(full_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                elif error_message:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🛡️ Protected by Azure Content Safety | "
        "🖼️ Images annotated with Llama3.1 8B | "
        f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    # Check credentials
    if not os.environ.get("CONTENT_SAFETY_ENDPOINT") or not os.environ.get("CONTENT_SAFETY_KEY"):
        st.error("⚠️ Azure Content Safety credentials not set!")
        st.info("Set CONTENT_SAFETY_ENDPOINT and CONTENT_SAFETY_KEY environment variables")
        st.stop()
    
    main()
