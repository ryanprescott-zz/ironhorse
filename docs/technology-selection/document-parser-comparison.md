# Open-Source Document Parser Comparison for Graph RAG Applications

## Executive Summary

Based on your Graph RAG architecture requirements, this comparison evaluates leading open-source document parsers suitable for the document processing pipeline. Your architecture emphasizes multi-format support, accurate text extraction, layout preservation, and integration with LangChain/LlamaIndex ecosystems.

**Top Recommendations:**
- **Primary Choice: Docling** - Best all-around balance of accuracy, speed, and RAG integration
- **High-Quality Alternative: MinerU** - Superior for complex tables and multilingual content (GPU required)
- **Lightweight Fallback: PyMuPDF** - Fast, reliable for text-heavy PDFs without complex layouts
- **Advanced Alternative: Unstructured** - Excellent semantic chunking for RAG workflows

---

## Comparison Matrix

| Parser | License | Speed (sec/page) | Formats | Table Extraction | Formula Support | RAG Integration | Best Use Case |
|--------|---------|------------------|---------|------------------|-----------------|-----------------|---------------|
| **Docling** | MIT | 1.27 (CPU M3)<br>0.49 (GPU L4) | PDF, DOCX, PPTX, XLSX, HTML, Images | Excellent | Excellent (LaTeX) | Native LangChain/LlamaIndex | General-purpose, production RAG |
| **MinerU** | Apache 2.0 | 3.3 (CPU)<br>0.21 (GPU L4) | PDF, DOCX, EPUB, MOBI | Excellent (HTML output) | Excellent (84 langs OCR) | Good | Complex tables, multilingual |
| **Unstructured** | Apache 2.0 | 4.2 (CPU)<br>2.7 (M3) | PDF, DOCX, PPTX, Images, HTML | Good | Moderate | Native LangChain/LlamaIndex | Semantic chunking, RAG workflows |
| **PyMuPDF** | AGPL/Commercial | 0.042 (text extract) | PDF, XPS, EPUB, CBZ | Good | Moderate | LangChain support | Fast text extraction, high volume |
| **Marker** | GPL 3.0 | 4.2 (CPU M3)<br>0.86 (GPU L4) | PDF, EPUB, MOBI | Good | Good | Moderate | PDF-to-Markdown conversion |
| **pypdfium2** | Apache 2.0/BSD | 0.003 (text only) | PDF | Moderate | Limited | LangChain support | Ultra-fast basic extraction |
| **pdfplumber** | MIT | Variable | PDF | Excellent | Limited | LangChain support | Table-focused extraction |
| **pymupdf4llm** | AGPL | 0.12 | PDF | Good | Good | LangChain/LlamaIndex | Markdown output for LLMs |

---

## Detailed Parser Analysis

### 1. Docling (IBM Research) ⭐ **RECOMMENDED**

**Overview:**  
Docling is a modern, AI-driven document conversion toolkit from IBM Research with native RAG integration and excellent performance characteristics.

**Key Strengths:**
- **MIT License** - Most permissive among advanced parsers
- **Multi-format support** - PDF, DOCX, PPTX, XLSX, HTML, Images, Markdown
- **Advanced PDF understanding** - Layout analysis, reading order, table structure, code blocks
- **Native RAG integration** - Built-in LangChain and LlamaIndex connectors
- **Balanced performance** - Fast on both CPU and GPU (1.27 sec/page on M3 Mac, 0.49 sec/page on L4 GPU)
- **Local execution** - No external API dependencies
- **Structured output** - Unified DoclingDocument format, exports to Markdown, HTML, JSON
- **Formula support** - LaTeX output for mathematical formulas
- **OCR capabilities** - Extensive OCR for scanned documents and images

**Performance:**
- CPU (x86): 3.1 sec/page
- CPU (M3 Max): 1.27 sec/page
- GPU (L4): 0.49 sec/page

**Weaknesses:**
- Relatively newer project (less mature ecosystem than PyMuPDF)
- Moderate resource requirements for advanced features
- Learning curve for advanced customization

**Integration with Your Architecture:**
```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")
markdown = result.document.export_to_markdown()

# Direct LangChain integration
from langchain_community.document_loaders import DoclingLoader
loader = DoclingLoader("document.pdf")
documents = loader.load()
```

**Best For:**
- Production RAG systems with LangChain/LlamaIndex
- Multi-format document processing pipelines
- Enterprise knowledge bases requiring structure preservation
- Projects needing permissive licensing

**Verdict:** **Primary recommendation** for your Graph RAG application due to MIT license, native RAG integration, strong performance, and comprehensive feature set.

---

### 2. MinerU (OpenDataLab)

**Overview:**  
MinerU is a sophisticated multi-module document parsing framework from Shanghai AI Lab, optimized for complex layouts and multilingual content.

**Key Strengths:**
- **Apache 2.0 License** - Permissive for commercial use
- **Superior table extraction** - Complex tables rendered as HTML with high fidelity
- **Excellent formula recognition** - LaTeX output with 84-language OCR support
- **Advanced layout analysis** - LayoutLMv3 + YOLOv8 for object detection
- **Header/footer cleanup** - Automatically removes non-content elements
- **Best GPU performance** - 0.21 sec/page on L4 GPU (fastest in class)
- **Multilingual OCR** - 84 languages with high accuracy
- **Structure preservation** - Maintains document hierarchy and semantics

**Performance:**
- CPU (x86): 3.3 sec/page
- GPU (L4): 0.21 sec/page (fastest tested)
- Note: Failed to run on M3 Mac in benchmarks

**Weaknesses:**
- **High resource requirements** - Requires GPU for optimal performance
- **Complex deployment** - Docker containerization, CUDA dependencies
- **Limited heading support** - Only first-level headings (critical for graph construction)
- **No vertical text** - Cannot process documents with vertical writing
- **Table complexity limits** - Row/column errors with very complex tables
- **CPU performance** - Slower than competitors without GPU acceleration

**Integration with Your Architecture:**
```python
from magic_pdf.pipe.UNIPipe import UNIPipe

# Requires substantial setup and GPU
pipe = UNIPipe(pdf_path)
result = pipe.pipe_parse()
markdown = result.get_markdown()
```

**Best For:**
- Academic papers with complex formulas
- Multilingual document collections
- Tables with merged cells and complex structures
- GPU-enabled production environments
- Chinese, Japanese, Korean document processing

**Verdict:** **High-quality alternative** when GPU resources are available and complex table/formula extraction is critical. Consider for graph construction phase where entity extraction quality matters most.

---

### 3. Unstructured

**Overview:**  
Unstructured is a comprehensive document preprocessing library designed specifically for RAG and NLP pipelines.

**Key Strengths:**
- **Apache 2.0 License** - Permissive for commercial use
- **Semantic chunking** - Creates meaningful content boundaries for embeddings
- **Element detection** - Identifies and labels document components (Title, NarrativeText, Table, etc.)
- **Native RAG integration** - Built for LangChain/LlamaIndex workflows
- **Multi-format support** - PDF, DOCX, PPTX, Images, HTML, Markdown
- **OCR support** - Built-in OCR for scanned documents
- **Layout analysis** - Identifies document structure and hierarchy
- **Production-ready** - Both open-source library and commercial API available

**Performance:**
- CPU (x86): 4.2 sec/page
- CPU (M3 Max): 2.7 sec/page
- No GPU acceleration

**Weaknesses:**
- Slower than PyMuPDF and pypdfium2 for simple extraction
- Higher memory usage for advanced features
- Table extraction accuracy lower than specialized tools (75% on complex tables)
- Configuration complexity for optimal results

**Integration with Your Architecture:**
```python
from unstructured.partition.auto import partition

# Semantic element detection
elements = partition(filename="document.pdf")

for element in elements:
    print(f"{element.category}: {element.text}")
    # Categories: Title, NarrativeText, Table, ListItem, etc.

# LangChain integration
from langchain_community.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader("document.pdf", mode="elements")
documents = loader.load()
```

**Best For:**
- RAG systems requiring semantic chunking
- Document analysis with element-level granularity
- Pipelines needing content classification
- When semantic boundaries matter more than raw speed

**Verdict:** **Specialized alternative** for the semantic chunking phase mentioned in your architecture. Particularly valuable for entity extraction and graph construction where understanding document structure is critical.

---

### 4. PyMuPDF (fitz)

**Overview:**  
PyMuPDF is a mature, high-performance Python binding to the MuPDF library, offering excellent speed and reliability for text extraction.

**Key Strengths:**
- **Blazing fast** - 42ms per document for text extraction
- **Mature and stable** - Battle-tested in production for years
- **Multi-format support** - PDF, XPS, EPUB, CBZ, FB2
- **Low resource usage** - Minimal memory footprint
- **Excellent documentation** - Comprehensive guides and examples
- **High accuracy** - Consistently outperforms in benchmark studies
- **Table extraction** - Good table detection and extraction
- **LangChain integration** - Native support for RAG workflows

**Performance:**
- Text extraction: 0.042 seconds per document
- Consistently fastest for basic text extraction tasks

**Weaknesses:**
- **AGPL/Commercial License** - Requires commercial license for proprietary use
- **Limited semantic understanding** - No automatic element classification
- **OCR not built-in** - Requires Tesseract integration for scanned docs
- **Complex table rendering** - May struggle with heavily formatted tables

**Integration with Your Architecture:**
```python
import pymupdf  # fitz

doc = pymupdf.open("document.pdf")
text = ""
for page in doc:
    text += page.get_text()

# LangChain integration
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("document.pdf")
documents = loader.load()
```

**Best For:**
- High-volume text extraction
- Text-heavy PDFs without complex layouts
- Resource-constrained environments
- When speed is critical and licensing is acceptable

**Verdict:** **Lightweight fallback** for the RecursiveCharacterTextSplitter phase. Excellent for baseline processing when document structure is less critical. Note: AGPL license requires careful consideration for commercial use.

---

### 5. Marker

**Overview:**  
Marker is an open-source tool specifically designed for PDF-to-Markdown conversion with OCR support, developed by Vik Paruchuri.

**Key Strengths:**
- **PDF-to-Markdown focus** - Optimized output for LLM consumption
- **Multi-format support** - PDF, EPUB, MOBI, DOCX, XLSX, Images
- **Good formula extraction** - Mathematical formula recognition
- **OCR support** - Tesseract OCR with optional Surya GPU acceleration
- **Clean output** - Well-structured Markdown with proper hierarchy
- **Active development** - Regular updates and improvements

**Performance:**
- CPU (x86): 16 sec/page
- CPU (M3 Mac): 4.2 sec/page
- GPU (L4): 0.86 sec/page

**Weaknesses:**
- **GPL 3.0 License** - Copyleft requirements may restrict commercial use
- **Slower on CPU** - Significantly slower than competitors without GPU
- **Limited language support** - Fewer languages than MinerU
- **Image handling** - Some issues with figure processing
- **Resource intensive** - Higher memory requirements

**Integration with Your Architecture:**
```python
from marker.convert import convert_single_pdf

# Convert to Markdown
markdown, images, metadata = convert_single_pdf(
    "document.pdf",
    output_format="markdown"
)
```

**Best For:**
- Converting PDF documentation to Markdown
- Projects already using GPL-licensed components
- When GPU acceleration is available
- Research and academic use cases

**Verdict:** **Consider for specialized use** in the PDF Strategy phase mentioned in your architecture document. GPL license may be problematic for commercial Graph RAG deployment.

---

### 6. pypdfium2

**Overview:**  
pypdfium2 is a Python binding to Google's PDFium library, offering ultra-fast basic text extraction.

**Key Strengths:**
- **Apache 2.0/BSD License** - Fully permissive
- **Ultra-fast** - 0.003 seconds for basic extraction (100x faster than others)
- **Low overhead** - Minimal resource requirements
- **Clean text output** - Simple, reliable text extraction
- **Good accuracy** - Consistently high scores in benchmarks
- **LangChain support** - Easy integration

**Performance:**
- Text extraction: 0.003 seconds per document (fastest of all tested tools)

**Weaknesses:**
- **Basic features only** - No semantic understanding or element detection
- **Limited table extraction** - Moderate table handling
- **No formula support** - Cannot process mathematical equations
- **No OCR** - Cannot handle scanned documents
- **Minimal structure preservation** - Loses complex layout information

**Integration with Your Architecture:**
```python
import pypdfium2 as pdfium

pdf = pdfium.PdfDocument("document.pdf")
text = ""
for page in pdf:
    textpage = page.get_textpage()
    text += textpage.get_text_range()

# LangChain integration
from langchain_community.document_loaders import PyPDFium2Loader
loader = PyPDFium2Loader("document.pdf")
documents = loader.load()
```

**Best For:**
- Ultra-fast baseline text extraction
- Development and testing environments
- Simple PDFs with primarily text content
- When speed is paramount and structure doesn't matter

**Verdict:** **Development/testing tool** for rapid iteration during initial phases. Not recommended for production due to limited feature set for Graph RAG requirements.

---

### 7. pdfplumber

**Overview:**  
pdfplumber is a MIT-licensed library built on pdfminer.six, specialized in table extraction with visual debugging capabilities.

**Key Strengths:**
- **MIT License** - Fully permissive
- **Excellent table extraction** - Best-in-class for structured tables
- **Visual debugging** - Interactive tools for extraction tuning
- **Character-level detail** - Access to detailed positioning data
- **Good documentation** - Clear examples and guides
- **Active maintenance** - Regular updates

**Performance:**
- Variable, typically 0.10 seconds per document for basic extraction

**Weaknesses:**
- **Slower than PyMuPDF** - Less optimized for pure speed
- **Configuration required** - Needs tuning for optimal table extraction
- **Limited structure understanding** - No semantic element detection
- **Basic text extraction** - Occasional spacing artifacts

**Integration with Your Architecture:**
```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        tables = page.extract_tables()

# LangChain integration
from langchain_community.document_loaders import PDFPlumberLoader
loader = PDFPlumberLoader("document.pdf")
documents = loader.load()
```

**Best For:**
- Table-heavy documents
- Financial reports with structured data
- When table accuracy is more important than speed
- Debugging complex extraction issues

**Verdict:** **Specialized tool** for table extraction phase. Consider using in combination with other parsers when tables are critical for entity extraction.

---

### 8. pymupdf4llm

**Overview:**  
pymupdf4llm is a specialized wrapper around PyMuPDF designed specifically for LLM-friendly Markdown output.

**Key Strengths:**
- **AGPL License** - Same as PyMuPDF
- **LLM-optimized output** - Clean Markdown with proper formatting
- **Fast processing** - 0.12 seconds per document
- **Preserves hierarchy** - Maintains headings and structure
- **Good table handling** - Tables formatted for LLM consumption
- **RAG-ready** - Designed for embedding and retrieval workflows

**Performance:**
- 0.12 seconds per document

**Weaknesses:**
- **AGPL License** - Requires commercial license for proprietary use
- **Less flexible** - Optimized for specific use case
- **Limited customization** - Fewer options than base PyMuPDF

**Integration with Your Architecture:**
```python
import pymupdf4llm

# Direct Markdown output
markdown = pymupdf4llm.to_markdown("document.pdf")

# Can be directly fed to LangChain text splitters
from langchain.text_splitter import MarkdownHeaderTextSplitter
splitter = MarkdownHeaderTextSplitter(...)
chunks = splitter.split_text(markdown)
```

**Best For:**
- Direct PDF-to-Markdown conversion for RAG
- When licensing constraints allow AGPL
- Fast preprocessing for LLM pipelines

**Verdict:** **Alternative to PyMuPDF** specifically for Markdown output in the MarkdownHeaderTextSplitter phase of your chunking strategy.

---

## Performance Benchmarks Summary

### Text Extraction Accuracy (DocLayNet Dataset)
Based on F1 scores across document categories:

**Financial Documents:**
1. PyMuPDF: 0.94
2. pypdfium2: 0.93
3. pdfplumber: 0.92
4. pdfminer.six: 0.91

**Scientific Documents (with formulas):**
1. Nougat (deep learning): 0.87
2. PyMuPDF: 0.68
3. pypdfium2: 0.67
4. Others: <0.65

**General Text:**
- PyMuPDF and pypdfium2 consistently achieve 90%+ accuracy
- All parsers struggle with Patents and Scientific documents
- Learning-based tools (Nougat, MinerU) excel with complex layouts

### Speed Comparison (seconds per page)

**CPU Performance (x86):**
1. pypdfium2: 0.003 (100x faster)
2. PyMuPDF: 0.042
3. pymupdf4llm: 0.12
4. Unstructured: 4.2
5. MinerU: 3.3
6. Docling: 3.1
7. Marker: 16.0

**GPU Performance (Nvidia L4):**
1. MinerU: 0.21
2. Docling: 0.49
3. Marker: 0.86

### Table Extraction (Complex Documents)

**Accuracy on Complex Tables:**
1. MinerU: 97.9% (HTML rendering)
2. Docling: 95%+
3. pdfplumber: 90%+
4. Unstructured: 75%
5. PyMuPDF: 70-80%

---

## Licensing Considerations

### Permissive Licenses (Commercial-Friendly)
- **MIT**: Docling, pdfplumber
- **Apache 2.0**: MinerU, Unstructured, pypdfium2
- **BSD**: pypdfium2 (PDFium component)

### Copyleft Licenses (Restrictions Apply)
- **GPL 3.0**: Marker (strong copyleft)
- **AGPL 3.0**: PyMuPDF, pymupdf4llm (requires commercial license for proprietary software)

**Recommendation for Your Architecture:** Prioritize MIT/Apache 2.0 licensed tools (Docling, Unstructured, pdfplumber) to avoid licensing complications in commercial Graph RAG deployment.

---

## Integration with Your Graph RAG Architecture

### Document Processing Pipeline Mapping

**Phase 1: Document Loading**
- **Primary**: Docling (multi-format, native integration)
- **Alternative**: Unstructured (semantic understanding)
- **Fallback**: PyMuPDF (speed for text-heavy docs)

**Phase 2: Text Chunking - RecursiveCharacterTextSplitter**
- **Primary**: Docling output → LangChain splitter
- **Fast option**: PyMuPDF + RecursiveCharacterTextSplitter
- **Quality option**: Unstructured elements → LangChain

**Phase 3: Semantic Chunking (High-Quality Strategy)**
- **Primary**: Docling + SemanticChunker
- **Alternative**: Unstructured native semantic elements
- **Tables**: MinerU for complex table structures

**Phase 4: Markdown Strategy**
- **Primary**: Docling → Markdown export
- **Alternative**: pymupdf4llm for AGPL-acceptable projects
- **Specialized**: Marker for pure PDF-to-Markdown

**Phase 5: PDF Strategy**
- **Primary**: Docling (layout-aware, comprehensive)
- **Complex tables**: MinerU with GPU acceleration
- **Simple PDFs**: PyMuPDF for speed

### Recommended Architecture Integration

```python
# config.yaml
document_parser:
  primary: "docling"
  fallback: "pymupdf"
  strategies:
    complex_tables: "mineru"
    markdown_conversion: "docling"
    high_volume: "pymupdf"
    semantic_chunking: "unstructured"

# app/document_parsing/factory.py
from typing import Protocol

class DocumentParser(Protocol):
    def parse(self, file_path: str) -> Document: ...

class DoclingParser:
    def __init__(self):
        from docling.document_converter import DocumentConverter
        self.converter = DocumentConverter()
    
    def parse(self, file_path: str) -> Document:
        result = self.converter.convert(file_path)
        return Document(
            text=result.document.export_to_markdown(),
            metadata=result.document.export_to_dict()
        )

class MinerUParser:  # For complex tables
    def parse(self, file_path: str) -> Document:
        # MinerU implementation
        pass

class UnstructuredParser:  # For semantic understanding
    def parse(self, file_path: str) -> Document:
        from unstructured.partition.auto import partition
        elements = partition(filename=file_path)
        return Document(
            text="\n".join([el.text for el in elements]),
            elements=elements  # Preserve semantic structure
        )

# Factory pattern for pluggable parsers
class ParserFactory:
    _parsers = {
        "docling": DoclingParser,
        "mineru": MinerUParser,
        "unstructured": UnstructuredParser,
        "pymupdf": PyMuPDFParser,
    }
    
    @classmethod
    def create(cls, parser_type: str) -> DocumentParser:
        return cls._parsers[parser_type]()
```

---

## Deployment Recommendations

### Development Environment
**Setup:**
```yaml
# docker-compose.yml additions
services:
  document-parser:
    image: python:3.10
    volumes:
      - ./data:/data
    environment:
      - PARSER_TYPE=docling
```

**Installation:**
```bash
# Primary parser (Docling)
pip install docling

# Alternative parsers
pip install pymupdf  # For fallback
pip install unstructured[all-docs]  # For semantic chunking
pip install pdfplumber  # For table extraction
```

### Production Environment (CPU-Only)

**Recommended Stack:**
- **Primary**: Docling (1.27 sec/page on M3, 3.1 sec/page on x86)
- **Fallback**: PyMuPDF (0.042 sec for simple extraction)
- **Table Specialist**: pdfplumber (when tables are critical)

**Resource Requirements:**
- CPU: 4+ cores recommended
- RAM: 8GB minimum, 16GB recommended
- Storage: 10GB for models and cache

### Production Environment (GPU-Enabled)

**Recommended Stack:**
- **Primary**: Docling (0.49 sec/page on L4 GPU)
- **Complex Documents**: MinerU (0.21 sec/page, best for tables/formulas)
- **Fallback**: PyMuPDF (for simple docs)

**Resource Requirements:**
- GPU: 16GB+ VRAM (for MinerU)
- GPU: 8GB VRAM sufficient (for Docling)
- CPU: 8+ cores
- RAM: 32GB+ recommended

---

## Decision Matrix

### Choose Docling If:
✓ You need native LangChain/LlamaIndex integration  
✓ MIT license is required for commercial use  
✓ Multi-format support (PDF, DOCX, PPTX) is essential  
✓ Balanced performance and accuracy are priorities  
✓ Local execution without external APIs is required  
✓ You want the best general-purpose solution

### Choose MinerU If:
✓ GPU resources are available  
✓ Complex table extraction is critical  
✓ Multilingual (especially CJK) content is common  
✓ Mathematical formula recognition is essential  
✓ Best-in-class table accuracy justifies GPU cost  
✓ Apache 2.0 license is acceptable

### Choose Unstructured If:
✓ Semantic chunking is more important than speed  
✓ Element-level document understanding is needed  
✓ RAG-specific features are prioritized  
✓ You need document component classification  
✓ Apache 2.0 license works for your project

### Choose PyMuPDF If:
✓ Speed is the top priority  
✓ Documents are primarily text-heavy  
✓ AGPL/Commercial licensing is acceptable  
✓ You need a mature, battle-tested solution  
✓ Resource constraints are significant

### Choose pdfplumber If:
✓ Table extraction accuracy is paramount  
✓ You need visual debugging capabilities  
✓ MIT license is required  
✓ Tables are the primary content type

---

## Final Recommendation for Your Graph RAG Architecture

### Primary Stack (Recommended)
```
Document Parser: Docling (MIT License)
├── Advantages: Native RAG integration, balanced performance, permissive license
├── Use for: 80% of documents across all phases
└── Fallback: PyMuPDF for simple text extraction

Specialized Parsers:
├── Complex Tables: MinerU (when GPU available)
├── Semantic Chunking: Unstructured (element detection)
└── Table Debugging: pdfplumber (visual tools)
```

### Rationale
1. **Docling** provides the best balance of features, performance, and licensing for a production Graph RAG system
2. **MIT license** ensures no commercial restrictions
3. **Native LangChain/LlamaIndex integration** aligns with your architecture
4. **Multi-format support** covers PDF, DOCX, PPTX needs
5. **Performance** is acceptable on both CPU (1.27s/page) and GPU (0.49s/page)
6. **Modular design** allows adding MinerU or Unstructured for specialized needs

### Implementation Priority
1. **Week 1-2**: Implement Docling as primary parser
2. **Week 3**: Add PyMuPDF fallback for simple documents
3. **Week 4**: Integrate pdfplumber for table-heavy content
4. **Week 5+**: Evaluate MinerU for complex documents (if GPU available)
5. **Ongoing**: Monitor accuracy and adjust based on document types

### Cost-Benefit Analysis
- **Development Time**: Docling has clear docs and examples (1-2 days integration)
- **Performance**: Good enough for real-time processing (<2s per page on CPU)
- **Maintenance**: Active IBM Research project with strong community
- **Licensing**: No restrictions or ongoing costs
- **Scalability**: Can handle 1M+ documents with appropriate infrastructure

---

## Additional Resources

### Documentation
- **Docling**: https://github.com/docling-project/docling
- **MinerU**: https://github.com/opendatalab/MinerU
- **Unstructured**: https://github.com/Unstructured-IO/unstructured
- **PyMuPDF**: https://pymupdf.readthedocs.io/
- **LangChain Document Loaders**: https://python.langchain.com/docs/modules/data_connection/document_loaders/

### Benchmarks
- **DocLayNet Benchmark**: Comprehensive evaluation across 6 document categories
- **OmniDocBench**: 1,355 PDF pages across 9 document types
- **Docling Technical Report**: Head-to-head comparison of Docling, Marker, MinerU, Unstructured

### Community Support
- **Docling**: Active IBM Research team, growing community
- **MinerU**: Large Chinese AI community, extensive documentation
- **PyMuPDF**: Mature community, extensive StackOverflow presence
- **Unstructured**: Strong RAG/NLP community focus

---

## Conclusion

For your Graph RAG application, **Docling** emerges as the clear primary choice due to its:
- MIT license (most permissive)
- Native RAG framework integration
- Balanced performance across hardware profiles
- Comprehensive feature set
- Active development and support

Supplement Docling with **MinerU** for GPU-enabled environments with complex tables, and **pdfplumber** for specialized table extraction when needed. This combination provides the flexibility mentioned in your architecture while maintaining the "pluggable" design principle.

The parser selection directly impacts the quality of your knowledge graph construction. Docling's superior layout understanding and structure preservation make it ideal for the entity extraction phase, while its speed ensures the system can scale to production workloads.
