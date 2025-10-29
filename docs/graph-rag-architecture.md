# Graph RAG Application Architecture

## Executive Summary

This document outlines a production-ready Graph RAG (Retrieval-Augmented Generation) system using best-in-class open source components. The architecture combines traditional vector search with knowledge graph capabilities to enable sophisticated question-answering over user-provided documents.

## Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│                    (Streamlit / Gradio)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Application Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Document   │  │    Query     │  │   Response   │     │
│  │  Ingestion   │  │  Processing  │  │  Generation  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Storage Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Vector DB  │  │  Knowledge   │  │   Document   │     │
│  │  (Qdrant)    │  │    Graph     │  │    Store     │     │
│  │              │  │  (Neo4j)     │  │  (MinIO/S3)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Document Processing Pipeline

#### **Document Loader: LangChain Document Loaders**
- **Why**: Comprehensive support for 100+ document formats
- **Features**: PDF, DOCX, HTML, Markdown, CSV, JSON, and more
- **Maturity**: Part of LangChain ecosystem, battle-tested
- **Integration**: Seamless with downstream components

#### **Text Chunking: Pluggable Strategy Architecture**

The system implements a **strategy pattern** for text chunking to enable easy switching between different chunking approaches based on document type and quality requirements.

**Primary Strategy: LangChain RecursiveCharacterTextSplitter**
- **Why**:
  - Best general-purpose chunking solution
  - Respects document structure (paragraphs, sentences)
  - Fast and production-ready
  - Handles 80% of use cases effectively
  - Battle-tested in production environments

**High-Quality Strategy: LangChain SemanticChunker**
- **Why**:
  - 15-25% better retrieval accuracy
  - Creates semantically coherent chunks
  - Respects topic boundaries naturally
  - Ideal for graph construction (better entity extraction)
  - Worth the computational cost for critical documents

**Specialized Strategy: MarkdownHeaderTextSplitter**
- **Why**:
  - Perfect for technical documentation
  - Preserves document hierarchy
  - Maintains header context as metadata
  - Essential for structured markdown content

**PDF Strategy: Unstructured.io**
- **Why**:
  - Best-in-class PDF handling
  - Layout-aware chunking
  - Preserves tables, images, and formatting
  - Critical for PDF-heavy workloads

**Design Principle**: Configuration-driven chunking strategy selection enables optimal chunking for different document types and quality requirements without code changes.

**Typical Configuration**:
- Chunk size: 512-1024 tokens (2048-4096 characters)
- Overlap: 10-20% for context continuity
- Strategy selection based on document type and processing stage
- Separate strategies for vector search vs. graph construction

#### **Metadata Extraction: LlamaIndex Extractors**
- **Why**: Rich metadata extraction capabilities
- **Components**:
  - `SummaryExtractor` - Generate chunk summaries
  - `QuestionsAnsweredExtractor` - Hypothetical questions
  - `KeywordExtractor` - Key terms and concepts
  - `EntityExtractor` - Named entities (people, places, organizations)

### 2. Embedding and Vector Storage

#### **Embedding Model: Pluggable Architecture**

**Primary Recommendation**: `intfloat/e5-mistral-7b-instruct`
- **Why**:
  - State-of-the-art quality (top MTEB benchmarks)
  - 7 billion parameters for superior understanding
  - 4096 dimensions for rich semantic representation
  - Excellent for complex domain-specific retrieval
  - Requires GPU (16GB+ VRAM recommended)
- **Performance**: ~20% better retrieval accuracy than smaller models
- **Usage Note**: Requires instruction prefixes ("query: " for queries, "passage: " for documents)

**High-Quality Alternative**: `thenlper/gte-large`
- **Why**:
  - Excellent MTEB performance (comparable to BGE)
  - 1024 dimensions (good balance)
  - No instruction prefix needed (simpler implementation)
  - Lower resource requirements (2-4GB VRAM)
  - Faster inference than e5-mistral

**Lightweight Fallback**: `BAAI/bge-base-en-v1.5` or `all-MiniLM-L6-v2`
- For CPU-only deployments or resource-constrained environments
- Still production-quality performance

**Design Principle**: The architecture uses a **pluggable embedding interface** to allow easy swapping of models without code changes. This enables:
- Testing different models for your specific domain
- Upgrading to better models as they're released
- Scaling down for development environments
- Supporting different models for different document types

#### **Vector Database: Pluggable Architecture**

The system implements a **provider abstraction** for vector databases to enable easy switching between different backends based on scale and feature requirements.

#### **Vector Database: Pluggable Provider Architecture**

The system implements a **provider abstraction** for vector databases to enable seamless switching between different vector store solutions based on scale, features, and operational requirements.

**Primary Provider: Qdrant** ⭐ RECOMMENDED
- **Why Qdrant**:
  - Best overall balance of performance and features
  - Excellent metadata filtering (critical for RAG)
  - Fast queries (5-15ms p95 latency)
  - Rich filtering capabilities without performance penalty
  - HNSW indexing for fast approximate search
  - Hybrid search support (dense + sparse vectors)
  - Quantization for reduced memory footprint (4-8x reduction)
  - Docker-ready deployment (single container)
  - Production-proven and stable
  - Handles 1M-100M vectors efficiently
  - Clean Python client and comprehensive documentation

**Scale Alternative: Milvus**
- **When to use**: >100M vectors, Kubernetes deployment, GPU acceleration needs
- **Why Milvus**:
  - Designed for 100M-10B+ vectors
  - Cloud-native Kubernetes deployment
  - GPU acceleration support
  - Enterprise features (multi-tenancy, time travel, CDC)
  - Distributed architecture (separate compute/storage)
  - When scale exceeds Qdrant's sweet spot
  - Requires more DevOps expertise

**Feature Alternative: Weaviate**
- **When to use**: Built-in hybrid search and reranking are must-haves
- **Why Weaviate**:
  - Built-in hybrid search (BM25 + vector)
  - Integrated reranking capabilities
  - Generative search features
  - GraphQL API
  - Good for mid-scale with rich features (1M-50M vectors)
  - When you want everything in one package

**Simplicity Alternative: pgvector**
- **When to use**: Already using PostgreSQL, small-medium scale (<10M vectors)
- **Why pgvector**:
  - PostgreSQL extension (use existing database)
  - ACID transactions
  - No additional infrastructure
  - SQL interface (familiar)
  - Standard PostgreSQL tooling (backup, monitoring)
  - When you want to minimize moving parts

**Design Principle**: Configuration-driven vector database selection enables switching between providers without application code changes. Start with Qdrant for development and migrate to alternatives only when specific needs arise (massive scale → Milvus, advanced features → Weaviate, infrastructure minimization → pgvector).

### 3. Knowledge Graph

#### **Graph Database: Neo4j Community Edition**
- **Why Neo4j**:
  - Industry standard for graph databases
  - Cypher query language (intuitive and powerful)
  - Mature ecosystem and extensive documentation
  - Excellent visualization tools (Neo4j Browser)
  - APOC library for advanced operations
  - Strong performance for graph traversals
- **License**: GPLv3 (Community Edition)
- **Deployment**: Docker container recommended

#### **Graph Construction: LlamaIndex Property Graph**
- **Why LlamaIndex**:
  - Purpose-built for RAG with graph support
  - Automatic entity and relationship extraction
  - Integration with multiple graph stores
  - Support for hybrid vector + graph retrieval
- **Entity Extraction Methods**:
  - LLM-based extraction (using Ollama - see below)
  - Simple keyword/NER-based extraction
  - Schema-guided extraction for domain-specific use cases

**Graph Schema Design**:
```cypher
// Nodes
(:Document {id, title, source, upload_date})
(:Chunk {id, content, embedding_id, position})
(:Entity {name, type, description})
(:Concept {name, description})

// Relationships
(:Document)-[:CONTAINS]->(:Chunk)
(:Chunk)-[:MENTIONS]->(:Entity)
(:Entity)-[:RELATED_TO]->(:Entity)
(:Chunk)-[:NEXT_CHUNK]->(:Chunk)
(:Entity)-[:INSTANCE_OF]->(:Concept)
```

### 4. LLM Integration

#### **Pluggable LLM Provider Architecture**

The system implements a **provider abstraction** to support multiple LLM backends without code changes.

**Primary Provider: Ollama** (Development & MVP)
- **Why Ollama**:
  - Runs LLMs locally (no API costs, data privacy)
  - Dead-simple setup and model management
  - GPU and CPU support (flexibility)
  - Easy model switching for experimentation
  - Active community and frequent updates
- **Best For**: Development, prototyping, low-concurrency deployments (<10 users)

**Production Provider: vLLM** (Scale & Performance)
- **Why vLLM**:
  - 2-4x higher throughput than Ollama
  - Continuous batching for concurrent requests
  - Superior GPU memory utilization (PagedAttention)
  - OpenAI-compatible API
  - Built-in monitoring and metrics
- **Best For**: Production deployments with 10+ concurrent users, high throughput needs

**Recommended Models** (compatible with both providers):
- **Llama 3.1 8B**: Best balance of quality and speed
- **Mistral 7B**: Fast inference, good for extraction tasks
- **Llama 3.1 70B**: Higher quality responses (requires more resources, vLLM recommended)
- **Phi-3**: Excellent for resource-constrained environments

**Design Principle**: Configuration-driven provider selection enables seamless switching between Ollama (development) and vLLM (production) without any application code changes.

#### **LLM Framework: LangChain**
- **Why LangChain**:
  - De facto standard for LLM applications
  - Rich ecosystem of integrations
  - Built-in RAG patterns and chains
  - Memory management
  - Extensive documentation and examples
  - Works seamlessly with both Ollama and vLLM
- **Key Components**:
  - `ChatOllama` or `ChatOpenAI` (vLLM compatible) for LLM interaction
  - `ConversationalRetrievalChain` for Q&A
  - `GraphCypherQAChain` for graph queries
  - Custom chains for hybrid retrieval

### 5. Orchestration and Retrieval

#### **RAG Framework: LlamaIndex**
- **Why LlamaIndex**:
  - Designed specifically for RAG applications
  - Native graph RAG support
  - Advanced retrieval strategies
  - Query pipeline abstraction
- **Retrieval Strategies**:
  - **Vector Similarity**: Dense retrieval from Qdrant
  - **Graph Traversal**: Entity-based exploration in Neo4j
  - **Hybrid Retrieval**: Combine vector + graph results
  - **Metadata Filtering**: Pre-filter by document properties

**Recommended Retrieval Pipeline**:
1. **Query Analysis**: Extract entities and intent
2. **Parallel Retrieval**:
   - Vector search in Qdrant (top-k chunks)
   - Graph traversal in Neo4j (related entities)
3. **Reranking**: Use cross-encoder or LLM-based reranking
4. **Context Assembly**: Merge and deduplicate results
5. **Response Generation**: Generate answer with citations

### 6. Document Storage

#### **Object Storage: MinIO**
- **Why MinIO**:
  - S3-compatible API (easy migration to cloud)
  - Self-hosted and fully open source
  - High performance
  - Simple deployment
- **Alternative**: Local filesystem for simple deployments

### 7. User Interface

#### **UI Framework: Streamlit**
- **Why Streamlit**:
  - Rapid development (pure Python)
  - Built-in components for file upload, chat interface
  - Session state management
  - Easy deployment
  - Large community and marketplace

**Alternative**: **Gradio** (similar benefits, different API style)

**Key UI Features**:
- File upload interface (drag-and-drop)
- Processing status indicators
- Chat interface with message history
- Source citation display
- Graph visualization (using Cytoscape.js or vis.js)
- Document management (view uploaded docs)

## Detailed Architecture Components

### Pluggable LLM Provider Architecture

The system implements a **strategy pattern** for LLM providers to enable seamless switching between Ollama and vLLM:

```python
# Abstract interface
from abc import ABC, abstractmethod
from typing import List, Dict, Iterator, Optional

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a complete response."""
        pass
    
    @abstractmethod
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate a streaming response."""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat completion with message history."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

# Ollama implementation
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, **kwargs):
        self.base_url = base_url
        self.model = model
        self.llm = Ollama(base_url=base_url, model=model, **kwargs)
        self.chat_llm = ChatOllama(base_url=base_url, model=model, **kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        return self.llm.invoke(prompt, **kwargs)
    
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        for chunk in self.llm.stream(prompt, **kwargs):
            yield chunk
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        response = self.chat_llm.invoke(langchain_messages, **kwargs)
        return response.content
    
    @property
    def model_name(self) -> str:
        return self.model

# vLLM implementation (using OpenAI-compatible API)
from langchain_openai import ChatOpenAI

class VLLMProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", **kwargs):
        self.base_url = base_url
        self.model = model
        # vLLM exposes OpenAI-compatible API
        self.chat_llm = ChatOpenAI(
            base_url=f"{base_url}/v1",
            api_key=api_key,  # vLLM doesn't require real API key
            model=model,
            **kwargs
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        messages = [{"role": "user", "content": prompt}]
        for chunk in self.chat_llm.stream(messages, **kwargs):
            yield chunk.content
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        response = self.chat_llm.invoke(langchain_messages, **kwargs)
        return response.content
    
    @property
    def model_name(self) -> str:
        return self.model

# Factory for provider selection
def get_llm_provider(config: dict) -> LLMProvider:
    """Factory function to create LLM provider based on config."""
    provider_map = {
        'ollama': OllamaProvider,
        'vllm': VLLMProvider,
    }
    
    provider_type = config.get('provider', 'ollama')
    provider_class = provider_map.get(provider_type)
    
    if not provider_class:
        raise ValueError(f"Unknown LLM provider: {provider_type}")
    
    return provider_class(
        base_url=config['base_url'],
        model=config['model'],
        temperature=config.get('temperature', 0.7),
        max_tokens=config.get('max_tokens', 2048),
    )
```

**Benefits of Pluggable Design**:
- Switch providers via config file (zero code changes)
- A/B test Ollama vs vLLM on your workload
- Use Ollama for development, vLLM for production
- Easy migration path as needs evolve
- Support for future LLM providers

**Configuration Example** (`config.yaml`):
```yaml
llm:
  # Development configuration (Ollama)
  provider: "ollama"
  base_url: "http://ollama:11434"
  model: "llama3.1:8b"
  temperature: 0.7
  max_tokens: 2048

  # Production configuration (vLLM) - commented out
  # provider: "vllm"
  # base_url: "http://vllm:8000"
  # model: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  # temperature: 0.7
  # max_tokens: 2048
```

**Usage in Application**:
```python
from app.config import load_config
from app.llm.factory import get_llm_provider

# Load configuration
config = load_config('config.yaml')

# Get LLM provider (works with Ollama or vLLM)
llm = get_llm_provider(config['llm'])

# Use the provider
response = llm.generate("Explain quantum computing")
print(response)

# Streaming
for chunk in llm.stream_generate("Write a poem about AI"):
    print(chunk, end='', flush=True)

# Chat with history
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is RAG?"}
]
response = llm.chat(messages)
```

### Pluggable Embedding Architecture

The system implements a **strategy pattern** for embeddings to allow seamless model swapping:

```python
# Abstract interface
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

# Concrete implementations
class E5MistralEmbedding(EmbeddingProvider):
    def __init__(self):
        self.model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
    
    def embed_query(self, text: str) -> List[float]:
        return self._encode("query: " + text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._encode("passage: " + t) for t in texts]
    
    @property
    def dimension(self) -> int:
        return 4096

class GTELargeEmbedding(EmbeddingProvider):
    def __init__(self):
        self.model = SentenceTransformer('thenlper/gte-large')
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    @property
    def dimension(self) -> int:
        return 1024

# Configuration-driven selection
def get_embedding_provider(config: dict) -> EmbeddingProvider:
    provider_map = {
        'e5-mistral': E5MistralEmbedding,
        'gte-large': GTELargeEmbedding,
        'bge-base': BGEBaseEmbedding,
        'minilm': MiniLMEmbedding,
    }
    return provider_map[config['embedding_model']]()
```

**Benefits**:
- Switch models via config file (no code changes)
- Easy A/B testing of different embeddings
- Graceful degradation (fallback to lighter models)
- Future-proof (add new models easily)

**Configuration Example** (`config.yaml`):
```yaml
embedding:
  model: "e5-mistral"  # or "gte-large", "bge-base", etc.
  device: "cuda"       # or "cpu"
  batch_size: 32
```

### Pluggable Text Chunking Architecture

The system implements a **strategy pattern** for text chunking to optimize for different document types and quality requirements:

```python
# Abstract interface
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk_document(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """Chunk document and return chunks with metadata."""
        pass
    
    @abstractmethod
    def chunk_documents(self, documents: List[str], metadatas: List[Dict] = None) -> List[Dict[str, Any]]:
        """Chunk multiple documents."""
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        pass

# Recursive Character Splitter (General Purpose)
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RecursiveChunker(ChunkingStrategy):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            **kwargs
        )
    
    def chunk_document(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        chunks = self.splitter.split_text(text)
        return [
            {
                'content': chunk,
                'metadata': {**(metadata or {}), 'chunk_index': i, 'strategy': self.strategy_name},
                'chunk_size': len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def chunk_documents(self, documents: List[str], metadatas: List[Dict] = None) -> List[Dict[str, Any]]:
        all_chunks = []
        for i, doc in enumerate(documents):
            meta = metadatas[i] if metadatas else None
            all_chunks.extend(self.chunk_document(doc, meta))
        return all_chunks
    
    @property
    def strategy_name(self) -> str:
        return "recursive"

# Semantic Chunker (High Quality)
from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker

class SemanticChunker(ChunkingStrategy):
    def __init__(self, embeddings, breakpoint_threshold_type: str = "percentile", 
                 breakpoint_threshold_amount: int = 90, **kwargs):
        self.splitter = LCSemanticChunker(
            embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            **kwargs
        )
    
    def chunk_document(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        chunks = self.splitter.split_text(text)
        return [
            {
                'content': chunk,
                'metadata': {**(metadata or {}), 'chunk_index': i, 'strategy': self.strategy_name},
                'chunk_size': len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def chunk_documents(self, documents: List[str], metadatas: List[Dict] = None) -> List[Dict[str, Any]]:
        all_chunks = []
        for i, doc in enumerate(documents):
            meta = metadatas[i] if metadatas else None
            all_chunks.extend(self.chunk_document(doc, meta))
        return all_chunks
    
    @property
    def strategy_name(self) -> str:
        return "semantic"

# Markdown Header Splitter (Structure-Aware)
from langchain.text_splitter import MarkdownHeaderTextSplitter

class MarkdownChunker(ChunkingStrategy):
    def __init__(self, headers_to_split_on: List[tuple] = None, **kwargs):
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            **kwargs
        )
        # Secondary splitter for size control
        self.size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
    
    def chunk_document(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        # First split by headers
        md_chunks = self.splitter.split_text(text)
        
        # Then ensure chunks aren't too large
        final_chunks = []
        for i, chunk in enumerate(md_chunks):
            if len(chunk) > 1200:
                # Split large sections further
                sub_chunks = self.size_splitter.split_text(chunk)
                for j, sub_chunk in enumerate(sub_chunks):
                    final_chunks.append({
                        'content': sub_chunk,
                        'metadata': {**(metadata or {}), 'chunk_index': f"{i}.{j}", 'strategy': self.strategy_name},
                        'chunk_size': len(sub_chunk)
                    })
            else:
                final_chunks.append({
                    'content': chunk,
                    'metadata': {**(metadata or {}), 'chunk_index': i, 'strategy': self.strategy_name},
                    'chunk_size': len(chunk)
                })
        return final_chunks
    
    def chunk_documents(self, documents: List[str], metadatas: List[Dict] = None) -> List[Dict[str, Any]]:
        all_chunks = []
        for i, doc in enumerate(documents):
            meta = metadatas[i] if metadatas else None
            all_chunks.extend(self.chunk_document(doc, meta))
        return all_chunks
    
    @property
    def strategy_name(self) -> str:
        return "markdown"

# Factory for chunking strategy selection
def get_chunking_strategy(config: dict, embeddings=None) -> ChunkingStrategy:
    """Factory function to create chunking strategy based on config."""
    strategy_map = {
        'recursive': RecursiveChunker,
        'semantic': SemanticChunker,
        'markdown': MarkdownChunker,
    }
    
    strategy_type = config.get('strategy', 'recursive')
    strategy_class = strategy_map.get(strategy_type)
    
    if not strategy_class:
        raise ValueError(f"Unknown chunking strategy: {strategy_type}")
    
    # Semantic chunker requires embeddings
    if strategy_type == 'semantic':
        if not embeddings:
            raise ValueError("Semantic chunking requires embeddings")
        return strategy_class(embeddings, **config.get('params', {}))
    
    return strategy_class(**config.get('params', {}))

# Multi-Strategy Chunking (for Graph RAG)
class HybridChunker:
    """Use different strategies for vector search vs. graph construction."""
    
    def __init__(self, vector_strategy: ChunkingStrategy, graph_strategy: ChunkingStrategy):
        self.vector_strategy = vector_strategy
        self.graph_strategy = graph_strategy
    
    def chunk_for_vector_search(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """Fast chunking optimized for vector similarity search."""
        return self.vector_strategy.chunk_document(text, metadata)
    
    def chunk_for_graph_construction(self, text: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """High-quality chunking optimized for entity extraction."""
        return self.graph_strategy.chunk_document(text, metadata)
```

**Benefits of Pluggable Chunking**:
- Switch strategies via configuration (no code changes)
- Use different strategies for different document types
- Optimize for vector search vs. graph construction separately
- A/B test chunking strategies on your data
- Easy to add new strategies (Unstructured.io, custom logic)

**Configuration Example** (`config.yaml`):
```yaml
chunking:
  # For vector search (fast, good coverage)
  vector:
    strategy: "recursive"
    params:
      chunk_size: 800
      chunk_overlap: 150
  
  # For graph construction (high quality, better entities)
  graph:
    strategy: "semantic"
    params:
      breakpoint_threshold_type: "percentile"
      breakpoint_threshold_amount: 90
  
  # Document-type specific overrides
  document_types:
    markdown:
      strategy: "markdown"
      params:
        headers_to_split_on:
          - ["#", "Header 1"]
          - ["##", "Header 2"]
          - ["###", "Header 3"]
    pdf:
      strategy: "recursive"  # Could use Unstructured.io
      params:
        chunk_size: 1000
        chunk_overlap: 200
```

**Usage in Application**:
```python
from app.config import load_config
from app.chunking.factory import get_chunking_strategy, HybridChunker
from app.embeddings.factory import get_embedding_provider

# Load configuration
config = load_config('config.yaml')

# Get embedding provider (needed for semantic chunking)
embeddings = get_embedding_provider(config['embedding'])

# Create chunking strategies
vector_chunker = get_chunking_strategy(config['chunking']['vector'])
graph_chunker = get_chunking_strategy(
    config['chunking']['graph'],
    embeddings=embeddings
)

# Create hybrid chunker
hybrid_chunker = HybridChunker(vector_chunker, graph_chunker)

# Chunk document
document = "Your document text here..."
metadata = {"source": "document.pdf", "page": 1}

# Get chunks for vector search (fast)
vector_chunks = hybrid_chunker.chunk_for_vector_search(document, metadata)

# Get chunks for graph construction (high quality)
graph_chunks = hybrid_chunker.chunk_for_graph_construction(document, metadata)
```

### Pluggable Vector Database Architecture

The system implements a **provider abstraction** for vector databases to enable seamless switching between Qdrant, Milvus, Weaviate, or pgvector:

```python
# Abstract interface
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorDBProvider(ABC):
    @abstractmethod
    def create_collection(self, name: str, vector_size: int, distance_metric: str = "cosine"):
        """Create a new collection/index."""
        pass
    
    @abstractmethod
    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        """Add vectors with metadata to collection."""
        pass
    
    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors with optional metadata filtering."""
        pass
    
    @abstractmethod
    def delete(self, collection_name: str, ids: List[str]):
        """Delete vectors by ID."""
        pass
    
    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        pass

# Qdrant implementation
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

class QdrantProvider(VectorDBProvider):
    def __init__(self, host: str = "localhost", port: int = 6333, **kwargs):
        self.client = QdrantClient(host=host, port=port, **kwargs)
    
    def create_collection(self, name: str, vector_size: int, distance_metric: str = "cosine"):
        distance_map = {
            "cosine": Distance.COSINE,
            "l2": Distance.EUCLID,
            "ip": Distance.DOT,
        }
        
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_map.get(distance_metric, Distance.COSINE),
            ),
        )
    
    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        if ids is None:
            ids = [str(i) for i in range(len(vectors))]
        
        points = [
            PointStruct(id=id, vector=vector, payload=metadata)
            for id, vector, metadata in zip(ids, vectors, metadatas)
        ]
        
        self.client.upsert(collection_name=collection_name, points=points)
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # Convert generic filter to Qdrant filter format
        qdrant_filter = None
        if filter:
            qdrant_filter = self._convert_filter_to_qdrant(filter)
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=k,
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "metadata": result.payload,
            }
            for result in results
        ]
    
    def delete(self, collection_name: str, ids: List[str]):
        self.client.delete(collection_name=collection_name, points_selector=ids)
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        info = self.client.get_collection(collection_name=collection_name)
        return {
            "name": collection_name,
            "vector_size": info.config.params.vectors.size,
            "count": info.points_count,
        }
    
    def _convert_filter_to_qdrant(self, filter: Dict[str, Any]) -> Filter:
        """Convert generic filter to Qdrant filter format."""
        conditions = []
        for key, value in filter.items():
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
        return Filter(must=conditions)

# Milvus implementation
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

class MilvusProvider(VectorDBProvider):
    def __init__(self, host: str = "localhost", port: int = 19530, **kwargs):
        connections.connect(host=host, port=port, **kwargs)
        self.collections = {}
    
    def create_collection(self, name: str, vector_size: int, distance_metric: str = "cosine"):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, description=f"Collection {name}")
        collection = Collection(name=name, schema=schema)
        
        # Create index
        metric_map = {"cosine": "COSINE", "l2": "L2", "ip": "IP"}
        index_params = {
            "index_type": "HNSW",
            "metric_type": metric_map.get(distance_metric, "COSINE"),
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        self.collections[name] = collection
    
    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        collection = self._get_collection(collection_name)
        data = [vectors, metadatas]
        collection.insert(data)
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        collection = self._get_collection(collection_name)
        collection.load()
        
        # Convert filter to Milvus expression
        expr = self._convert_filter_to_milvus(filter) if filter else None
        
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=k,
            expr=expr,
            output_fields=["metadata"]
        )
        
        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "metadata": hit.entity.get("metadata"),
            }
            for hit in results[0]
        ]
    
    def delete(self, collection_name: str, ids: List[str]):
        collection = self._get_collection(collection_name)
        collection.delete(expr=f"id in [{','.join(ids)}]")
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        collection = self._get_collection(collection_name)
        return {
            "name": collection_name,
            "count": collection.num_entities,
        }
    
    def _get_collection(self, name: str) -> Collection:
        if name not in self.collections:
            self.collections[name] = Collection(name)
        return self.collections[name]
    
    def _convert_filter_to_milvus(self, filter: Dict[str, Any]) -> str:
        """Convert generic filter to Milvus expression."""
        conditions = [f'metadata["{k}"] == "{v}"' for k, v in filter.items()]
        return " and ".join(conditions)

# Weaviate implementation
import weaviate
from weaviate.classes.config import Configure

class WeaviateProvider(VectorDBProvider):
    def __init__(self, host: str = "localhost", port: int = 8080, **kwargs):
        self.client = weaviate.connect_to_local(host=host, port=port)
    
    def create_collection(self, name: str, vector_size: int, distance_metric: str = "cosine"):
        self.client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=distance_metric,
            ),
        )
    
    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        collection = self.client.collections.get(collection_name)
        
        for vector, metadata in zip(vectors, metadatas):
            collection.data.insert(
                properties=metadata,
                vector=vector,
            )
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        collection = self.client.collections.get(collection_name)
        
        # Weaviate filtering uses where clause
        where_filter = self._convert_filter_to_weaviate(filter) if filter else None
        
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=k,
            where=where_filter,
        )
        
        return [
            {
                "id": str(obj.uuid),
                "score": obj.metadata.distance if hasattr(obj.metadata, 'distance') else None,
                "metadata": obj.properties,
            }
            for obj in response.objects
        ]
    
    def delete(self, collection_name: str, ids: List[str]):
        collection = self.client.collections.get(collection_name)
        for id in ids:
            collection.data.delete_by_id(id)
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        collection = self.client.collections.get(collection_name)
        return {
            "name": collection_name,
            # Count would require aggregation query
        }
    
    def _convert_filter_to_weaviate(self, filter: Dict[str, Any]):
        """Convert generic filter to Weaviate where clause."""
        from weaviate.classes.query import Filter as WeaviateFilter
        # Simplified - would need more complex conversion for production
        conditions = []
        for key, value in filter.items():
            conditions.append(WeaviateFilter.by_property(key).equal(value))
        return conditions[0] if len(conditions) == 1 else None

# pgvector implementation (using psycopg)
import psycopg
from pgvector.psycopg import register_vector

class PgvectorProvider(VectorDBProvider):
    def __init__(self, connection_string: str = "postgresql://localhost/vectordb", **kwargs):
        self.conn_string = connection_string
    
    def create_collection(self, name: str, vector_size: int, distance_metric: str = "cosine"):
        with psycopg.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                # Create table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {name} (
                        id SERIAL PRIMARY KEY,
                        vector vector({vector_size}),
                        metadata JSONB
                    )
                """)
                
                # Create index
                operator_map = {"cosine": "vector_cosine_ops", "l2": "vector_l2_ops", "ip": "vector_ip_ops"}
                operator = operator_map.get(distance_metric, "vector_cosine_ops")
                
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {name}_vector_idx 
                    ON {name} USING hnsw (vector {operator})
                """)
                conn.commit()
    
    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        with psycopg.connect(self.conn_string) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for vector, metadata in zip(vectors, metadatas):
                    cur.execute(
                        f"INSERT INTO {collection_name} (vector, metadata) VALUES (%s, %s)",
                        (vector, psycopg.types.json.Json(metadata))
                    )
                conn.commit()
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        with psycopg.connect(self.conn_string) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                # Build filter clause
                where_clause = self._build_where_clause(filter) if filter else ""
                
                query = f"""
                    SELECT id, metadata, 1 - (vector <=> %s::vector) AS score
                    FROM {collection_name}
                    {where_clause}
                    ORDER BY vector <=> %s::vector
                    LIMIT %s
                """
                
                cur.execute(query, (query_vector, query_vector, k))
                results = cur.fetchall()
                
                return [
                    {
                        "id": str(row[0]),
                        "metadata": row[1],
                        "score": float(row[2]),
                    }
                    for row in results
                ]
    
    def delete(self, collection_name: str, ids: List[str]):
        with psycopg.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {collection_name} WHERE id = ANY(%s)", (ids,))
                conn.commit()
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        with psycopg.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {collection_name}")
                count = cur.fetchone()[0]
                return {"name": collection_name, "count": count}
    
    def _build_where_clause(self, filter: Dict[str, Any]) -> str:
        """Build SQL WHERE clause from filter dict."""
        conditions = [f"metadata->>'{k}' = '{v}'" for k, v in filter.items()]
        return "WHERE " + " AND ".join(conditions) if conditions else ""

# Factory for vector DB provider selection
def get_vector_db_provider(config: dict) -> VectorDBProvider:
    """Factory function to create vector DB provider based on config."""
    provider_map = {
        'qdrant': QdrantProvider,
        'milvus': MilvusProvider,
        'weaviate': WeaviateProvider,
        'pgvector': PgvectorProvider,
    }
    
    provider_type = config.get('provider', 'qdrant')
    provider_class = provider_map.get(provider_type)
    
    if not provider_class:
        raise ValueError(f"Unknown vector DB provider: {provider_type}")
    
    return provider_class(**config.get('params', {}))
```

**Benefits of Pluggable Vector DB**:
- Switch providers via configuration (no code changes)
- Start with Qdrant, migrate to Milvus only if scale demands it
- Test different providers on your workload
- Minimize vendor lock-in
- Easy to add new providers

**Configuration Example** (`config.yaml`):
```yaml
vector_db:
  # Development/Production (default)
  provider: "qdrant"
  params:
    host: "qdrant"
    port: 6333
  
  # Massive Scale (>100M vectors)
  # provider: "milvus"
  # params:
  #   host: "milvus"
  #   port: 19530
  
  # Advanced Features (hybrid search)
  # provider: "weaviate"
  # params:
  #   host: "weaviate"
  #   port: 8080
  
  # Minimal Infrastructure (use existing Postgres)
  # provider: "pgvector"
  # params:
  #   connection_string: "postgresql://localhost/graphrag"

# Collection settings
collections:
  documents:
    vector_size: 1024  # e5-mistral dimension
    distance_metric: "cosine"
```

**Usage in Application**:
```python
from app.config import load_config
from app.vector_db.factory import get_vector_db_provider
from app.embeddings.factory import get_embedding_provider

# Load configuration
config = load_config('config.yaml')

# Get vector DB provider
vector_db = get_vector_db_provider(config['vector_db'])

# Create collection
vector_db.create_collection(
    name="documents",
    vector_size=config['collections']['documents']['vector_size'],
    distance_metric=config['collections']['documents']['distance_metric']
)

# Get embedding provider
embeddings = get_embedding_provider(config['embedding'])

# Add document chunks
chunks = ["chunk 1", "chunk 2", "chunk 3"]
vectors = embeddings.embed_documents(chunks)
metadatas = [
    {"source": "doc1.pdf", "page": 1, "chunk_index": 0},
    {"source": "doc1.pdf", "page": 1, "chunk_index": 1},
    {"source": "doc1.pdf", "page": 2, "chunk_index": 2},
]

vector_db.add_vectors(
    collection_name="documents",
    vectors=vectors,
    metadatas=metadatas
)

# Search with filtering
query = "What is quantum computing?"
query_vector = embeddings.embed_query(query)

results = vector_db.search(
    collection_name="documents",
    query_vector=query_vector,
    k=5,
    filter={"source": "doc1.pdf"}  # Filter by source document
)

for result in results:
    print(f"Score: {result['score']}, Metadata: {result['metadata']}")
```

### Pluggable Vector Database Architecture

The system implements a **provider abstraction** for vector databases to enable switching between different backends based on scale and requirements:

```python
# Abstract interface
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorDBProvider(ABC):
    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int, distance_metric: str):
        """Create a collection/index for vectors."""
        pass
    
    @abstractmethod
    def add_vectors(self, collection_name: str, vectors: List[List[float]], 
                   metadatas: List[Dict], ids: List[str]):
        """Add vectors with metadata to the collection."""
        pass
    
    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], 
              k: int, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors with optional metadata filtering."""
        pass
    
    @abstractmethod
    def delete(self, collection_name: str, ids: List[str]):
        """Delete vectors by ID."""
        pass
    
    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics and info."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

# Qdrant implementation (Primary)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition

class QdrantProvider(VectorDBProvider):
    def __init__(self, host: str, port: int = 6333, **kwargs):
        self.client = QdrantClient(host=host, port=port, **kwargs)
        self.host = host
        self.port = port
    
    def create_collection(self, collection_name: str, vector_size: int, distance_metric: str = "cosine"):
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_map.get(distance_metric, Distance.COSINE),
            ),
        )
    
    def add_vectors(self, collection_name: str, vectors: List[List[float]], 
                   metadatas: List[Dict], ids: List[str]):
        points = [
            PointStruct(
                id=id,
                vector=vector,
                payload=metadata
            )
            for id, vector, metadata in zip(ids, vectors, metadatas)
        ]
        self.client.upsert(collection_name=collection_name, points=points)
    
    def search(self, collection_name: str, query_vector: List[float], 
              k: int, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        # Convert generic filter to Qdrant filter format
        qdrant_filter = self._convert_filter(filter) if filter else None
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=k,
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "metadata": result.payload,
            }
            for result in results
        ]
    
    def delete(self, collection_name: str, ids: List[str]):
        self.client.delete(
            collection_name=collection_name,
            points_selector=ids,
        )
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        info = self.client.get_collection(collection_name)
        return {
            "vector_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance_metric": info.config.params.vectors.distance,
        }
    
    def _convert_filter(self, filter: Dict) -> Filter:
        """Convert generic filter format to Qdrant filter."""
        # Example implementation - expand based on your needs
        conditions = []
        for key, value in filter.items():
            conditions.append(
                FieldCondition(key=key, match={"value": value})
            )
        return Filter(must=conditions) if conditions else None
    
    @property
    def provider_name(self) -> str:
        return "qdrant"

# Milvus implementation (Massive Scale)
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

class MilvusProvider(VectorDBProvider):
    def __init__(self, host: str, port: int = 19530, **kwargs):
        connections.connect(host=host, port=port, **kwargs)
        self.host = host
        self.port = port
        self._collections = {}
    
    def create_collection(self, collection_name: str, vector_size: int, distance_metric: str = "cosine"):
        metric_map = {
            "cosine": "COSINE",
            "euclidean": "L2",
            "dot": "IP",
        }
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, description="Document embeddings")
        
        collection = Collection(name=collection_name, schema=schema)
        
        # Create index
        index_params = {
            "index_type": "HNSW",
            "metric_type": metric_map.get(distance_metric, "COSINE"),
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        self._collections[collection_name] = collection
    
    def add_vectors(self, collection_name: str, vectors: List[List[float]], 
                   metadatas: List[Dict], ids: List[str]):
        collection = self._get_collection(collection_name)
        data = [ids, vectors, metadatas]
        collection.insert(data)
    
    def search(self, collection_name: str, query_vector: List[float], 
              k: int, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        collection = self._get_collection(collection_name)
        collection.load()
        
        # Convert filter to Milvus expression format
        expr = self._convert_filter(filter) if filter else None
        
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=k,
            expr=expr,
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "metadata": hit.entity.get("metadata"),
            }
            for hit in results[0]
        ]
    
    def delete(self, collection_name: str, ids: List[str]):
        collection = self._get_collection(collection_name)
        expr = f"id in {ids}"
        collection.delete(expr)
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        collection = self._get_collection(collection_name)
        stats = collection.num_entities
        return {
            "vector_count": stats,
            "vector_size": collection.schema.fields[1].params.get("dim"),
        }
    
    def _get_collection(self, collection_name: str) -> Collection:
        if collection_name not in self._collections:
            self._collections[collection_name] = Collection(collection_name)
        return self._collections[collection_name]
    
    def _convert_filter(self, filter: Dict) -> str:
        """Convert generic filter to Milvus expression."""
        # Example: {"document_type": "research_paper"} → "metadata['document_type'] == 'research_paper'"
        conditions = []
        for key, value in filter.items():
            if isinstance(value, str):
                conditions.append(f"metadata['{key}'] == '{value}'")
            else:
                conditions.append(f"metadata['{key}'] == {value}")
        return " and ".join(conditions) if conditions else None
    
    @property
    def provider_name(self) -> str:
        return "milvus"

# Weaviate implementation (Hybrid Search)
import weaviate

class WeaviateProvider(VectorDBProvider):
    def __init__(self, host: str, port: int = 8080, **kwargs):
        self.client = weaviate.connect_to_local(host=host, port=port)
        self.host = host
        self.port = port
    
    def create_collection(self, collection_name: str, vector_size: int, distance_metric: str = "cosine"):
        # Weaviate uses class-based schema
        self.client.collections.create(
            name=collection_name,
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
            vector_index_config=weaviate.classes.config.Configure.VectorIndex.hnsw(
                distance_metric=distance_metric,
            ),
        )
    
    def add_vectors(self, collection_name: str, vectors: List[List[float]], 
                   metadatas: List[Dict], ids: List[str]):
        collection = self.client.collections.get(collection_name)
        with collection.batch.dynamic() as batch:
            for id, vector, metadata in zip(ids, vectors, metadatas):
                batch.add_object(
                    properties=metadata,
                    vector=vector,
                    uuid=id,
                )
    
    def search(self, collection_name: str, query_vector: List[float], 
              k: int, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        collection = self.client.collections.get(collection_name)
        
        # Weaviate filter conversion
        where_filter = self._convert_filter(filter) if filter else None
        
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=k,
            where=where_filter,
        )
        
        return [
            {
                "id": obj.uuid,
                "score": obj.metadata.distance,
                "metadata": obj.properties,
            }
            for obj in response.objects
        ]
    
    def delete(self, collection_name: str, ids: List[str]):
        collection = self.client.collections.get(collection_name)
        for id in ids:
            collection.data.delete_by_id(id)
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        # Weaviate collection info retrieval
        collection = self.client.collections.get(collection_name)
        aggregate = collection.aggregate.over_all()
        return {
            "vector_count": aggregate.total_count,
        }
    
    def _convert_filter(self, filter: Dict):
        """Convert generic filter to Weaviate format."""
        # Simplified - expand based on needs
        from weaviate.classes.query import Filter as WeaviateFilter
        conditions = []
        for key, value in filter.items():
            conditions.append(WeaviateFilter.by_property(key).equal(value))
        return conditions[0] if len(conditions) == 1 else None
    
    @property
    def provider_name(self) -> str:
        return "weaviate"

# Factory for vector DB provider selection
def get_vector_db_provider(config: dict) -> VectorDBProvider:
    """Factory function to create vector DB provider based on config."""
    provider_map = {
        'qdrant': QdrantProvider,
        'milvus': MilvusProvider,
        'weaviate': WeaviateProvider,
        # 'pgvector': PgvectorProvider,  # Can add if needed
    }
    
    provider_type = config.get('provider', 'qdrant')
    provider_class = provider_map.get(provider_type)
    
    if not provider_class:
        raise ValueError(f"Unknown vector DB provider: {provider_type}")
    
    return provider_class(
        host=config['host'],
        port=config.get('port'),
        **config.get('params', {})
    )
```

**Benefits of Pluggable Vector DB**:
- Switch providers via configuration (no code changes)
- Start with Qdrant, migrate to Milvus if scale demands
- Use Weaviate if hybrid search is critical
- Easy to benchmark different providers on your data
- Future-proof (add new providers as they emerge)

**Configuration Example** (`config.yaml`):
```yaml
vector_db:
  # Development/Production (recommended)
  provider: "qdrant"
  host: "qdrant"
  port: 6333
  params:
    collection_name: "documents"
    vector_size: 1024
    distance_metric: "cosine"
  
  # For massive scale (100M+ vectors)
  # provider: "milvus"
  # host: "milvus"
  # port: 19530
  # params:
  #   collection_name: "documents"
  #   vector_size: 1024
  #   distance_metric: "cosine"
  
  # For hybrid search features
  # provider: "weaviate"
  # host: "weaviate"
  # port: 8080
  # params:
  #   collection_name: "documents"
  #   vector_size: 1024
  #   distance_metric: "cosine"
```

**Usage in Application**:
```python
from app.config import load_config
from app.vector_db.factory import get_vector_db_provider
from app.embeddings.factory import get_embedding_provider

# Load configuration
config = load_config('config.yaml')

# Get vector DB provider
vector_db = get_vector_db_provider(config['vector_db'])

# Get embedding provider
embeddings = get_embedding_provider(config['embedding'])

# Create collection
vector_db.create_collection(
    collection_name="documents",
    vector_size=embeddings.dimension,
    distance_metric="cosine"
)

# Add vectors
vectors = embeddings.embed_documents(["text1", "text2"])
metadatas = [
    {"source": "doc1.pdf", "page": 1},
    {"source": "doc2.pdf", "page": 1},
]
ids = ["id1", "id2"]
vector_db.add_vectors("documents", vectors, metadatas, ids)

# Search with filtering
query_vector = embeddings.embed_query("search query")
results = vector_db.search(
    collection_name="documents",
    query_vector=query_vector,
    k=5,
    filter={"source": "doc1.pdf"}
)
```

### Data Ingestion Flow

```python
# Conceptual Flow
1. User uploads document(s)
   ↓
2. Document loader parses file
   ↓
3. Text splitter creates chunks
   ↓
4. Parallel processing:
   a) Generate embeddings → Store in Qdrant
   b) Extract entities/relations → Store in Neo4j
   c) Store original document → MinIO
   ↓
5. Update metadata and indexing
   ↓
6. Confirm completion to user
```

### Query Processing Flow

```python
# Conceptual Flow
1. User submits question
   ↓
2. Query analysis:
   - Extract entities
   - Determine query type (factual, analytical, etc.)
   ↓
3. Hybrid retrieval:
   a) Vector search: Query → Embedding → Qdrant top-k
   b) Graph search: Entities → Neo4j traversal → Related chunks
   ↓
4. Result fusion and reranking
   ↓
5. Context construction (with citations)
   ↓
6. LLM generates response
   ↓
7. Display answer with source references
```

## Technology Stack Summary

| Component | Technology | License | Why Chosen |
|-----------|-----------|---------|------------|
| **Document Loading** | LangChain Loaders | MIT | Format support, maturity |
| **Text Chunking** | LangChain Splitters (pluggable) | MIT | Multiple strategies, pluggable |
| **Embeddings** | e5-mistral-7b / gte-large | MIT | State-of-the-art quality, pluggable |
| **Vector DB** | Qdrant / Milvus / Weaviate | Apache 2.0 | Performance, features, pluggable |
| **Graph DB** | Neo4j CE | GPLv3 | Industry standard, maturity |
| **Graph RAG** | LlamaIndex | MIT | Purpose-built for RAG |
| **LLM** | Ollama / vLLM + Llama 3.1 | MIT / Apache 2.0 | Local, pluggable, scalable |
| **Orchestration** | LangChain | MIT | Ecosystem, patterns |
| **Object Storage** | MinIO | AGPLv3 | S3-compatible, self-hosted |
| **UI** | Streamlit | Apache 2.0 | Rapid development |
| **Container** | Docker Compose | Apache 2.0 | Easy deployment |

## Deployment Architecture

### Docker Compose Stack

```yaml
# Simplified structure showing key services
version: '3.8'

services:
  app:
    # Main Python application (Streamlit UI)
    build: .
    ports:
      - "8501:8501"
    environment:
      - CONFIG_PATH=/app/config.yaml
    volumes:
      - ./config.yaml:/app/config.yaml
      - ./data:/app/data
    depends_on:
      - qdrant  # or milvus/weaviate
      - neo4j
      - minio
      - ollama  # or vllm
    
  # Development/Production: Use Qdrant (recommended)
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC API
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    deploy:
      resources:
        limits:
          memory: 4G
  
  # For massive scale (100M+ vectors) - Use Milvus
  # milvus:
  #   image: milvusdb/milvus:latest
  #   ports:
  #     - "19530:19530"  # gRPC
  #     - "9091:9091"    # Metrics
  #   volumes:
  #     - milvus_storage:/var/lib/milvus
  #   environment:
  #     - ETCD_ENDPOINTS=etcd:2379
  #     - MINIO_ADDRESS=minio:9000
  #   depends_on:
  #     - etcd
  #     - minio
  
  # For hybrid search features - Use Weaviate
  # weaviate:
  #   image: semitechnologies/weaviate:latest
  #   ports:
  #     - "8080:8080"
  #   environment:
  #     - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
  #     - PERSISTENCE_DATA_PATH=/var/lib/weaviate
  #   volumes:
  #     - weaviate_storage:/var/lib/weaviate
    
  neo4j:
    # Graph database
    image: neo4j:5-community
    ports:
      - "7474:7474"  # Browser
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    
  minio:
    # Object storage
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_storage:/data
    
  # Development/MVP: Use Ollama
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    environment:
      - OLLAMA_NUM_PARALLEL=4
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  # Production: Use vLLM (comment out Ollama, uncomment this)
  # vllm:
  #   image: vllm/vllm-openai:latest
  #   ports:
  #     - "8000:8000"
  #   volumes:
  #     - vllm_models:/root/.cache/huggingface
  #   command: >
  #     --model meta-llama/Meta-Llama-3.1-8B-Instruct
  #     --dtype auto
  #     --max-model-len 4096
  #     --gpu-memory-utilization 0.9
  #   deploy:
  #     resources:
  #       reservations:
  #         devices:
  #           - driver: nvidia
  #             count: 1
  #             capabilities: [gpu]
  #   environment:
  #     - HF_TOKEN=${HF_TOKEN}  # For gated models

volumes:
  qdrant_storage:
  # milvus_storage:  # Uncomment for Milvus
  # weaviate_storage:  # Uncomment for Weaviate
  neo4j_data:
  neo4j_logs:
  minio_storage:
  ollama_models:
  # vllm_models:  # Uncomment for vLLM
```

**Switching Between Providers**:
1. **Vector DB**:
   - **Qdrant** (default): Use qdrant service
   - **Milvus** (massive scale): Comment out qdrant, uncomment milvus + dependencies
   - **Weaviate** (hybrid search): Comment out qdrant, uncomment weaviate
   - Update config.yaml to match selected provider

2. **LLM**:
   - **Ollama** (default): Use ollama service
   - **vLLM** (production): Comment out ollama, uncomment vllm
   - Update config.yaml to match selected provider

3. **No application code changes needed** - only config.yaml updates

### System Requirements

**Development Configuration (Ollama + lighter embeddings)**:
- 16-24 GB RAM
- 4-8 CPU cores
- 100 GB storage
- Optional: NVIDIA GPU with 8+ GB VRAM

**Production Configuration (vLLM + e5-mistral embeddings)**:
- 32-48 GB RAM
- 8-16 CPU cores
- 500 GB storage
- **Required: NVIDIA GPU with 24+ GB VRAM** (16GB for embeddings + 8GB for vLLM)
  - Alternative: 2 separate GPUs (1 for embeddings, 1 for LLM)

**Balanced Configuration (Ollama + gte-large embeddings)**:
- 24-32 GB RAM
- 8 CPU cores
- 300 GB storage
- NVIDIA GPU with 12-16 GB VRAM

**Resource Allocation Guidance**:
- **Embedding Model**: 4-16 GB VRAM (e5-mistral: 16GB, gte-large: 4-8GB)
- **LLM (Ollama)**: 8-12 GB VRAM for 8B model
- **LLM (vLLM)**: 8-10 GB VRAM for 8B model (more efficient)
- **Vector DB + Graph DB**: 8-16 GB RAM
- **Application**: 4-8 GB RAM

**Note**: The pluggable architecture allows graceful degradation - start with available resources and upgrade components as needed.

## Advanced Features

### 0. Model Selection Guides

#### Vector Database Selection

Choose your vector database based on scale and feature requirements:

| Database | Performance | Scale | Best For | Trade-offs |
|----------|-------------|-------|----------|------------|
| **Qdrant** | ⭐⭐⭐⭐⭐ | 1M-100M | General purpose, rich filtering | None for most use cases |
| **Milvus** | ⭐⭐⭐⭐ | 100M-10B+ | Massive scale, GPU acceleration | Complex deployment |
| **Weaviate** | ⭐⭐⭐ | 1M-50M | Hybrid search, built-in reranking | Higher resource usage |
| **pgvector** | ⭐⭐ | 100k-10M | Minimize infrastructure, existing Postgres | Lower performance |

**Recommendation Flow**:
1. **Start with Qdrant** for all deployments (1M-100M vectors)
2. **Migrate to Milvus** only if you exceed 100M vectors
3. **Use Weaviate** if hybrid search (keyword + semantic) is critical
4. **Use pgvector** only if you want to minimize infrastructure

**Performance Comparison** (1M vectors, p95 latency):
- Qdrant: 5-15ms (⭐⭐⭐⭐⭐)
- Milvus: 10-20ms (⭐⭐⭐⭐)
- Weaviate: 15-30ms (⭐⭐⭐)
- pgvector: 30-100ms (⭐⭐)

**Key Features Comparison**:
- **Metadata Filtering**: Qdrant ⭐⭐⭐⭐⭐, Milvus ⭐⭐⭐⭐, Weaviate ⭐⭐⭐⭐
- **Hybrid Search**: Weaviate ⭐⭐⭐⭐⭐, Qdrant ⭐⭐⭐⭐⭐, Milvus ⭐⭐⭐
- **Ease of Deployment**: Qdrant ⭐⭐⭐⭐⭐, Weaviate ⭐⭐⭐⭐, Milvus ⭐⭐⭐

#### Text Chunking Strategy Selection

Choose your chunking strategy based on document type and quality requirements:

| Strategy | Quality | Speed | Best For | Document Types |
|----------|---------|-------|----------|----------------|
| **RecursiveCharacterTextSplitter** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | General purpose, vector search | All types |
| **SemanticChunker** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Graph construction, entity extraction | Research papers, technical docs |
| **MarkdownHeaderTextSplitter** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Structured docs | Markdown, documentation |
| **Unstructured.io** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | PDFs, complex layouts | PDFs, scanned docs |

**Recommendation Flow**:
1. **Start with RecursiveCharacterTextSplitter** for all documents
2. **Add SemanticChunker** for graph construction (entities)
3. **Use MarkdownHeaderTextSplitter** for markdown/technical docs
4. **Integrate Unstructured.io** if PDFs are >30% of workload

**Hybrid Approach for Graph RAG**:
- **Vector Search**: RecursiveCharacterTextSplitter (fast, good coverage)
- **Graph Construction**: SemanticChunker (better entity extraction)
- **Result**: 15-25% improvement in retrieval accuracy

**Configuration Impact**:
- Chunk size: 512-1024 tokens optimal for most use cases
- Overlap: 10-20% prevents information loss
- Strategy choice: Can improve retrieval by 15-25%

#### Embedding Model Selection

Choose your embedding model based on your priorities:

| Model | Quality | Speed | GPU Required | Best For |
|-------|---------|-------|--------------|----------|
| **e5-mistral-7b** | ⭐⭐⭐⭐⭐ | ⭐⭐ | 16GB+ VRAM | Maximum accuracy, complex domains |
| **gte-large** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 4-8GB VRAM | Great balance, simpler than e5 |
| **bge-base** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Optional | Production-ready, efficient |
| **all-MiniLM** | ⭐⭐ | ⭐⭐⭐⭐⭐ | No | Development, CPU-only |

**Recommendation Flow**:
1. **Start with e5-mistral-7b** if you have GPU resources
2. **Fall back to gte-large** if GPU memory limited
3. **Use bge-base** for CPU deployment or high-throughput needs
4. **Switch via config** - test different models on your data

**Performance Impact on Retrieval**:
- e5-mistral-7b: +20-25% accuracy vs MiniLM
- gte-large: +15-18% accuracy vs MiniLM
- bge-base: +12-15% accuracy vs MiniLM

#### LLM Provider Selection

Choose your LLM provider based on your deployment stage and scale:

| Aspect | Ollama | vLLM | Winner |
|--------|--------|------|--------|
| **Setup Difficulty** | ⭐⭐⭐⭐⭐ Easy | ⭐⭐ Hard | Ollama |
| **Throughput** | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent | vLLM |
| **Multi-User Support** | ⭐⭐ Limited | ⭐⭐⭐⭐⭐ Excellent | vLLM |
| **CPU Support** | ✅ Yes | ❌ No | Ollama |
| **Model Management** | ⭐⭐⭐⭐⭐ Automatic | ⭐⭐ Manual | Ollama |
| **Production Ready** | ⭐⭐⭐ Decent | ⭐⭐⭐⭐⭐ Yes | vLLM |
| **Dev Experience** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Okay | Ollama |

**Recommendation Flow**:
1. **Start with Ollama** for development and MVP (0-10 users)
2. **Migrate to vLLM** when you reach production scale (10+ users)
3. **Switch via config** - no code changes needed

**Performance Comparison** (Llama 3.1 8B on A100):
- **Single request**: Ollama 50ms vs vLLM 40ms (similar)
- **10 concurrent users**: Ollama 350 tok/s vs vLLM 800 tok/s (2.3x better)
- **50 concurrent users**: Ollama 400 tok/s vs vLLM 1200 tok/s (3x better)

### 1. Reranking
- **Reranker Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Purpose**: Improve relevance of top-k results
- **Integration**: Post-retrieval, pre-generation step

### 2. Query Decomposition
- Break complex questions into sub-questions
- Route sub-questions to appropriate retrieval methods
- Synthesize answers from multiple sources

### 3. Citation and Provenance
- Track which chunks contributed to each answer
- Display source documents with highlights
- Enable user verification of claims

### 4. Conversation Memory
- Store conversation history in application state
- Use recent context for follow-up questions
- Implement conversation summarization for long chats

### 5. Graph Visualization
- Interactive graph explorer
- Show entity relationships
- Highlight relevant subgraphs for queries

## Implementation Roadmap

### Phase 1: Core RAG with Ollama (Week 1-2)
- [ ] Set up Docker infrastructure
- [ ] Implement pluggable text chunking architecture
- [ ] Configure RecursiveCharacterTextSplitter (baseline)
- [ ] Implement pluggable embedding architecture
- [ ] Configure and test embedding models (start with e5-mistral or gte-large)
- [ ] Implement pluggable vector database architecture
- [ ] Deploy Qdrant (recommended starting point)
- [ ] Implement pluggable LLM architecture
- [ ] Deploy Ollama and test with Llama 3.1 8B
- [ ] Implement document ingestion pipeline (chunking + embedding + vector storage)
- [ ] Test end-to-end RAG pipeline with basic vector search
- [ ] Create simple Streamlit UI
- [ ] Verify query performance (<50ms p95 latency)

### Phase 2: Graph Integration (Week 3-4)
- [ ] Deploy Neo4j
- [ ] Add SemanticChunker for high-quality entity extraction
- [ ] Implement entity extraction (using Ollama + semantic chunks)
- [ ] Build graph construction pipeline
- [ ] Integrate graph retrieval with vector search
- [ ] Implement hybrid retrieval strategy (vector + graph)
- [ ] Add graph visualization to UI
- [ ] Test hybrid chunking approach (recursive for vectors, semantic for graph)
- [ ] Benchmark retrieval quality improvements

### Phase 3: Enhancement (Week 5-6)
- [ ] Add MarkdownHeaderTextSplitter for documentation
- [ ] Implement document-type detection and strategy routing
- [ ] Add reranking
- [ ] Implement citation tracking
- [ ] Improve graph visualization
- [ ] Add conversation memory
- [ ] Optimize chunking parameters (size, overlap)
- [ ] Load testing with Ollama
- [ ] Monitor vector DB performance metrics

### Phase 4: Production Readiness (Week 7-8)
- [ ] **Evaluate chunking strategy performance** (A/B test on real queries)
- [ ] **Evaluate vector DB performance** under production load
- [ ] **Evaluate migration to vLLM** (if concurrent users > 10)
- [ ] Set up vLLM if needed (update docker-compose, config.yaml)
- [ ] Performance testing: Compare chunking strategies on retrieval metrics
- [ ] Performance testing: Verify vector DB meets latency requirements
- [ ] User acceptance testing
- [ ] Documentation (including chunking, embedding, vector DB, and LLM provider guides)
- [ ] Deployment automation
- [ ] Monitoring and logging setup (vector DB metrics: latency, throughput, cache hits)
- [ ] Security hardening

### Phase 5: Scale & Optimize (Ongoing)
- [ ] Monitor vector DB performance:
  - [ ] Query latency (p50, p95, p99)
  - [ ] Throughput (queries/second)
  - [ ] Memory usage and growth
  - [ ] Vector count and collection size
- [ ] **Migrate to Milvus** if vectors exceed 100M:
  - [ ] Setup Kubernetes cluster
  - [ ] Deploy Milvus with dependencies
  - [ ] Data migration from Qdrant
  - [ ] Performance validation
- [ ] Monitor retrieval quality metrics by chunking strategy
- [ ] Monitor concurrent user metrics
- [ ] Migrate to vLLM when thresholds met:
  - [ ] Average concurrent users > 10
  - [ ] Response time degradation under load
  - [ ] GPU utilization < 70% with Ollama
- [ ] Optimize vLLM configuration (if using)
- [ ] A/B test chunking strategies on production queries
- [ ] A/B test embedding models on real queries
- [ ] Fine-tune chunk sizes based on retrieval metrics
- [ ] Continuous performance tuning

## Key Design Decisions

### Why Pluggable Vector Database Architecture?

**Rationale**:
1. **Scale Evolution**: Start with Qdrant, migrate to Milvus if dataset exceeds 100M vectors
2. **Feature Requirements**: Switch to Weaviate if hybrid search becomes critical
3. **Performance Testing**: Easy to benchmark different databases on your workload
4. **Infrastructure Flexibility**: Use pgvector to minimize moving parts in early stages
5. **Future-Proofing**: New vector databases can be added without changing core logic

**Implementation Benefits**:
- Switch databases via configuration file (no code changes)
- All downstream components work with any vector DB through abstraction
- Easy to optimize for your specific scale and feature needs
- Vendor independence - not locked into any single solution

**Key Insight**: Vector database choice significantly impacts query latency (5ms vs 100ms) and scalability. The pluggable architecture allows you to start with the best general-purpose option (Qdrant) and migrate only if specific needs arise (massive scale → Milvus, hybrid search → Weaviate).

### Why Pluggable Text Chunking Architecture?

**Rationale**:
1. **Document Type Optimization**: Different documents benefit from different chunking strategies
2. **Quality vs. Speed Trade-offs**: Use fast chunking for vector search, semantic chunking for graph construction
3. **Experimentation**: Easy to A/B test chunking strategies on your data
4. **Future-Proofing**: New chunking algorithms can be added without changing core logic
5. **Specialized Handling**: Markdown, PDFs, code all benefit from tailored strategies

**Implementation Benefits**:
- Switch strategies via configuration file (no code changes)
- Use different strategies for different purposes (vector search vs. entity extraction)
- Hybrid approach: fast + accurate chunking in one system
- Metadata preservation across all strategies

**Key Insight**: Chunking quality directly impacts retrieval accuracy by 15-25%. Using the right strategy for each document type and purpose is critical for Graph RAG performance.

### Why Pluggable Embedding Architecture?

**Rationale**:
1. **Future-Proofing**: New embedding models are released frequently; easy to upgrade
2. **Domain Optimization**: Different document types may benefit from different embeddings
3. **Resource Flexibility**: Scale between quality and resource usage based on deployment environment
4. **A/B Testing**: Compare embedding models on your specific data
5. **Graceful Degradation**: Fallback to lighter models when GPU unavailable

**Implementation Benefits**:
- Change models via configuration file (no code changes)
- All downstream components (Qdrant, retrieval, etc.) work with any embedding model
- Vector dimension is automatically handled by the provider interface

### Why Pluggable LLM Architecture?

**Rationale**:
1. **Development Speed**: Start with Ollama's simple setup, migrate to vLLM when needed
2. **Performance Scaling**: vLLM provides 2-4x better throughput for production workloads
3. **Cost Optimization**: Use the right tool for the right stage (Ollama for dev, vLLM for scale)
4. **Flexibility**: Same application code works with both providers
5. **Risk Mitigation**: Easy to switch providers if issues arise

**Migration Path**:
- **Phase 1 (Weeks 1-8)**: Use Ollama for rapid development
- **Phase 2 (Month 3+)**: Switch to vLLM when concurrent users exceed 10
- **Zero Code Changes**: Only configuration file updates needed

### Why Graph RAG over Standard RAG?

**Advantages**:
1. **Better Entity Understanding**: Graph captures explicit relationships between entities
2. **Multi-Hop Reasoning**: Navigate relationships to find indirect connections
3. **Structured Knowledge**: Entities and relationships are queryable
4. **Disambiguation**: Entity nodes help resolve ambiguous references
5. **Explainability**: Graph paths show reasoning chains

**Trade-offs**:
- More complex infrastructure
- Higher computational cost for graph construction
- Requires careful schema design
- Best for entity-rich, relationship-heavy documents

### Why These Specific Tools?

1. **Qdrant over Alternatives**: Best balance of performance, features, and ease of use
2. **Neo4j over Alternatives**: Maturity, Cypher language, visualization tools
3. **LlamaIndex + LangChain**: Complementary strengths (LlamaIndex for indexing, LangChain for chains)
4. **Ollama over Cloud APIs**: Privacy, cost control, no rate limits
5. **Streamlit over Custom**: Development speed, built-in components

## Code Structure

```
graph-rag-app/
├── docker-compose.yml
├── docker-compose.vllm.yml  # vLLM variant for production
├── docker-compose.milvus.yml  # Milvus variant for massive scale
├── requirements.txt
├── README.md
├── .env.example
├── config.yaml              # Configuration including all pluggable components
├── app/
│   ├── main.py              # Streamlit entry point
│   ├── config.py            # Configuration management
│   ├── chunking/
│   │   ├── base.py          # Abstract ChunkingStrategy interface
│   │   ├── recursive.py     # RecursiveCharacterTextSplitter implementation
│   │   ├── semantic.py      # SemanticChunker implementation
│   │   ├── markdown.py      # MarkdownHeaderTextSplitter implementation
│   │   ├── hybrid.py        # HybridChunker for multi-strategy
│   │   └── factory.py       # Chunking strategy factory
│   ├── embeddings/
│   │   ├── base.py          # Abstract EmbeddingProvider interface
│   │   ├── e5_mistral.py    # E5-Mistral implementation
│   │   ├── gte_large.py     # GTE-Large implementation
│   │   ├── bge_base.py      # BGE-Base implementation
│   │   └── factory.py       # Embedding provider factory
│   ├── vector_db/
│   │   ├── base.py          # Abstract VectorDBProvider interface
│   │   ├── qdrant.py        # Qdrant implementation
│   │   ├── milvus.py        # Milvus implementation
│   │   ├── weaviate.py      # Weaviate implementation
│   │   └── factory.py       # Vector DB provider factory
│   ├── llm/
│   │   ├── base.py          # Abstract LLMProvider interface
│   │   ├── ollama.py        # Ollama implementation
│   │   ├── vllm.py          # vLLM implementation
│   │   └── factory.py       # LLM provider factory
│   ├── ingestion/
│   │   ├── loader.py        # Document loading
│   │   ├── processor.py     # Orchestrates chunking, embedding, storage
│   │   └── graph_builder.py # Graph construction (uses LLM & chunking factories)
│   ├── retrieval/
│   │   ├── vector_search.py # Vector DB operations (uses factory)
│   │   ├── graph_search.py  # Neo4j operations
│   │   └── hybrid.py        # Hybrid retrieval (vector + graph)
│   ├── generation/
│   │   ├── response.py      # Response generation (uses LLM factory)
│   │   └── chains.py        # LangChain chains
│   ├── ui/
│   │   ├── chat.py          # Chat interface
│   │   ├── upload.py        # File upload
│   │   └── visualize.py     # Graph visualization
│   └── utils/
│       ├── logging.py
│       └── helpers.py
├── data/                    # Local data directory
├── models/                  # Downloaded models
└── tests/
    ├── test_chunking.py     # Test different chunking strategies
    ├── test_embeddings.py   # Test different embedding models
    ├── test_vector_db.py    # Test different vector databases
    └── test_llm.py          # Test different LLM providers
```

## Security Considerations

1. **Data Privacy**: All processing happens locally
2. **Access Control**: Implement user authentication (e.g., using Streamlit auth)
3. **Input Validation**: Sanitize uploaded files
4. **Resource Limits**: Set max file sizes, query limits
5. **Secrets Management**: Use environment variables for credentials

## Monitoring and Observability

- **Application Logs**: Python logging to files
- **Metrics**: Track query latency, chunk retrieval times
- **Health Checks**: Monitor service availability
- **Cost Tracking**: Monitor storage usage, compute time

## Next Steps

1. **Set up development environment**: Install Docker, Python 3.10+
2. **Clone starter repositories**: LangChain and LlamaIndex examples
3. **Deploy infrastructure**: Run docker-compose up
4. **Build MVP**: Implement Phase 1 features
5. **Iterate based on testing**: Add advanced features as needed

## Resources

### Core Technologies
- **LangChain**: https://github.com/langchain-ai/langchain
- **LlamaIndex**: https://github.com/run-llama/llama_index
- **Qdrant**: https://qdrant.tech/documentation/
- **Neo4j**: https://neo4j.com/docs/
- **Ollama**: https://ollama.ai/
- **vLLM**: https://docs.vllm.ai/
- **Streamlit**: https://docs.streamlit.io/

### Text Chunking
- **LangChain Text Splitters**: https://python.langchain.com/docs/modules/data_connection/document_transformers/
- **LlamaIndex Node Parsers**: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
- **Unstructured.io**: https://unstructured-io.github.io/unstructured/
- **Semantic Chunking Research**: https://arxiv.org/abs/2404.09640

### Embedding Models
- **E5 Models**: https://huggingface.co/intfloat/e5-mistral-7b-instruct
- **GTE Models**: https://huggingface.co/thenlper/gte-large
- **BGE Models**: https://huggingface.co/BAAI/bge-large-en-v1.5
- **Sentence Transformers**: https://www.sbert.net/

### Related Documentation
- **Text Chunking Comparison**: See companion document for detailed chunking strategy comparison
- **Vector Database Comparison**: See companion document for detailed vector DB comparison
- **vLLM vs Ollama Comparison**: See companion document for detailed provider comparison
- **Graph RAG Paper**: https://arxiv.org/abs/2404.16130
- **LangChain RAG Tutorial**: https://python.langchain.com/docs/use_cases/question_answering/
- **LlamaIndex Graph RAG**: https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_rag_query_engine/

## Conclusion

This architecture provides a production-ready foundation for a Graph RAG application using proven open source components. The design prioritizes:
- **Modularity**: Easy to swap components - text chunking, embeddings, vector database, and LLM providers
- **Quality**: State-of-the-art models for each component
  - SemanticChunker for high-quality entity extraction
  - e5-mistral-7b embeddings for best retrieval accuracy
  - Qdrant for fast, feature-rich vector search
  - vLLM for production-scale LLM serving
- **Flexibility**: Quadruple pluggable architecture adapts to different:
  - Document types (markdown, PDFs, code)
  - Processing stages (vector search vs. graph construction)
  - Scale requirements (1M vs. 100M+ vectors)
  - Resource constraints (development vs. production)
- **Development Speed**: Start with simple, fast options
  - RecursiveChunker for baseline chunking
  - Qdrant for vector storage
  - Ollama for LLM serving
- **Production Scalability**: Migrate to optimized options when needed
  - SemanticChunker for better entity extraction
  - Milvus if vectors exceed 100M
  - vLLM when concurrent users exceed 10
- **Cost Optimization**: Right-size infrastructure for each development stage
- **Privacy**: Fully self-hosted option with no external API dependencies
- **Developer Experience**: Well-documented tools with active communities

The combination of vector search (Qdrant/Milvus/Weaviate) and graph database (Neo4j) enables sophisticated retrieval strategies that outperform traditional RAG for complex, entity-rich documents. 

**The quadruple pluggable architecture** ensures you can:
1. **Start simple**: RecursiveChunker + gte-large + Qdrant + Ollama for rapid MVP development
2. **Scale smart**: SemanticChunker + e5-mistral + Milvus + vLLM for production performance
3. **Adapt easily**: Change any component via configuration, not code rewrites
4. **Optimize continuously**: A/B test different combinations on your data

**Component Impact on System Performance**:
- **Chunking Strategy**: 15-25% improvement in retrieval accuracy
- **Embedding Model**: 15-20% improvement in semantic matching
- **Vector Database**: 5-20x difference in query latency (Qdrant vs. pgvector)
- **LLM Provider**: 2-3x difference in concurrent user capacity (vLLM vs. Ollama)

**Deployment Path**:
- **Weeks 1-2**: Develop with RecursiveChunker + Qdrant + Ollama (baseline)
- **Weeks 3-4**: Add SemanticChunker for graph construction
- **Weeks 5-6**: Add specialized strategies (Markdown, PDF handling)
- **Week 7-8**: Production readiness, evaluate migrations
- **Month 3+**: Migrate to vLLM when concurrent users exceed 10
- **Only if needed**: Migrate to Milvus if vectors exceed 100M
- **Ongoing**: Fine-tune all components based on production metrics

This phased approach balances development velocity with production requirements, ensuring you can ship quickly while maintaining clear paths to scale and optimize. The pluggable architecture means you're never locked in - each component can evolve independently based on your actual needs.
