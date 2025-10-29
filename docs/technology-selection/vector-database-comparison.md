# Vector Database Comparison for Graph RAG Applications

## Executive Summary

Vector databases are critical infrastructure for RAG systems, storing embeddings and enabling fast similarity search. This document compares the major open-source vector database solutions, their strengths, weaknesses, and suitability for Graph RAG applications.

**Quick Recommendation**: Use **Qdrant** for most Graph RAG applications due to its excellent balance of performance, features, and ease of use. Consider **Milvus** for very large-scale deployments (100M+ vectors) or **Weaviate** if you need built-in hybrid search and reranking.

---

## Why Vector Databases Matter in Graph RAG

### Core Requirements

A good vector database for Graph RAG must provide:

1. **Fast Similarity Search**: Sub-100ms queries even with millions of vectors
2. **Metadata Filtering**: Filter by document, date, type, etc. before similarity search
3. **Scalability**: Handle growing document collections
4. **Reliability**: Data persistence and backup/restore
5. **Integration**: Easy to use with LangChain/LlamaIndex
6. **Hybrid Search**: Combine dense vectors with keyword search (optional but valuable)

### Impact on System Performance

**Vector database affects**:
- **Query Latency**: How fast users get responses
- **Retrieval Quality**: Better indexing = more relevant results
- **System Scalability**: Can it handle your data growth?
- **Operational Overhead**: Ease of deployment and maintenance
- **Cost**: Resource requirements and efficiency

---

## Top Open Source Vector Databases

## 1. Qdrant ⭐ RECOMMENDED

**Repository**: https://github.com/qdrant/qdrant  
**License**: Apache 2.0  
**Language**: Rust  
**Maturity**: ⭐⭐⭐⭐⭐ Production-ready

### Overview

Qdrant is a vector search engine built from the ground up for high performance and developer experience. Written in Rust, it offers excellent performance with a clean API.

### Key Features

**Core Capabilities**:
- **HNSW Indexing**: Fast approximate nearest neighbor search
- **Rich Filtering**: Complex metadata filters with high performance
- **Hybrid Search**: Sparse + dense vector support
- **Quantization**: Reduce memory footprint (Scalar, Product, Binary)
- **Sharding & Replication**: Built-in horizontal scaling
- **Snapshots**: Full backup and restore capabilities
- **Payload Storage**: Store metadata with vectors

**Advanced Features**:
- **Multiple Vector Per Point**: Store multiple embeddings for same document
- **Named Vectors**: Different embedding types in same collection
- **Query Recommendations**: Built-in similar item recommendations
- **Batch Operations**: Efficient bulk inserts and updates
- **Change Data Capture**: Track all changes for audit/replay

### Performance

**Benchmarks** (1M vectors, 768 dimensions):
- **Single Query Latency**: 5-15ms (p95)
- **Throughput**: 1000+ queries/second
- **Insert Speed**: 10,000+ vectors/second
- **Memory Usage**: ~1.5GB per 1M vectors (with quantization)
- **Indexing Time**: ~2 minutes for 1M vectors

**Scaling**:
- Tested up to 100M+ vectors
- Horizontal scaling via sharding
- Replication for high availability

### Pros

✅ **Excellent Performance**: One of the fastest vector databases  
✅ **Rich Filtering**: Best-in-class metadata filtering without compromising speed  
✅ **Clean API**: Well-designed REST and gRPC APIs  
✅ **Great Documentation**: Comprehensive docs with examples  
✅ **Active Development**: Frequent updates and new features  
✅ **Easy Deployment**: Single Docker container  
✅ **Quantization**: Multiple strategies to reduce memory  
✅ **Hybrid Search**: Native sparse vector support  
✅ **Python Client**: Excellent Python SDK  
✅ **Production Ready**: Used by many companies in production  

### Cons

❌ **Relatively New**: Less mature than Elasticsearch (but very stable)  
❌ **Rust-based**: Harder to contribute code (though not an issue for users)  
❌ **Memory Usage**: Can be high for very large datasets without quantization  
❌ **Limited Ecosystem**: Fewer third-party tools vs. established databases  

### Best For

- General-purpose RAG applications
- When metadata filtering is important
- Medium to large scale (1M-100M vectors)
- When you want excellent performance without complexity
- Graph RAG applications (recommended)

### Configuration Example

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Initialize client
client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1024,  # Embedding dimension
        distance=Distance.COSINE,
    ),
    # Optional: Enable scalar quantization
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            always_ram=True,
        ),
    ),
)

# Insert vectors
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],  # Your embedding
            payload={
                "text": "Document content",
                "source": "document.pdf",
                "page": 1,
                "document_type": "research_paper",
            }
        ),
    ]
)

# Search with filtering
results = client.search(
    collection_name="documents",
    query_vector=[0.1, 0.2, ...],
    query_filter=Filter(
        must=[
            FieldCondition(
                key="document_type",
                match=MatchValue(value="research_paper"),
            ),
        ]
    ),
    limit=5,
)
```

### Docker Deployment

```yaml
services:
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
```

---

## 2. Milvus

**Repository**: https://github.com/milvus-io/milvus  
**License**: Apache 2.0  
**Language**: Go, C++, Python  
**Maturity**: ⭐⭐⭐⭐⭐ Production-ready

### Overview

Milvus is a highly scalable vector database designed for massive-scale similarity search. Built for cloud-native deployments, it excels at handling billions of vectors.

### Key Features

**Core Capabilities**:
- **Multiple Index Types**: HNSW, IVF_FLAT, IVF_SQ8, ANNOY, etc.
- **GPU Support**: GPU-accelerated indexing and search
- **Hybrid Search**: Vector + scalar filtering
- **Consistency Levels**: Tunable consistency (strong, bounded, eventually)
- **Time Travel**: Query historical data
- **Dynamic Schema**: Add fields without rebuilding
- **Multi-Tenancy**: Isolation for different users/applications

**Advanced Features**:
- **Distributed Architecture**: Separates compute and storage
- **Message Queue**: Kafka/Pulsar integration
- **Object Storage**: S3/MinIO for vector storage
- **Kubernetes Native**: Designed for K8s deployments
- **CDC**: Change data capture for streaming

### Performance

**Benchmarks** (1M vectors, 768 dimensions):
- **Single Query Latency**: 10-20ms (p95)
- **Throughput**: 500-1000 queries/second
- **Insert Speed**: 5,000-10,000 vectors/second
- **Memory Usage**: ~2GB per 1M vectors
- **Indexing Time**: ~5 minutes for 1M vectors

**Scaling**:
- Designed for 100M-10B+ vectors
- Cloud-native horizontal scaling
- Can scale to multiple nodes easily

### Pros

✅ **Massive Scale**: Best for very large datasets (billions of vectors)  
✅ **Cloud Native**: Excellent Kubernetes deployment  
✅ **Flexible Architecture**: Separate compute and storage  
✅ **GPU Support**: Accelerated operations with GPU  
✅ **Multiple Indexes**: Choose optimal index for use case  
✅ **Enterprise Features**: Multi-tenancy, time travel, etc.  
✅ **Strong Ecosystem**: Good tooling and integrations  
✅ **Backed by Zilliz**: Commercial support available  

### Cons

❌ **Complexity**: More complex to deploy and operate than Qdrant  
❌ **Resource Heavy**: Requires more resources (memory, storage)  
❌ **Steeper Learning Curve**: More concepts to understand  
❌ **Slower for Small Scale**: Overkill for <10M vectors  
❌ **Operational Overhead**: Requires expertise for production deployment  

### Best For

- Very large scale (100M+ vectors)
- Enterprise deployments with dedicated DevOps
- When GPU acceleration is needed
- Multi-tenant applications
- Cloud-native microservices architecture

### Configuration Example

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# Connect
connections.connect(host="localhost", port=19530)

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
]
schema = CollectionSchema(fields, description="Document embeddings")

# Create collection
collection = Collection(name="documents", schema=schema)

# Create index
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index(field_name="embedding", index_params=index_params)

# Insert data
collection.insert([
    [0.1, 0.2, ...],  # embeddings
    ["Document text"],  # text
    ["document.pdf"],  # source
])

# Search
collection.load()
results = collection.search(
    data=[[0.1, 0.2, ...]],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=5,
    expr='source == "document.pdf"',  # Filtering
)
```

---

## 3. Weaviate

**Repository**: https://github.com/weaviate/weaviate  
**License**: BSD 3-Clause  
**Language**: Go  
**Maturity**: ⭐⭐⭐⭐⭐ Production-ready

### Overview

Weaviate is a vector database with built-in modules for vectorization, hybrid search, and advanced query capabilities. It emphasizes AI-native features and ease of use.

### Key Features

**Core Capabilities**:
- **Built-in Vectorization**: Optional modules for automatic embedding generation
- **Hybrid Search**: BM25 + vector search with alpha tuning
- **GraphQL API**: Flexible query language
- **Reranking**: Built-in reranking with Cohere/others
- **Multi-Vector**: Multiple embeddings per object
- **Modules System**: Pluggable architecture for extensions

**Advanced Features**:
- **Generative Search**: LLM integration for RAG
- **Named Vectors**: Multiple vector spaces per class
- **Cross-References**: Link between objects (graph-like)
- **Conditional Filtering**: Complex where clauses
- **Aggregate Queries**: Analytics on vector data

### Performance

**Benchmarks** (1M vectors, 768 dimensions):
- **Single Query Latency**: 15-30ms (p95)
- **Throughput**: 300-600 queries/second
- **Insert Speed**: 5,000 vectors/second
- **Memory Usage**: ~2.5GB per 1M vectors
- **Indexing Time**: ~3-4 minutes for 1M vectors

**Scaling**:
- Good for 1M-50M vectors
- Horizontal scaling via sharding
- Replication for HA

### Pros

✅ **Hybrid Search**: Excellent built-in BM25 + vector search  
✅ **Built-in Reranking**: No need for external reranker  
✅ **GraphQL API**: Powerful and flexible querying  
✅ **Vectorization Modules**: Can auto-generate embeddings  
✅ **Generative Search**: Built-in RAG capabilities  
✅ **Good Documentation**: Comprehensive guides  
✅ **Active Community**: Strong user base  
✅ **Cloud Offering**: Managed service available  

### Cons

❌ **Performance**: Slower than Qdrant for pure vector search  
❌ **Memory Usage**: Higher than alternatives  
❌ **Complexity**: Many features = steeper learning curve  
❌ **Resource Heavy**: Requires more resources  
❌ **GraphQL**: Some developers prefer REST/gRPC  

### Best For

- When you need hybrid search (vector + keyword)
- Built-in reranking requirements
- Generative search / RAG use cases
- When GraphQL is preferred
- Medium scale with rich features

### Configuration Example

```python
import weaviate
from weaviate.classes.config import Configure

# Connect
client = weaviate.connect_to_local()

# Create collection
client.collections.create(
    name="Document",
    vectorizer_config=Configure.Vectorizer.none(),  # Using external embeddings
    vector_index_config=Configure.VectorIndex.hnsw(
        distance_metric="cosine",
    ),
    properties=[
        weaviate.classes.config.Property(
            name="text",
            data_type=weaviate.classes.config.DataType.TEXT,
        ),
        weaviate.classes.config.Property(
            name="source",
            data_type=weaviate.classes.config.DataType.TEXT,
        ),
    ]
)

# Insert
documents = client.collections.get("Document")
documents.data.insert(
    properties={
        "text": "Document content",
        "source": "document.pdf",
    },
    vector=[0.1, 0.2, ...],
)

# Hybrid search (vector + keyword)
response = documents.query.hybrid(
    query="search query",
    vector=[0.1, 0.2, ...],
    alpha=0.5,  # 0=keyword only, 1=vector only
    limit=5,
)
```

---

## 4. Chroma

**Repository**: https://github.com/chroma-core/chroma  
**License**: Apache 2.0  
**Language**: Python  
**Maturity**: ⭐⭐⭐ Growing

### Overview

Chroma is the "AI-native open-source embedding database" designed to be simple and developer-friendly. It emphasizes ease of use and quick integration.

### Key Features

**Core Capabilities**:
- **Simple API**: Minimal setup, easy to use
- **Built-in Embeddings**: Can generate embeddings automatically
- **Metadata Filtering**: Basic filtering support
- **In-Memory Mode**: Fast prototyping without persistence
- **Persistent Storage**: DuckDB backend for persistence
- **LangChain Integration**: First-class support

**Advanced Features**:
- **Collections**: Organize vectors into namespaces
- **Distance Metrics**: Cosine, L2, IP
- **Where Filtering**: SQL-like filtering
- **Document Store**: Store original documents

### Performance

**Benchmarks** (1M vectors, 768 dimensions):
- **Single Query Latency**: 20-50ms (p95)
- **Throughput**: 200-400 queries/second
- **Insert Speed**: 2,000-5,000 vectors/second
- **Memory Usage**: ~3GB per 1M vectors
- **Indexing Time**: ~5-7 minutes for 1M vectors

**Scaling**:
- Best for <1M vectors
- Limited scaling options
- Not designed for massive scale

### Pros

✅ **Extremely Simple**: Easiest to get started  
✅ **Python-First**: Great Python developer experience  
✅ **LangChain Native**: Excellent integration  
✅ **In-Memory Mode**: Perfect for prototyping  
✅ **Auto Embeddings**: Can generate embeddings for you  
✅ **Lightweight**: Minimal dependencies  
✅ **Open Development**: Active GitHub community  

### Cons

❌ **Performance**: Slower than production-grade alternatives  
❌ **Limited Scale**: Not suitable for large datasets  
❌ **Basic Features**: Missing advanced capabilities  
❌ **Immature**: Still evolving, API changes  
❌ **No Distributed Mode**: Single-node only  
❌ **DuckDB Backend**: Not as robust as dedicated vector DBs  

### Best For

- Prototyping and development
- Small-scale applications (<100k vectors)
- Learning and experimentation
- When simplicity is more important than performance
- LangChain-based projects

### Configuration Example

```python
import chromadb
from chromadb.config import Settings

# Initialize
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# Add documents (with auto-ID generation)
collection.add(
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["Document 1", "Document 2"],
    metadatas=[
        {"source": "doc1.pdf", "page": 1},
        {"source": "doc2.pdf", "page": 1},
    ],
    ids=["id1", "id2"]
)

# Query with filtering
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=5,
    where={"source": "doc1.pdf"},
)
```

---

## 5. pgvector (PostgreSQL Extension)

**Repository**: https://github.com/pgvector/pgvector  
**License**: PostgreSQL License  
**Language**: C  
**Maturity**: ⭐⭐⭐⭐ Production-ready

### Overview

pgvector adds vector similarity search capabilities to PostgreSQL. It's ideal when you want to keep everything in your existing PostgreSQL database.

### Key Features

**Core Capabilities**:
- **PostgreSQL Native**: Vectors stored in Postgres
- **Multiple Distance Metrics**: L2, Inner Product, Cosine
- **Indexes**: IVFFlat and HNSW indexes
- **SQL Interface**: Use standard SQL for queries
- **ACID Transactions**: Full Postgres transaction support
- **Existing Tools**: Use all Postgres tooling

**Advanced Features**:
- **Hybrid Queries**: Easily combine vector and relational queries
- **JSON Support**: Store rich metadata
- **Full-Text Search**: Native PostgreSQL FTS
- **Aggregations**: SQL aggregations on vectors
- **Replication**: Postgres streaming replication

### Performance

**Benchmarks** (1M vectors, 768 dimensions):
- **Single Query Latency**: 30-100ms (p95) - depends on index
- **Throughput**: 100-300 queries/second
- **Insert Speed**: 1,000-3,000 vectors/second
- **Memory Usage**: ~4GB per 1M vectors
- **Indexing Time**: ~10-15 minutes for 1M vectors

**Scaling**:
- Good for <10M vectors
- Limited by PostgreSQL scaling
- Can use read replicas

### Pros

✅ **Use Existing Postgres**: No new database to manage  
✅ **ACID Transactions**: Full transactional support  
✅ **Mature Tooling**: All Postgres tools work  
✅ **Hybrid Queries**: Easy to join vectors with relational data  
✅ **Simple Deployment**: Just an extension  
✅ **SQL Interface**: Familiar query language  
✅ **Backup/Restore**: Standard Postgres tools  

### Cons

❌ **Performance**: Slower than purpose-built vector DBs  
❌ **Limited Scale**: Not for massive vector collections  
❌ **Memory Usage**: Higher than specialized solutions  
❌ **Index Build**: Slower indexing than alternatives  
❌ **Resource Contention**: Shares resources with relational queries  

### Best For

- When you already use PostgreSQL
- Small to medium scale (<1M vectors)
- Need ACID transactions
- Want to avoid additional infrastructure
- Hybrid relational + vector queries

### Configuration Example

```sql
-- Enable extension
CREATE EXTENSION vector;

-- Create table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    embedding vector(1024),
    text TEXT,
    source VARCHAR(512),
    metadata JSONB
);

-- Create index
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- Insert data
INSERT INTO documents (embedding, text, source, metadata)
VALUES ('[0.1,0.2,...]'::vector, 'Document content', 'doc.pdf', '{"page": 1}');

-- Query
SELECT id, text, source, 
       1 - (embedding <=> '[0.1,0.2,...]'::vector) AS similarity
FROM documents
WHERE metadata->>'page' = '1'
ORDER BY embedding <=> '[0.1,0.2,...]'::vector
LIMIT 5;
```

---

## 6. LanceDB

**Repository**: https://github.com/lancedb/lancedb  
**License**: Apache 2.0  
**Language**: Rust, Python  
**Maturity**: ⭐⭐⭐ Growing

### Overview

LanceDB is a developer-friendly, embedded vector database built on the Lance columnar format. It's designed for AI applications with a focus on performance and simplicity.

### Key Features

**Core Capabilities**:
- **Embedded Mode**: Run in-process (like SQLite)
- **Serverless**: Also offers cloud mode
- **Disk-based**: Efficient storage, not all in RAM
- **Versioning**: Built-in data versioning
- **Zero-Copy**: Efficient data access
- **Multi-Modal**: Handles various data types

**Advanced Features**:
- **Lance Format**: Efficient columnar storage
- **Incremental Updates**: Fast updates without reindexing
- **Time Travel**: Query historical versions
- **Python/JS SDKs**: Multiple language support

### Performance

**Benchmarks** (1M vectors, 768 dimensions):
- **Single Query Latency**: 20-40ms (p95)
- **Throughput**: 300-500 queries/second
- **Insert Speed**: 5,000-8,000 vectors/second
- **Disk Usage**: ~2GB per 1M vectors
- **Indexing Time**: ~3-4 minutes for 1M vectors

**Scaling**:
- Good for 1M-10M vectors
- Embedded mode limited by single machine
- Cloud mode scales better

### Pros

✅ **Embedded Mode**: No server needed  
✅ **Simple Setup**: Easy to get started  
✅ **Disk-Based**: Lower memory requirements  
✅ **Versioning**: Built-in data versioning  
✅ **Fast Development**: Active improvements  
✅ **Multi-Language**: Python and JavaScript  

### Cons

❌ **Newer Project**: Less mature than alternatives  
❌ **Limited Ecosystem**: Fewer integrations  
❌ **Embedded Limitations**: Single-machine in embedded mode  
❌ **Documentation**: Still evolving  

### Best For

- Embedded applications
- When you want simplicity of SQLite
- Development and prototyping
- Budget-conscious projects (low memory)

---

## Side-by-Side Comparison

### Feature Comparison Matrix

| Feature | Qdrant | Milvus | Weaviate | Chroma | pgvector | LanceDB |
|---------|--------|--------|----------|--------|----------|---------|
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Scalability** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Metadata Filtering** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Hybrid Search** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Maturity** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Documentation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Resource Usage** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Deployment** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Performance Comparison (1M vectors, 768d, HNSW index)

| Metric | Qdrant | Milvus | Weaviate | Chroma | pgvector | LanceDB |
|--------|--------|--------|----------|--------|----------|---------|
| **Query Latency (p95)** | 5-15ms | 10-20ms | 15-30ms | 20-50ms | 30-100ms | 20-40ms |
| **Throughput (QPS)** | 1000+ | 500-1000 | 300-600 | 200-400 | 100-300 | 300-500 |
| **Insert Speed (vec/s)** | 10,000+ | 5-10k | 5,000 | 2-5k | 1-3k | 5-8k |
| **Memory Usage** | 1.5GB | 2GB | 2.5GB | 3GB | 4GB | 1GB (disk) |
| **Index Build Time** | 2 min | 5 min | 3-4 min | 5-7 min | 10-15 min | 3-4 min |

### Scale Comparison

| Database | Ideal Range | Maximum Tested | Notes |
|----------|-------------|----------------|-------|
| **Qdrant** | 1M-100M | 100M+ | Excellent up to 50M, good beyond |
| **Milvus** | 10M-10B | 10B+ | Designed for massive scale |
| **Weaviate** | 1M-50M | 100M | Good mid-scale performance |
| **Chroma** | 10k-1M | ~5M | Limited scalability |
| **pgvector** | 100k-10M | ~30M | Postgres scaling limits |
| **LanceDB** | 100k-10M | ~20M | Embedded mode limited |

---

## Detailed Comparison by Use Case

### For Graph RAG Applications

**Primary Recommendation: Qdrant**

**Why Qdrant for Graph RAG:**
1. ✅ **Excellent Metadata Filtering**: Critical for filtering by document, section, entity type
2. ✅ **Fast Performance**: Low latency for interactive queries
3. ✅ **Hybrid Search**: Combine semantic and keyword search
4. ✅ **Easy Integration**: Works seamlessly with LangChain/LlamaIndex
5. ✅ **Reliable**: Production-ready with good operational characteristics
6. ✅ **Quantization**: Reduce memory for large document collections

**Alternative: Weaviate**
- Choose if you need built-in reranking
- Generative search integration
- Prefer GraphQL API

**For Massive Scale: Milvus**
- Choose if you have 100M+ vectors
- Need GPU acceleration
- Have DevOps resources for Kubernetes

### By Deployment Context

#### **Development & Prototyping**
1. **Chroma** - Simplest to get started
2. **LanceDB** - Embedded, no server needed
3. **Qdrant** - Still easy, more features

#### **Production (Small-Medium)**
1. **Qdrant** - Best balance
2. **Weaviate** - If hybrid search critical
3. **pgvector** - If already using Postgres

#### **Production (Large Scale)**
1. **Milvus** - Designed for billions of vectors
2. **Qdrant** - Excellent up to 100M
3. **Weaviate** - Good up to 50M

#### **Embedded Applications**
1. **LanceDB** - Purpose-built for embedded
2. **Chroma** - In-memory mode
3. **pgvector** - If SQLite too limited

---

## Decision Framework

### Quick Decision Tree

```
Do you already use PostgreSQL extensively?
├─ Yes → Consider pgvector
└─ No → Continue

Is your dataset < 100k vectors?
├─ Yes → Chroma or LanceDB (simplicity)
└─ No → Continue

Is your dataset > 100M vectors?
├─ Yes → Milvus
└─ No → Continue

Do you need built-in hybrid search + reranking?
├─ Yes → Weaviate
└─ No → Qdrant (RECOMMENDED)
```

### Selection Criteria

| Priority | Best Choice |
|----------|-------------|
| **Overall Best** | Qdrant |
| **Simplicity** | Chroma |
| **Massive Scale** | Milvus |
| **Hybrid Search** | Weaviate |
| **Use Existing DB** | pgvector |
| **Embedded** | LanceDB |
| **Performance** | Qdrant |
| **Metadata Filtering** | Qdrant |
| **GPU Acceleration** | Milvus |
| **Developer Experience** | Qdrant or Chroma |

---

## Integration with Graph RAG Stack

### How Vector DB Fits in Graph RAG

```
Document Ingestion:
1. Document → Chunks → Embeddings → [Vector DB]
2. Document → Chunks → Entities → [Graph DB]

Query Processing:
1. Query → Query Embedding → [Vector DB Search] → Relevant Chunks
2. Query → Entity Extraction → [Graph DB Traversal] → Related Entities
3. Combine Results → Rerank → LLM Context
```

### Qdrant Integration Example

```python
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
client = QdrantClient(host="localhost", port=6333)

# Create LangChain vector store
vectorstore = Qdrant(
    client=client,
    collection_name="documents",
    embeddings=embeddings,
)

# Add documents
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
vectorstore.add_documents(chunks)

# Search with metadata filtering
results = vectorstore.similarity_search(
    query="What is quantum computing?",
    k=5,
    filter={
        "document_type": "research_paper",
        "year": {"$gte": 2020}
    }
)
```

### Pluggable Vector DB Architecture

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorDBProvider(ABC):
    @abstractmethod
    def add_vectors(self, vectors: List[List[float]], metadatas: List[Dict], ids: List[str]):
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], k: int, filter: Dict = None) -> List[Dict]:
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]):
        pass

class QdrantProvider(VectorDBProvider):
    def __init__(self, host: str, port: int, collection_name: str):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
    
    def add_vectors(self, vectors: List[List[float]], metadatas: List[Dict], ids: List[str]):
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(id=id, vector=vec, payload=meta)
            for id, vec, meta in zip(ids, vectors, metadatas)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
    
    def search(self, query_vector: List[float], k: int, filter: Dict = None) -> List[Dict]:
        # Convert filter to Qdrant format
        qdrant_filter = self._convert_filter(filter) if filter else None
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=k,
        )
        return [{"id": r.id, "score": r.score, "metadata": r.payload} for r in results]

# Factory
def get_vector_db_provider(config: Dict) -> VectorDBProvider:
    providers = {
        'qdrant': QdrantProvider,
        'milvus': MilvusProvider,
        'weaviate': WeaviateProvider,
        # ... others
    }
    provider_type = config.get('provider', 'qdrant')
    return providers[provider_type](**config.get('params', {}))
```

---

## Best Practices

### 1. Index Selection

**HNSW (Hierarchical Navigable Small World)**:
- Best for: Most use cases
- Pros: Fast queries, good recall
- Cons: Slower index build, more memory
- Recommended for: Graph RAG

**IVF (Inverted File)**:
- Best for: Very large datasets, can sacrifice accuracy
- Pros: Fast build, less memory
- Cons: Lower recall, slower queries
- Recommended for: >100M vectors

### 2. Distance Metrics

**Cosine Similarity**: Recommended for most text embeddings
- Normalizes vector length
- Focus on direction similarity
- Most common for RAG

**L2 (Euclidean)**: When magnitude matters
**Inner Product**: For specific embedding models

### 3. Quantization

**When to use**:
- Memory is constrained
- Have >1M vectors
- Can accept small accuracy trade-off (2-3%)

**Types**:
- **Scalar Quantization**: Best balance (Qdrant)
- **Product Quantization**: More aggressive (Milvus)
- **Binary Quantization**: Most aggressive (Qdrant)

### 4. Metadata Strategy

**Always store**:
- Source document ID
- Chunk index
- Document type
- Creation timestamp
- Section/page information

**For Graph RAG**:
- Entity mentions in chunk
- Graph node IDs related to chunk
- Quality scores
- Chunk creation strategy used

### 5. Monitoring

**Key Metrics**:
- Query latency (p50, p95, p99)
- Throughput (queries per second)
- Memory usage
- Index size
- Cache hit rate

---

## Common Pitfalls & Solutions

### Pitfall 1: No Metadata Filtering
**Problem**: Retrieving irrelevant chunks from wrong documents  
**Solution**: Always use metadata filtering (document_id, type, etc.)

### Pitfall 2: Wrong Distance Metric
**Problem**: Poor retrieval quality  
**Solution**: Use Cosine for most text embeddings

### Pitfall 3: No Index Tuning
**Problem**: Slow queries or poor recall  
**Solution**: Tune HNSW parameters (M, efConstruction)

### Pitfall 4: Insufficient Memory
**Problem**: OOM errors or thrashing  
**Solution**: Use quantization or choose disk-based solution

### Pitfall 5: No Backup Strategy
**Problem**: Data loss  
**Solution**: Regular snapshots (Qdrant) or backups

---

## Migration Path

### Start with Qdrant, Scale if Needed

**Phase 1: Development (Weeks 1-2)**
- Use Qdrant with default settings
- Get end-to-end pipeline working
- Establish baseline performance

**Phase 2: Optimization (Weeks 3-4)**
- Tune index parameters
- Add quantization if memory-constrained
- Implement metadata filtering strategy

**Phase 3: Production (Week 5-8)**
- Set up monitoring
- Configure backups/snapshots
- Load test with realistic traffic

**Phase 4: Scale (Month 3+)**
- If vectors > 100M → Consider Milvus
- If hybrid search critical → Consider Weaviate
- Otherwise → Stay with Qdrant

---

## Conclusion

### Recommended Solution for Graph RAG

**Primary: Qdrant**
- Best overall balance for Graph RAG applications
- Excellent performance with rich features
- Easy to deploy and operate
- Strong metadata filtering (critical for RAG)
- Production-ready reliability

**When to Consider Alternatives**:
- **Milvus**: If you have >100M vectors and Kubernetes expertise
- **Weaviate**: If built-in hybrid search and reranking are must-haves
- **pgvector**: If you want to minimize infrastructure (already use Postgres)
- **Chroma**: Only for prototyping or very small scale

### Key Takeaways

✅ **Qdrant** is the best general-purpose choice for Graph RAG  
✅ **Metadata filtering** is critical - ensure your vector DB supports it well  
✅ **HNSW indexing** with Cosine similarity for most use cases  
✅ **Quantization** can reduce memory usage by 4-8x with minimal quality loss  
✅ Start simple, scale when needed - premature optimization is costly  
✅ **Integration** with LangChain/LlamaIndex matters - choose supported options  

The vector database is a critical component that directly impacts retrieval quality and system performance. Choosing the right one and configuring it properly can improve your Graph RAG system's accuracy by 10-15%.

---

## Resources

### Official Documentation
- **Qdrant**: https://qdrant.tech/documentation/
- **Milvus**: https://milvus.io/docs
- **Weaviate**: https://weaviate.io/developers/weaviate
- **Chroma**: https://docs.trychroma.com/
- **pgvector**: https://github.com/pgvector/pgvector
- **LanceDB**: https://lancedb.github.io/lancedb/

### Benchmarks
- **ANN Benchmarks**: http://ann-benchmarks.com/
- **VectorDBBench**: https://github.com/zilliztech/VectorDBBench

### Comparisons
- **Qdrant vs Others**: https://qdrant.tech/benchmarks/
- **Milvus Benchmarks**: https://milvus.io/docs/benchmark.md

### Integration Guides
- **LangChain Vector Stores**: https://python.langchain.com/docs/modules/data_connection/vectorstores/
- **LlamaIndex Vector Stores**: https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/
