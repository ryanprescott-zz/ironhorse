# Open Source Graph Database Comparison for Graph RAG Application

## Executive Summary

This document compares leading open source graph databases suitable for the Graph RAG architecture described in your application design. Based on current market analysis (2025), we evaluate eight major solutions across critical dimensions including performance, scalability, licensing, query language, and RAG-specific requirements.

**Top Recommendation**: **Neo4j Community Edition** remains the best choice for your Graph RAG application due to its mature ecosystem, excellent Cypher query language, strong LangChain/LlamaIndex integration, and production-proven reliability. For performance-critical scenarios requiring real-time analytics, **Memgraph** offers a compelling alternative.

---

## Comparison Matrix

| Database | License | Architecture | Query Language | Scale | Performance | RAG Integration | Maintenance |
|----------|---------|--------------|----------------|-------|-------------|-----------------|-------------|
| **Neo4j Community** | GPLv3 | Disk-based, single-node | Cypher | 1M-100M nodes | Good | ⭐ Excellent | ⭐ Excellent |
| **Memgraph** | BSL/Community | In-memory | Cypher (OpenCypher) | 1M-100M nodes | ⭐ Excellent | Good | Good |
| **ArangoDB** | ArangoDB Community License | Multi-model, disk-based | AQL | 1M-1B+ nodes | Very Good | Good | Good |
| **JanusGraph** | Apache 2.0 | Distributed | Gremlin | 100M-1B+ nodes | Good | Fair | Fair |
| **NebulaGraph** | Apache 2.0 | Distributed | nGQL (openCypher-based) | 100M-10B+ nodes | Very Good | Fair | Good |
| **Dgraph** | Apache 2.0 (v25) | Distributed | GraphQL+/DQL | 1M-1B+ nodes | Very Good | Good | Fair |
| **OrientDB** | Apache 2.0 | Multi-model, disk-based | SQL-like | 1M-100M nodes | Fair | Fair | Poor |
| **HugeGraph** | Apache 2.0 | Distributed | Gremlin | 100M-1B+ nodes | Fair | Fair | Fair |

---

## Detailed Analysis

### 1. Neo4j Community Edition ⭐ RECOMMENDED

**Overview**: The most mature and widely-adopted graph database with the largest community and ecosystem.

**Strengths**:
- **Best-in-class documentation and learning resources**: Extensive tutorials, certification programs, and 100K+ community members
- **Cypher query language**: Intuitive, SQL-like syntax specifically designed for graphs, making it easy for developers to learn
- **Native graph storage**: Purpose-built for graph data with optimized traversals (1 billion relationships in <500ms)
- **Excellent RAG ecosystem integration**: 
  - First-class support in LangChain and LlamaIndex
  - Rich community examples for Graph RAG patterns
  - Strong Python client (Neo4j Python driver + graph-ml integrations)
- **Production-proven**: Used by thousands of enterprises for mission-critical applications
- **ACID compliance**: Full transactional guarantees for data integrity
- **Visualization tools**: Neo4j Browser and Bloom for graph exploration

**Limitations**:
- **Single-node only**: Community Edition does not support clustering or high availability
- **No hot backups**: Requires downtime for backups in Community Edition
- **GPLv3 license**: May have implications for some commercial use cases (though typically fine for internal applications)
- **Resource intensive**: Java-based implementation can consume significant memory
- **Disk-based**: Slower than in-memory solutions for real-time workloads

**Best For**:
- General-purpose Graph RAG applications
- Development and medium-scale production deployments (<50M nodes)
- Teams wanting mature tooling and extensive documentation
- Projects requiring strong Python/LangChain integration

**License Considerations**:
- GPLv3 allows free use for internal applications
- Does not require source code disclosure if used via REST/Bolt API (clear separation)
- Commercial license available for clustering and enterprise features
- No size limitations in Community Edition (historical limits removed)

**Performance Characteristics**:
- Query latency: 10-100ms for typical graph traversals
- Throughput: Good for read-heavy workloads, moderate for writes
- Memory: ~4-8GB minimum, scales with dataset size

---

### 2. Memgraph

**Overview**: High-performance in-memory graph database optimized for real-time analytics and streaming data.

**Strengths**:
- **Exceptional performance**: 8-50x faster than Neo4j for many workloads
  - 8x faster for read-heavy operations
  - 50x faster for write-heavy operations
  - Sub-millisecond query latency for in-memory operations
- **Neo4j compatibility**: Drop-in replacement supporting Cypher and Bolt protocol
  - Easy migration from Neo4j
  - Same drivers and tools work with both
- **Streaming integration**: Native Kafka and Pulsar connectors out-of-the-box
- **Memory efficiency**: Consumes 25% of the memory compared to Neo4j for equivalent datasets
- **C++ implementation**: More optimized than Java-based alternatives
- **Python ecosystem**: Strong Python support for custom query modules (MAGE library)
- **Real-time focus**: Built specifically for streaming and real-time graph analytics

**Limitations**:
- **Memory constraints**: All data must fit in RAM (with periodic snapshots to disk)
- **Smaller ecosystem**: Fewer community examples and third-party integrations compared to Neo4j
- **Younger product**: Less mature than Neo4j (though battle-tested in production)
- **Community Edition restrictions**: Some enterprise features (LDAP, auditing) only in Enterprise
- **Limited historical query support**: Optimized for current state, not time-travel queries

**Best For**:
- Real-time streaming Graph RAG applications
- Performance-critical scenarios requiring <10ms query latency
- Knowledge graph construction from streaming data sources
- Projects where dataset fits comfortably in memory (<100GB)
- Applications requiring high write throughput

**License**:
- Business Source License (BSL) for Community Edition
- Converts to Apache 2.0 after 4 years
- Permissive for most use cases, but verify terms for your needs

**Performance Characteristics**:
- Query latency: 1-10ms for typical graph traversals
- Throughput: Excellent for both reads and writes
- Memory: Dataset must fit in RAM (50-200GB typical)

---

### 3. ArangoDB

**Overview**: Multi-model database supporting graph, document, and key-value models in a unified system.

**Strengths**:
- **Multi-model flexibility**: Single database for graphs, documents, and KV data
  - Store text chunks as documents
  - Maintain entity relationships as graphs
  - Use single query language (AQL) across all models
- **Unified query language**: AQL supports traversals, joins, and aggregations across data models
- **Good performance**: Competitive with Neo4j in benchmarks for graph operations
- **Distributed clustering**: Built-in sharding and replication (in Enterprise Edition)
- **JSON-native**: Natural fit for document storage alongside graph data
- **Foxx microservices**: Build APIs directly in the database
- **Kubernetes integration**: Cloud-native deployment options

**Limitations**:
- **License restrictions**: ArangoDB Community License has 100GB limit and commercial use restrictions (since Oct 2023)
- **Learning curve**: Multi-model capabilities add complexity even if not needed
- **AQL learning curve**: Different from standard Cypher or SQL
- **Smaller RAG ecosystem**: Fewer LangChain/LlamaIndex examples compared to Neo4j
- **Graph performance**: Not as optimized for pure graph workloads as specialized solutions

**Best For**:
- Projects requiring both document and graph storage
- Applications needing flexibility to switch between data models
- Teams wanting to minimize database infrastructure
- Scenarios where documents and relationships are equally important

**License**:
- ArangoDB Community License (restrictive)
- 100GB single cluster limit in Community Edition
- Enterprise Edition available for commercial use

**Performance Characteristics**:
- Query latency: 15-50ms for graph traversals
- Throughput: Very good for mixed workloads
- Memory: Moderate requirements (2-4x data size)

---

### 4. JanusGraph

**Overview**: Distributed graph database designed for massive-scale graphs using pluggable storage backends.

**Strengths**:
- **Apache 2.0 license**: Fully open source without restrictions
- **Massive scale**: Designed for hundreds of billions of vertices and trillions of edges
- **Pluggable backends**: Works with Cassandra, HBase, or Berkeley DB
- **TinkerPop integration**: Full Apache TinkerPop and Gremlin support
- **Distributed architecture**: Separates compute and storage for independent scaling
- **ACID transactions**: Maintains consistency in distributed environment
- **Apache Spark integration**: Global graph analytics capabilities
- **Active development**: Backed by Linux Foundation with multiple contributors

**Limitations**:
- **Gremlin complexity**: Steeper learning curve compared to Cypher
- **Setup complexity**: Requires configuring and managing multiple systems (JanusGraph + storage backend + indexing)
- **Operational overhead**: More moving parts to monitor and maintain
- **Performance**: Slower than specialized solutions for small-to-medium graphs
- **RAG integration**: Less mature Python ecosystem compared to Neo4j
- **Community size**: Smaller than Neo4j, though growing

**Best For**:
- Applications requiring >100M nodes from day one
- Organizations already using Cassandra or HBase
- Projects needing Apache 2.0 licensing
- Scenarios requiring petabyte-scale graph data
- Teams comfortable with distributed systems operations

**License**:
- Apache 2.0 (fully permissive)

**Performance Characteristics**:
- Query latency: 50-200ms for distributed queries
- Throughput: Excellent for write-heavy workloads at scale
- Memory: Depends on backend; generally moderate

---

### 5. NebulaGraph

**Overview**: Distributed graph database designed for ultra-large-scale graphs with strong performance characteristics.

**Strengths**:
- **Massive scalability**: Handles 100B+ vertices and trillions of edges
- **Strong performance**: Better than Dgraph and JanusGraph in benchmarks
  - Balanced data distribution across nodes
  - Low storage amplification ratio (2.67x)
  - Fast bulk loading (3.4 hours for 194GB dataset)
- **Apache 2.0 license**: Fully permissive open source
- **Separation of concerns**: Storage, compute, and meta services separated for independent scaling
- **nGQL query language**: Based on openCypher, familiar to Cypher users
- **Active development**: Strong backing from Chinese tech companies (Vesoft)
- **Cloud-native**: Kubernetes-ready with good container support

**Limitations**:
- **Complexity**: Requires understanding of distributed architecture
- **Smaller Western ecosystem**: Primarily documented and used in Chinese market
- **RAG integration**: Limited LangChain/LlamaIndex examples and documentation
- **Learning curve**: nGQL similar but not identical to Cypher
- **Operational overhead**: Need expertise in distributed systems
- **Resource requirements**: Minimum 3-node cluster for high availability

**Best For**:
- Applications requiring extreme scale (>100M nodes)
- Organizations with distributed systems expertise
- Projects needing Apache 2.0 licensing with enterprise features
- Asian markets with strong local support

**License**:
- Apache 2.0 (fully permissive)

**Performance Characteristics**:
- Query latency: 20-100ms for distributed queries
- Throughput: Excellent for massive-scale operations
- Memory: Moderate per-node requirements with horizontal scaling

---

### 6. Dgraph

**Overview**: Distributed native graph database with GraphQL support and focus on modern API design.

**Strengths**:
- **GraphQL native**: Built-in GraphQL support simplifies API integration
- **DQL query language**: Powerful custom query language for complex traversals
- **Distributed architecture**: Horizontal scaling and high availability built-in
- **ACID transactions**: Full consistency guarantees in distributed setup
- **JSON data model**: Natural for document-based applications
- **Good performance**: Fast for real-time queries over large datasets
- **Apache 2.0 (upcoming)**: v25 release moving all features to Apache 2.0

**Limitations**:
- **Data distribution issues**: Poor load balancing in some benchmarks
  - Predicate-based sharding can create hotspots
  - Observed storage imbalance (1GB/1GB/22GB across 3 nodes in tests)
- **Bulk loading challenges**: OOM errors with large datasets in tests
- **Smaller ecosystem**: Less mature than Neo4j or JanusGraph
- **RAG integration**: Limited documentation for Graph RAG patterns
- **GraphQL+/DQL**: Requires learning custom query syntax
- **License transition**: Currently in flux (moving to full Apache 2.0)

**Best For**:
- GraphQL-first applications
- Projects requiring modern API design
- Teams familiar with GraphQL ecosystem
- Medium-to-large scale deployments (1M-100M nodes)

**License**:
- Core: Apache 2.0
- Moving to fully Apache 2.0 in v25 (early 2025)

**Performance Characteristics**:
- Query latency: 20-80ms for typical queries
- Throughput: Good for read-heavy workloads
- Memory: Can be memory-intensive for large graphs

---

### 7. OrientDB

**Overview**: Multi-model database supporting graph, document, and key-value models with SQL-like query language.

**Strengths**:
- **Multi-model support**: Graph, document, key-value, and object models
- **SQL familiarity**: SQL-like query syntax lowers learning curve
- **Apache 2.0 license**: Community Edition fully open source
- **Good performance**: 220,000 records/second throughput
- **ACID compliance**: Full transactional support
- **Embedded mode**: Can be embedded in Java applications
- **Distributed clustering**: Support for multi-master replication

**Limitations**:
- **Declining activity**: Less active development and community engagement
- **Migration pain points**: Reports of performance bottlenecks and single-point failures
- **Documentation gaps**: Incomplete documentation for advanced features
- **Smaller ecosystem**: Limited Graph RAG examples and integrations
- **Storage issues**: High storage amplification and poor load balancing in tests
- **Limited enterprise support**: Weaker than Neo4j or ArangoDB

**Best For**:
- Legacy system migrations (though NebulaGraph is often chosen as replacement)
- Teams requiring SQL-like syntax
- Projects needing multi-model flexibility on a budget
- Java-embedded applications

**License**:
- Community Edition: Apache 2.0
- Enterprise Edition: Commercial

**Performance Characteristics**:
- Query latency: 30-100ms for graph operations
- Throughput: Good for transactional workloads
- Memory: Moderate requirements

**Note**: While still viable, OrientDB is often replaced by newer solutions (NebulaGraph, ArangoDB) due to performance and maintainability concerns.

---

### 8. HugeGraph

**Overview**: Distributed graph database developed by Baidu, designed for massive-scale Chinese internet applications.

**Strengths**:
- **Apache 2.0 license**: Fully open source
- **TinkerPop compliant**: Standard Gremlin support
- **Baidu backing**: Strong support from major tech company
- **Chinese market**: Good documentation and support in Chinese

**Limitations**:
- **Poor performance metrics**: Worst results in comparative benchmarks
  - Highest storage amplification
  - Severe data distribution imbalance
  - Failed to complete full imports in tests (1000GB node full at 194GB dataset)
- **Western ecosystem**: Limited documentation and support outside China
- **RAG integration**: Very few examples or integrations
- **Operational complexity**: Difficult to configure and optimize

**Best For**:
- Chinese market deployments with local support
- Organizations already using Baidu ecosystem
- Specialized use cases requiring Baidu integration

**License**:
- Apache 2.0

**Performance Characteristics**:
- Query latency: Highly variable, often poor
- Throughput: Limited compared to alternatives
- Memory: High storage amplification (worst in benchmarks)

**Recommendation**: Not recommended for most Graph RAG applications due to performance concerns and limited Western ecosystem.

---

## Graph RAG-Specific Considerations

### Integration with LangChain/LlamaIndex

**Excellent Support**:
1. **Neo4j**: First-class integration with extensive examples
   - `Neo4jGraph` class in LangChain
   - Rich documentation for RAG patterns
   - Community examples for knowledge graph construction

2. **Memgraph**: Good Cypher compatibility means Neo4j examples work
   - Neo4j drivers compatible via Bolt protocol
   - Growing library of Python integrations

**Good Support**:
3. **ArangoDB**: Python drivers available, some RAG examples
4. **Dgraph**: GraphQL integration, limited RAG-specific docs

**Limited Support**:
5. **JanusGraph, NebulaGraph, OrientDB, HugeGraph**: Require custom integration work

### Entity Extraction and Knowledge Graph Construction

**Best Choices**:
- **Neo4j**: Mature patterns for entity extraction with LLMs
- **Memgraph**: Fast writes ideal for streaming entity extraction
- **ArangoDB**: Good for hybrid document + graph entity storage

**Considerations**:
- Graph construction typically write-heavy during ingestion
- Need good Python client libraries for LLM integration
- Schema flexibility important for evolving entity types

### Query Performance for RAG Retrieval

For typical RAG query patterns (finding related entities, path traversal, neighborhood retrieval):

**Fastest** (1-10ms):
- Memgraph (in-memory)

**Fast** (10-50ms):
- Neo4j (optimized native graph storage)
- ArangoDB (good indexes)

**Moderate** (50-200ms):
- JanusGraph, NebulaGraph, Dgraph (distributed systems)

**Variable**:
- OrientDB, HugeGraph (depends on configuration)

---

## Licensing Deep Dive

### Fully Permissive (Best for Commercial Use)
- **Apache 2.0**: JanusGraph, NebulaGraph, Dgraph (v25), OrientDB CE, HugeGraph
  - No restrictions on use, modification, or distribution
  - Can be used in commercial products
  - No copyleft requirements

### Copyleft (Requires Careful Consideration)
- **GPLv3** (Neo4j Community Edition):
  - Free for internal use and SaaS applications
  - Source disclosure only required if you modify Neo4j itself AND distribute
  - Using via REST/Bolt API (as designed) doesn't trigger copyleft
  - Many commercial SaaS companies use Neo4j CE successfully
  - Consider: Your application is separate from Neo4j; just talks to it

### Restricted (Read Terms Carefully)
- **Business Source License** (Memgraph Community):
  - Production use allowed
  - Converts to Apache 2.0 after 4 years
  - Some commercial restrictions during BSL period
  
- **ArangoDB Community License**:
  - 100GB single cluster limit
  - Restrictions on commercial use
  - Enterprise Edition required for production at scale

---

## Deployment and Operational Considerations

### Easiest to Deploy and Manage
1. **Neo4j**: Single binary, Docker container, excellent documentation
2. **Memgraph**: Single container, minimal configuration
3. **ArangoDB**: Single instance or cluster, good Docker support

### More Complex
4. **Dgraph**: Requires understanding of distributed architecture
5. **JanusGraph**: Multiple components (JanusGraph + storage + indexing)
6. **NebulaGraph**: 3+ node cluster with separated services

### Most Complex
7. **OrientDB**: Configuration challenges reported
8. **HugeGraph**: Complex setup with limited English documentation

---

## Performance Summary (Based on Benchmarks)

### LDBC SNB Benchmark Results (2023)
For graph operations at scale:

**Query Execution Time** (average, lower is better):
1. Neo4j: ⭐ Best overall
2. TigerGraph: Very good (not open source)
3. JanusGraph: Good
4. NebulaGraph: Good

**Data Loading** (194GB dataset):
1. NebulaGraph: 3.4 hours total
2. Dgraph: Partial success only (OOM errors)
3. HugeGraph: Failed to complete

**Memory Usage**:
1. Memgraph: 25% of Neo4j memory for same dataset
2. Neo4j: Moderate
3. NebulaGraph: Efficient per-node
4. Others: Variable

### Real-World Performance Characteristics

**Read-Heavy Workloads**:
1. Memgraph (in-memory advantage)
2. Neo4j (native graph storage)
3. ArangoDB
4. NebulaGraph, JanusGraph (distributed overhead)

**Write-Heavy Workloads**:
1. Memgraph (50x faster than Neo4j in some tests)
2. NebulaGraph (good bulk loading)
3. JanusGraph (distributed writes)
4. Neo4j (moderate write performance)

**Mixed Workloads**:
1. Neo4j (balanced)
2. ArangoDB (multi-model optimization)
3. Memgraph (excellent overall)

---

## Recommendations by Use Case

### 🏆 General Purpose Graph RAG (Recommended)
**Choice**: Neo4j Community Edition

**Why**:
- Mature ecosystem with extensive documentation
- Excellent LangChain/LlamaIndex integration
- Intuitive Cypher query language
- Production-proven reliability
- Strong community support
- Good balance of features and complexity

**When to reconsider**: Dataset >50M nodes or need clustering

---

### ⚡ Performance-Critical Real-Time RAG
**Choice**: Memgraph

**Why**:
- 8-50x faster than alternatives
- Sub-10ms query latency
- Excellent for streaming data ingestion
- Neo4j compatible (easy migration if needed)
- Strong write throughput for entity extraction

**When to reconsider**: Dataset won't fit in memory or need mature ecosystem

---

### 📊 Hybrid Document + Graph Storage
**Choice**: ArangoDB

**Why**:
- Single database for documents (text chunks) and graphs (entities)
- Unified query language across models
- Good performance for both access patterns
- Natural fit for RAG hybrid storage

**When to reconsider**: License limitations (100GB) or need pure graph optimization

---

### 🚀 Massive Scale (>100M nodes)
**Choice**: NebulaGraph or JanusGraph

**Why**:
- Designed for billion+ node graphs
- Apache 2.0 licensing
- Proven at scale in production
- Horizontal scalability

**JanusGraph if**: Already using Cassandra/HBase, need TinkerPop ecosystem
**NebulaGraph if**: Starting fresh, want better performance, comfortable with nGQL

**When to reconsider**: Dataset <100M nodes (overhead not worth it)

---

### 💰 Budget-Conscious with Apache 2.0 Required
**Choice**: JanusGraph or NebulaGraph

**Why**:
- Fully open source with no commercial restrictions
- Can scale to enterprise needs without licensing costs
- Active communities and ongoing development

---

### 🔧 Development/Prototyping
**Choice**: Neo4j Community Edition or Memgraph

**Why**:
- Quick setup and minimal configuration
- Excellent documentation and learning resources
- Easy to transition to production
- Strong Python ecosystem

---

## Migration Paths

### Starting with Neo4j
**Easy Migration To**:
- Memgraph (Cypher/Bolt compatible)

**Moderate Effort**:
- ArangoDB (different query language, document model)
- Dgraph (different query paradigm)

**Significant Effort**:
- JanusGraph, NebulaGraph (distributed architecture)

### Starting with Memgraph
**Easy Migration To**:
- Neo4j (Cypher/Bolt compatible)

### Starting with Others
Generally requires significant query language translation and data export/import.

---

## Decision Framework

### Start Here: Answer These Questions

1. **What is your expected scale?**
   - <10M nodes → Neo4j or Memgraph
   - 10-100M nodes → Neo4j, Memgraph, or ArangoDB
   - >100M nodes → NebulaGraph or JanusGraph

2. **What are your performance requirements?**
   - Real-time (<10ms) → Memgraph
   - Standard (<100ms) → Neo4j or ArangoDB
   - Batch analytics → Any distributed option

3. **What is your licensing requirement?**
   - Must be Apache 2.0 → JanusGraph, NebulaGraph, Dgraph (v25)
   - GPLv3 acceptable → Neo4j
   - Any open source → Most options work

4. **What is your team's expertise?**
   - SQL background → OrientDB or ArangoDB
   - Graph database experience → Neo4j or Memgraph
   - Distributed systems → JanusGraph or NebulaGraph
   - GraphQL background → Dgraph

5. **Do you need multi-model support?**
   - Yes → ArangoDB or OrientDB
   - No → Neo4j, Memgraph, or others

6. **What is your operational capacity?**
   - Minimal DevOps → Neo4j or Memgraph (single instance)
   - Strong DevOps team → Any distributed option
   - Managed service required → Consider Neo4j AuraDB (not open source)

---

## Conclusion

For most Graph RAG applications, **Neo4j Community Edition** remains the best choice due to its:
- Mature ecosystem and extensive documentation
- Excellent integration with LangChain and LlamaIndex  
- Intuitive Cypher query language
- Production-proven reliability at scale
- Strong community support
- Good balance between features and operational complexity

**Memgraph** is an excellent choice for performance-critical applications where:
- Sub-10ms query latency is required
- High write throughput is needed (streaming entity extraction)
- The dataset fits comfortably in available memory
- Neo4j compatibility provides a safety net for migration

**For specific scenarios**:
- **Massive scale (>100M nodes)**: NebulaGraph or JanusGraph with Apache 2.0 licensing
- **Multi-model needs**: ArangoDB for unified document + graph storage
- **Apache 2.0 requirement with moderate scale**: JanusGraph
- **Development/prototyping**: Neo4j or Memgraph for quick setup

The graph database landscape in 2025 offers mature, production-ready options for Graph RAG applications. The choice depends primarily on your scale requirements, performance needs, licensing constraints, and team expertise. Start with Neo4j or Memgraph for rapid development, and plan migration paths to distributed solutions (NebulaGraph, JanusGraph) only when scale requirements justify the additional operational complexity.

---

## Additional Resources

### Neo4j
- Documentation: https://neo4j.com/docs/
- LangChain Integration: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher
- Community Forum: https://community.neo4j.com/

### Memgraph
- Documentation: https://memgraph.com/docs
- MAGE Library: https://github.com/memgraph/mage
- Benchmarks vs Neo4j: https://memgraph.com/blog/neo4j-vs-memgraph

### ArangoDB
- Documentation: https://www.arangodb.com/docs/
- Graph Queries: https://www.arangodb.com/docs/stable/graphs.html

### JanusGraph
- Documentation: https://docs.janusgraph.org/
- GitHub: https://github.com/JanusGraph/janusgraph

### NebulaGraph
- Documentation: https://docs.nebula-graph.io/
- Benchmarks: https://www.nebula-graph.io/posts/nebula-graph-benchmark-performace-against-janusgraph-dgraph

### Comparative Benchmarks
- LDBC SNB Benchmark Study (2023): Academic comparison of JanusGraph, NebulaGraph, Neo4j, TigerGraph
- Meituan's Real-World Comparison: NebulaGraph vs Dgraph vs JanusGraph for production knowledge graphs

---

*Document Version: 1.0*  
*Last Updated: October 2025*  
*Based on: Current documentation, 2023-2025 benchmarks, community feedback, and production deployment reports*
