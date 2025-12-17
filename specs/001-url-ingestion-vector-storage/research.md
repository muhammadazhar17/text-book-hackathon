# Research Document: URL Ingestion & Vector Storage

## Research Findings Summary

### 1. Cohere Embedding Model Selection

**Decision**: Use Cohere's `embed-english-v3.0` model
**Rationale**: This is Cohere's latest embedding model optimized for retrieval tasks, which is perfect for RAG applications. It supports both classification and similarity tasks.
**Alternatives considered**:
- `embed-multilingual-v3.0` (for multilingual content, but our content appears to be English)
- `embed-english-light-v3.0` (lighter version, but potentially less accurate)

### 2. Text Chunking Strategy

**Decision**: Use 512-token chunks with 50-token overlap
**Rationale**: This size provides semantic coherence while staying well below Cohere's 5,120-token limit. The overlap ensures context preservation across chunks.
**Alternatives considered**:
- Character-based chunking (less semantic coherence)
- Sentence-based chunking (potentially too small for context)
- Larger chunks (risk of exceeding token limits)

### 3. Qdrant Collection Configuration

**Decision**: Create collection named "as_embeddingone" with 1024 dimensions
**Rationale**: Cohere's embedding model produces 1024-dimensional vectors. The name is descriptive and follows standard naming conventions.
**Configuration**:
- Vector size: 1024
- Distance metric: Cosine
- Collection name: "as_embeddingone"

### 4. Frontend Components

**Decision**: Command-line interface for triggering ingestion, no web UI needed initially
**Rationale**: The user requested a single backend file in `/backend/main.py`, suggesting a simple implementation. A CLI approach is sufficient for the ingestion process.
**Alternatives considered**:
- Web interface (would require additional dependencies and complexity)

### 5. Text Processing Libraries

**Decision**: Use BeautifulSoup for HTML parsing and custom logic for text cleaning
**Rationale**: BeautifulSoup is the standard for HTML parsing in Python. It handles malformed HTML well and provides easy navigation of DOM elements.
**Alternatives considered**:
- lxml (more complex setup)
- html2text (simpler but less control)

### 6. Rate Limiting and Best Practices

**Decision**: Implement 1 request per second with exponential backoff for API calls
**Rationale**: This respects web server resources and avoids being blocked. Cohere's rate limits are typically generous but implementing backoff is a best practice.
**Best practices**:
- Respect robots.txt
- Add user-agent identification
- Implement proper error handling and retries

### 7. Environment Variables

**Decision**: Define standard environment variables for configuration
**Configuration**:
- COHERE_API_KEY: API key for Cohere
- QDRANT_URL: URL for Qdrant Cloud instance
- QDRANT_API_KEY: API key for Qdrant Cloud
- COLLECTION_NAME: Name of the Qdrant collection (default: "as_embeddingone")

### 8. Error Handling Strategy

**Decision**: Comprehensive error handling with logging and retry mechanisms
**Approach**:
- Network errors: Retry with exponential backoff
- API errors: Log and continue with next item
- Data errors: Clean/transform where possible, skip if not