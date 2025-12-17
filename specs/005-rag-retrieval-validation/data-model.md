# Data Model: RAG Retrieval & Pipeline Validation

## Entities

### Query
- **Description**: The input text that needs to be matched against stored embeddings for retrieval
- **Fields**:
  - text: string (the query text)
  - embedding: list[float] (vector representation of the query)
- **Validation**: Must be non-empty and contain valid text content

### ContentChunk
- **Description**: Segments of original documents that are stored as vectors in the Qdrant collection
- **Fields**:
  - id: string (unique identifier)
  - content: string (the text content of the chunk)
  - embedding: list[float] (vector representation of the content)
  - similarity_score: float (cosine similarity score relative to the query)
- **Validation**: Must have content and valid embedding vector

### Metadata
- **Description**: Associated information that provides context and attribution for retrieved content
- **Fields**:
  - url: string (source URL of the content)
  - chunk_index: int (index position of the chunk in the original document)
  - source_document: string (optional reference to original document)
- **Validation**: URL must be a valid format, chunk_index must be non-negative

### SearchResult
- **Description**: Container for the results of a similarity search operation
- **Fields**:
  - query: Query (the original query)
  - results: list[ContentChunk] (top-k most relevant content chunks)
  - search_time_ms: float (time taken to perform the search)
  - collection_name: string (name of the Qdrant collection searched)
- **Validation**: Must contain at least one result or indicate why no results were found

## Relationships

- A Query is associated with multiple ContentChunk objects through similarity search
- Each ContentChunk has associated Metadata
- SearchResult aggregates Query and multiple ContentChunk objects