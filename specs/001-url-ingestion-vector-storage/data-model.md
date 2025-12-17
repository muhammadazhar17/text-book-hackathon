# Data Model: URL Ingestion & Vector Storage

## Entity: ContentChunk

### Attributes
- **id** (string): Unique identifier for the chunk (auto-generated)
- **content** (string): The actual text content of the chunk
- **url** (string): The source URL where the content was extracted from
- **chunk_index** (integer): The position of this chunk in the original document
- **embedding** (list[float]): 1024-dimensional vector representation of the content
- **metadata** (dict): Additional information including:
  - title: The page title
  - created_at: Timestamp when the chunk was processed
  - source_document: Name or identifier of the original document
- **created_at** (datetime): Timestamp when the chunk was created

### Relationships
- Each ContentChunk belongs to one source URL
- Multiple ContentChunks can originate from the same URL (if the document was chunked)

### Validation Rules
- content must not exceed 512 tokens when processed by the embedding model
- url must be a valid URL format
- embedding must be a 1024-element array of floats (for Cohere embeddings)
- created_at must be set when the chunk is processed

## Entity: EmbeddingCollection

### Attributes
- **name** (string): Name of the Qdrant collection (e.g., "embeddingone")
- **dimension** (integer): Dimension of the vectors (1024 for Cohere embeddings)
- **size** (integer): Number of vectors currently stored
- **created_at** (datetime): Timestamp when the collection was created

### Relationships
- Contains multiple ContentChunk entities as vector records

## Entity: ProcessingJob

### Attributes
- **id** (string): Unique identifier for the ingestion job
- **status** (string): Current status (e.g., "pending", "processing", "completed", "failed")
- **total_urls** (integer): Total number of URLs to process
- **processed_urls** (integer): Number of URLs processed so far
- **total_chunks** (integer): Total number of content chunks created
- **started_at** (datetime): When the job started
- **completed_at** (datetime): When the job completed (if finished)
- **error** (string): Error message if the job failed

### Relationships
- Associated with multiple ContentChunk entities created during the job