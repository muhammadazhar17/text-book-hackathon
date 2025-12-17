# Implementation Plan: URL Ingestion & Vector Storage

## Technical Context

**Feature**: URL Ingestion & Vector Storage
**Branch**: 001-url-ingestion-vector-storage
**Created**: 2025-12-14

### System Architecture
- **Backend**: Single Python file in  `/backend/main.py`, inside backend foalder, the "uv package" 
- **Vector Database**: Qdrant Cloud
- **Embedding Service**: Cohere API

### Technology Stack
- **Language**: Python
- **Web Scraping**: requests, BeautifulSoup
- **XML Parsing**: xml.etree.ElementTree
- **Text Processing**: BeautifulSoup for HTML parsing and custom logic for text cleaning
- **Embeddings**: Cohere Python SDK
- **Vector DB**: Qdrant Python client
- **Configuration**: python-dotenv for environment management

### External Dependencies
- **Cohere API**: For generating embeddings
- **Qdrant Cloud**: Vector database service
- **Sitemap URL**: https://pre-hackathon-text-book-as.vercel.app/sitemap.xml

### Infrastructure Requirements
- **Environment Variables**:
  - COHERE_API_KEY
  - QDRANT_URL
  - QDRANT_API_KEY
- **Qdrant Collection**: Named "as_embeddingone"

## Constitution Check

### Compliance Analysis
- **Academic Accuracy and Integrity**: The system will store content accurately and maintain proper metadata linking to original sources
- **Interdisciplinary Collaboration**: The system bridges web technologies, AI/ML, and database systems
- **Ethical AI Principles**: Proper handling of web content with respect to usage rights and privacy
- **Robustness and Safety**: Error handling for network requests and API calls
- **Human-Centered Design**: Simple interface for triggering the ingestion process
- **Technical Excellence**: Following best practices for API integration and data processing

### Gate Evaluation
- **Pass**: The feature aligns with core principles of the project
- **Pass**: Technical approach follows established standards
- **Pass**: Implementation plan includes safety and error handling considerations

## Phase 0: Research & Discovery

### Research Tasks
1. Determine optimal text chunking strategy for semantic consistency
2. Identify appropriate Cohere embedding model for the use case
3. Define Qdrant collection schema and configuration
4. Research best practices for web scraping with respect to robots.txt and rate limiting

### Dependencies Resolved
- Use Cohere's `embed-english-v3.0` model for embeddings
- Use 512-token chunks with 50-token overlap for text processing
- Collection name: "as_embeddingone" with 1024 dimensions (matching Cohere embeddings)
- Command-line interface for triggering ingestion

## Phase 1: Design & Architecture

### Data Model
- **ContentChunk**:
  - id: unique identifier
  - content: text content of the chunk
  - url: source URL
  - chunk_index: position in original document
  - embedding: vector representation
  - metadata: additional information
  - created_at: timestamp

### Functions to Implement in main.py
1. **get_all_urls**: Fetch and parse sitemap.xml to extract all URLs
2. **extract_text_from_urls**: Fetch content from URLs and clean HTML to extract text
3. **chunk_text**: Split text into semantically meaningful chunks
4. **embed**: Generate embeddings using Cohere API
5. **create_collection**: Create Qdrant collection named "as_embeddingone"
6. **save_chunk_to_qdrant**: Store embeddings with metadata in Qdrant
7. **ingest_book**: Main function that orchestrates the entire process

### Implementation Approach
1. Setup Cohere and Qdrant clients with API keys
2. Fetch and clean data from fetched URLs
3. Generate embeddings using Cohere
4. Upsert embeddings into Qdrant with metadata
5. Implement the specific functions as requested in a single main.py file
6. Deploy URL: https://pre-hackathon-text-book-as.vercel.app/

## Phase 2: Implementation Plan

### Development Tasks
1. Set up project structure and dependencies in `/backend/main.py`
2. Implement Cohere and Qdrant client setup
3. Create `get_all_urls` function to parse sitemap
4. Create `extract_text_from_urls` function to fetch and clean content
5. Create `chunk_text` function to split content into chunks
6. Create `embed` function to generate embeddings
7. Create `create_collection` function to set up Qdrant collection
8. Create `save_chunk_to_qdrant` function to store embeddings
9. Create main `ingest_book` function to orchestrate the process
10. Add error handling and logging

### Testing Strategy
- Unit tests for individual functions
- Integration test for the full pipeline
- Validation tests to ensure content accuracy

## Risk Analysis

### Technical Risks
- API rate limits from Cohere or web sources
- Large documents exceeding embedding model limits
- Network reliability issues during content fetching

### Mitigation Strategies
- Implement exponential backoff for API calls
- Chunk large documents appropriately
- Add retry logic with circuit breakers