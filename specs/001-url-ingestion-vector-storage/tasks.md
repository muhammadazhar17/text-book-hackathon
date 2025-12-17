# Implementation Tasks: URL Ingestion & Vector Storage

## Feature Overview

**Feature**: URL Ingestion & Vector Storage
**Branch**: 001-url-ingestion-vector-storage
**Created**: 2025-12-14
**Goal**: Ingest website URLs from sitemap, generate embeddings with Cohere, and store in Qdrant cloud for RAG applications

## Implementation Strategy

- **MVP First**: Implement User Story 1 (core ingestion) first as a complete, testable increment
- **Incremental Delivery**: Each user story builds on the previous with additional capabilities
- **Single File**: All code in `/backend/main.py` as specified
- **Parallel Opportunities**: Dependencies and utility functions can be developed in parallel

## Dependencies

- **User Story 1 (P1)**: Core ingestion - Foundation for all other stories
- **User Story 2 (P2)**: Vector search - Depends on User Story 1 (needs ingested data)
- **User Story 3 (P3)**: Monitoring - Can be implemented in parallel with User Story 1

## Parallel Execution Examples

- **Setup Tasks**: Can run in parallel with foundational tasks
- **Utility Functions**: chunk_text, extract_text_from_urls can be developed in parallel
- **Error Handling**: Can be added to each function as it's developed

---

## Phase 1: Setup

Initialize project structure and install dependencies.

- [X] T001 Create backend directory structure: `/backend/` and initialized **uv package** load .venv\scripts\activate than install all things under the virtual environment.
- [X] T002 Install required Python packages: requests, beautifulsoup4, cohere, qdrant-client, python-dotenv, lxml
- [X] T003 Create initial `/backend/main.py` file with imports and configuration under the uv initialized package
- [X] T004 Set up environment variable loading for COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY

---

## Phase 2: Foundational

Core infrastructure and utility functions needed by all user stories.

- [X] T005 [P] Initialize Cohere client with API key from environment
- [X] T006 [P] Initialize Qdrant client with URL and API key from environment
- [X] T007 [P] Create helper function to validate URL format
- [X] T008 [P] Create helper function for exponential backoff retry mechanism
- [X] T009 [P] Create logging configuration for the application

---

## Phase 3: User Story 1 - Ingest Website Content for RAG (Priority: P1)

As a user, I want to provide a website URL so that its content can be ingested, embedded, and stored in a vector database for use in a Retrieval-Augmented Generation (RAG) system.

**Independent Test**: Can be fully tested by providing a website URL and verifying that its content is properly ingested, chunked, embedded, and stored in the vector database with appropriate metadata.

- [X] T010 [US1] Implement `get_all_urls` function to fetch and parse sitemap.xml from https://pre-hackathon-text-book-as.vercel.app/sitemap.xml
- [X] T011 [US1] Implement `extract_text_from_urls` function to fetch content from URLs and clean HTML to extract text using BeautifulSoup
- [X] T012 [US1] Implement `chunk_text` function to split text into 512-token chunks with 50-token overlap
- [X] T013 [US1] Implement `embed` function to generate embeddings using Cohere's embed-english-v3.0 model
- [X] T014 [US1] Implement `create_collection` function to create Qdrant collection named "as_embeddingone" with 1024 dimensions
- [X] T015 [US1] Implement `save_chunk_to_qdrant` function to store embeddings with metadata in Qdrant
- [X] T016 [US1] Implement main `ingest_book` function to orchestrate the entire process
- [X] T017 [US1] Add error handling to each function with retry mechanism for network/API calls
- [X] T018 [US1] Test User Story 1: Run full ingestion pipeline and verify content is stored in Qdrant

---

## Phase 4: User Story 2 - Access Ingested Content via Vector Search (Priority: P2)

As a user, I want to be able to search through the ingested content using semantic queries so that I can retrieve relevant information for the RAG system.

**Independent Test**: Can be tested by performing vector searches against the stored embeddings and verifying that relevant content is returned.

- [X] T019 [US2] Implement search function to query Qdrant collection "as_embeddingone" with semantic search
- [X] T020 [US2] Implement function to convert search query to embedding using Cohere
- [X] T021 [US2] Implement function to format search results with metadata
- [X] T022 [US2] Add search functionality to main.py with command-line interface
- [X] T023 [US2] Test User Story 2: Perform search queries and verify relevant content chunks are returned

---

## Phase 5: User Story 3 - Monitor Ingestion Process (Priority: P3)

As an administrator, I want to monitor the ingestion process so that I can track progress and identify any issues.

**Independent Test**: Can be tested by observing the ingestion process and verifying that progress metrics and error handling work correctly.

- [X] T024 [US3] Implement progress tracking for URL processing in `ingest_book` function
- [X] T025 [US3] Add progress logging to show total URLs, processed URLs, and current status
- [X] T026 [US3] Implement error tracking and reporting during ingestion process
- [X] T027 [US3] Add summary statistics at the end of ingestion (total chunks created, success rate, etc.)
- [X] T028 [US3] Test User Story 3: Run ingestion and verify progress monitoring works correctly

---

## Phase 6: Polish & Cross-Cutting Concerns

Final touches and quality improvements.

- [X] T029 Add comprehensive error handling throughout the application
- [X] T030 Add input validation for URLs and configuration parameters
- [X] T031 Optimize performance for large sitemaps (batch processing, etc.)
- [X] T032 Add configuration options for chunk size and other parameters
- [X] T033 Update documentation in main.py with usage instructions
- [X] T034 Run complete end-to-end test of the ingestion pipeline
- [X] T035 Verify 95% of URLs from sitemap are successfully ingested (SC-001)
- [X] T036 Verify content ingestion completes within 30 minutes for websites up to 100 pages (SC-002)
- [X] T037 Verify embeddings are stored with 99% success rate in Qdrant cloud (SC-003)