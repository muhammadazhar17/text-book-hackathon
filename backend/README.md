# URL Ingestion & Vector Storage System

This system ingests website URLs from a sitemap, generates embeddings using Cohere, and stores them in Qdrant cloud for RAG applications.

## Setup

1. Install dependencies:
   ```bash
   cd backend
   pip install requests beautifulsoup4 cohere qdrant-client python-dotenv lxml
   ```

2. Create a `.env` file in the backend directory with your API keys:
   ```env
   COHERE_API_KEY=your_cohere_api_key_here
   QDRANT_URL=your_qdrant_url_here
   QDRANT_API_KEY=your_qdrant_api_key_here
   ```

## Usage

### Ingest content from sitemap:
```bash
python main.py --mode ingest --sitemap-url https://pre-hackathon-text-book-as.vercel.app/sitemap.xml
```

### Search in the ingested content:
```bash
python main.py --mode search --query "your search query here"
```

### Additional options:
- `--collection-name`: Name of the Qdrant collection (default: as_embeddingone)
- `--chunk-size`: Size of text chunks in tokens (default: 512)
- `--overlap`: Overlap between chunks in tokens (default: 50)
- `--batch-size`: Number of URLs to process in each batch (default: 5)

## Features

- **URL Ingestion**: Fetches and parses sitemap.xml to extract all URLs
- **Content Extraction**: Downloads and extracts text content from each URL
- **Text Chunking**: Splits content into 512-token chunks with 50-token overlap
- **Embedding Generation**: Creates vector embeddings using Cohere's embed-english-v3.0 model
- **Vector Storage**: Stores embeddings in Qdrant cloud with metadata
- **Semantic Search**: Performs semantic search on ingested content
- **Progress Monitoring**: Tracks ingestion progress and provides summary statistics
- **Error Handling**: Comprehensive error handling with retry mechanisms