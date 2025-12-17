# Quickstart Guide: URL Ingestion & Vector Storage

## Prerequisites

- Python 3.8+
- pip package manager
- Access to Cohere API (API key)
- Access to Qdrant Cloud (URL and API key)

## Setup

1. **Install dependencies**:
   ```bash
   pip install requests beautifulsoup4 cohere qdrant-client python-dotenv lxml
   ```

2. **Set up environment variables** by creating a `.env` file:
   ```
   COHERE_API_KEY=your_cohere_api_key_here
   QDRANT_URL=your_qdrant_cloud_url_here
   QDRANT_API_KEY=your_qdrant_api_key_here
   ```

3. **Create the backend directory**:
   ```bash
   mkdir -p backend
   ```

## Usage

1. **Run the ingestion process**:
   ```bash
   cd backend
   python main.py
   ```

2. **The main.py file will**:
   - Connect to Cohere and Qdrant services
   - Fetch URLs from the sitemap: https://pre-hackathon-text-book-as.vercel.app/sitemap.xml
   - Extract and clean text content from each URL
   - Chunk the text into semantically meaningful pieces
   - Generate embeddings using Cohere
   - Store the embeddings in the "as_embeddingone" Qdrant collection with metadata

## Expected Output

- All content from the sitemap URLs will be processed and stored as embeddings
- Each chunk will be stored with its source URL and other metadata
- The Qdrant collection "as_embeddingone" will contain searchable vector representations

## Troubleshooting

- If you get API key errors, verify your environment variables are set correctly
- If URLs fail to fetch, check network connectivity and the target site's robots.txt
- If embedding generation fails, ensure your Cohere API key has sufficient quota