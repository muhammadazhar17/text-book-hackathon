# Quickstart Guide: Gemini RAG Agent

## Prerequisites

- Python 3.11+
- Access to Google Gemini API
- Access to Cohere API
- Access to Qdrant Cloud with existing `as_embeddingone` collection

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn google-generativeai cohere qdrant-client python-dotenv
   ```

4. **Set up environment variables**
   Create a `.env` file in the backend directory with the following:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   COHERE_API_KEY=your_cohere_api_key
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_HOST=your_qdrant_cloud_host
   QDRANT_COLLECTION_NAME=as_embeddingone
   ```

## Running the Service

1. **Start the server**
   ```bash
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **Access the API**
   - API documentation: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`
   - Chat endpoint: `http://localhost:8000/chat` (POST)

## Example Usage

### Using curl
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key principles of physical AI?"
  }'
```

### Python client
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"query": "What are the key principles of physical AI?"}
)
print(response.json())
```

## Configuration

- **Environment Variables**:
  - `GEMINI_API_KEY`: Google Gemini API key
  - `COHERE_API_KEY`: Cohere API key
  - `QDRANT_API_KEY`: Qdrant Cloud API key
  - `QDRANT_HOST`: Qdrant Cloud host URL
  - `QDRANT_COLLECTION_NAME`: Name of the collection to query (default: as_embeddingone)
  - `TOP_K`: Number of chunks to retrieve (default: 3)
  - `MAX_TOKENS`: Maximum tokens for Gemini response (default: 1000)
  - `TEMPERATURE`: Gemini temperature setting (default: 0.7)

## Architecture

The service follows a RAG (Retrieval-Augmented Generation) pattern:

1. **Query Processing**: Accept and validate user query
2. **Embedding Generation**: Create embedding for query using Cohere
3. **Context Retrieval**: Search Qdrant Cloud for relevant document chunks
4. **Response Generation**: Use Gemini to generate response based on query and retrieved context
5. **Response Formatting**: Return structured response with sources