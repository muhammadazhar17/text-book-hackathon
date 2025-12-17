# Quickstart: RAG Retrieval & Pipeline Validation

## Setup

1. **Install Dependencies**
   ```bash
   pip install qdrant-client cohere python-dotenv
   ```

2. **Environment Variables**
   Create a `.env` file with the following variables:
   ```env
   QDRANT_URL=your_qdrant_cloud_url
   QDRANT_API_KEY=your_qdrant_api_key
   COHERE_API_KEY=your_cohere_api_key
   ```

3. **Run Validation**
   ```bash
   cd backend
   python main.py
   ```

## Usage

The validation script will:
1. Initialize connections to Qdrant Cloud and Cohere
2. Generate embeddings for test queries
3. Perform similarity searches against the `as_embeddingone` collection
4. Validate retrieved results for accuracy and metadata integrity
5. Output validation results and metrics

## Configuration

- Adjust the top-k value in the code to control how many results to retrieve
- Modify test queries to validate different scenarios
- Configure timeout values for API calls as needed

## Expected Output

The script will output:
- Retrieval accuracy metrics
- Metadata integrity validation results
- Performance timing information
- Error rates and edge case handling results