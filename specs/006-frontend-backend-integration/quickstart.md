# Quickstart: Frontend ↔ Backend Integration

## Prerequisites

- Node.js (v16 or higher) for the frontend
- Python (v3.11) for the backend
- npm or yarn package managers
- Git for version control

## Setup Instructions

### 1. Start the Backend Server

First, navigate to the backend directory and start the FastAPI server:

```bash
cd backend
pip install -r requirements.txt  # Install dependencies
python -m src.main  # Start the backend server
```

The backend server should now be running on `http://localhost:8000`.

### 2. Start the Frontend Development Server

In a separate terminal, navigate to the documentation site directory:

```bash
cd physical-ai-robotics-docs
npm install  # Install dependencies
npm start  # Start the Docusaurus development server
```

The frontend should now be running on `http://localhost:3000`.

### 3. Verify the Integration

1. Open your browser to `http://localhost:3000`
2. You should see a chatbot button in the bottom-right corner of the page
3. Click the button to open the chat panel
4. Type a query and submit it
5. The query should be sent to the backend at `http://localhost:8000/chat`
6. The response should appear in the chat panel

## Configuration

### CORS Settings

The backend is configured to allow requests from `http://localhost:3000` by default. If you need to run the frontend on a different port, update the CORS settings in the backend configuration.

### Environment Variables

The frontend may require the following environment variables:

```env
REACT_APP_BACKEND_URL=http://localhost:8000  # Backend API URL
```

## Troubleshooting

### Common Issues

1. **CORS Error**: Make sure the backend server is running and CORS is configured correctly
2. **Connection Refused**: Verify both servers are running on the correct ports
3. **API Endpoint Not Found**: Ensure the backend has the `/chat` endpoint implemented

### Testing the API Directly

You can test the backend API directly using curl:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, RAG agent!"}'
```

## Development Workflow

1. Make changes to the frontend components in `physical-ai-robotics-docs/src/components/`
2. Make changes to the backend API in `backend/src/api/`
3. Test the integration by sending queries through the chat interface
4. Verify that responses are displayed correctly in the chat panel