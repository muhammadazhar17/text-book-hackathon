# Quickstart: Chatbot Integration

## Overview
This guide provides instructions for setting up and running the chatbot integration in the Docusaurus documentation site.

## Prerequisites
- Node.js 18+ installed
- Yarn or npm package manager
- Access to the backend chatbot API
- Git for version control

## Getting Started

### 1. Clone the Repository
```bash
git clone [repository-url]
cd [repository-name]
```

### 2. Install Dependencies
```bash
cd physical-ai-robotics-docs  # Navigate to the Docusaurus directory
yarn install
# or
npm install
```

### 3. Configure Environment Variables
Create a `.env` file in the `physical-ai-robotics-docs` directory:

```env
# Backend API URL
REACT_APP_CHATBOT_API_URL=http://localhost:8000/api/chat
# For production, use:
# REACT_APP_CHATBOT_API_URL=https://your-backend-domain.com/api/chat

# Optional: API key if required by backend
REACT_APP_CHATBOT_API_KEY=your-api-key-if-required
```

### 4. Run the Development Server
```bash
yarn start
# or
npm run start
```

The Docusaurus site will start at `http://localhost:3000`, with the chatbot widget available on all pages.

## Integration Points

### Chat Widget Components
The chatbot integration includes:

1. **Floating Button**: Always visible in the bottom-left corner
2. **Chat Panel**: Expands when the button is clicked
3. **Message History**: Shows conversation between user and bot
4. **Input Area**: Where users can type their messages

### Docusaurus Plugin
The chat widget is integrated via a Docusaurus plugin that automatically adds the functionality to all pages.

## API Communication
The frontend communicates with the backend via the configured API endpoint. Ensure your backend service is running and accessible from the frontend.

For development, if your backend runs on a different port, you may encounter CORS issues. Configure your backend to allow requests from `http://localhost:3000`.

## Building for Production
```bash
yarn build
# or
npm run build
```

The built site can be served from the `build/` directory and will include the chatbot functionality.

## Troubleshooting

### Chat Widget Not Appearing
- Ensure you've properly installed all dependencies
- Check that the plugin is correctly registered in `docusaurus.config.js`

### API Connection Issues
- Verify the `REACT_APP_CHATBOT_API_URL` is set correctly
- Check that the backend service is running
- Verify CORS configuration on the backend

### Build Issues
- Ensure all required environment variables are available during build
- Check that all dependencies are properly configured in `package.json`