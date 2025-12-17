// API client service for backend communication
import { ChatMessage } from '../types/chat';

// Define the API response types
export interface ChatRequest {
  message: string;
  sessionId?: string;
}

export interface ChatResponse {
  response?: string;
  sources?: string[];
  error?: string;
}

class ApiClient {
  private baseUrl: string;

  constructor() {
    // Use environment variable or default to localhost:8000
    // Safely check for process.env to avoid ReferenceError in some environments
    const env = typeof process !== 'undefined' ? process.env : {};
    this.baseUrl = 'https://backend-bmai.onrender.com';
  }

  async sendChatMessage(message: string, sessionId?: string): Promise<ChatResponse> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const response = await fetch(`${this.baseUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message,
          user_id: sessionId
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        // Handle different HTTP error statuses
        if (response.status === 400) {
          const errorData = await response.json();
          return { error: errorData.detail || 'Invalid request parameters' };
        } else if (response.status === 500) {
          return { error: 'The server encountered an error processing your request. Please try again.' };
        } else {
          return { error: `Server error: ${response.status} - ${response.statusText}` };
        }
      }

      const data: ChatResponse = await response.json();
      return data;
    } catch (error: any) {
      // Handle network errors and timeouts
      if (error.name === 'AbortError') {
        return { error: 'Render server is curently shutdown,upgrade plan or wait for server starting.' };
      } else if (error.message && error.message.includes('fetch')) {
        return { error: 'Network error: Unable to connect to the server. Please check your connection.' };
      } else {
        console.error('Error sending chat message:', error);
        return { error: error.message || 'An error occurred while sending your message.' };
      }
    }
  }

  // Check if the backend is available
  async checkBackendStatus(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      return response.ok;
    } catch (error) {
      console.error('Error checking backend status:', error);
      return false;
    }
  }
}

export default new ApiClient();
