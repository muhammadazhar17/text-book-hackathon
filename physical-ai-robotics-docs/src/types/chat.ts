// ChatMessage interface definition
export interface ChatMessage {
  id: string; // unique identifier for the message
  sender: 'user' | 'agent'; // indicates who sent the message
  content: string; // the actual message content
  timestamp: string; // when the message was created in ISO 8601 format
  status?: 'sent' | 'delivered' | 'error'; // delivery status for user messages
  sources?: string[]; // source documents referenced in the response (for agent messages)
}

// ChatSession interface definition
export interface ChatSession {
  sessionId: string; // unique identifier for the session
  messages: ChatMessage[]; // ordered list of messages in the conversation
  createdAt: string; // when the session was started in ISO 8601 format
  lastActive: string; // when the last message was sent/received in ISO 8601 format
}