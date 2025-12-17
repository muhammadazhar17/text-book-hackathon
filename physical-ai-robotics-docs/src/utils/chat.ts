// Utility functions for message formatting and validation
import { ChatMessage } from '../types/chat';

// Generate a unique ID for messages
export function generateMessageId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Validate message content
export function validateMessageContent(content: string): boolean {
  if (!content || content.trim().length === 0) {
    return false; // Message cannot be empty
  }

  if (content.length > 2000) {
    return false; // Message exceeds 2000 characters
  }

  return true;
}

// Format message for display
export function formatMessageForDisplay(message: ChatMessage): string {
  const timestamp = new Date(message.timestamp).toLocaleTimeString();
  return `[${timestamp}] ${message.sender}: ${message.content}`;
}

// Create a new message object
export function createNewMessage(
  content: string,
  sender: 'user' | 'agent'
): ChatMessage {
  return {
    id: generateMessageId(),
    sender,
    content,
    timestamp: new Date().toISOString(),
    status: sender === 'user' ? 'sent' : undefined
  };
}

// Validate that a ChatMessage object has required properties
export function isValidChatMessage(message: any): message is ChatMessage {
  return (
    typeof message.id === 'string' &&
    (message.sender === 'user' || message.sender === 'agent') &&
    typeof message.content === 'string' &&
    typeof message.timestamp === 'string' &&
    new Date(message.timestamp).toString() !== 'Invalid Date'
  );
}