import React, { useState, useEffect } from 'react';
import MessageInput from './MessageInput';
import MessageList from './MessageList';
import { ChatMessage } from '../../types/chat';
import { createNewMessage, validateMessageContent } from '../../utils/chat';
import ApiClient from '../../services/api-client';
import './ChatPanel.css';

interface ChatPanelProps {
  isOpen: boolean;
  onClose: () => void;
  sessionId?: string;
}

const ChatPanel: React.FC<ChatPanelProps> = ({ isOpen, onClose, sessionId }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Add welcome message when panel opens
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      const welcomeMessage = createNewMessage(
        "Hello! I'm your AI assistant for Physical AI and Humanoid Robotics. How can I help you today?",
        'agent'
      );
      setMessages([welcomeMessage]);
    }
  }, [isOpen, messages.length]);

  // Focus the input field when the panel opens
  useEffect(() => {
    if (isOpen) {
      const inputField = document.querySelector('.message-input') as HTMLTextAreaElement;
      if (inputField) {
        setTimeout(() => {
          inputField.focus();
        }, 300); // Small delay to allow panel to render
      }
    }
  }, [isOpen]);

  const handleSendMessage = async (content: string) => {
    if (!validateMessageContent(content)) {
      setError('Message cannot be empty and must be less than 2000 characters');
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      // Add user message to the chat
      const userMessage = createNewMessage(content, 'user');
      setMessages(prev => [...prev, userMessage]);

      // Send to backend with session ID
      const response = await ApiClient.sendChatMessage(content, sessionId);

      // Check if the response contains an error
      if (response.error) {
        // Add error response to the chat
        const errorMessage = createNewMessage(response.error, 'agent');
        setMessages(prev => [...prev, errorMessage]);
      } else {
        // Add agent response to the chat
        const agentMessage = createNewMessage(response.response, 'agent');
        // Add sources if they exist in the response
        if (response.sources && response.sources.length > 0) {
          agentMessage.sources = response.sources;
        }
        setMessages(prev => [...prev, agentMessage]);
      }
    } catch (err: any) {
      console.error('Error sending message:', err);
      setError(err.message || 'Failed to get response from the AI assistant. Please try again.');

      // Add error message to chat
      const errorMessage = createNewMessage(
        err.message || 'Sorry, I encountered an issue processing your request. Please try again.',
        'agent'
      );
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div
      className="chat-panel"
      id="chat-panel"
      role="dialog"
      aria-modal="true"
      aria-label="AI Assistant Chat"
    >
      <div className="chat-header">
        <h3>AI Assistant</h3>
        <button
          className="close-button"
          onClick={onClose}
          aria-label="Close chat"
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              onClose();
            }
          }}
        >
          ×
        </button>
      </div>
      <div className="chat-content" role="region" aria-label="Chat messages">
        <MessageList messages={messages} />
        <MessageInput
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          error={error}
        />
      </div>
    </div>
  );
};

export default ChatPanel;