import React, { useState, useEffect } from 'react';
import ChatBotButton from './ChatBotButton';
import ChatPanel from './ChatPanel';
import { ChatSession } from '../../types/chat';

const ChatContainer: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  // Initialize session ID on component mount
  useEffect(() => {
    // Generate a unique session ID or retrieve from localStorage
    const storedSessionId = localStorage.getItem('chat-session-id');
    if (storedSessionId) {
      setSessionId(storedSessionId);
    } else {
      const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('chat-session-id', newSessionId);
      setSessionId(newSessionId);
    }
  }, []);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const closeChat = () => {
    setIsOpen(false);
  };

  return (
    <>
      <ChatBotButton onClick={toggleChat} isOpen={isOpen} />
      <ChatPanel isOpen={isOpen} onClose={closeChat} sessionId={sessionId} />
    </>
  );
};

export default ChatContainer;