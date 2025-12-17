import React, { useState, KeyboardEvent } from 'react';
import './MessageInput.css';

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  error: string | null;
}

const MessageInput: React.FC<MessageInputProps> = ({ onSendMessage, isLoading, error }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = () => {
    if (inputValue.trim() && !isLoading) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="message-input-container">
      {error && <div className="error-message">{error}</div>}
      <div className="message-input-wrapper">
        <textarea
          className="message-input"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message here..."
          disabled={isLoading}
          rows={1}
        />
        <button
          className={`send-button ${isLoading ? 'loading' : ''}`}
          onClick={handleSubmit}
          disabled={isLoading || !inputValue.trim()}
          aria-label="Send message"
        >
          {isLoading ? (
            <span className="loading-spinner">●●●</span>
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          )}
        </button>
      </div>
    </div>
  );
};

export default MessageInput;