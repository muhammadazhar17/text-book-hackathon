import React from 'react';
import './ChatBotButton.css';

interface ChatBotButtonProps {
  onClick: () => void;
  isOpen: boolean;
}

const ChatBotButton: React.FC<ChatBotButtonProps> = ({ onClick, isOpen }) => {
  return (
    <button
      className={`chatbot-button ${isOpen ? 'hidden' : ''}`}
      onClick={onClick}
      aria-label={isOpen ? 'Close chat' : 'Open chat'}
      title={isOpen ? 'Close chat' : 'Open chat'}
      aria-expanded={!isOpen}
      aria-controls="chat-panel"
    >
      <div className="chatbot-icon" aria-hidden="true">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
      </div>
    </button>
  );
};

export default ChatBotButton;