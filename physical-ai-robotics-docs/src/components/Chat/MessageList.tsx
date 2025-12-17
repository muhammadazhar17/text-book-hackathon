import React from 'react';
import { ChatMessage } from '../../types/chat';
import './MessageList.css';

interface MessageListProps {
  messages: ChatMessage[];
}

const MessageList: React.FC<MessageListProps> = ({ messages }) => {
  // Limit the number of messages displayed for performance (keep last 50)
  const displayMessages = messages.length > 50 ? messages.slice(-50) : messages;

  return (
    <div className="message-list">
      {displayMessages.map((message) => (
        <div
          key={message.id}
          className={`message ${message.sender === 'user' ? 'user-message' : 'agent-message'}`}
        >
          <div className="message-content">
            {message.content}
          </div>
          {message.sources && message.sources.length > 0 && (
            <div className="message-sources">
              <div className="sources-label">Sources:</div>
              <ul className="sources-list">
                {message.sources.map((source, index) => (
                  <li key={index} className="source-item">{source}</li>
                ))}
              </ul>
            </div>
          )}
          <div className="message-timestamp">
            {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      ))}
    </div>
  );
};

export default MessageList;