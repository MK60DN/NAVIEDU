import React from 'react';
import { Brain, HelpCircle, MessageCircle } from 'lucide-react';

const MessageList = ({ messages, type }) => {
  const getTypeIcon = (msgType) => {
    switch (msgType) {
      case 'learning': return <Brain size={16} />;
      case 'questioning': return <HelpCircle size={16} />;
      case 'chat': return <MessageCircle size={16} />;
      default: return null;
    }
  };

  const getTypeColor = (msgType) => {
    switch (msgType) {
      case 'learning': return 'text-blue-400';
      case 'questioning': return 'text-red-400';
      case 'chat': return 'text-green-400';
      default: return 'text-yellow-400';
    }
  };

  return (
    <div className="message-list">
      {messages.map(message => (
        <div
          key={message.id}
          className={`message ${message.sender === 'user' ? 'user-message' : 'assistant-message'}`}
        >
          {message.sender === 'assistant' && (
            <div className={`message-type ${getTypeColor(message.type)}`}>
              {getTypeIcon(message.type)}
              <span className="type-label">
                {message.type === 'learning' && '学习辅导'}
                {message.type === 'questioning' && '批判思考'}
                {message.type === 'chat' && 'Navi'}
              </span>
            </div>
          )}

          <div className="message-content">
            {message.content}
          </div>

          <div className="message-time">
            {new Date(message.timestamp).toLocaleTimeString()}
          </div>
        </div>
      ))}
    </div>
  );
};

export default MessageList;