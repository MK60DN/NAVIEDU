import React, { useState } from 'react';
import { MessageCircle, ArrowLeft } from 'lucide-react';
import InputBar from './InputBar';
import MessageList from './MessageList';
import { useNaviAPI } from '../hooks/useNaviAPI';

const ChatMode = ({
  currentQuestion,
  onQuestionChange,
  onSwitchToLearning,
  sessions,
  onUpdateSession,
  chatMessages,
  setChatMessages
}) => {
  const [inputText, setInputText] = useState(currentQuestion || '');
  const { callChatAPI } = useNaviAPI();

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      id: Date.now(),
      content: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    console.log('添加聊天用户消息:', userMessage);
    setChatMessages(prev => [...prev, userMessage]);
    onQuestionChange(inputText);

    try {
      const response = await callChatAPI(inputText, {
        context: chatMessages.slice(-5)
      });

      console.log('收到聊天API响应:', response);

      const assistantMessage = {
        id: Date.now() + 1,
        content: response.content,
        sender: 'assistant',
        type: 'chat',
        timestamp: new Date()
      };

      setChatMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error('对话API调用失败:', error);

      const errorMessage = {
        id: Date.now() + 1,
        content: `抱歉，调用失败: ${error.message}`,
        sender: 'assistant',
        type: 'error',
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
    }

    setInputText('');
  };

  return (
    <div className="chat-mode">
      {/* 模式切换提示 */}
      <div className="mode-switcher">
        <div className="current-question">
          {currentQuestion && (
            <div className="question-display">
              <span>当前问题: {currentQuestion}</span>
              <button
                className="switch-to-learning"
                onClick={onSwitchToLearning}
              >
                <ArrowLeft size={16} />
                切换到学习模式
              </button>
            </div>
          )}
        </div>
      </div>

      {/* 单栏对话 */}
      <div className="chat-panel">
        <div className="panel-header">
          <MessageCircle className="panel-icon" size={20} />
          <h3>自由对话</h3>
          <span className="message-count">({chatMessages.length})</span>
        </div>

        <MessageList
          messages={chatMessages}
          type="chat"
        />

        <InputBar
          value={inputText}
          onChange={setInputText}
          onSend={sendMessage}
          placeholder="随便聊聊..."
          showVoice={true}
          showCamera={true}
          showFile={true}
        />
      </div>
    </div>
  );
};

export default ChatMode;