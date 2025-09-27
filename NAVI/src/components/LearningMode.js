import React, { useState } from 'react';
import { Brain, HelpCircle, ArrowRight } from 'lucide-react';
import InputBar from './InputBar';
import MessageList from './MessageList';
import { useNaviAPI } from '../hooks/useNaviAPI';

const LearningMode = ({
  currentQuestion,
  onQuestionChange,
  onSwitchToChat,
  sessions,
  onUpdateSession,
  knowledgeGraph,
  onUpdateKnowledge,
  learningMessages,
  setLearningMessages,
  questioningMessages,
  setQuestioningMessages
}) => {
  const [inputText, setInputText] = useState(currentQuestion || '');
  const { callLearningAPI, callQuestioningAPI } = useNaviAPI();

  const sendToLearning = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      id: Date.now(),
      content: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    console.log('添加学习用户消息:', userMessage);
    setLearningMessages(prev => [...prev, userMessage]);
    onQuestionChange(inputText);

    try {
      const response = await callLearningAPI(inputText, {
        context: learningMessages.slice(-3),
        knowledgeGraph: knowledgeGraph
      });

      console.log('收到学习API响应:', response);

      const assistantMessage = {
        id: Date.now() + 1,
        content: response.content,
        sender: 'assistant',
        type: 'learning',
        timestamp: new Date()
      };

      setLearningMessages(prev => [...prev, assistantMessage]);

      // 更新知识图谱
      updateKnowledgeGraphFromLearning(inputText, response.content);

    } catch (error) {
      console.error('学习API调用失败:', error);

      const errorMessage = {
        id: Date.now() + 1,
        content: `抱歉，调用失败: ${error.message}`,
        sender: 'assistant',
        type: 'error',
        timestamp: new Date()
      };
      setLearningMessages(prev => [...prev, errorMessage]);
    }

    setInputText('');
  };

  const sendToQuestioning = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      id: Date.now(),
      content: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    console.log('添加质疑用户消息:', userMessage);
    setQuestioningMessages(prev => [...prev, userMessage]);
    onQuestionChange(inputText);

    try {
      const response = await callQuestioningAPI(inputText, {
        context: questioningMessages.slice(-3),
        learningContext: learningMessages.slice(-2)
      });

      console.log('收到质疑API响应:', response);

      const assistantMessage = {
        id: Date.now() + 1,
        content: response.content,
        sender: 'assistant',
        type: 'questioning',
        timestamp: new Date()
      };

      setQuestioningMessages(prev => [...prev, assistantMessage]);

      // 质疑内容也参与知识图谱构建
      updateKnowledgeGraphFromQuestioning(inputText, response.content);

    } catch (error) {
      console.error('质疑API调用失败:', error);

      const errorMessage = {
        id: Date.now() + 1,
        content: `抱歉，调用失败: ${error.message}`,
        sender: 'assistant',
        type: 'error',
        timestamp: new Date()
      };
      setQuestioningMessages(prev => [...prev, errorMessage]);
    }

    setInputText('');
  };

  const updateKnowledgeGraphFromLearning = (question, response) => {
    const newNode = {
      id: `learning_${Date.now()}`,
      title: question.slice(0, 30) + (question.length > 30 ? '...' : ''),
      content: `问题: ${question}\n学习指导: ${response.slice(0, 200)}${response.length > 200 ? '...' : ''}`,
      type: 'learning',
      timestamp: new Date().toISOString(),
      children: []
    };

    console.log('添加学习知识节点:', newNode);
    onUpdateKnowledge(prev => ({
      ...prev,
      children: [...prev.children, newNode]
    }));
  };

  const updateKnowledgeGraphFromQuestioning = (question, response) => {
    const newNode = {
      id: `questioning_${Date.now()}`,
      title: `质疑: ${question.slice(0, 25)}${question.length > 25 ? '...' : ''}`,
      content: `原问题: ${question}\n批判思考: ${response.slice(0, 200)}${response.length > 200 ? '...' : ''}`,
      type: 'questioning',
      timestamp: new Date().toISOString(),
      children: []
    };

    console.log('添加质疑知识节点:', newNode);
    onUpdateKnowledge(prev => ({
      ...prev,
      children: [...prev.children, newNode]
    }));
  };

  return (
    <div className="learning-mode">
      {/* 模式切换提示 */}
      <div className="mode-switcher">
        <div className="current-question">
          {currentQuestion && (
            <div className="question-display">
              <span>当前问题: {currentQuestion}</span>
              <button
                className="switch-to-chat"
                onClick={onSwitchToChat}
              >
                <ArrowRight size={16} />
                切换到对话模式
              </button>
            </div>
          )}
        </div>
      </div>

      {/* 双栏布局 */}
      <div className="dual-panel">
        {/* 学习面板 */}
        <div className="learning-panel">
          <div className="panel-header">
            <Brain className="panel-icon" size={20} />
            <h3>学习辅导</h3>
            <span className="message-count">({learningMessages.length})</span>
          </div>

          <MessageList
            messages={learningMessages}
            type="learning"
          />

          <InputBar
            value={inputText}
            onChange={setInputText}
            onSend={sendToLearning}
            placeholder="请输入您想学习的内容..."
            showVoice={true}
            showCamera={true}
            showFile={true}
          />
        </div>

        {/* 质疑面板 */}
        <div className="questioning-panel">
          <div className="panel-header">
            <HelpCircle className="panel-icon" size={20} />
            <h3>批判思考</h3>
            <span className="message-count">({questioningMessages.length})</span>
          </div>

          <MessageList
            messages={questioningMessages}
            type="questioning"
          />

          <InputBar
            value={inputText}
            onChange={setInputText}
            onSend={sendToQuestioning}
            placeholder="请输入您想质疑思考的内容..."
            showVoice={true}
            showCamera={true}
            showFile={true}
          />
        </div>
      </div>
    </div>
  );
};

export default LearningMode;