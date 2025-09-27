import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import LearningMode from './components/LearningMode';
import ChatMode from './components/ChatMode';
import KnowledgeGraph from './components/KnowledgeGraph';
import { useLocalStorage } from './hooks/useLocalStorage';
import './styles/index.css';

const App = () => {
  const [currentMode, setCurrentMode] = useState('learning');
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [sessions, setSessions] = useLocalStorage('navi_sessions', {});

  // 添加消息持久化
  const [learningMessages, setLearningMessages] = useLocalStorage('navi_learning_messages', []);
  const [questioningMessages, setQuestioningMessages] = useLocalStorage('navi_questioning_messages', []);
  const [chatMessages, setChatMessages] = useLocalStorage('navi_chat_messages', []);

  const [knowledgeGraph, setKnowledgeGraph] = useLocalStorage('navi_knowledge_graph', {
    id: 'root',
    title: '我的知识库',
    content: '个人知识图谱根节点',
    children: []
  });

  const switchMode = (mode, question = null) => {
    setCurrentMode(mode);
    if (question) setCurrentQuestion(question);
  };

  const updateSession = (sessionId, data) => {
    setSessions(prev => ({
      ...prev,
      [sessionId]: { ...prev[sessionId], ...data }
    }));
  };

  // 清空所有对话功能
  const clearAllConversations = () => {
    if (window.confirm('确定要清空所有对话记录吗？此操作不可撤销。')) {
      setLearningMessages([]);
      setQuestioningMessages([]);
      setChatMessages([]);
      setCurrentQuestion('');
      console.log('所有对话记录已清空');
    }
  };

  // 清空知识图谱功能
  const clearKnowledgeGraph = () => {
    if (window.confirm('确定要重置知识图谱吗？此操作不可撤销。')) {
      setKnowledgeGraph({
        id: 'root',
        title: '我的知识库',
        content: '个人知识图谱根节点',
        children: []
      });
      console.log('知识图谱已重置');
    }
  };

  return (
    <div className="navi-app">
      <Header
        currentMode={currentMode}
        onModeChange={switchMode}
        currentQuestion={currentQuestion}
        onClearConversations={clearAllConversations}
        onClearKnowledgeGraph={clearKnowledgeGraph}
      />

      <main className="app-main">
        {currentMode === 'learning' && (
          <LearningMode
            currentQuestion={currentQuestion}
            onQuestionChange={setCurrentQuestion}
            onSwitchToChat={() => switchMode('chat', currentQuestion)}
            sessions={sessions}
            onUpdateSession={updateSession}
            knowledgeGraph={knowledgeGraph}
            onUpdateKnowledge={setKnowledgeGraph}
            learningMessages={learningMessages}
            setLearningMessages={setLearningMessages}
            questioningMessages={questioningMessages}
            setQuestioningMessages={setQuestioningMessages}
          />
        )}

        {currentMode === 'chat' && (
          <ChatMode
            currentQuestion={currentQuestion}
            onQuestionChange={setCurrentQuestion}
            onSwitchToLearning={() => switchMode('learning', currentQuestion)}
            sessions={sessions}
            onUpdateSession={updateSession}
            chatMessages={chatMessages}
            setChatMessages={setChatMessages}
          />
        )}

        {currentMode === 'knowledge' && (
          <KnowledgeGraph
            knowledgeGraph={knowledgeGraph}
            onUpdateKnowledge={setKnowledgeGraph}
            onSwitchToLearning={(question) => switchMode('learning', question)}
          />
        )}
      </main>
    </div>
  );
};

export default App;