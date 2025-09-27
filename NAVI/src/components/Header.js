import React, { useState } from 'react';
import { Brain, MessageCircle, GitBranch, Menu, Settings, Trash2, MoreVertical } from 'lucide-react';

const Header = ({
  currentMode,
  onModeChange,
  currentQuestion,
  onClearConversations,
  onClearKnowledgeGraph
}) => {
  const [showMenu, setShowMenu] = useState(false);

  const modes = [
    { id: 'learning', name: '学习模式', icon: Brain },
    { id: 'chat', name: '对话模式', icon: MessageCircle },
    { id: 'knowledge', name: '知识图谱', icon: GitBranch }
  ];

  const toggleMenu = () => {
    setShowMenu(!showMenu);
  };

  const handleClearConversations = () => {
    onClearConversations();
    setShowMenu(false);
  };

  const handleClearKnowledgeGraph = () => {
    onClearKnowledgeGraph();
    setShowMenu(false);
  };

  return (
    <header className="app-header">
      <div className="header-top">
        <div className="header-left">
          <Menu className="menu-icon" size={24} />
          <h1 className="app-title">Navi</h1>
        </div>

        <div className="header-right">
          <div className="menu-container">
            <button
              className="menu-button"
              onClick={toggleMenu}
            >
              <MoreVertical size={20} />
            </button>

            {showMenu && (
              <div className="dropdown-menu">
                <button
                  className="menu-item"
                  onClick={handleClearConversations}
                >
                  <Trash2 size={16} />
                  清空对话
                </button>
                <button
                  className="menu-item"
                  onClick={handleClearKnowledgeGraph}
                >
                  <Trash2 size={16} />
                  重置知识图谱
                </button>
                <div className="menu-divider"></div>
                <button className="menu-item">
                  <Settings size={16} />
                  设置
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      <nav className="mode-nav">
        {modes.map(mode => (
          <button
            key={mode.id}
            className={`mode-button ${currentMode === mode.id ? 'active' : ''}`}
            onClick={() => onModeChange(mode.id)}
          >
            <mode.icon size={18} />
            <span>{mode.name}</span>
          </button>
        ))}
      </nav>

      {currentQuestion && (
        <div className="current-question">
          <span className="question-label">当前问题:</span>
          <span className="question-text">{currentQuestion}</span>
        </div>
      )}
    </header>
  );
};

export default Header;