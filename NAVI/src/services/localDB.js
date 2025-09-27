class LocalDB {
  constructor() {
    this.dbName = 'NaviDB';
    this.version = 1;
  }

  // 存储会话数据
  saveSession(sessionId, sessionData) {
    try {
      const sessions = this.getSessions();
      sessions[sessionId] = {
        ...sessionData,
        updatedAt: new Date().toISOString()
      };
      localStorage.setItem('navi_sessions', JSON.stringify(sessions));
      return true;
    } catch (error) {
      console.error('保存会话失败:', error);
      return false;
    }
  }

  // 获取所有会话
  getSessions() {
    try {
      const sessions = localStorage.getItem('navi_sessions');
      return sessions ? JSON.parse(sessions) : {};
    } catch (error) {
      console.error('读取会话失败:', error);
      return {};
    }
  }

  // 获取特定会话
  getSession(sessionId) {
    const sessions = this.getSessions();
    return sessions[sessionId] || null;
  }

  // 删除会话
  deleteSession(sessionId) {
    try {
      const sessions = this.getSessions();
      delete sessions[sessionId];
      localStorage.setItem('navi_sessions', JSON.stringify(sessions));
      return true;
    } catch (error) {
      console.error('删除会话失败:', error);
      return false;
    }
  }

  // 存储知识图谱
  saveKnowledgeGraph(graphData) {
    try {
      localStorage.setItem('navi_knowledge_graph', JSON.stringify(graphData));
      return true;
    } catch (error) {
      console.error('保存知识图谱失败:', error);
      return false;
    }
  }

  // 获取知识图谱
  getKnowledgeGraph() {
    try {
      const graph = localStorage.getItem('navi_knowledge_graph');
      return graph ? JSON.parse(graph) : null;
    } catch (error) {
      console.error('读取知识图谱失败:', error);
      return null;
    }
  }

  // 存储用户设置
  saveSettings(settings) {
    try {
      localStorage.setItem('navi_settings', JSON.stringify(settings));
      return true;
    } catch (error) {
      console.error('保存设置失败:', error);
      return false;
    }
  }

  // 获取用户设置
  getSettings() {
    try {
      const settings = localStorage.getItem('navi_settings');
      return settings ? JSON.parse(settings) : {
        darkMode: true,
        voiceEnabled: true,
        apiKey: '',
        language: 'zh-CN'
      };
    } catch (error) {
      console.error('读取设置失败:', error);
      return {};
    }
  }

  // 清空所有数据
  clearAll() {
    try {
      localStorage.removeItem('navi_sessions');
      localStorage.removeItem('navi_knowledge_graph');
      localStorage.removeItem('navi_settings');
      return true;
    } catch (error) {
      console.error('清空数据失败:', error);
      return false;
    }
  }

  // 导出数据
  exportData() {
    return {
      sessions: this.getSessions(),
      knowledgeGraph: this.getKnowledgeGraph(),
      settings: this.getSettings(),
      exportTime: new Date().toISOString()
    };
  }

  // 导入数据
  importData(data) {
    try {
      if (data.sessions) {
        localStorage.setItem('navi_sessions', JSON.stringify(data.sessions));
      }
      if (data.knowledgeGraph) {
        localStorage.setItem('navi_knowledge_graph', JSON.stringify(data.knowledgeGraph));
      }
      if (data.settings) {
        localStorage.setItem('navi_settings', JSON.stringify(data.settings));
      }
      return true;
    } catch (error) {
      console.error('导入数据失败:', error);
      return false;
    }
  }
}

export const localDB = new LocalDB();