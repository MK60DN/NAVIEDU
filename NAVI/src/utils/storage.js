import { STORAGE_KEYS, APP_CONFIG } from './constants';

class StorageManager {
  constructor() {
    this.quota = this.getStorageQuota();
  }

  // 获取存储配额
  getStorageQuota() {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      navigator.storage.estimate().then(estimate => {
        this.quota = estimate.quota;
        this.usage = estimate.usage;
      });
    }
    return null;
  }

  // 检查存储空间
  async checkStorageSpace() {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      const usage = estimate.usage || 0;
      const quota = estimate.quota || 0;

      return {
        usage,
        quota,
        available: quota - usage,
        usagePercentage: quota > 0 ? (usage / quota) * 100 : 0
      };
    }
    return null;
  }

  // 清理旧数据
  async cleanupOldData() {
    try {
      const sessions = this.getSessions();
      const now = new Date();
      const cutoffDate = new Date(now.getTime() - (30 * 24 * 60 * 60 * 1000)); // 30天前

      const cleanedSessions = {};
      let deletedCount = 0;

      Object.keys(sessions).forEach(sessionId => {
        const session = sessions[sessionId];
        const sessionDate = new Date(session.updatedAt || session.createdAt);

        if (sessionDate > cutoffDate) {
          cleanedSessions[sessionId] = session;
        } else {
          deletedCount++;
        }
      });

      if (deletedCount > 0) {
        this.setSessions(cleanedSessions);
        console.log(`清理了 ${deletedCount} 个过期会话`);
      }

      return deletedCount;
    } catch (error) {
      console.error('清理数据失败:', error);
      return 0;
    }
  }

  // 压缩数据
  compressData(data) {
    // 简单的数据压缩：移除不必要的字段
    if (Array.isArray(data)) {
      return data.map(item => this.compressData(item));
    }

    if (data && typeof data === 'object') {
      const compressed = {};
      Object.keys(data).forEach(key => {
        const value = data[key];

        // 跳过空值或很长的字符串（可能是调试信息）
        if (value !== null && value !== undefined) {
          if (typeof value === 'string' && value.length > 10000) {
            compressed[key] = value.substring(0, 1000) + '...';
          } else {
            compressed[key] = this.compressData(value);
          }
        }
      });
      return compressed;
    }

    return data;
  }

  // 获取会话数据
  getSessions() {
    try {
      const data = localStorage.getItem(STORAGE_KEYS.SESSIONS);
      return data ? JSON.parse(data) : {};
    } catch (error) {
      console.error('读取会话数据失败:', error);
      return {};
    }
  }

  // 保存会话数据
  setSessions(sessions) {
    try {
      const compressed = this.compressData(sessions);
      localStorage.setItem(STORAGE_KEYS.SESSIONS, JSON.stringify(compressed));
      return true;
    } catch (error) {
      if (error.name === 'QuotaExceededError') {
        // 存储空间不足，尝试清理
        this.cleanupOldData().then(() => {
          try {
            localStorage.setItem(STORAGE_KEYS.SESSIONS, JSON.stringify(compressed));
          } catch (e) {
            console.error('清理后仍然无法保存:', e);
          }
        });
      }
      console.error('保存会话数据失败:', error);
      return false;
    }
  }

  // 获取知识图谱
  getKnowledgeGraph() {
    try {
      const data = localStorage.getItem(STORAGE_KEYS.KNOWLEDGE_GRAPH);
      return data ? JSON.parse(data) : this.getDefaultKnowledgeGraph();
    } catch (error) {
      console.error('读取知识图谱失败:', error);
      return this.getDefaultKnowledgeGraph();
    }
  }

  // 保存知识图谱
  setKnowledgeGraph(graph) {
    try {
      const compressed = this.compressData(graph);
      localStorage.setItem(STORAGE_KEYS.KNOWLEDGE_GRAPH, JSON.stringify(compressed));
      return true;
    } catch (error) {
      console.error('保存知识图谱失败:', error);
      return false;
    }
  }

  // 获取默认知识图谱
  getDefaultKnowledgeGraph() {
    return {
      id: 'root',
      title: '我的知识库',
      content: '个人知识图谱根节点',
      type: 'system',
      children: [
        {
          id: 'learning_notes',
          title: '学习笔记',
          content: '学习过程中的知识点记录',
          type: 'system',
          children: []
        },
        {
          id: 'questions',
          title: '问题思考',
          content: '质疑和批判性思考记录',
          type: 'system',
          children: []
        }
      ]
    };
  }

  // 获取设置
  getSettings() {
    try {
      const data = localStorage.getItem(STORAGE_KEYS.SETTINGS);
      return data ? { ...this.getDefaultSettings(), ...JSON.parse(data) } : this.getDefaultSettings();
    } catch (error) {
      console.error('读取设置失败:', error);
      return this.getDefaultSettings();
    }
  }

  // 保存设置
  setSettings(settings) {
    try {
      localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
      return true;
    } catch (error) {
      console.error('保存设置失败:', error);
      return false;
    }
  }

  // 获取默认设置
  getDefaultSettings() {
    return {
      theme: 'dark',
      language: 'zh-CN',
      voiceEnabled: true,
      autoSave: true,
      apiKey: '',
      maxHistoryLength: APP_CONFIG.MAX_CONVERSATION_HISTORY,
      notifications: true
    };
  }

  // 导出所有数据
  exportAllData() {
    return {
      sessions: this.getSessions(),
      knowledgeGraph: this.getKnowledgeGraph(),
      settings: this.getSettings(),
      exportTime: new Date().toISOString(),
      version: '1.0.0'
    };
  }

  // 导入数据
  importData(data) {
    try {
      if (data.sessions) {
        this.setSessions(data.sessions);
      }
      if (data.knowledgeGraph) {
        this.setKnowledgeGraph(data.knowledgeGraph);
      }
      if (data.settings) {
        this.setSettings(data.settings);
      }
      return true;
    } catch (error) {
      console.error('导入数据失败:', error);
      return false;
    }
  }

  // 清空所有数据
  clearAll() {
    try {
      Object.values(STORAGE_KEYS).forEach(key => {
        localStorage.removeItem(key);
      });
      return true;
    } catch (error) {
      console.error('清空数据失败:', error);
      return false;
    }
  }

  // 获取存储使用情况
  getStorageInfo() {
    let totalSize = 0;
    const info = {};

    Object.entries(STORAGE_KEYS).forEach(([name, key]) => {
      const data = localStorage.getItem(key);
      const size = data ? new Blob([data]).size : 0;
      info[name] = { size, sizeFormatted: this.formatBytes(size) };
      totalSize += size;
    });

    return {
      ...info,
      total: { size: totalSize, sizeFormatted: this.formatBytes(totalSize) }
    };
  }

  // 格式化字节数
  formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }
}

export const storageManager = new StorageManager();