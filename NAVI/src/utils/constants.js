export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  TIMEOUT: 30000,
  RETRY_ATTEMPTS: 3
};

export const STORAGE_KEYS = {
  SESSIONS: 'navi_sessions',
  KNOWLEDGE_GRAPH: 'navi_knowledge_graph',
  SETTINGS: 'navi_settings',
  USER_PROFILE: 'navi_user_profile'
};

export const AGENT_TYPES = {
  LEARNING: 'learning',
  QUESTIONING: 'questioning',
  BALANCING: 'balancing',
  CHAT: 'chat'
};

export const MESSAGE_TYPES = {
  TEXT: 'text',
  VOICE: 'voice',
  IMAGE: 'image',
  FILE: 'file'
};

export const KNOWLEDGE_NODE_TYPES = {
  LEARNING: 'learning',
  QUESTIONING: 'questioning',
  MANUAL: 'manual',
  SYSTEM: 'system'
};

export const APP_CONFIG = {
  MAX_MESSAGE_LENGTH: 2000,
  MAX_CONVERSATION_HISTORY: 50,
  AUTO_SAVE_INTERVAL: 30000, // 30秒
  VOICE_TIMEOUT: 10000, // 10秒
  MAX_FILE_SIZE: 10 * 1024 * 1024 // 10MB
};

export const UI_CONFIG = {
  COLORS: {
    PRIMARY: '#fbbf24', // 黄色
    SECONDARY: '#1a1a1a', // 深灰
    BACKGROUND: '#000000', // 黑色
    TEXT_PRIMARY: '#ffffff', // 白色
    TEXT_SECONDARY: '#9ca3af', // 浅灰
    SUCCESS: '#10b981',
    ERROR: '#ef4444',
    WARNING: '#f59e0b'
  },
  BREAKPOINTS: {
    SM: '640px',
    MD: '768px',
    LG: '1024px',
    XL: '1280px'
  }
};

export const ERROR_MESSAGES = {
  NETWORK_ERROR: '网络连接失败，请检查网络设置',
  API_ERROR: 'API调用失败，请稍后重试',
  VOICE_NOT_SUPPORTED: '当前浏览器不支持语音功能',
  FILE_TOO_LARGE: '文件大小超过限制',
  INVALID_FILE_TYPE: '不支持的文件类型',
  STORAGE_FULL: '本地存储空间不足'
};

export const SUCCESS_MESSAGES = {
  DATA_SAVED: '数据保存成功',
  SETTINGS_UPDATED: '设置更新成功',
  FILE_UPLOADED: '文件上传成功',
  KNOWLEDGE_UPDATED: '知识图谱更新成功'
};