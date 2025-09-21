// ====== web/js/aiTutor.js ======
// AI学习导师核心功能模块

class AITutorService {
    constructor() {
        this.apiBase = API_CONFIG.BASE_URL || 'http://localhost:8000/api';
        this.conversationHistory = [];
        this.currentSession = {
            id: null,
            startTime: Date.now(),
            messageCount: 0,
            topics: []
        };
        this.cache = new Map();
        this.retryCount = 0;
        this.maxRetries = 3;
    }

    // 获取认证头部
    getAuthHeaders() {
        const token = localStorage.getItem('token');
        return {
            'Content-Type': 'application/json',
            'Authorization': token ? `Bearer ${token}` : ''
        };
    }

    // API请求封装
    async makeRequest(endpoint, options = {}) {
        const url = `${this.apiBase}${endpoint}`;
        const defaultOptions = {
            method: 'GET',
            headers: this.getAuthHeaders(),
            ...options
        };

        try {
            const response = await fetch(url, defaultOptions);
            
            if (!response.ok) {
                if (response.status === 401) {
                    // Token过期，重定向到登录
                    this.handleAuthError();
                    throw new Error('认证失败');
                }
                
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail?.message || errorData.message || `HTTP ${response.status}`);
            }

            const data = await response.json();
            this.retryCount = 0; // 请求成功，重置重试计数
            return data;

        } catch (error) {
            console.error(`API请求失败 ${endpoint}:`, error);
            
            // 网络错误重试机制
            if (this.isNetworkError(error) && this.retryCount < this.maxRetries) {
                this.retryCount++;
                console.log(`重试 ${this.retryCount}/${this.maxRetries}: ${endpoint}`);
                await this.delay(1000 * this.retryCount); // 递增延迟
                return this.makeRequest(endpoint, options);
            }
            
            throw error;
        }
    }

    // 判断是否为网络错误
    isNetworkError(error) {
        return error.message.includes('fetch') || 
               error.message.includes('network') ||
               error.message.includes('timeout');
    }

    // 延迟函数
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // 处理认证错误
    handleAuthError() {
        localStorage.removeItem('token');
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 2000);
    }

    // 发送聊天消息
    async sendChatMessage(message, conversationHistory = []) {
        // 优化对话历史
        const optimizedHistory = this.optimizeConversationHistory(conversationHistory);
        
        const response = await this.makeRequest('/ai-tutor/chat', {
            method: 'POST',
            body: JSON.stringify({
                message: message.trim(),
                conversation_history: optimizedHistory
            })
        });

        if (response.success) {
            // 更新会话统计
            this.currentSession.messageCount++;
            this.updateSessionTopics(response.data);
            
            // 缓存常见问题的答案
            if (this.isCommonQuestion(message)) {
                this.cache.set(message.toLowerCase(), response.data);
            }
        }

        return response;
    }

    // 优化对话历史，控制Token使用
    optimizeConversationHistory(history, maxMessages = 10) {
        if (history.length <= maxMessages) {
            return history;
        }

        // 保留最近的对话
        const recentHistory = history.slice(-maxMessages);
        
        // 生成上下文摘要
        const contextSummary = this.generateContextSummary(history.slice(0, -maxMessages));
        
        if (contextSummary) {
            return [
                { role: 'system', content: `对话上下文：${contextSummary}` },
                ...recentHistory
            ];
        }
        
        return recentHistory;
    }

    // 生成上下文摘要
    generateContextSummary(history) {
        const topics = [];
        const keyPhrases = ['学习', '理解', '问题', '解释', '教程', '概念'];
        
        history.forEach(msg => {
            if (msg.role === 'user') {
                keyPhrases.forEach(phrase => {
                    if (msg.content.includes(phrase)) {
                        const snippet = msg.content.substring(0, 50);
                        if (!topics.includes(snippet)) {
                            topics.push(snippet);
                        }
                    }
                });
            }
        });
        
        return topics.slice(0, 3).join('; ');
    }

    // 判断是否为常见问题
    isCommonQuestion(message) {
        const commonPatterns = [
            /什么是.*?/,
            /如何.*?/,
            /怎么.*?/,
            /为什么.*?/,
            /解释.*?/
        ];
        
        return commonPatterns.some(pattern => pattern.test(message));
    }

    // 更新会话主题
    updateSessionTopics(responseData) {
        if (responseData.type === 'learning_assistance' && responseData.data?.current_topic) {
            const topic = responseData.data.current_topic;
            if (!this.currentSession.topics.includes(topic)) {
                this.currentSession.topics.push(topic);
            }
        }
    }

    // 快速搜索知识点
    async searchKnowledge(query) {
        // 检查缓存
        const cacheKey = `search:${query.toLowerCase()}`;
        if (this.cache.has(cacheKey)) {
            return { success: true, data: this.cache.get(cacheKey) };
        }

        const response = await this.makeRequest('/ai-tutor/search', {
            method: 'POST',
            body: JSON.stringify({ query })
        });

        if (response.success) {
            // 缓存搜索结果
            this.cache.set(cacheKey, response.data);
        }

        return response;
    }

    // 获取学习路径
    async getLearningPath(startTopic, endTopic) {
        const cacheKey = `path:${startTopic}->${endTopic}`;
        if (this.cache.has(cacheKey)) {
            return { success: true, data: this.cache.get(cacheKey) };
        }

        const response = await this.makeRequest(`/ai-tutor/learning-path/${encodeURIComponent(startTopic)}/${encodeURIComponent(endTopic)}`);

        if (response.success) {
            this.cache.set(cacheKey, response.data);
        }

        return response;
    }

    // 提交知识贡献
    async submitContribution(contributionData) {
        const response = await this.makeRequest('/ai-tutor/contribute', {
            method: 'POST',
            body: JSON.stringify(contributionData)
        });
        
        return response;
    }

    // 提交用户反馈
    async submitFeedback(feedbackData) {
        const response = await this.makeRequest('/ai-tutor/feedback', {
            method: 'POST',
            body: JSON.stringify({
                ...feedbackData,
                session_info: this.currentSession,
                timestamp: Date.now()
            })
        });
        
        return response;
    }

    // 获取AI导师统计信息
    async getStats() {
        try {
            const response = await this.makeRequest('/ai-tutor/stats');
            return response;
        } catch (error) {
            console.warn('获取统计信息失败:', error);
            return { success: false, data: {} };
        }
    }

    // 清除缓存
    clearCache() {
        this.cache.clear();
        this.conversationHistory = [];
    }

    // 获取会话信息
    getSessionInfo() {
        return {
            ...this.currentSession,
            duration: Date.now() - this.currentSession.startTime,
            cacheSize: this.cache.size
        };
    }
}

// ====== 消息格式化工具 ======
class MessageFormatter {
    static formatContent(content) {
        if (!content) return '';
        
        // HTML转义
        content = this.escapeHtml(content);
        
        // 换行符转换
        content = content.replace(/\n/g, '<br>');
        
        // Markdown样式转换
        content = this.convertMarkdown(content);
        
        // 代码高亮
        content = this.highlightCode(content);
        
        // 链接转换
        content = this.convertLinks(content);
        
        return content;
    }

    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    static convertMarkdown(text) {
        // 粗体
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // 斜体
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // 代码块
        text = text.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // 行内代码
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // 标题
        text = text.replace(/^### (.*$)/gm, '<h3>$1</h3>');
        text = text.replace(/^## (.*$)/gm, '<h2>$1</h2>');
        text = text.replace(/^# (.*$)/gm, '<h1>$1</h1>');
        
        // 列表
        text = text.replace(/^\* (.*$)/gm, '<li>$1</li>');
        text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        
        return text;
    }

    static highlightCode(text) {
        return text.replace(/<code>(.*?)<\/code>/g, (match, code) => {
            // 简单的语法高亮
            let highlighted = code;
            
            // Python关键字
            highlighted = highlighted.replace(/\b(def|class|if|else|elif|for|while|import|from|return|try|except|with|as)\b/g, '<span class="keyword">$1</span>');
            
            // 字符串
            highlighted = highlighted.replace(/(["'])(.*?)\1/g, '<span class="string">$1$2$1</span>');
            
            // 数字
            highlighted = highlighted.replace(/\b\d+\b/g, '<span class="number">$&</span>');
            
            // 注释
            highlighted = highlighted.replace(/(#.*$)/gm, '<span class="comment">$1</span>');
            
            return `<code>${highlighted}</code>`;
        });
    }

    static convertLinks(text) {
        // URL链接
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        return text.replace(urlRegex, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
    }

    static formatTime(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        
        if (date.toDateString() === now.toDateString()) {
            // 今天 - 显示时间
            return date.toLocaleTimeString('zh-CN', {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit'
            });
        } else if (date.getFullYear() === now.getFullYear()) {
            // 今年 - 显示月日时间
            return date.toLocaleDateString('zh-CN', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } else {
            // 其他年份 - 显示完整日期
            return date.toLocaleDateString('zh-CN');
        }
    }

    static truncateText(text, maxLength = 100) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }
}

// ====== 本地存储管理器 ======
class LocalStorageManager {
    static keys = {
        CHAT_HISTORY: 'aiTutor_chatHistory',
        USER_PREFERENCES: 'aiTutor_preferences',
        CHAT_STATS: 'aiTutor_stats',
        CACHE_DATA: 'aiTutor_cache',
        LEARNING_PROGRESS: 'aiTutor_progress'
    };

    static saveChatHistory(messages) {
        try {
            // 只保存最近50条消息
            const limitedMessages = messages.slice(-50);
            localStorage.setItem(this.keys.CHAT_HISTORY, JSON.stringify(limitedMessages));
        } catch (error) {
            console.warn('保存聊天历史失败:', error);
            // 清理旧数据
            this.clearOldData();
        }
    }

    static loadChatHistory() {
        try {
            const data = localStorage.getItem(this.keys.CHAT_HISTORY);
            return data ? JSON.parse(data) : [];
        } catch (error) {
            console.warn('加载聊天历史失败:', error);
            return [];
        }
    }

    static saveUserPreferences(preferences) {
        try {
            localStorage.setItem(this.keys.USER_PREFERENCES, JSON.stringify(preferences));
        } catch (error) {
            console.warn('保存用户偏好失败:', error);
        }
    }

    static loadUserPreferences() {
        try {
            const data = localStorage.getItem(this.keys.USER_PREFERENCES);
            return data ? JSON.parse(data) : {
                theme: 'dark',
                autoScroll: true,
                soundEnabled: false,
                animationsEnabled: true
            };
        } catch (error) {
            console.warn('加载用户偏好失败:', error);
            return {};
        }
    }

    static saveChatStats(stats) {
        try {
            const existing = this.loadChatStats();
            const updated = { ...existing, ...stats, lastUpdated: Date.now() };
            localStorage.setItem(this.keys.CHAT_STATS, JSON.stringify(updated));
        } catch (error) {
            console.warn('保存聊天统计失败:', error);
        }
    }

    static loadChatStats() {
        try {
            const data = localStorage.getItem(this.keys.CHAT_STATS);
            return data ? JSON.parse(data) : {
                totalChats: 0,
                learnedTopics: 0,
                contributions: 0,
                totalTimeSpent: 0,
                lastUpdated: Date.now()
            };
        } catch (error) {
            console.warn('加载聊天统计失败:', error);
            return {};
        }
    }

    static saveLearningProgress(progress) {
        try {
            const today = new Date().toDateString();
            const existing = this.loadLearningProgress();
            existing[today] = { ...existing[today], ...progress, lastUpdated: Date.now() };
            localStorage.setItem(this.keys.LEARNING_PROGRESS, JSON.stringify(existing));
        } catch (error) {
            console.warn('保存学习进度失败:', error);
        }
    }

    static loadLearningProgress() {
        try {
            const data = localStorage.getItem(this.keys.LEARNING_PROGRESS);
            return data ? JSON.parse(data) : {};
        } catch (error) {
            console.warn('加载学习进度失败:', error);
            return {};
        }
    }

    static getTodayProgress() {
        const today = new Date().toDateString();
        const allProgress = this.loadLearningProgress();
        return allProgress[today] || {
            learned: 0,
            target: 5,
            topics: [],
            timeSpent: 0
        };
    }

    static clearOldData() {
        const cutoffDate = Date.now() - (30 * 24 * 60 * 60 * 1000); // 30天前
        
        try {
            // 清理旧的学习进度
            const progress = this.loadLearningProgress();
            const filteredProgress = {};
            
            Object.keys(progress).forEach(date => {
                const progressData = progress[date];
                if (progressData.lastUpdated && progressData.lastUpdated > cutoffDate) {
                    filteredProgress[date] = progressData;
                }
            });
            
            localStorage.setItem(this.keys.LEARNING_PROGRESS, JSON.stringify(filteredProgress));
            
        } catch (error) {
            console.warn('清理旧数据失败:', error);
        }
    }

    static exportData() {
        const data = {
            chatHistory: this.loadChatHistory(),
            preferences: this.loadUserPreferences(),
            stats: this.loadChatStats(),
            progress: this.loadLearningProgress(),
            exportTime: Date.now()
        };
        
        return JSON.stringify(data, null, 2);
    }

    static importData(jsonData) {
        try {
            const data = JSON.parse(jsonData);
            
            if (data.chatHistory) {
                this.saveChatHistory(data.chatHistory);
            }
            
            if (data.preferences) {
                this.saveUserPreferences(data.preferences);
            }
            
            if (data.stats) {
                this.saveChatStats(data.stats);
            }
            
            if (data.progress) {
                localStorage.setItem(this.keys.LEARNING_PROGRESS, JSON.stringify(data.progress));
            }
            
            return true;
        } catch (error) {
            console.error('导入数据失败:', error);
            return false;
        }
    }

    static clearAllData() {
        Object.values(this.keys).forEach(key => {
            localStorage.removeItem(key);
        });
    }
}

// ====== 音效管理器 ======
class SoundManager {
    constructor() {
        this.sounds = {};
        this.enabled = LocalStorageManager.loadUserPreferences().soundEnabled || false;
        this.volume = 0.5;
        
        // 预加载音效
        this.loadSounds();
    }

    loadSounds() {
        const soundFiles = {
            message: 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMaAzKPzu7BZikHM4fN8N+WQA', // 消息提示音
            success: 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMaAzKPzu7BZikHM4fN8N+WQA', // 成功音效
            error: 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmMaAzKPzu7BZikHM4fN8N+WQA'  // 错误音效
        };

        Object.entries(soundFiles).forEach(([name, data]) => {
            try {
                const audio = new Audio(data);
                audio.volume = this.volume;
                this.sounds[name] = audio;
            } catch (error) {
                console.warn(`加载音效失败: ${name}`, error);
            }
        });
    }

    play(soundName) {
        if (!this.enabled || !this.sounds[soundName]) return;
        
        try {
            const sound = this.sounds[soundName].cloneNode();
            sound.volume = this.volume;
            sound.play().catch(e => console.warn('播放音效失败:', e));
        } catch (error) {
            console.warn(`播放音效失败: ${soundName}`, error);
        }
    }

    setEnabled(enabled) {
        this.enabled = enabled;
        const preferences = LocalStorageManager.loadUserPreferences();
        preferences.soundEnabled = enabled;
        LocalStorageManager.saveUserPreferences(preferences);
    }

    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume));
        Object.values(this.sounds).forEach(sound => {
            if (sound) sound.volume = this.volume;
        });
    }
}

// ====== 主题管理器 ======
class ThemeManager {
    constructor() {
        this.themes = {
            dark: {
                name: '深色主题',
                isDark: true
            },
            light: {
                name: '浅色主题', 
                isDark: false
            },
            binance: {
                name: '币安主题',
                isDark: true
            }
        };
        
        this.currentTheme = this.loadTheme();
        this.applyTheme(this.currentTheme);
    }

    loadTheme() {
        const preferences = LocalStorageManager.loadUserPreferences();
        return preferences.theme || 'binance';
    }

    applyTheme(themeName) {
        const theme = this.themes[themeName];
        if (!theme) return;

        document.body.className = `theme-${themeName}`;
        
        // 更新meta标签
        const metaTheme = document.querySelector('meta[name="theme-color"]');
        if (metaTheme) {
            metaTheme.content = theme.isDark ? '#0B0E11' : '#FFFFFF';
        }

        this.currentTheme = themeName;
        this.saveTheme(themeName);
    }

    saveTheme(themeName) {
        const preferences = LocalStorageManager.loadUserPreferences();
        preferences.theme = themeName;
        LocalStorageManager.saveUserPreferences(preferences);
    }

    toggleTheme() {
        const themes = Object.keys(this.themes);
        const currentIndex = themes.indexOf(this.currentTheme);
        const nextIndex = (currentIndex + 1) % themes.length;
        this.applyTheme(themes[nextIndex]);
    }

    getCurrentTheme() {
        return this.themes[this.currentTheme];
    }
}

// ====== 性能监控器 ======
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            apiCalls: [],
            renderTimes: [],
            memoryUsage: [],
            errors: []
        };
        
        this.startMonitoring();
    }

    startMonitoring() {
        // 监控API调用性能
        this.monitorApiCalls();
        
        // 监控内存使用
        if (performance.memory) {
            setInterval(() => {
                this.recordMemoryUsage();
            }, 30000); // 每30秒记录一次
        }
        
        // 监控错误
        window.addEventListener('error', (event) => {
            this.recordError({
                message: event.message,
                filename: event.filename,
                lineno: event.lineno,
                colno: event.colno,
                timestamp: Date.now()
            });
        });
    }

    monitorApiCalls() {
        // 劫持fetch来监控API性能
        const originalFetch = window.fetch;
        window.fetch = async (...args) => {
            const start = performance.now();
            const url = args[0];
            
            try {
                const response = await originalFetch.apply(this, args);
                const duration = performance.now() - start;
                
                this.recordApiCall({
                    url,
                    duration,
                    status: response.status,
                    success: response.ok,
                    timestamp: Date.now()
                });
                
                return response;
            } catch (error) {
                const duration = performance.now() - start;
                
                this.recordApiCall({
                    url,
                    duration,
                    status: 0,
                    success: false,
                    error: error.message,
                    timestamp: Date.now()
                });
                
                throw error;
            }
        };
    }

    recordApiCall(data) {
        this.metrics.apiCalls.push(data);
        
        // 只保留最近100次调用
        if (this.metrics.apiCalls.length > 100) {
            this.metrics.apiCalls = this.metrics.apiCalls.slice(-50);
        }
    }

    recordRenderTime(componentName, duration) {
        this.metrics.renderTimes.push({
            component: componentName,
            duration,
            timestamp: Date.now()
        });
        
        // 只保留最近50次记录
        if (this.metrics.renderTimes.length > 50) {
            this.metrics.renderTimes = this.metrics.renderTimes.slice(-25);
        }
    }

    recordMemoryUsage() {
        if (!performance.memory) return;
        
        this.metrics.memoryUsage.push({
            used: performance.memory.usedJSHeapSize,
            total: performance.memory.totalJSHeapSize,
            limit: performance.memory.jsHeapSizeLimit,
            timestamp: Date.now()
        });
        
        // 只保留最近20次记录
        if (this.metrics.memoryUsage.length > 20) {
            this.metrics.memoryUsage = this.metrics.memoryUsage.slice(-10);
        }
    }

    recordError(error) {
        this.metrics.errors.push(error);
        
        // 只保留最近20个错误
        if (this.metrics.errors.length > 20) {
            this.metrics.errors = this.metrics.errors.slice(-10);
        }
    }

    getMetrics() {
        return {
            ...this.metrics,
            summary: this.generateSummary()
        };
    }

    generateSummary() {
        const apiCalls = this.metrics.apiCalls;
        const renderTimes = this.metrics.renderTimes;
        const memoryUsage = this.metrics.memoryUsage;
        
        return {
            avgApiResponseTime: apiCalls.length > 0 ? 
                apiCalls.reduce((sum, call) => sum + call.duration, 0) / apiCalls.length : 0,
            apiSuccessRate: apiCalls.length > 0 ?
                apiCalls.filter(call => call.success).length / apiCalls.length : 1,
            avgRenderTime: renderTimes.length > 0 ?
                renderTimes.reduce((sum, time) => sum + time.duration, 0) / renderTimes.length : 0,
            currentMemoryUsage: memoryUsage.length > 0 ? 
                memoryUsage[memoryUsage.length - 1].used : 0,
            errorCount: this.metrics.errors.length
        };
    }

    exportMetrics() {
        return JSON.stringify(this.getMetrics(), null, 2);
    }
}

// ====== 全局实例 ======
let aiTutorService;
let messageFormatter;
let soundManager;
let themeManager;
let performanceMonitor;

// 初始化所有服务
function initializeAITutorServices() {
    try {
        aiTutorService = new AITutorService();
        messageFormatter = new MessageFormatter();
        soundManager = new SoundManager();
        themeManager = new ThemeManager();
        performanceMonitor = new PerformanceMonitor();
        
        console.log('AI导师服务初始化完成');
        
        // 定期清理数据
        setInterval(() => {
            LocalStorageManager.clearOldData();
        }, 24 * 60 * 60 * 1000); // 每24小时清理一次
        
    } catch (error) {
        console.error('AI导师服务初始化失败:', error);
    }
}

// 页面加载时自动初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeAITutorServices);
} else {
    initializeAITutorServices();
}

// 导出到全局
window.AITutorService = AITutorService;
window.MessageFormatter = MessageFormatter;
window.LocalStorageManager = LocalStorageManager;
window.SoundManager = SoundManager;
window.ThemeManager = ThemeManager;
window.PerformanceMonitor = PerformanceMonitor;

// 工具函数导出
window.aiTutorUtils = {
    formatMessage: MessageFormatter.formatContent,
    formatTime: MessageFormatter.formatTime,
    truncateText: MessageFormatter.truncateText,
    playSound: (sound) => soundManager?.play(sound),
    toggleTheme: () => themeManager?.toggleTheme(),
    exportData: () => LocalStorageManager.exportData(),
    importData: (data) => LocalStorageManager.importData(data),
    getMetrics: () => performanceMonitor?.getMetrics()
};