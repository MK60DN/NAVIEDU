// EduPath 应用配置文件

// ===== 应用配置 =====
const APP_CONFIG = {
    // 应用信息
    APP_NAME: 'EduPath',
    APP_VERSION: '1.0.0',
    APP_DESCRIPTION: '知识经济学习平台',

    // API配置
    API_BASE_URL: 'http://localhost:8000/api',
    API_TIMEOUT: 30000, // 30秒超时

    // 存储配置
    STORAGE_KEYS: {
        AUTH_TOKEN: 'edupath_auth_token',
        USER_INFO: 'edupath_user_info',
        SETTINGS: 'edupath_settings',
        THEME: 'edupath_theme',
        LANGUAGE: 'edupath_language'
    },

    // 主题配置
    THEMES: {
        DEFAULT: 'binance-dark',
        LIGHT: 'binance-light',
        DARK: 'binance-dark'
    },

    // 语言配置
    LANGUAGES: {
        ZH_CN: 'zh-CN',
        EN_US: 'en-US'
    },

    // 页面配置
    VIEWS: {
        WELCOME: 'welcome',
        DASHBOARD: 'dashboard',
        LEARN: 'learn',
        MARKET: 'market',
        CONTRIBUTE: 'contribute',
        WALLET: 'wallet',
        PROFILE: 'profile',
        SETTINGS: 'settings'
    },

    // 用户角色
    USER_ROLES: {
        STUDENT: 'student',
        TEACHER: 'teacher',
        ADMIN: 'admin'
    },

    // 代币配置
    TOKENS: {
        EDP: {
            symbol: 'EDP',
            name: 'EduPath Token',
            decimals: 18,
            icon: '🎓'
        },
        KCT: {
            symbol: 'KCT',
            name: 'Knowledge Contribution Token',
            decimals: 18,
            icon: '💡'
        }
    },

    // 知识胶囊配置
    CAPSULE_TYPES: {
        VIDEO: 'video',
        ARTICLE: 'article',
        QUIZ: 'quiz',
        COURSE: 'course',
        TUTORIAL: 'tutorial'
    },

    // 分类配置
    CATEGORIES: [
        { id: 'programming', name: '编程开发', icon: '💻' },
        { id: 'design', name: '设计创意', icon: '🎨' },
        { id: 'business', name: '商业管理', icon: '📊' },
        { id: 'language', name: '语言学习', icon: '🗣️' },
        { id: 'science', name: '科学技术', icon: '🔬' },
        { id: 'arts', name: '人文艺术', icon: '🎭' },
        { id: 'health', name: '健康生活', icon: '🏃' },
        { id: 'finance', name: '金融投资', icon: '💰' }
    ],

    // 难度等级
    DIFFICULTY_LEVELS: [
        { id: 'beginner', name: '初级', color: '#02c076' },
        { id: 'intermediate', name: '中级', color: '#fcd535' },
        { id: 'advanced', name: '高级', color: '#f84960' },
        { id: 'expert', name: '专家', color: '#8b5cf6' }
    ],

    // 分页配置
    PAGINATION: {
        DEFAULT_PAGE_SIZE: 10,
        MAX_PAGE_SIZE: 100,
        PAGE_SIZE_OPTIONS: [10, 20, 50, 100]
    },

    // 文件上传配置
    UPLOAD: {
        MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
        ALLOWED_IMAGE_TYPES: ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
        ALLOWED_VIDEO_TYPES: ['video/mp4', 'video/webm', 'video/ogg'],
        ALLOWED_DOCUMENT_TYPES: ['application/pdf', 'text/plain', 'application/msword']
    },

    // 通知配置
    NOTIFICATIONS: {
        AUTO_HIDE_DELAY: 3000,
        MAX_NOTIFICATIONS: 5,
        POSITION: 'top-right'
    },

    // 图表配置
    CHARTS: {
        DEFAULT_COLORS: [
            '#fcd535', // 币安黄
            '#02c076', // 成功绿
            '#f84960', // 错误红
            '#0099ff', // 信息蓝
            '#8b5cf6', // 紫色
            '#f59e0b', // 橙色
            '#ef4444', // 红色
            '#10b981'  // 青色
        ],
        ANIMATION_DURATION: 1000,
        RESPONSIVE: true
    },

    // 搜索配置
    SEARCH: {
        MIN_QUERY_LENGTH: 2,
        DEBOUNCE_DELAY: 300,
        MAX_RESULTS: 20
    },

    // 验证规则
    VALIDATION: {
        USERNAME: {
            MIN_LENGTH: 3,
            MAX_LENGTH: 20,
            PATTERN: /^[a-zA-Z0-9_]+$/
        },
        PASSWORD: {
            MIN_LENGTH: 6,
            MAX_LENGTH: 50,
            REQUIRE_UPPERCASE: false,
            REQUIRE_LOWERCASE: false,
            REQUIRE_NUMBERS: false,
            REQUIRE_SYMBOLS: false
        },
        EMAIL: {
            PATTERN: /^[^\s@]+@[^\s@]+\.[^\s@]+$/
        }
    },

    // 缓存配置
    CACHE: {
        USER_DATA_TTL: 30 * 60 * 1000, // 30分钟
        MARKET_DATA_TTL: 5 * 60 * 1000,  // 5分钟
        STATIC_DATA_TTL: 60 * 60 * 1000  // 1小时
    },

    // 功能开关
    FEATURES: {
        AI_TUTOR: true,
        DARK_MODE: true,
        REAL_TIME_CHAT: false,
        PUSH_NOTIFICATIONS: false,
        OFFLINE_MODE: false,
        ANALYTICS: true
    },

    // 社交链接
    SOCIAL_LINKS: {
        WEBSITE: 'https://edupath.example.com',
        TWITTER: 'https://twitter.com/edupath',
        TELEGRAM: 'https://t.me/edupath',
        DISCORD: 'https://discord.gg/edupath',
        GITHUB: 'https://github.com/edupath'
    },

    // 错误信息
    ERROR_MESSAGES: {
        NETWORK_ERROR: '网络连接失败，请检查网络设置',
        SERVER_ERROR: '服务器繁忙，请稍后重试',
        UNAUTHORIZED: '登录已过期，请重新登录',
        FORBIDDEN: '权限不足，无法执行此操作',
        NOT_FOUND: '请求的资源不存在',
        VALIDATION_ERROR: '输入信息不符合要求',
        UNKNOWN_ERROR: '未知错误，请联系客服'
    },

    // 成功信息
    SUCCESS_MESSAGES: {
        LOGIN_SUCCESS: '登录成功',
        REGISTER_SUCCESS: '注册成功',
        LOGOUT_SUCCESS: '已安全退出',
        SAVE_SUCCESS: '保存成功',
        DELETE_SUCCESS: '删除成功',
        UPDATE_SUCCESS: '更新成功',
        UPLOAD_SUCCESS: '上传成功'
    }
};

// ===== 环境配置 =====
const ENV_CONFIG = {
    // 开发环境
    development: {
        API_BASE_URL: 'http://localhost:8000/api',
        DEBUG: true,
        LOG_LEVEL: 'debug'
    },

    // 测试环境
    testing: {
        API_BASE_URL: 'https://test-api.edupath.com/api',
        DEBUG: true,
        LOG_LEVEL: 'info'
    },

    // 生产环境
    production: {
        API_BASE_URL: 'https://api.edupath.com/api',
        DEBUG: false,
        LOG_LEVEL: 'error'
    }
};

// ===== 当前环境检测 =====
const getCurrentEnvironment = () => {
    const hostname = window.location.hostname;

    if (hostname === 'localhost' || hostname === '127.0.0.1') {
        return 'development';
    } else if (hostname.includes('test') || hostname.includes('staging')) {
        return 'testing';
    } else {
        return 'production';
    }
};

// ===== 合并配置 =====
const CURRENT_ENV = getCurrentEnvironment();
const MERGED_CONFIG = {
    ...APP_CONFIG,
    ...ENV_CONFIG[CURRENT_ENV],
    ENVIRONMENT: CURRENT_ENV
};

// ===== 导出配置 =====
window.APP_CONFIG = MERGED_CONFIG;

// ===== 配置验证 =====
const validateConfig = () => {
    const requiredKeys = ['API_BASE_URL', 'STORAGE_KEYS', 'VIEWS'];

    for (const key of requiredKeys) {
        if (!MERGED_CONFIG[key]) {
            console.error(`Missing required config: ${key}`);
            return false;
        }
    }

    return true;
};

// ===== 配置初始化 =====
const initConfig = () => {
    if (!validateConfig()) {
        throw new Error('Configuration validation failed');
    }

    // 设置全局错误处理
    window.addEventListener('unhandledrejection', (event) => {
        console.error('Unhandled promise rejection:', event.reason);

        if (MERGED_CONFIG.DEBUG) {
            alert(`Unhandled error: ${event.reason}`);
        }
    });

    // 设置全局配置
    if (MERGED_CONFIG.DEBUG) {
        console.log('EduPath Config Initialized:', MERGED_CONFIG);
    }
};

// 初始化配置
initConfig();