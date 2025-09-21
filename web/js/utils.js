// EduPath 工具函数文件

// ===== 通用工具类 =====
class Utils {
    // ===== 字符串工具 =====

    // 截断字符串
    static truncate(str, length = 50, suffix = '...') {
        if (!str || str.length <= length) return str;
        return str.substring(0, length) + suffix;
    }

    // 首字母大写
    static capitalize(str) {
        if (!str) return '';
        return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
    }

    // 驼峰转下划线
    static camelToSnake(str) {
        return str.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
    }

    // 下划线转驼峰
    static snakeToCamel(str) {
        return str.replace(/_([a-z])/g, (match, letter) => letter.toUpperCase());
    }

    // 生成随机字符串
    static randomString(length = 10) {
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        let result = '';
        for (let i = 0; i < length; i++) {
            result += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return result;
    }

    // ===== 数字工具 =====

    // 格式化数字
    static formatNumber(num, decimals = 2) {
        if (isNaN(num)) return '0';
        return Number(num).toLocaleString('zh-CN', {
            minimumFractionDigits: 0,
            maximumFractionDigits: decimals
        });
    }

    // 格式化货币
    static formatCurrency(amount, currency = 'EDP', decimals = 2) {
        const formatted = this.formatNumber(amount, decimals);
        return `${formatted} ${currency}`;
    }

    // 格式化百分比
    static formatPercentage(value, decimals = 2) {
        if (isNaN(value)) return '0%';
        return `${(value * 100).toFixed(decimals)}%`;
    }

    // 格式化文件大小
    static formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // 生成随机数
    static randomNumber(min = 0, max = 100) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    // ===== 日期时间工具 =====

    // 格式化日期
    static formatDate(date, format = 'YYYY-MM-DD') {
        if (!date) return '';

        const d = new Date(date);
        if (isNaN(d.getTime())) return '';

        const year = d.getFullYear();
        const month = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        const hours = String(d.getHours()).padStart(2, '0');
        const minutes = String(d.getMinutes()).padStart(2, '0');
        const seconds = String(d.getSeconds()).padStart(2, '0');

        return format
            .replace('YYYY', year)
            .replace('MM', month)
            .replace('DD', day)
            .replace('HH', hours)
            .replace('mm', minutes)
            .replace('ss', seconds);
    }

    // 格式化日期时间
    static formatDateTime(date) {
        return this.formatDate(date, 'YYYY-MM-DD HH:mm:ss');
    }

    // 相对时间
    static timeAgo(date) {
        if (!date) return '';

        const now = new Date();
        const past = new Date(date);
        const diffMs = now - past;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);

        if (diffMins < 1) return '刚刚';
        if (diffMins < 60) return `${diffMins}分钟前`;
        if (diffHours < 24) return `${diffHours}小时前`;
        if (diffDays < 7) return `${diffDays}天前`;

        return this.formatDate(date);
    }

    // ===== 验证工具 =====

    // 验证邮箱
    static isValidEmail(email) {
        return APP_CONFIG.VALIDATION.EMAIL.PATTERN.test(email);
    }

    // 验证用户名
    static isValidUsername(username) {
        const { MIN_LENGTH, MAX_LENGTH, PATTERN } = APP_CONFIG.VALIDATION.USERNAME;
        return username &&
               username.length >= MIN_LENGTH &&
               username.length <= MAX_LENGTH &&
               PATTERN.test(username);
    }

    // 验证密码
    static isValidPassword(password) {
        const { MIN_LENGTH, MAX_LENGTH } = APP_CONFIG.VALIDATION.PASSWORD;
        return password &&
               password.length >= MIN_LENGTH &&
               password.length <= MAX_LENGTH;
    }

    // 验证URL
    static isValidUrl(url) {
        try {
            new URL(url);
            return true;
        } catch {
            return false;
        }
    }

    // ===== 存储工具 =====

    // 设置本地存储
    static setStorage(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (error) {
            console.error('Storage set error:', error);
            return false;
        }
    }

    // 获取本地存储
    static getStorage(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (error) {
            console.error('Storage get error:', error);
            return defaultValue;
        }
    }

    // 删除本地存储
    static removeStorage(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (error) {
            console.error('Storage remove error:', error);
            return false;
        }
    }

    // 清空本地存储
    static clearStorage() {
        try {
            localStorage.clear();
            return true;
        } catch (error) {
            console.error('Storage clear error:', error);
            return false;
        }
    }

    // ===== DOM工具 =====

    // 添加类名
    static addClass(element, className) {
        if (element && className) {
            element.classList.add(className);
        }
    }

    // 移除类名
    static removeClass(element, className) {
        if (element && className) {
            element.classList.remove(className);
        }
    }

    // 切换类名
    static toggleClass(element, className) {
        if (element && className) {
            element.classList.toggle(className);
        }
    }

    // 查询元素
    static $(selector) {
        return document.querySelector(selector);
    }

    // 查询所有元素
    static $$(selector) {
        return document.querySelectorAll(selector);
    }

    // ===== 通知工具 =====

    // 显示成功提示
    static showSuccess(message) {
        this.showToast(message, 'success');
    }

    // 显示错误提示
    static showError(message) {
        this.showToast(message, 'error');
    }

    // 显示警告提示
    static showWarning(message) {
        this.showToast(message, 'warning');
    }

    // 显示信息提示
    static showInfo(message) {
        this.showToast(message, 'info');
    }

    // 显示Toast
    static showToast(message, type = 'info') {
        // 创建toast元素
        const toast = document.createElement('div');
        toast.className = `toast toast-${type} notification-slide-in`;
        toast.textContent = message;

        // 添加到页面
        document.body.appendChild(toast);

        // 自动移除
        setTimeout(() => {
            toast.classList.add('notification-slide-out');
            setTimeout(() => {
                if (document.body.contains(toast)) {
                    document.body.removeChild(toast);
                }
            }, 300);
        }, APP_CONFIG.NOTIFICATIONS.AUTO_HIDE_DELAY);
    }

    // ===== 防抖和节流 =====

    // 防抖
    static debounce(func, wait = 300) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // 节流
    static throttle(func, limit = 300) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    // ===== 数组工具 =====

    // 数组去重
    static uniqueArray(arr, key = null) {
        if (!Array.isArray(arr)) return [];

        if (key) {
            const seen = new Set();
            return arr.filter(item => {
                const keyValue = item[key];
                if (seen.has(keyValue)) {
                    return false;
                }
                seen.add(keyValue);
                return true;
            });
        }

        return [...new Set(arr)];
    }

    // 数组分组
    static groupBy(arr, key) {
        if (!Array.isArray(arr)) return {};

        return arr.reduce((groups, item) => {
            const group = item[key];
            groups[group] = groups[group] || [];
            groups[group].push(item);
            return groups;
        }, {});
    }

    // 数组排序
    static sortBy(arr, key, desc = false) {
        if (!Array.isArray(arr)) return [];

        return [...arr].sort((a, b) => {
            const aVal = a[key];
            const bVal = b[key];

            if (aVal < bVal) return desc ? 1 : -1;
            if (aVal > bVal) return desc ? -1 : 1;
            return 0;
        });
    }

    // ===== 对象工具 =====

    // 深拷贝
    static deepClone(obj) {
        if (obj === null || typeof obj !== 'object') return obj;
        if (obj instanceof Date) return new Date(obj.getTime());
        if (obj instanceof Array) return obj.map(item => this.deepClone(item));

        const cloned = {};
        for (const key in obj) {
            if (obj.hasOwnProperty(key)) {
                cloned[key] = this.deepClone(obj[key]);
            }
        }
        return cloned;
    }

    // 对象合并
    static mergeObjects(...objects) {
        return Object.assign({}, ...objects);
    }

    // 获取嵌套属性
    static getNestedProperty(obj, path, defaultValue = undefined) {
        const keys = path.split('.');
        let result = obj;

        for (const key of keys) {
            if (result === null || result === undefined) {
                return defaultValue;
            }
            result = result[key];
        }

        return result !== undefined ? result : defaultValue;
    }

    // ===== 文件工具 =====

    // 读取文件为Base64
    static readFileAsBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    // 验证文件类型
    static isValidFileType(file, allowedTypes) {
        return allowedTypes.includes(file.type);
    }

    // 验证文件大小
    static isValidFileSize(file, maxSize) {
        return file.size <= maxSize;
    }

    // ===== URL工具 =====

    // 获取URL参数
    static getUrlParams() {
        const params = new URLSearchParams(window.location.search);
        const result = {};
        for (const [key, value] of params) {
            result[key] = value;
        }
        return result;
    }

    // 设置URL参数
    static setUrlParams(params) {
        const url = new URL(window.location);
        Object.keys(params).forEach(key => {
            if (params[key] !== null && params[key] !== undefined) {
                url.searchParams.set(key, params[key]);
            } else {
                url.searchParams.delete(key);
            }
        });
        window.history.replaceState({}, '', url);
    }

    // ===== 加密工具 =====

    // 简单哈希
    static simpleHash(str) {
        let hash = 0;
        if (str.length === 0) return hash;

        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // 转换为32位整数
        }

        return Math.abs(hash).toString(36);
    }

    // ===== 设备检测 =====

    // 检测移动设备
    static isMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }

    // 检测平板设备
    static isTablet() {
        return /iPad|Android/i.test(navigator.userAgent) && !this.isMobile();
    }

    // 检测桌面设备
    static isDesktop() {
        return !this.isMobile() && !this.isTablet();
    }

    // ===== 性能工具 =====

    // 性能计时
    static time(label) {
        console.time(label);
    }

    static timeEnd(label) {
        console.timeEnd(label);
    }

    // 延迟执行
    static delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // ===== 颜色工具 =====

    // 生成随机颜色
    static randomColor() {
        return '#' + Math.floor(Math.random() * 16777215).toString(16).padStart(6, '0');
    }

    // 颜色转换
    static hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }
}

// ===== 导出工具类 =====
window.Utils = Utils;

// ===== 全局扩展 =====

// 扩展Date原型
Date.prototype.format = function(format = 'YYYY-MM-DD') {
    return Utils.formatDate(this, format);
};

// 扩展Array原型
Array.prototype.unique = function(key = null) {
    return Utils.uniqueArray(this, key);
};

Array.prototype.groupBy = function(key) {
    return Utils.groupBy(this, key);
};

Array.prototype.sortBy = function(key, desc = false) {
    return Utils.sortBy(this, key, desc);
};

// ===== 初始化 =====
console.log('Utils initialized');