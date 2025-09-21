// EduPath API封装文件

// ===== API客户端类 =====
class ApiClient {
    constructor() {
        this.baseURL = APP_CONFIG.API_BASE_URL;
        this.timeout = APP_CONFIG.API_TIMEOUT;
        this.token = this.getAuthToken();
    }

    // 获取认证token
    getAuthToken() {
        return localStorage.getItem(APP_CONFIG.STORAGE_KEYS.AUTH_TOKEN);
    }

    // 设置认证token
    setAuthToken(token) {
        this.token = token;
        if (token) {
            localStorage.setItem(APP_CONFIG.STORAGE_KEYS.AUTH_TOKEN, token);
        } else {
            localStorage.removeItem(APP_CONFIG.STORAGE_KEYS.AUTH_TOKEN);
        }
    }

    // 获取请求头
    getHeaders(contentType = 'application/json') {
        const headers = {
            'Content-Type': contentType,
        };

        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }

        return headers;
    }

    // 处理响应
    async handleResponse(response) {
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new ApiError(response.status, error.detail || response.statusText, error);
        }

        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        }

        return await response.text();
    }

    // 基础请求方法
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            timeout: this.timeout,
            headers: this.getHeaders(options.contentType),
            ...options
        };

        if (APP_CONFIG.DEBUG) {
            console.log(`API Request: ${config.method || 'GET'} ${url}`, config);
        }

        try {
            const response = await fetch(url, config);
            const result = await this.handleResponse(response);

            if (APP_CONFIG.DEBUG) {
                console.log(`API Response: ${config.method || 'GET'} ${url}`, result);
            }

            return result;
        } catch (error) {
            if (APP_CONFIG.DEBUG) {
                console.error(`API Error: ${config.method || 'GET'} ${url}`, error);
            }

            // 处理认证失败
            if (error.status === 401) {
                this.setAuthToken(null);
                window.location.reload();
            }

            throw error;
        }
    }

    // GET请求
    async get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const url = queryString ? `${endpoint}?${queryString}` : endpoint;

        return this.request(url, {
            method: 'GET'
        });
    }

    // POST请求
    async post(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    // PUT请求
    async put(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    // DELETE请求
    async delete(endpoint) {
        return this.request(endpoint, {
            method: 'DELETE'
        });
    }

    // 文件上传
    async upload(endpoint, file, additionalData = {}) {
        const formData = new FormData();
        formData.append('file', file);

        // 添加额外数据
        Object.keys(additionalData).forEach(key => {
            formData.append(key, additionalData[key]);
        });

        return this.request(endpoint, {
            method: 'POST',
            body: formData,
            contentType: undefined // 让浏览器自动设置multipart/form-data
        });
    }
}

// ===== API错误类 =====
class ApiError extends Error {
    constructor(status, message, data = {}) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.data = data;
    }
}

// ===== API服务类 =====
class ApiService {
    constructor() {
        this.client = new ApiClient();
    }

    // ===== 认证相关API =====

    // 用户登录
    async login(username, password) {
        const response = await this.client.post('/auth/login', {
            username,
            password
        });

        if (response.access_token) {
            this.client.setAuthToken(response.access_token);
        }

        return response;
    }

    // 用户注册
    async register(userData) {
        return await this.client.post('/auth/register', userData);
    }

    // 用户登出
    async logout() {
        try {
            await this.client.post('/auth/logout');
        } finally {
            this.client.setAuthToken(null);
        }
    }

    // 刷新token
    async refreshToken() {
        const response = await this.client.post('/auth/refresh');
        if (response.access_token) {
            this.client.setAuthToken(response.access_token);
        }
        return response;
    }

    // 获取当前用户信息
    async getCurrentUser() {
        return await this.client.get('/auth/me');
    }

    // ===== 用户相关API =====

    // 获取用户列表
    async getUsers(params = {}) {
        return await this.client.get('/users', params);
    }

    // 获取用户详情
    async getUser(userId) {
        return await this.client.get(`/users/${userId}`);
    }

    // 更新用户信息
    async updateUser(userId, userData) {
        return await this.client.put(`/users/${userId}`, userData);
    }

    // 删除用户
    async deleteUser(userId) {
        return await this.client.delete(`/users/${userId}`);
    }

    // ===== 知识胶囊相关API =====

    // 获取知识胶囊列表
    async getCapsules(params = {}) {
        return await this.client.get('/capsules', params);
    }

    // 获取知识胶囊详情
    async getCapsule(capsuleId) {
        return await this.client.get(`/capsules/${capsuleId}`);
    }

    // 创建知识胶囊
    async createCapsule(capsuleData) {
        return await this.client.post('/capsules', capsuleData);
    }

    // 更新知识胶囊
    async updateCapsule(capsuleId, capsuleData) {
        return await this.client.put(`/capsules/${capsuleId}`, capsuleData);
    }

    // 删除知识胶囊
    async deleteCapsule(capsuleId) {
        return await this.client.delete(`/capsules/${capsuleId}`);
    }

    // 购买知识胶囊
    async purchaseCapsule(capsuleId) {
        return await this.client.post(`/capsules/${capsuleId}/purchase`);
    }

    // 评价知识胶囊
    async rateCapsule(capsuleId, rating, comment) {
        return await this.client.post(`/capsules/${capsuleId}/rate`, {
            rating,
            comment
        });
    }

    // ===== 代币相关API =====

    // 获取代币列表
    async getTokens() {
        return await this.client.get('/tokens');
    }

    // 获取代币详情
    async getToken(tokenId) {
        return await this.client.get(`/tokens/${tokenId}`);
    }

    // 获取代币价格历史
    async getTokenPriceHistory(tokenId, timeframe = '24h') {
        return await this.client.get(`/tokens/${tokenId}/price-history`, {
            timeframe
        });
    }

    // ===== 钱包相关API =====

    // 获取钱包信息
    async getWallet() {
        return await this.client.get('/wallet');
    }

    // 获取钱包余额
    async getWalletBalance() {
        return await this.client.get('/wallet/balance');
    }

    // 获取交易历史
    async getTransactionHistory(params = {}) {
        return await this.client.get('/wallet/transactions', params);
    }

    // 转账
    async transfer(toAddress, amount, tokenSymbol) {
        return await this.client.post('/wallet/transfer', {
            to_address: toAddress,
            amount,
            token_symbol: tokenSymbol
        });
    }

    // ===== 市场相关API =====

    // 获取市场数据
    async getMarketData() {
        return await this.client.get('/market/overview');
    }

    // 获取交易对列表
    async getTradingPairs() {
        return await this.client.get('/market/pairs');
    }

    // 获取订单簿
    async getOrderBook(symbol) {
        return await this.client.get(`/market/orderbook/${symbol}`);
    }

    // 创建订单
    async createOrder(orderData) {
        return await this.client.post('/market/orders', orderData);
    }

    // 取消订单
    async cancelOrder(orderId) {
        return await this.client.delete(`/market/orders/${orderId}`);
    }

    // ===== 贡献相关API =====

    // 提交内容贡献
    async submitContribution(contributionData) {
        return await this.client.post('/contributions', contributionData);
    }

    // 获取贡献列表
    async getContributions(params = {}) {
        return await this.client.get('/contributions', params);
    }

    // 审核贡献
    async reviewContribution(contributionId, action, comment) {
        return await this.client.post(`/contributions/${contributionId}/review`, {
            action,
            comment
        });
    }

    // ===== 学习相关API =====

    // 获取学习进度
    async getLearningProgress() {
        return await this.client.get('/learning/progress');
    }

    // 更新学习进度
    async updateLearningProgress(capsuleId, progress) {
        return await this.client.post('/learning/progress', {
            capsule_id: capsuleId,
            progress
        });
    }

    // 获取推荐内容
    async getRecommendations(params = {}) {
        return await this.client.get('/learning/recommendations', params);
    }

    // ===== 管理员相关API =====

    // 获取系统统计
    async getSystemStats() {
        return await this.client.get('/admin/stats');
    }

    // 获取用户管理列表
    async getAdminUsers(params = {}) {
        return await this.client.get('/admin/users', params);
    }

    // 获取内容管理列表
    async getAdminContent(params = {}) {
        return await this.client.get('/admin/content', params);
    }

    // ===== 文件上传API =====

    // 上传头像
    async uploadAvatar(file) {
        return await this.client.upload('/upload/avatar', file);
    }

    // 上传内容文件
    async uploadContent(file, type) {
        return await this.client.upload('/upload/content', file, { type });
    }

    // ===== 搜索API =====

    // 搜索内容
    async search(query, filters = {}) {
        return await this.client.get('/search', {
            q: query,
            ...filters
        });
    }

    // 获取搜索建议
    async getSearchSuggestions(query) {
        return await this.client.get('/search/suggestions', { q: query });
    }

    // ===== 系统API =====

    // 健康检查
    async healthCheck() {
        return await this.client.get('/system/health');
    }

    // 获取系统版本
    async getSystemVersion() {
        return await this.client.get('/system/version');
    }
}

// ===== 创建全局API实例 =====
const api = new ApiService();

// ===== 导出API =====
window.api = api;
window.ApiError = ApiError;

// ===== API响应拦截器 =====
const setupApiInterceptors = () => {
    // 可以在这里添加全局的请求/响应拦截逻辑
    console.log('API interceptors initialized');
};

// 初始化API拦截器
setupApiInterceptors();