// EduPath Vue应用主文件

// ===== Vue应用配置 =====
const { createApp } = Vue;

// ===== 创建Vue应用 =====
const app = createApp({
    // ===== 数据状态 =====
    data() {
        return {
            // 应用状态
            isLoading: false,
            currentView: 'welcome',
            user: null,
            isAuthenticated: false,

            // 登录表单
            loginForm: {
                username: '',
                password: ''
            },

            // 注册表单
            registerForm: {
                username: '',
                email: '',
                password: '',
                confirmPassword: ''
            },

            // 仪表盘数据
            dashboardData: {
                stats: {
                    totalEarnings: 0,
                    learningProgress: 0,
                    contributedContent: 0,
                    activeStreaks: 0
                },
                recentActivities: [],
                earningsChart: null,
                progressChart: null
            },

            // 学习中心数据
            learningData: {
                categories: [],
                featuredCapsules: [],
                myCapsules: [],
                searchQuery: '',
                selectedCategory: 'all',
                sortBy: 'latest'
            },

            // 市场数据
            marketData: {
                tokens: [],
                pairs: [],
                myOrders: [],
                selectedPair: null,
                orderBook: { bids: [], asks: [] }
            },

            // 钱包数据
            walletData: {
                balances: [],
                transactions: [],
                transferForm: {
                    toAddress: '',
                    amount: '',
                    tokenSymbol: 'EDP'
                }
            },

            // 贡献数据
            contributeData: {
                submissions: [],
                pendingReviews: [],
                submitForm: {
                    title: '',
                    description: '',
                    content: '',
                    category: '',
                    difficulty: 'beginner',
                    price: 0
                }
            },

            // 模态框状态
            modals: {
                login: false,
                register: false,
                capsuleDetail: false,
                submitContent: false,
                transfer: false
            },

            // 当前选中的项目
            selectedCapsule: null,
            selectedTransaction: null
        };
    },

    // ===== 计算属性 =====
    computed: {
        // 格式化用户名显示
        displayUsername() {
            return this.user ? Utils.capitalize(this.user.username) : '';
        },

        // 用户头像首字母
        userInitial() {
            return this.user ? this.user.username.charAt(0).toUpperCase() : '';
        },

        // 钱包总价值
        totalWalletValue() {
            if (!this.walletData.balances.length) return 0;
            return this.walletData.balances.reduce((total, balance) => {
                return total + (balance.amount * balance.price_usd);
            }, 0);
        },

        // 过滤后的知识胶囊
        filteredCapsules() {
            let capsules = this.learningData.featuredCapsules;

            // 分类过滤
            if (this.learningData.selectedCategory !== 'all') {
                capsules = capsules.filter(capsule =>
                    capsule.category === this.learningData.selectedCategory
                );
            }

            // 搜索过滤
            if (this.learningData.searchQuery) {
                const query = this.learningData.searchQuery.toLowerCase();
                capsules = capsules.filter(capsule =>
                    capsule.title.toLowerCase().includes(query) ||
                    capsule.description.toLowerCase().includes(query)
                );
            }

            // 排序
            const sortBy = this.learningData.sortBy;
            if (sortBy === 'price_low') {
                capsules.sort((a, b) => a.price - b.price);
            } else if (sortBy === 'price_high') {
                capsules.sort((a, b) => b.price - a.price);
            } else if (sortBy === 'rating') {
                capsules.sort((a, b) => b.rating - a.rating);
            } else {
                capsules.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
            }

            return capsules;
        }
    },

    // ===== 生命周期钩子 =====
    async mounted() {
        await this.initializeApp();
    },

    // ===== 方法 =====
    methods: {
        // ===== 应用初始化 =====
        async initializeApp() {
            try {
                // 检查本地存储中的登录状态
                const token = Utils.getStorage(APP_CONFIG.STORAGE_KEYS.AUTH_TOKEN);
                const userInfo = Utils.getStorage(APP_CONFIG.STORAGE_KEYS.USER_INFO);

                if (token && userInfo) {
                    this.user = userInfo;
                    this.isAuthenticated = true;
                    this.currentView = 'dashboard';

                    // 验证token有效性
                    await this.validateToken();
                }

                // 初始化模拟数据
                this.initializeMockData();

            } catch (error) {
                console.error('App initialization error:', error);
                Utils.showError('应用初始化失败');
            }
        },

        // 验证token有效性
        async validateToken() {
            try {
                const userData = await api.getCurrentUser();
                this.user = userData;
                Utils.setStorage(APP_CONFIG.STORAGE_KEYS.USER_INFO, userData);
            } catch (error) {
                // Token无效，清除登录状态
                this.logout();
            }
        },

        // 初始化模拟数据
        initializeMockData() {
            // 分类数据
            this.learningData.categories = APP_CONFIG.CATEGORIES;

            // 模拟知识胶囊数据
            this.learningData.featuredCapsules = this.generateMockCapsules();

            // 模拟市场数据
            this.marketData.tokens = this.generateMockTokens();

            // 模拟钱包数据
            if (this.isAuthenticated) {
                this.walletData.balances = this.generateMockBalances();
                this.walletData.transactions = this.generateMockTransactions();
            }
        },

        // ===== 认证相关 =====

        // 用户登录
        async login() {
            if (!this.validateLoginForm()) return;

            this.isLoading = true;
            try {
                const response = await api.login(
                    this.loginForm.username,
                    this.loginForm.password
                );

                this.user = response.user;
                this.isAuthenticated = true;
                Utils.setStorage(APP_CONFIG.STORAGE_KEYS.USER_INFO, response.user);

                Utils.showSuccess('登录成功！');
                this.closeModal('login');
                this.switchView('dashboard');

                // 重置表单
                this.resetLoginForm();

                // 加载用户数据
                await this.loadUserData();

            } catch (error) {
                Utils.showError(error.message || '登录失败，请重试');
            } finally {
                this.isLoading = false;
            }
        },

        // 用户注册
        async register() {
            if (!this.validateRegisterForm()) return;

            this.isLoading = true;
            try {
                await api.register({
                    username: this.registerForm.username,
                    email: this.registerForm.email,
                    password: this.registerForm.password
                });

                Utils.showSuccess('注册成功！请使用新账号登录');
                this.closeModal('register');
                this.showModal('login');

                // 重置表单
                this.resetRegisterForm();

            } catch (error) {
                Utils.showError(error.message || '注册失败，请重试');
            } finally {
                this.isLoading = false;
            }
        },

        // 用户登出
        logout() {
            this.user = null;
            this.isAuthenticated = false;
            this.currentView = 'welcome';

            // 清除本地存储
            Utils.removeStorage(APP_CONFIG.STORAGE_KEYS.AUTH_TOKEN);
            Utils.removeStorage(APP_CONFIG.STORAGE_KEYS.USER_INFO);

            Utils.showInfo('已安全退出');
        },

        // ===== 表单验证 =====

        // 验证登录表单
        validateLoginForm() {
            if (!this.loginForm.username.trim()) {
                Utils.showWarning('请输入用户名');
                return false;
            }

            if (!this.loginForm.password) {
                Utils.showWarning('请输入密码');
                return false;
            }

            return true;
        },

        // 验证注册表单
        validateRegisterForm() {
            const { username, email, password, confirmPassword } = this.registerForm;

            if (!Utils.isValidUsername(username)) {
                Utils.showWarning('用户名长度应为3-20个字符，只能包含字母、数字和下划线');
                return false;
            }

            if (!Utils.isValidEmail(email)) {
                Utils.showWarning('请输入有效的邮箱地址');
                return false;
            }

            if (!Utils.isValidPassword(password)) {
                Utils.showWarning('密码长度至少6位');
                return false;
            }

            if (password !== confirmPassword) {
                Utils.showWarning('两次输入的密码不一致');
                return false;
            }

            return true;
        },

        // ===== 数据加载 =====

        // 加载用户数据
        async loadUserData() {
            try {
                // 加载仪表盘数据
                await this.loadDashboardData();

                // 加载钱包数据
                await this.loadWalletData();

                // 加载学习进度
                await this.loadLearningProgress();

            } catch (error) {
                console.error('Failed to load user data:', error);
            }
        },

        // 加载仪表盘数据
        async loadDashboardData() {
            // 模拟API调用
            this.dashboardData.stats = {
                totalEarnings: Utils.randomNumber(100, 1000),
                learningProgress: Utils.randomNumber(30, 95),
                contributedContent: Utils.randomNumber(5, 25),
                activeStreaks: Utils.randomNumber(1, 30)
            };

            // 初始化图表
            await Utils.delay(100);
            this.initializeCharts();
        },

        // 加载钱包数据
        async loadWalletData() {
            if (!this.isAuthenticated) return;

            try {
                // 这里可以调用真实API
                // const walletData = await api.getWallet();
                // this.walletData = walletData;

                // 模拟数据
                this.walletData.balances = this.generateMockBalances();
                this.walletData.transactions = this.generateMockTransactions();

            } catch (error) {
                console.error('Failed to load wallet data:', error);
            }
        },

        // 加载学习进度
        async loadLearningProgress() {
            try {
                // const progress = await api.getLearningProgress();
                // 模拟学习进度数据
            } catch (error) {
                console.error('Failed to load learning progress:', error);
            }
        },

        // ===== 视图切换 =====

        // 切换视图
        switchView(view) {
            this.currentView = view;

            // 根据视图加载相应数据
            switch (view) {
                case 'market':
                    this.loadMarketData();
                    break;
                case 'learn':
                    this.loadLearningData();
                    break;
                case 'contribute':
                    this.loadContributeData();
                    break;
            }
        },

        // ===== 模态框管理 =====

        // 显示模态框
        showModal(modalName) {
            this.modals[modalName] = true;
            document.body.style.overflow = 'hidden';
        },

        // 关闭模态框
        closeModal(modalName) {
            this.modals[modalName] = false;
            document.body.style.overflow = 'auto';
        },

        // 关闭所有模态框
        closeAllModals() {
            Object.keys(this.modals).forEach(key => {
                this.modals[key] = false;
            });
            document.body.style.overflow = 'auto';
        },

        // ===== 表单重置 =====

        // 重置登录表单
        resetLoginForm() {
            this.loginForm = {
                username: '',
                password: ''
            };
        },

        // 重置注册表单
        resetRegisterForm() {
            this.registerForm = {
                username: '',
                email: '',
                password: '',
                confirmPassword: ''
            };
        },

        // ===== 图表初始化 =====

        // 初始化图表
        initializeCharts() {
            this.$nextTick(() => {
                this.initEarningsChart();
                this.initProgressChart();
            });
        },

        // 收益图表
        initEarningsChart() {
            const ctx = document.getElementById('earningsChart');
            if (!ctx) return;

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['1月', '2月', '3月', '4月', '5月', '6月'],
                    datasets: [{
                        label: '收益 (EDP)',
                        data: [65, 89, 80, 81, 156, 155],
                        borderColor: '#fcd535',
                        backgroundColor: 'rgba(252, 213, 53, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#848e9c'
                            }
                        }
                    },
                    scales: {
                        y: {
                            ticks: {
                                color: '#848e9c'
                            },
                            grid: {
                                color: '#2b3139'
                            }
                        },
                        x: {
                            ticks: {
                                color: '#848e9c'
                            },
                            grid: {
                                color: '#2b3139'
                            }
                        }
                    }
                }
            });
        },

        // 进度图表
        initProgressChart() {
            const ctx = document.getElementById('progressChart');
            if (!ctx) return;

            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['已完成', '进行中', '未开始'],
                    datasets: [{
                        data: [65, 25, 10],
                        backgroundColor: ['#02c076', '#fcd535', '#5e6673']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#848e9c'
                            }
                        }
                    }
                }
            });
        },

        // ===== 模拟数据生成 =====

        // 生成模拟知识胶囊
        generateMockCapsules() {
            const titles = [
                'Vue.js 3完整教程',
                'Python数据分析入门',
                'React Hooks深度解析',
                'JavaScript ES6+新特性',
                'Node.js后端开发',
                '机器学习基础',
                'UI/UX设计原理',
                '区块链技术详解'
            ];

            return titles.map((title, index) => ({
                id: index + 1,
                title,
                description: `这是关于${title}的详细教程，适合初学者到进阶者学习。`,
                price: Utils.randomNumber(10, 100),
                rating: (4 + Math.random()).toFixed(1),
                duration: Utils.randomNumber(30, 180),
                level: APP_CONFIG.DIFFICULTY_LEVELS[Utils.randomNumber(0, 3)].id,
                category: APP_CONFIG.CATEGORIES[Utils.randomNumber(0, 7)].id,
                author: `作者${index + 1}`,
                students: Utils.randomNumber(100, 1000),
                created_at: new Date(Date.now() - Utils.randomNumber(1, 30) * 24 * 60 * 60 * 1000),
                image: `https://picsum.photos/300/200?random=${index + 1}`
            }));
        },

        // 生成模拟代币数据
        generateMockTokens() {
            return [
                {
                    symbol: 'EDP',
                    name: 'EduPath Token',
                    price: 1.25,
                    change_24h: 5.67,
                    volume_24h: 15420000,
                    market_cap: 45600000
                },
                {
                    symbol: 'KCT',
                    name: 'Knowledge Token',
                    price: 0.85,
                    change_24h: -2.34,
                    volume_24h: 8950000,
                    market_cap: 21300000
                }
            ];
        },

        // 生成模拟余额数据
        generateMockBalances() {
            return [
                {
                    token_symbol: 'EDP',
                    token_name: 'EduPath Token',
                    amount: 1250.75,
                    price_usd: 1.25,
                    value_usd: 1563.44
                },
                {
                    token_symbol: 'KCT',
                    token_name: 'Knowledge Token',
                    amount: 3420.50,
                    price_usd: 0.85,
                    value_usd: 2907.43
                }
            ];
        },

        // 生成模拟交易记录
        generateMockTransactions() {
            const types = ['receive', 'send', 'purchase', 'reward'];
            const tokens = ['EDP', 'KCT'];

            return Array.from({ length: 10 }, (_, index) => ({
                id: index + 1,
                type: types[Utils.randomNumber(0, 3)],
                token_symbol: tokens[Utils.randomNumber(0, 1)],
                amount: Utils.randomNumber(10, 500),
                from_address: `0x${Utils.randomString(40)}`,
                to_address: `0x${Utils.randomString(40)}`,
                timestamp: new Date(Date.now() - Utils.randomNumber(1, 7) * 24 * 60 * 60 * 1000),
                status: 'confirmed',
                tx_hash: `0x${Utils.randomString(64)}`
            }));
        },

        // ===== 市场相关方法 =====

        async loadMarketData() {
            // 加载市场数据
            try {
                // const marketData = await api.getMarketData();
                // 模拟市场数据
                this.marketData.tokens = this.generateMockTokens();
            } catch (error) {
                console.error('Failed to load market data:', error);
            }
        },

        // ===== 学习相关方法 =====

        async loadLearningData() {
            // 加载学习数据
            try {
                // const capsules = await api.getCapsules();
                // 模拟数据已在初始化时生成
            } catch (error) {
                console.error('Failed to load learning data:', error);
            }
        },

        // 查看知识胶囊详情
        viewCapsuleDetail(capsule) {
            this.selectedCapsule = capsule;
            this.showModal('capsuleDetail');
        },

        // 购买知识胶囊
        async purchaseCapsule(capsuleId) {
            try {
                // await api.purchaseCapsule(capsuleId);
                Utils.showSuccess('购买成功！');
                this.closeModal('capsuleDetail');
            } catch (error) {
                Utils.showError('购买失败，请重试');
            }
        },

        // ===== 贡献相关方法 =====

        async loadContributeData() {
            // 加载贡献数据
            try {
                // const contributions = await api.getContributions();
                // 模拟数据
            } catch (error) {
                console.error('Failed to load contribution data:', error);
            }
        },

        // 提交内容贡献
        async submitContribution() {
            try {
                // const response = await api.submitContribution(this.contributeData.submitForm);
                Utils.showSuccess('提交成功，等待审核！');
                this.closeModal('submitContent');
            } catch (error) {
                Utils.showError('提交失败，请重试');
            }
        },

        // ===== 钱包相关方法 =====

        // 转账
        async transfer() {
            const { toAddress, amount, tokenSymbol } = this.walletData.transferForm;

            if (!toAddress || !amount) {
                Utils.showWarning('请填写完整的转账信息');
                return;
            }

            try {
                // await api.transfer(toAddress, amount, tokenSymbol);
                Utils.showSuccess('转账成功！');
                this.closeModal('transfer');

                // 重置表单
                this.walletData.transferForm = {
                    toAddress: '',
                    amount: '',
                    tokenSymbol: 'EDP'
                };

                // 刷新钱包数据
                await this.loadWalletData();

            } catch (error) {
                Utils.showError('转账失败，请重试');
            }
        }
    }
});

// ===== 挂载应用 =====
app.mount('#app');