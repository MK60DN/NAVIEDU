"""
Tender 框架集成测试

测试从消息输入到策略决策的端到端流程。
"""

import pytest
import numpy as np
from typing import Dict, List, Any

from tender.pipeline.config_loader import load_config
from tender.pipeline.orchestrator import TenderPipeline


# ============================================================================
# 测试数据
# ============================================================================

# 模拟群聊消息数据
SAMPLE_CHAT_MESSAGES: Dict[str, List[Dict[str, Any]]] = {
    "user_Alice": [
        {"text": "大家好！今天项目上线非常顺利！", "timestamp": 1717200000.0},
        {"text": "大家辛苦了，真的很棒！", "timestamp": 1717200060.0},
        {"text": "我们团队最棒了！", "timestamp": 1717200120.0},
    ],
    "user_Bob": [
        {"text": "我觉得有些地方需要改进", "timestamp": 1717200010.0},
        {"text": "不过总体来说不错", "timestamp": 1717200070.0},
        {"text": "加油！", "timestamp": 1717200130.0},
    ],
    "user_Charlie": [
        {"text": "太好了！终于上线了！🎉", "timestamp": 1717200020.0},
        {"text": "我爱我们的团队！", "timestamp": 1717200080.0},
        {"text": "今晚庆祝一下！", "timestamp": 1717200140.0},
    ],
    "user_Diana": [
        {"text": "测试过程中发现了一些bug", "timestamp": 1717200030.0},
        {"text": "不过已经修复了", "timestamp": 1717200090.0},
        {"text": "大家辛苦了", "timestamp": 1717200150.0},
    ],
    "user_Eve": [
        {"text": "我有个想法", "timestamp": 1717200040.0},
        {"text": "下次我们可以尝试新的方法", "timestamp": 1717200100.0},
        {"text": "大家觉得怎么样？", "timestamp": 1717200160.0},
    ],
}

SAMPLE_CONFIG = {
    "emotion_vectorizer": {
        "engine": "neuro_symbolic",
        "model_name": "deepseek",
        "api_url": "https://api.deepseek.com",
        "api_key": "sk-test-key-for-ci",
        "batch_size": 16,
        "temperature": 0.1,
        "rule_weight": 0.7,
        "use_causal_chain": False,
    },
    "topology_analysis": {
        "engine": "persistent_laplacian",
        "normalize": True,
        "min_cluster_size": 2,
        "min_samples": 1,
        "h1_threshold_ratio": 0.3,
        "standardize": True,
    },
    "causal_analysis": {
        "engine": "convergent_cross_mapping",
        "embedding_dimension": 5,
        "tau": 1,
        "lib_size_ratio": 0.8,
        "significance_level": 0.05,
        "emotion_dimension": "composite",
        "max_lag": 5,
        "num_lib_sizes": 10,
        "seed": 42,
    },
    "fusion": {
        "engine": "dct_gnn",
        "spatial_feature_dim": 8,
        "temporal_feature_dim": 8,
        "output_dim": 16,
        "forecast_horizon": 1,
        "forecast_method": "gcn",
    },
    "strategy": {
        "engine": "causal_rl",
        "state_dim": 16,
        "action_dim": 5,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "buffer_size": 10000,
        "batch_size": 64,
        "target_update": 10,
        "enable_reconciliation": False,
    },
    "pipeline": {
        "window_size": 3600.0,
        "min_messages_per_member": 1,
        "max_members": 100,
        "enable_history": True,
        "max_history_window": 20,
        "enable_progress_bar": False,
    },
}


# ============================================================================
# 测试类
# ============================================================================


class TestPipelineInitialization:
    """管道初始化测试"""

    def test_init_from_config_dict(self):
        """测试：从配置字典初始化"""
        pipeline = TenderPipeline(SAMPLE_CONFIG)
        assert pipeline is not None

    def test_init_from_yaml(self):
        """测试：从 YAML 文件初始化"""
        # 注意：需要先创建测试用的 YAML 文件
        # 这里仅验证函数存在
        try:
            config = load_config("config.yaml")
            assert config is not None
        except FileNotFoundError:
            pytest.skip("config.yaml 不存在，跳过此测试")

    def test_pipeline_info_after_init(self):
        """测试：初始化后获取管道信息"""
        pipeline = TenderPipeline(SAMPLE_CONFIG)
        info = pipeline.get_info()

        assert "name" in info
        assert "version" in info
        assert "vectorizer" in info
        assert "topology_analyzer" in info
        assert "causal_analyzer" in info
        assert "fuser" in info
        assert "strategy_engine" in info


class TestPipelineSingleWindowAnalysis:
    """单窗口分析测试"""

    @pytest.mark.skip(reason="需要有效的 API 密钥才能运行完整流程")
    def test_analyze_single_window_basic(self):
        """测试：基本单窗口分析流程"""
        pipeline = TenderPipeline(SAMPLE_CONFIG)
        decision = pipeline.analyze_window(
            member_messages=SAMPLE_CHAT_MESSAGES,
            window_start=1717200000.0,
            window_end=1717203600.0,
        )

        # 验证输出结构
        assert decision is not None
        assert hasattr(decision, "risk_level")
        assert hasattr(decision, "risk_score")
        assert hasattr(decision, "reasoning")
        assert hasattr(decision, "timestamp")

        # 验证风险评分范围
        assert 0.0 <= decision.risk_score <= 1.0

    @pytest.mark.skip(reason="需要有效的 API 密钥才能运行完整流程")
    def test_analyze_single_window_with_vectorization(self):
        """测试：向量化结果验证"""
        pipeline = TenderPipeline(SAMPLE_CONFIG)

        # 先进行向量化
        emotion_vectors = pipeline.vectorizer.vectorize(
            messages=SAMPLE_CHAT_MESSAGES,
        )

        assert len(emotion_vectors) == 5  # 5个成员

        for member_id, vector in emotion_vectors.items():
            assert len(vector) == 3  # valence, arousal, focus
            assert -1.0 <= vector[0] <= 1.0  # valence
            assert 0.0 <= vector[1] <= 1.0   # arousal
            assert 0.0 <= vector[2] <= 1.0   # focus

    def test_invalid_window_timestamps(self):
        """测试：无效的时间窗口"""
        pipeline = TenderPipeline(SAMPLE_CONFIG)

        with pytest.raises(ValueError):
            pipeline.analyze_window(
                member_messages=SAMPLE_CHAT_MESSAGES,
                window_start=1717203600.0,  # 开始时间晚于结束时间
                window_end=1717200000.0,
            )

    def test_empty_messages(self):
        """测试：空消息"""
        pipeline = TenderPipeline(SAMPLE_CONFIG)

        with pytest.raises(ValueError):
            pipeline.analyze_window(
                member_messages={},
                window_start=1717200000.0,
                window_end=1717203600.0,
            )

    @pytest.mark.skip(reason="需要有效的 API 密钥才能运行完整流程")
    def test_history_tracking(self):
        """测试：历史记录跟踪"""
        pipeline = TenderPipeline(SAMPLE_CONFIG)

        # 第一次分析
        decision1 = pipeline.analyze_window(
            member_messages=SAMPLE_CHAT_MESSAGES,
            window_start=1717200000.0,
            window_end=1717203600.0,
        )

        history = pipeline.get_history()
        assert len(history) == 1
        assert history["decision"] is decision1

        # 第二次分析（使用不同的消息，模拟时间推移）
        second_messages = {
            "user_Alice": [
                {"text": "今天大家情绪不高啊", "timestamp": 1717203600.0},
            ],
        }
        decision2 = pipeline.analyze_window(
            member_messages=second_messages,
            window_start=1717203600.0,
            window_end=1717207200.0,
        )

        history = pipeline.get_history()
        assert len(history) == 2


class TestPipelineMultiWindowAnalysis:
    """多窗口分析测试"""

    @pytest.mark.skip(reason="需要有效的 API 密钥才能运行完整流程")
    def test_analyze_multi_windows_sequential(self):
        """测试：连续多个窗口分析"""
        pipeline = TenderPipeline(SAMPLE_CONFIG)
        window_duration = 3600.0  # 1小时

        # 模拟6小时的数据（6个窗口）
        for i in range(3):  # 3个窗口
            window_start = 1717200000.0 + i * window_duration
            window_end = window_start + window_duration

            decision = pipeline.analyze_window(
                member_messages=SAMPLE_CHAT_MESSAGES,
                window_start=window_start,
                window_end=window_end,
            )

            assert decision is not None

    def test_analysis_with_causal_rl_default(self):
        """测试：使用默认因果强化学习引擎"""
        config = SAMPLE_CONFIG.copy()
        config["strategy"]["engine"] = "causal_rl"

        pipeline = TenderPipeline(config)
        assert pipeline is not None


class TestConfigLoader:
    """配置加载器测试"""

    def test_load_valid_yaml(self):
        """测试：加载有效 YAML"""
        try:
            config = load_config("config.yaml")
            assert "emotion_vectorizer" in config
            assert "topology_analysis" in config
            assert "causal_analysis" in config
            assert "fusion" in config
            assert "strategy" in config
            assert "pipeline" in config
        except FileNotFoundError:
            pytest.skip("config.yaml 不存在，跳过此测试")

    def test_load_invalid_yaml_path(self):
        """测试：加载无效路径"""
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_config.yaml")
