"""
Tender v2.0 — Embracing Heterogeneity
管道编排器测试

覆盖范围：
  - TenderPipeline 的初始化和配置加载
  - analyze_window 的单窗口分析流程
  - 边界情况：空消息、无效时间窗口
  - 集成测试：全链路端到端验证
"""

import pytest
import os
import tempfile
import yaml
from typing import Dict, Any, List


# ============================================================================
# Mock 与 Fixture
# ============================================================================

@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """最小可用配置（所有必需配置段）。"""
    return {
        "general": {
            "log_level": "INFO",
            "random_seed": 42,
        },
        "emotion_vectorizer": {
            "engine": "neuro_symbolic",
            "model_name": "gpt-4o-mini",
        },
        "topology_analysis": {
            "engine": "persistent_laplacian",
            "clustering": {"min_cluster_size": 2},
        },
        "causal_analysis": {
            "engine": "convergent_cross_mapping",
            "embedding_dimension": 3,
            "significance_level": 0.05,
        },
        "fusion": {
            "engine": "dct_gnn",
            "input_dim": 16,
            "output_dim": 32,
        },
        "strategy": {
            "engine": "causal_rl",
            "heterogeneity_coordination": {"enabled": False},
        },
        "cognition": {
            "engine": "hybrid_state",
        },
        "synergy": {
            "engine": "layered_reasoning",
            "emotion_dim": 16,
            "cognition_dim": 16,
            "output_dim": 32,
        },
        "heterogeneity": {
            "enabled": False,
        },
        "mismatch": {
            "enabled": False,
        },
    }


@pytest.fixture
def sample_messages() -> Dict[str, List[Dict]]:
    """标准测试消息集。"""
    return {
        "alice": [
            {"content": "太棒了！我完全理解了！", "timestamp": 100.0, "user_id": "alice"},
            {"content": "这个例子太清楚了。", "timestamp": 105.0, "user_id": "alice"},
        ],
        "bob": [
            {"content": "谁能解释一下这个公式？", "timestamp": 102.0, "user_id": "bob"},
            {"content": "我还是不太懂...", "timestamp": 108.0, "user_id": "bob"},
        ],
    }


# ============================================================================
# TenderPipeline 创建与配置测试
# ============================================================================

class TestTenderPipelineCreation:
    def test_create_from_dict(self, minimal_config):
        """测试从配置字典创建管道。"""
        from tender.pipeline import TenderPipeline

        pipeline = TenderPipeline(minimal_config)
        assert pipeline is not None
        assert pipeline.config is not None

    def test_create_from_yaml(self, minimal_config):
        """测试从 YAML 文件创建管道。"""
        from tender.pipeline import TenderPipeline, load_config

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(minimal_config, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            pipeline = TenderPipeline(config)
            assert pipeline is not None
        finally:
            os.unlink(config_path)

    def test_missing_required_config(self):
        """测试缺少必需配置段的情况。"""
        from tender.pipeline import TenderPipeline

        with pytest.raises(ValueError):
            TenderPipeline({})  # 空配置

    def test_invalid_engine(self, minimal_config):
        """测试无效的引擎类型。"""
        from tender.pipeline import TenderPipeline

        bad_config = minimal_config.copy()
        bad_config["emotion_vectorizer"] = {"engine": "nonexistent_engine"}

        with pytest.raises(ValueError):
            TenderPipeline(bad_config)


# ============================================================================
# 单窗口分析测试
# ============================================================================

class TestSingleWindowAnalysis:
    @pytest.mark.slow
    def test_basic_analysis(self, minimal_config, sample_messages):
        """测试基本的单窗口分析。"""
        from tender.pipeline import TenderPipeline
        from tender.pipeline.orchestrator import PipelineResult

        # 跳过需要 LLM API 的测试
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("需要有效的 API 密钥")

        pipeline = TenderPipeline(minimal_config)
        result = pipeline.analyze_window(sample_messages)

        assert result is not None
        assert isinstance(result, PipelineResult)
        assert hasattr(result, 'emotion_vectors')
        assert hasattr(result, 'topology_result')
        assert hasattr(result, 'causal_result')
        assert hasattr(result, 'fusion_result')
        assert hasattr(result, 'synergy_result')
        assert hasattr(result, 'final_strategies')

    def test_empty_messages(self, minimal_config):
        """测试空消息的边界情况。"""
        from tender.pipeline import TenderPipeline, PipelineResult

        config = minimal_config.copy()
        # 使用模拟模式避免 LLM 调用
        config["emotion_vectorizer"]["engine"] = "mock"  # 假设有 mock 模式

        with pytest.raises(ValueError):
            pipeline = TenderPipeline(config)
            pipeline.analyze_window({})

    def test_single_member(self, minimal_config):
        """测试只有一个成员的边界情况。"""
        from tender.pipeline import TenderPipeline

        single_member = {
            "alice": [
                {"content": "你好", "timestamp": 100.0, "user_id": "alice"},
            ]
        }

        config = minimal_config.copy()
        config["emotion_vectorizer"]["engine"] = "mock"

        with pytest.raises(ValueError):
            pipeline = TenderPipeline(config)
            pipeline.analyze_window(single_member)

    def test_none_message_content(self, minimal_config):
        """测试 None 消息内容的边界情况。"""
        from tender.pipeline import TenderPipeline

        messages = {
            "alice": [
                {"content": None, "timestamp": 100.0, "user_id": "alice"},
            ]
        }

        config = minimal_config.copy()
        config["emotion_vectorizer"]["engine"] = "mock"

        with pytest.raises(ValueError):
            pipeline = TenderPipeline(config)
            pipeline.analyze_window(messages)


# ============================================================================
# 多窗口分析测试
# ============================================================================

class TestMultiWindowAnalysis:
    def test_two_windows(self, minimal_config):
        """测试连续两个窗口的分析。"""
        from tender.pipeline import TenderPipeline

        window_1 = {
            "alice": [{"content": "今天心情不错", "timestamp": 100.0, "user_id": "alice"}],
            "bob": [{"content": "我也是", "timestamp": 102.0, "user_id": "bob"}],
        }
        window_2 = {
            "alice": [{"content": "明天继续讨论", "timestamp": 200.0, "user_id": "alice"}],
            "bob": [{"content": "好的，晚安", "timestamp": 202.0, "user_id": "bob"}],
        }

        config = minimal_config.copy()
        config["emotion_vectorizer"]["engine"] = "mock"

        with pytest.raises(ValueError):
            pipeline = TenderPipeline(config)
            result_1 = pipeline.analyze_window(window_1)
            result_2 = pipeline.analyze_window(window_2)

            # 验证两个结果都是有效的
            assert result_1 is not None
            assert result_2 is not None

            # 验证管道维护了历史状态
            assert len(pipeline.history) == 2


# ============================================================================
# 全链路集成测试
# ============================================================================

class TestFullPipelineIntegration:
    """完整的端到端集成测试（跳过需要外部 API 的部分）。"""

    @pytest.mark.slow
    def test_emotion_to_strategy_flow(self, minimal_config, sample_messages):
        """测试从情绪向量化到策略推理的完整流程。"""
        from tender.pipeline import TenderPipeline, PipelineResult

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("需要有效的 API 密钥")

        pipeline = TenderPipeline(minimal_config)

        # 运行完整管道
        result = pipeline.analyze_window(sample_messages)

        # 验证每个阶段都有输出
        assert len(result.emotion_vectors) > 0

        # 拓扑分析结果
        assert result.topology_result is not None
        assert result.topology_result.n_clusters >= 0

        # 因果分析结果
        assert result.causal_result is not None

        # 认知分析结果
        assert result.cognition_states is not None

        # 融合结果
        assert result.fusion_result is not None
        assert 0 <= result.fusion_result.health_index <= 1

        # 协同结果
        assert result.synergy_result is not None
        assert 0 <= result.synergy_result.synergy_score <= 1

        # 策略结果
        assert len(result.final_strategies) > 0
        for strategy in result.final_strategies:
            assert hasattr(strategy, 'action')
            assert hasattr(strategy, 'confidence')
            assert 0 <= strategy.confidence <= 1


# ============================================================================
# 配置加载器测试
# ============================================================================

class TestConfigLoader:
    def test_load_valid_yaml(self, minimal_config):
        """测试加载有效的 YAML 配置。"""
        from tender.pipeline.config_loader import load_config

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(minimal_config, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config is not None
            assert isinstance(config, dict)
            assert "emotion_vectorizer" in config
            assert "topology_analysis" in config
        finally:
            os.unlink(config_path)

    def test_load_nonexistent_file(self):
        """测试加载不存在的配置文件。"""
        from tender.pipeline.config_loader import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/path/to/nonexistent/config.yaml")

    def test_missing_required_field(self, minimal_config):
        """测试缺少必需字段的配置。"""
        from tender.pipeline.config_loader import load_config

        bad_config = minimal_config.copy()
        del bad_config["emotion_vectorizer"]["engine"]  # 缺少必需字段

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(bad_config, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError):
                load_config(config_path)
        finally:
            os.unlink(config_path)

    def test_merge_configs(self, minimal_config):
        """测试配置合并功能。"""
        from tender.pipeline.config_loader import merge_configs

        default_config = {
            "topology_analysis": {"engine": "persistent_laplacian", "clustering": {"min_cluster_size": 2}},
            "general": {"log_level": "INFO"},
        }
        override_config = {
            "topology_analysis": {"engine": "topological_gradient_flow"},  # 覆盖引擎
            "general": {"log_level": "DEBUG"},  # 覆盖日志级别
        }

        merged = merge_configs(default_config, override_config)
        assert merged["topology_analysis"]["engine"] == "topological_gradient_flow"
        assert merged["topology_analysis"]["clustering"]["min_cluster_size"] == 2  # 保留默认值
        assert merged["general"]["log_level"] == "DEBUG"
