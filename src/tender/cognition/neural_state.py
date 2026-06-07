""" 基于神经网络的隐状态分析引擎——认知状态分析模块

该模块实现了基于轻量级神经网络的认知状态分析策略（Strategy 4）。
与前三种策略不同，它不依赖预设的知识图谱或显式的行为规则，
而是使用预训练的神经网络模型，直接从成员的文本消息中提取认知状态的隐表示。

核心思想：
- 成员的认知状态可以通过其语言表达中的深层语义特征来推断
- 例如：困惑的语气、理解时的知识整合表述、疲劳时的简短回应等
- 神经网络可以自动学习这些从文本到认知状态的映射
- 预训练模型（如 Sentence-BERT、MiniLM 等）提供了良好的语义基础

工作流程：
1. 使用预训练的语言模型将消息文本编码为语义向量
2. 通过轻量级的认知状态预测头（全连接网络）从语义向量中提取认知指标
3. 输出包括认知负荷、理解水平、注意力得分等连续指标
4. 使用 softmax 分类头输出认知阶段概率分布

适用场景：
- 大规模的文本数据，有足够的数据进行模型微调
- 需要端到端训练的认知分析流水线
- 作为知识图谱分析和行为分析的上游语义特征提取器

学术基础：
- 基于 Transformer 的语义表示 (Vaswani et al., 2017)
- 情感-认知的神经计算模型 (Ochsner & Phelps, 2007)
- 文本的认知特征提取 (Pennington et al., 2014)

注意事项：
- 该引擎在 "no_pretrained" 模式下会使用基于统计特征的启发式方法
- 完整的预训练模型需要在外部训练后通过 config.neural_model_path 加载
- 当前实现提供了一个可运行的基础版本，适合原型开发和测试
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from tender.cognition.base import (
    BaseCognitionAnalyzer,
    CognitionState,
    BehaviorProfile,
    KnowledgeGraphConfig,
    CognitivePhase,
    EngagementType,
)


class NeuralStateAnalyzer(BaseCognitionAnalyzer):
    """基于神经网络的隐状态分析引擎

    使用神经网络从文本中提取认知状态。支持预训练模型和启发式回退。

    两种运行模式：
    1. 预训练模式（neural_use_pretrained=True）：加载外部训练好的模型
    2. 启发式模式（默认）：基于统计特征的轻量级分析，无需外部模型

    Args:
        config: 配置字典，包含以下字段：
            - feature_dim: 认知特征维度（默认 16）
            - output_dim: 输出维度（默认 16）
            - neural_model_path: 预训练模型路径（可选）
            - neural_hidden_dim: 神经网络隐藏层维度（默认 64）
            - neural_dropout: Dropout 比率（默认 0.2）
            - neural_use_pretrained: 是否使用预训练模型（默认 False）
            - batch_size: 批处理大小（默认 32）
            - max_seq_length: 最大序列长度（默认 128）
            - aggregation_strategy: 群体聚合策略（默认 "weighted"）
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化神经网络隐状态分析引擎

        Args:
            config: 配置字典
        """
        # 核心维度参数
        self.feature_dim = config.get("feature_dim", 16)
        self.output_dim = config.get("output_dim", 16)

        # 神经网络参数
        self.neural_model_path = config.get("neural_model_path", None)
        self.neural_hidden_dim = config.get("neural_hidden_dim", 64)
        self.neural_dropout = config.get("neural_dropout", 0.2)
        self.neural_use_pretrained = config.get("neural_use_pretrained", False)
        self.batch_size = config.get("batch_size", 32)
        self.max_seq_length = config.get("max_seq_length", 128)

        # 聚合参数
        self.aggregation_strategy = config.get("aggregation_strategy", "weighted")

        # 内部模型状态
        self._model = None          # 预训练模型（如果加载）
        self._tokenizer = None      # 分词器（如果加载）
        self._is_model_loaded = False

        # 统计特征提取器（启发式模式使用）
        self._statistical_extractor = _StatisticalFeatureExtractor()

        # 尝试加载预训练模型
        if self.neural_use_pretrained and self.neural_model_path:
            self._load_pretrained_model(self.neural_model_path)

        # 记录初始化信息
        self._init_info = (
            f"NeuralStateAnalyzer initialized with "
            f"use_pretrained={self.neural_use_pretrained}, "
            f"model_path={self.neural_model_path}, "
            f"hidden_dim={self.neural_hidden_dim}, "
            f"feature_dim={self.feature_dim}"
        )

    def _load_pretrained_model(self, model_path: str) -> None:
        """加载预训练模型和分词器

        支持加载 Sentence-Transformer、Transformers 或 ONNX 格式的模型。

        Args:
            model_path: 模型文件路径
        """
        try:
            # 尝试加载 Sentence-Transformer 模型
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_path)
                self._is_model_loaded = True
                print(f"Sentence-Transformer 模型加载成功: {model_path}")
                return
            except ImportError:
                pass

            # 尝试加载 Transformers 模型
            try:
                from transformers import AutoModel, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                self._model = AutoModel.from_pretrained(model_path)
                self._is_model_loaded = True
                print(f"Transformers 模型加载成功: {model_path}")
                return
            except ImportError:
                pass

            print(f"警告：无法加载模型，缺少必要的库。"
                  f"请安装 sentence-transformers 或 transformers。"
                  f"将使用启发式模式运行。")

        except Exception as e:
            print(f"警告：加载模型失败: {e}。将使用启发式模式运行。")
            self._is_model_loaded = False

    def analyze(
        self,
        member_messages: Dict[str, List[Dict[str, Any]]],
        knowledge_graph_config: Optional[KnowledgeGraphConfig] = None,
        behavior_profiles: Optional[Dict[str, BehaviorProfile]] = None,
    ) -> Dict[str, CognitionState]:
        """分析所有成员的认知状态（神经网络分析）

        Args:
            member_messages: 成员消息字典
            knowledge_graph_config: 可选的知识图谱配置（此引擎不使用）
            behavior_profiles: 成员行为档案（可选，用于辅助分析）

        Returns:
            Dict[str, CognitionState]: 成员 ID 到认知状态的映射
        """
        # 验证输入
        self.validate_inputs(member_messages)

        # 对每个成员进行分析
        member_states = {}
        for member_id, messages in member_messages.items():
            behavior = behavior_profiles.get(member_id) if behavior_profiles else None
            state = self.analyze_single(
                member_id=member_id,
                messages=messages,
                behavior_profile=behavior,
            )
            member_states[member_id] = state

        # 计算群体状态（如果多个成员）
        if len(member_states) > 1:
            group_state = self.compute_group_state(member_states)
            member_states["__group__"] = group_state

        return member_states

    def analyze_single(
        self,
        member_id: str,
        messages: List[Dict[str, Any]],
        knowledge_graph_config: Optional[KnowledgeGraphConfig] = None,
        behavior_profile: Optional[BehaviorProfile] = None,
    ) -> CognitionState:
        """分析单个成员的认知状态

        Args:
            member_id: 成员唯一标识
            messages: 该成员的消息列表
            knowledge_graph_config: 可选的知识图谱配置（此引擎不使用）
            behavior_profile: 该成员的行为档案（可选，用于辅助分析）

        Returns:
            CognitionState: 该成员的认知状态
        """
        # 步骤1：计算文本语义特征
        text_features = self._compute_text_features(messages)

        # 步骤2：如果提供了行为档案，融合行为特征
        combined_features = text_features
        if behavior_profile is not None:
            behavior_features = self._extract_behavior_features(behavior_profile)
            combined_features = self._fuse_features(text_features, behavior_features)

        # 步骤3：通过神经网络（或启发式）预测认知指标
        predictions = self._predict_cognitive_indicators(
            combined_features, messages, behavior_profile
        )

        # 步骤4：提取各项指标
        cognitive_load = predictions.get("cognitive_load", 0.5)
        understanding_level = predictions.get("understanding_level", 0.5)
        attention_score = predictions.get("attention_score", 0.5)
        confusion_level = predictions.get("confusion_level", 0.3)
        cognitive_flexibility = predictions.get("cognitive_flexibility", 0.5)

        # 步骤5：分类认知阶段
        phase = self._categorize_phase(cognitive_load, understanding_level)

        # 步骤6：计算参与类型
        if behavior_profile:
            question_ratio = behavior_profile.question_count / max(1, behavior_profile.message_count)
            engagement = self._compute_engagement_type(
                len(messages),
                question_ratio,
                behavior_profile.avg_message_length,
            )
        else:
            # 基于消息特征的简单判断
            if len(messages) > 10:
                engagement = EngagementType.FOCUSED
            elif len(messages) > 5:
                engagement = EngagementType.MODERATE
            else:
                engagement = EngagementType.PASSIVE

        # 步骤7：计算置信度
        confidence = self._compute_confidence(text_features, predictions)

        # 步骤8：获取时间戳
        timestamp = max(msg.get("timestamp", 0.0) for msg in messages) if messages else 0.0

        # 构建认知状态
        state = CognitionState(
            member_id=member_id,
            cognitive_load=float(np.clip(cognitive_load, 0.0, 1.0)),
            understanding_level=float(np.clip(understanding_level, 0.0, 1.0)),
            cognitive_phase=phase,
            engagement_type=engagement,
            attention_score=float(np.clip(attention_score, 0.0, 1.0)),
            confusion_level=float(np.clip(confusion_level, 0.0, 1.0)),
            cognitive_flexibility=float(np.clip(cognitive_flexibility, 0.0, 1.0)),
            phase_confidence=float(np.clip(confidence, 0.0, 1.0)),
            source_engine="neural_state",
            timestamp=timestamp,
            knowledge_nodes=[],  # 神经引擎不直接关联知识节点
            raw_embedding=combined_features,
            metadata={
                "text_feature_norm": float(np.linalg.norm(text_features)) if text_features is not None else 0.0,
                "model_loaded": self._is_model_loaded,
                "behavior_features_used": behavior_profile is not None,
                "prediction_details": {k: float(v) for k, v in predictions.items()},
            },
        )

        return state

    def compute_group_state(
        self,
        member_states: Dict[str, CognitionState],
    ) -> CognitionState:
        """从成员状态计算群体认知状态

        Args:
            member_states: 成员状态字典

        Returns:
            CognitionState: 群体的认知状态
        """
        return self._aggregate_states(member_states, self.aggregation_strategy)

    def _compute_text_features(
        self,
        messages: List[Dict[str, Any]],
    ) -> np.ndarray:
        """计算消息的文本特征

        如果加载了预训练模型，使用模型编码；
        否则使用基于统计特征的启发式方法。

        Args:
            messages: 消息列表

        Returns:
            np.ndarray: 文本特征向量
        """
        if not messages:
            return np.zeros(self.feature_dim)

        # 合并所有消息文本
        all_text = " ".join(msg.get("text", "") for msg in messages)

        if not all_text:
            return np.zeros(self.feature_dim)

        if self._is_model_loaded and self._model is not None:
            # 使用预训练模型编码
            return self._encode_with_model(all_text)
        else:
            # 使用统计特征提取器
            return self._statistical_extractor.extract(all_text, self.feature_dim)

    def _encode_with_model(self, text: str) -> np.ndarray:
        """使用预训练模型编码文本

        Args:
            text: 输入文本

        Returns:
            np.ndarray: 编码向量
        """
        try:
            # 根据模型类型选择编码方式
            if hasattr(self._model, 'encode'):
                # Sentence-Transformer 风格
                embedding = self._model.encode(text)
                return np.array(embedding).flatten()

            elif self._tokenizer is not None:
                # Transformers 风格
                inputs = self._tokenizer(
                    text,
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )

                import torch
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    # 使用 [CLS] token 的输出作为句子表示
                    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()

                return embedding

            else:
                raise ValueError("无法识别的模型类型")

        except Exception as e:
            print(f"警告：模型编码失败: {e}。使用统计特征回退。")
            return self._statistical_extractor.extract(text, self.feature_dim)

    def _extract_behavior_features(
        self,
        profile: BehaviorProfile,
    ) -> np.ndarray:
        """从行为档案中提取数值特征

        Args:
            profile: 行为档案

        Returns:
            np.ndarray: 行为特征向量
        """
        # 提取关键行为指标
        n_msg = profile.message_count
        question_ratio = profile.question_count / max(1, n_msg)
        reply_ratio = profile.answer_count / max(1, n_msg)
        n_partners = len(profile.interaction_partners)
        interaction_breadth = min(1.0, n_partners / 10.0)

        features = np.array([
            min(1.0, n_msg / 50.0),                                    # 活跃度
            min(1.0, profile.avg_message_length / 200.0),              # 消息长度
            question_ratio,                                             # 提问比例
            reply_ratio,                                                # 回复比例
            min(1.0, profile.response_time / 120.0),                   # 响应时间
            min(1.0, profile.emoji_count / max(1, n_msg)),             # 表情使用比例
            interaction_breadth,                                        # 交互广度
        ])

        return features

    def _fuse_features(
        self,
        text_features: np.ndarray,
        behavior_features: np.ndarray,
    ) -> np.ndarray:
        """融合文本特征和行为特征

        Args:
            text_features: 文本特征向量
            behavior_features: 行为特征向量

        Returns:
            np.ndarray: 融合后的特征向量
        """
        # 确保文本特征有足够的维度
        if len(text_features) < self.feature_dim:
            text_features = np.pad(
                text_features,
                (0, self.feature_dim - len(text_features))
            )

        # 将行为特征编码到文本特征的后段
        max_behavior_dims = min(
            len(behavior_features),
            self.feature_dim - 8  # 保留前 8 维给文本特征
        )

        fused = text_features.copy()
        if max_behavior_dims > 0:
            fused[-max_behavior_dims:] = (
                fused[-max_behavior_dims:] * 0.6
                + behavior_features[:max_behavior_dims] * 0.4
            )

        return fused[:self.feature_dim]

    def _predict_cognitive_indicators(
        self,
        features: np.ndarray,
        messages: List[Dict[str, Any]],
        behavior_profile: Optional[BehaviorProfile] = None,
    ) -> Dict[str, float]:
        """从特征预测认知指标

        在预训练模式下，使用神经网络的预测头；
        在启发式模式下，使用基于规则的映射。

        Args:
            features: 特征向量
            messages: 消息列表
            behavior_profile: 行为档案（可选）

        Returns:
            Dict[str, float]: 认知指标字典
        """
        if self._is_model_loaded:
            # 使用预训练模型的预测头（简化版：线性映射）
            return self._predict_with_head(features)
        else:
            # 使用启发式方法从特征中映射
            return self._heuristic_predict(features, messages, behavior_profile)

    def _predict_with_head(
        self,
        features: np.ndarray,
    ) -> Dict[str, float]:
        """使用预测头进行预测（简化版）

        在实际应用中，这里应该是一个预训练好的全连接网络。
        当前实现提供了一个简单的线性映射作为占位。

        Args:
            features: 特征向量

        Returns:
            Dict[str, float]: 认知指标
        """
        # 简化版：使用特征的统计信息作为预测
        feature_mean = float(np.mean(features))
        feature_std = float(np.std(features)) if len(features) > 1 else 0.0
        feature_norm = float(np.linalg.norm(features))

        # 基于特征的统计信息推断认知指标
        cognitive_load = np.clip(feature_norm * 0.3 + feature_std * 0.5, 0.0, 1.0)
        understanding_level = np.clip(feature_mean * 0.5 + 0.3, 0.0, 1.0)
        attention_score = np.clip(1.0 - feature_std * 0.8, 0.0, 1.0)
        confusion_level = np.clip(feature_std * 1.2 - feature_mean * 0.3, 0.0, 1.0)
        cognitive_flexibility = np.clip(feature_norm * 0.4 + 0.2, 0.0, 1.0)

        return {
            "cognitive_load": float(cognitive_load),
            "understanding_level": float(understanding_level),
            "attention_score": float(attention_score),
            "confusion_level": float(confusion_level),
            "cognitive_flexibility": float(cognitive_flexibility),
        }

    def _heuristic_predict(
        self,
        features: np.ndarray,
        messages: List[Dict[str, Any]],
        behavior_profile: Optional[BehaviorProfile] = None,
    ) -> Dict[str, float]:
        """启发式预测认知指标

        基于特征的统计属性和消息内容进行推断。

        Args:
            features: 特征向量
            messages: 消息列表
            behavior_profile: 行为档案（可选）

        Returns:
            Dict[str, float]: 认知指标
        """
        # 1. 从特征中提取统计量
        feature_sum = float(np.sum(features))
        feature_var = float(np.var(features)) if len(features) > 1 else 0.0
        feature_max = float(np.max(features)) if len(features) > 0 else 0.0

        # 2. 从消息中提取统计量
        n_msg = len(messages)
        avg_length = np.mean([len(msg.get("text", "")) for msg in messages]) if messages else 0

        # 3. 检测提问和困惑信号
        question_count = 0
        confusion_signals = 0
        for msg in messages:
            text = msg.get("text", "")
            # 提问检测
            if any(q in text for q in ["?", "？", "为什么", "怎么", "如何"]):
                question_count += 1
            # 困惑信号检测
            if any(c in text for c in ["不懂", "不明白", "难", "奇怪", "confuse"]):
                confusion_signals += 1

        question_ratio = question_count / max(1, n_msg)
        confusion_ratio = confusion_signals / max(1, n_msg)

        # 4. 认知负荷计算
        load_from_features = min(1.0, feature_var * 2.0 + feature_sum * 0.1)
        load_from_messages = min(1.0, (avg_length / 200.0) * 0.5 + question_ratio * 0.5)
        cognitive_load = (load_from_features * 0.4 + load_from_messages * 0.6)

        # 5. 理解水平计算
        understanding_from_features = min(1.0, feature_max * 0.8)
        understanding_from_questions = max(0.0, 1.0 - question_ratio * 2.0)
        understanding_level = (understanding_from_features * 0.4 + understanding_from_questions * 0.6)

        # 6. 注意力得分
        attention_from_variance = max(0.0, 1.0 - feature_var * 3.0)
        attention_from_count = min(1.0, n_msg / 15.0)
        attention_score = (attention_from_variance * 0.5 + attention_from_count * 0.5)

        # 7. 困惑水平
        confusion_from_signals = confusion_ratio
        confusion_from_questions = min(1.0, question_ratio * 1.5)
        confusion_from_variance = min(1.0, feature_var * 2.0)
        confusion_level = (
            confusion_from_signals * 0.4
            + confusion_from_questions * 0.3
            + confusion_from_variance * 0.3
        )

        # 8. 认知灵活性
        flexibility_from_variance = min(1.0, feature_var * 2.0)
        flexibility_from_length = min(1.0, avg_length / 100.0)
        cognitive_flexibility = (flexibility_from_variance * 0.5 + flexibility_from_length * 0.5)

        return {
            "cognitive_load": float(np.clip(cognitive_load, 0.0, 1.0)),
            "understanding_level": float(np.clip(understanding_level, 0.0, 1.0)),
            "attention_score": float(np.clip(attention_score, 0.0, 1.0)),
            "confusion_level": float(np.clip(confusion_level, 0.0, 1.0)),
            "cognitive_flexibility": float(np.clip(cognitive_flexibility, 0.0, 1.0)),
        }

    def _compute_confidence(
        self,
        text_features: np.ndarray,
        predictions: Dict[str, float],
    ) -> float:
        """计算分析结果的置信度

        Args:
            text_features: 文本特征向量
            predictions: 预测的认知指标

        Returns:
            float: 置信度 (0-1)
        """
        # 因素1：特征的有效性
        feature_norm = np.linalg.norm(text_features)
        if feature_norm > 0:
            feature_quality = min(1.0, feature_norm / 5.0)
        else:
            feature_quality = 0.0

        # 因素2：预测的确定性（极端值越少，置信度越高）
        extreme_count = sum(
            1 for v in predictions.values()
            if v > 0.95 or v < 0.05
        )
        certainty = 1.0 - (extreme_count / len(predictions)) * 0.5

        # 因素3：模型状态
        model_factor = 0.8 if self._is_model_loaded else 0.4

        # 综合
        confidence = feature_quality * 0.3 + certainty * 0.4 + model_factor * 0.3

        return float(np.clip(confidence, 0.0, 1.0))


class _StatisticalFeatureExtractor:
    """统计特征提取器（内部辅助类）

    在不使用预训练模型时，从文本中提取数值化的统计特征。
    这些特征将作为认知状态预测的基础。

    提取的特征包括：
    - 文本长度特征（字符数、词数等）
    - 词汇多样性（唯一词比例）
    - 句法特征（标点符号使用、平均句长等）
    - 情感词汇比例（正/负向词汇）
    - 复杂度指标（长词比例、句式复杂度等）
    """

    def __init__(self):
        """初始化特征提取器"""
        # 预定义的认知相关词汇集
        self._understanding_words = {
            "明白", "懂了", "理解了", "明白了", "知道", "清楚",
            "understand", "got it", "i see", "makes sense",
        }
        self._confusion_words = {
            "不懂", "不明白", "困惑", "奇怪", "奇怪了", "搞不懂",
            "confuse", "confused", "weird", "strange",
        }
        self._question_markers = ["?", "？", "为什么", "怎么", "如何", "what", "how", "why"]

    def extract(self, text: str, target_dim: int = 16) -> np.ndarray:
        """提取统计特征

        Args:
            text: 输入文本
            target_dim: 目标特征维度

        Returns:
            np.ndarray: 特征向量
        """
        if not text:
            return np.zeros(target_dim)

        # 计算各种统计指标
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = max(1, text.count(".") + text.count("!") + text.count("?") + text.count("。") + text.count("！") + text.count("？") + 1)

        # 词汇多样性
        unique_words = len(set(text.split()))
        word_diversity = unique_words / max(1, word_count)

        # 平均句长（字符数）
        avg_sentence_length = char_count / sentence_count

        # 标点符号密度
        punctuation_count = sum(1 for c in text if c in ".,!?;:，。！？；：、")
        punctuation_density = punctuation_count / max(1, char_count)

        # 长词比例（>3 个字符的中文词或 >6 个字符的英文词）
        long_words = sum(1 for w in text.split() if len(w) > 3)
        long_word_ratio = long_words / max(1, word_count)

        # 提问信号密度
        question_count = sum(text.count(q) for q in self._question_markers)
        question_density = question_count / max(1, sentence_count)

        # 理解词汇比例
        understanding_count = sum(1 for w in text.split() if w in self._understanding_words)
        understanding_ratio = understanding_count / max(1, word_count)

        # 困惑词汇比例
        confusion_count = sum(1 for w in text.split() if w in self._confusion_words)
        confusion_ratio = confusion_count / max(1, word_count)

        # 构建特征向量（基础 10 维）
        base_features = np.array([
            min(1.0, char_count / 500.0),            # 文本长度
            min(1.0, word_count / 100.0),             # 词数
            min(1.0, avg_sentence_length / 100.0),    # 平均句长
            word_diversity,                            # 词汇多样性
            punctuation_density * 5,                   # 标点密度
            min(1.0, long_word_ratio * 3),            # 长词比例
            min(1.0, question_density),               # 提问密度
            min(1.0, understanding_ratio * 5),        # 理解词汇比例
            min(1.0, confusion_ratio * 5),            # 困惑词汇比例
            1.0 - punctuation_density * 3,            # 文本流畅度
        ])

        # 填充或截断到目标维度
        if len(base_features) >= target_dim:
            return base_features[:target_dim]
        else:
            padded = np.zeros(target_dim)
            padded[:len(base_features)] = base_features
            return padded
