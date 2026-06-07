""" 基于知识图谱的认知状态分析引擎——认知状态分析模块

该模块实现了基于预设知识图谱的认知状态分析策略（Strategy 1）。
它根据成员在知识点图谱中的位置和交互行为，推断其当前的认知状态。

核心思想：
- 知识图谱是教学目标的显式结构化表示
- 成员的消息可以映射到知识图谱中的特定节点
- 通过分析成员在知识图谱中的分布，可以推断其理解水平、认知负荷和认知阶段

工作流程：
1. 加载预设的知识图谱（从 YAML/JSON 文件或配置对象）
2. 对每个成员的消息进行知识点匹配（基于关键词或语义嵌入）
3. 计算成员在知识图谱中的覆盖率和深度
4. 基于覆盖率、深度和难度推断认知状态
5. 聚合为群体认知状态

适用场景：
- 在线教育平台，教师已预设好课程知识图谱
- 企业培训，有明确的技能图谱
- 任何有结构化知识体系的场景

学术基础：
- 知识追踪模型 (Corbett & Anderson, 1995): 基于知识点的认知状态追踪
- 知识图谱嵌入 (Bordes et al., 2013): 将知识图谱映射到低维向量空间
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict

from tender.cognition.base import (
    BaseCognitionAnalyzer,
    CognitionState,
    KnowledgeNode,
    BehaviorProfile,
    KnowledgeGraphConfig,
    CognitivePhase,
    EngagementType,
)


class KnowledgeStateAnalyzer(BaseCognitionAnalyzer):
    """基于知识图谱的认知状态分析引擎

    使用预设的知识图谱来分析成员的认知状态。
    该引擎的核心是：将成员的发言匹配到知识图谱中的节点，
    然后通过节点的难度、深度和覆盖率来推断认知状态。

    Args:
        config: 配置字典，包含以下字段：
            - feature_dim: 认知特征维度（默认 16）
            - output_dim: 输出维度（默认 16）
            - use_knowledge_graph: 是否使用知识图谱（默认 True）
            - knowledge_graph_path: 知识图谱配置文件路径（可选）
            - kg_embedding_method: 知识图谱嵌入方法（默认 "node2vec"）
            - kg_embedding_dim: 嵌入维度（默认 16）
            - node_difficulty_key: 节点字典中的难度字段名（默认 "difficulty"）
            - node_prerequisites_key: 节点字典中的前置字段名（默认 "prerequisites"）
            - aggregation_strategy: 群体聚合策略（默认 "weighted"）
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化知识图谱分析引擎

        Args:
            config: 配置字典
        """
        # 核心维度参数
        self.feature_dim = config.get("feature_dim", 16)
        self.output_dim = config.get("output_dim", 16)

        # 知识图谱相关参数
        self.use_knowledge_graph = config.get("use_knowledge_graph", True)
        self.knowledge_graph_path = config.get("knowledge_graph_path", None)
        self.kg_embedding_method = config.get("kg_embedding_method", "node2vec")
        self.kg_embedding_dim = config.get("kg_embedding_dim", 16)

        # 节点字段映射
        self.node_difficulty_key = config.get("node_difficulty_key", "difficulty")
        self.node_prerequisites_key = config.get("node_prerequisites_key", "prerequisites")

        # 聚合参数
        self.aggregation_strategy = config.get("aggregation_strategy", "weighted")

        # 内部状态：知识图谱数据结构
        self._kg_config: Optional[KnowledgeGraphConfig] = None
        self._node_map: Dict[str, KnowledgeNode] = {}          # node_id -> KnowledgeNode
        self._node_embeddings: Dict[str, np.ndarray] = {}       # node_id -> embedding
        self._parent_map: Dict[str, List[str]] = {}             # node_id -> parent nodes
        self._keyword_map: Dict[str, str] = {}                  # keyword -> node_id

        # 尝试加载知识图谱
        if self.use_knowledge_graph and self.knowledge_graph_path:
            self._load_knowledge_graph_from_path(self.knowledge_graph_path)

        # 记录初始化信息
        self._init_info = (
            f"KnowledgeStateAnalyzer initialized with "
            f"use_kg={self.use_knowledge_graph}, "
            f"kg_nodes={len(self._node_map)}, "
            f"embedding_method={self.kg_embedding_method}, "
            f"feature_dim={self.feature_dim}"
        )

    def _load_knowledge_graph_from_path(self, path: str) -> None:
        """从文件路径加载知识图谱

        支持 .yaml 和 .json 格式。
        文件应包含 nodes 列表和 edges 列表。

        Args:
            path: 知识图谱文件路径
        """
        try:
            import yaml
            import json

            if path.endswith(('.yaml', '.yml')):
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            elif path.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"不支持的知识图谱文件格式: {path}。仅支持 .yaml, .yml, .json")

            # 解析节点
            nodes = []
            for node_data in data.get("nodes", []):
                node = KnowledgeNode(
                    node_id=node_data.get("id", ""),
                    name=node_data.get("name", ""),
                    difficulty=node_data.get(self.node_difficulty_key, 0.5),
                    prerequisites=node_data.get(self.node_prerequisites_key, []),
                    children=node_data.get("children", []),
                    metadata=node_data.get("metadata", {}),
                )
                nodes.append(node)

                # 建立关键词映射
                keywords = node_data.get("keywords", [])
                for kw in keywords:
                    self._keyword_map[kw] = node.node_id

            # 解析边（如果单独定义）
            edges = data.get("edges", [])

            # 构建知识图谱配置
            kg_config = KnowledgeGraphConfig(
                nodes=nodes,
                edges=edges,
                embedding_dim=self.kg_embedding_dim,
            )

            # 配置引擎
            self.configure(kg_config)

            print(f"知识图谱加载完成：{len(nodes)} 个节点，{len(edges)} 条边")

        except ImportError:
            print("警告：缺少 yaml 库，请通过 `pip install pyyaml` 安装。跳过知识图谱加载。")
        except FileNotFoundError:
            print(f"警告：知识图谱文件未找到: {path}。引擎将以空图谱运行。")
        except Exception as e:
            print(f"警告：加载知识图谱时出错: {e}。引擎将以空图谱运行。")

    def configure(self, kg_config: KnowledgeGraphConfig) -> None:
        """配置知识图谱

        在初始化后或需要更换知识图谱时调用。

        Args:
            kg_config: 知识图谱配置
        """
        self._kg_config = kg_config

        # 构建节点映射表和父节点映射表
        self._node_map = {}
        self._parent_map = defaultdict(list)

        for node in kg_config.nodes:
            self._node_map[node.node_id] = node

            # 构建前置知识点的反向映射（父节点）
            for prereq_id in node.prerequisites:
                self._parent_map[prereq_id].append(node.node_id)

        # 生成节点嵌入
        self._generate_embeddings()

        print(f"知识图谱配置完成：{len(self._node_map)} 个节点嵌入可用")

    def _generate_embeddings(self) -> None:
        """生成知识图谱节点的低维嵌入向量

        根据配置的嵌入方法，将知识图谱节点映射到低维向量空间。
        这些嵌入将用于计算节点之间的语义距离。
        """
        if self.kg_embedding_method == "onehot":
            # 独热编码：每个节点一个唯一的 one-hot 向量
            node_ids = list(self._node_map.keys())
            n_nodes = len(node_ids)
            for i, node_id in enumerate(node_ids):
                embedding = np.zeros(n_nodes)
                embedding[i] = 1.0
                self._node_embeddings[node_id] = embedding

        elif self.kg_embedding_method == "node2vec":
            # 简化版 Node2Vec：使用图的拉普拉斯特征映射
            n_nodes = len(self._node_map)
            if n_nodes == 0:
                return

            node_ids = list(self._node_map.keys())
            node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

            # 构建邻接矩阵
            adj_matrix = np.zeros((n_nodes, n_nodes))
            for node_id, node in self._node_map.items():
                i = node_to_idx[node_id]
                for child_id in node.children:
                    if child_id in node_to_idx:
                        j = node_to_idx[child_id]
                        adj_matrix[i, j] = 1.0
                        adj_matrix[j, i] = 1.0  # 无向图

            # 计算拉普拉斯矩阵的特征向量
            degree = np.sum(adj_matrix, axis=1) + 1e-8
            d_inv = np.diag(1.0 / np.sqrt(degree))
            laplacian = np.eye(n_nodes) - d_inv @ adj_matrix @ d_inv

            try:
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

                # 取最小的 k 个非零特征值对应的特征向量
                k = min(self.kg_embedding_dim, n_nodes - 1)
                if k > 0:
                    # 排除最小的特征值（接近 0）
                    eigen_idx = np.argsort(eigenvalues)[1:k+1]
                    embeddings = eigenvectors[:, eigen_idx]

                    for i, node_id in enumerate(node_ids):
                        self._node_embeddings[node_id] = embeddings[i, :]

            except np.linalg.LinAlgError:
                # 如果特征分解失败，使用随机嵌入作为回退
                rng = np.random.RandomState(42)
                for node_id in node_ids:
                    self._node_embeddings[node_id] = rng.randn(self.kg_embedding_dim) * 0.1

        elif self.kg_embedding_method == "graphsage":
            # GraphSAGE 风格的嵌入（简化版）
            # 使用节点属性和邻居信息的简单聚合
            rng = np.random.RandomState(42)

            for node_id, node in self._node_map.items():
                # 基础属性编码
                base_embed = np.array([
                    node.difficulty,
                    len(node.prerequisites) / max(1, len(self._node_map)),
                    len(node.children) / max(1, len(self._node_map)),
                ])

                # 邻居信息聚合
                neighbor_features = []
                for neighbor_id in node.prerequisites + node.children:
                    if neighbor_id in self._node_map:
                        neighbor = self._node_map[neighbor_id]
                        neighbor_features.extend([
                            neighbor.difficulty,
                            len(neighbor.prerequisites) / max(1, len(self._node_map)),
                            len(neighbor.children) / max(1, len(self._node_map)),
                        ])

                # 填充或截断到目标维度
                embed_dim = self.kg_embedding_dim
                full_features = list(base_embed) + neighbor_features

                if len(full_features) >= embed_dim:
                    embedding = np.array(full_features[:embed_dim])
                else:
                    # 不足时使用随机噪声填充
                    noise = rng.randn(embed_dim - len(full_features)) * 0.01
                    embedding = np.concatenate([
                        np.array(full_features), noise
                    ])

                self._node_embeddings[node_id] = embedding

    def analyze(
        self,
        member_messages: Dict[str, List[Dict[str, Any]]],
        knowledge_graph_config: Optional[KnowledgeGraphConfig] = None,
        behavior_profiles: Optional[Dict[str, BehaviorProfile]] = None,
    ) -> Dict[str, CognitionState]:
        """分析所有成员的认知状态

        Args:
            member_messages: 成员消息字典
                {member_id: [{text: str, timestamp: float, ...}]}
            knowledge_graph_config: 可选的动态知识图谱配置
                如果提供，将替换或补充已有的知识图谱
            behavior_profiles: 成员行为档案（可选，用于细化分析）

        Returns:
            Dict[str, CognitionState]: 成员 ID 到认知状态的映射
        """
        # 验证输入
        self.validate_inputs(member_messages)

        # 如果提供了新的知识图谱配置，进行配置
        if knowledge_graph_config is not None:
            self.configure(knowledge_graph_config)

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
            knowledge_graph_config: 可选的知识图谱配置（此处不使用）
            behavior_profile: 该成员的行为档案（可选，用于细化分析）

        Returns:
            CognitionState: 该成员的认知状态
        """
        # 步骤1：知识点匹配
        matched_nodes, match_counts = self._match_knowledge_nodes(messages)

        # 步骤2：计算知识覆盖率和深度
        coverage = self._compute_coverage(matched_nodes)
        depth = self._compute_depth(matched_nodes)

        # 步骤3：计算难度得分
        difficulty_score = self._compute_difficulty(matched_nodes, match_counts)

        # 步骤4：计算认知负荷
        cognitive_load = self._compute_cognitive_load(
            matched_nodes, match_counts, messages, behavior_profile
        )

        # 步骤5：计算理解水平
        understanding_level = self._compute_understanding(
            coverage, depth, match_counts
        )

        # 步骤6：计算注意力集中程度
        attention_score = self._compute_attention(messages, behavior_profile)

        # 步骤7：计算困惑水平
        confusion_level = self._compute_confusion(
            messages, matched_nodes, match_counts
        )

        # 步骤8：计算认知灵活性
        cognitive_flexibility = self._compute_flexibility(matched_nodes, match_counts)

        # 步骤9：分类认知阶段
        phase = self._categorize_phase(cognitive_load, understanding_level)

        # 步骤10：计算参与类型
        if behavior_profile:
            question_ratio = behavior_profile.question_count / max(1, behavior_profile.message_count)
            engagement = self._compute_engagement_type(
                len(messages), question_ratio,
                behavior_profile.avg_message_length
            )
        else:
            engagement = EngagementType.MODERATE

        # 步骤11：计算置信度
        confidence = self._compute_confidence(matched_nodes, match_counts)

        # 步骤12：获取时间戳
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
            source_engine="knowledge_state",
            timestamp=timestamp,
            knowledge_nodes=list(matched_nodes),
            raw_embedding=self._build_embedding(matched_nodes, match_counts),
            metadata={
                "coverage": float(coverage),
                "depth": float(depth),
                "difficulty_score": float(difficulty_score),
                "matched_nodes": len(matched_nodes),
                "total_messages": len(messages),
            },
        )

        return state

    def compute_group_state(
        self,
        member_states: Dict[str, CognitionState],
    ) -> CognitionState:
        """从成员状态计算群体认知状态

        使用配置中设定的聚合策略。

        Args:
            member_states: 成员状态字典 {member_id: CognitionState}

        Returns:
            CognitionState: 群体的认知状态
        """
        return self._aggregate_states(member_states, self.aggregation_strategy)

    def _match_knowledge_nodes(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[Set[str], Dict[str, int]]:
        """将消息匹配到知识图谱中的节点

        使用关键词匹配和语义嵌入匹配相结合的方式。

        Args:
            messages: 成员的消息列表

        Returns:
            Tuple[Set[str], Dict[str, int]]:
                - 匹配到的节点 ID 集合
                - 每个节点被匹配的次数映射
        """
        matched_nodes = set()
        match_counts = defaultdict(int)

        for msg in messages:
            text = msg.get("text", "")
            if not text:
                continue

            # 关键词匹配（快速、高精度）
            for keyword, node_id in self._keyword_map.items():
                if keyword in text:
                    matched_nodes.add(node_id)
                    match_counts[node_id] += 1

            # 语义匹配（如果关键词匹配不足且嵌入可用）
            if len(matched_nodes) < 3 and self._node_embeddings:
                # 使用简单的词重叠率作为回退匹配策略
                words = set(text.split())
                for node_id, node in self._node_map.items():
                    node_name_words = set(node.name)
                    overlap = len(words & node_name_words)
                    if overlap > 0:
                        matched_nodes.add(node_id)
                        match_counts[node_id] += 1

        return matched_nodes, dict(match_counts)

    def _compute_coverage(self, matched_nodes: Set[str]) -> float:
        """计算知识覆盖率

        覆盖率 = 匹配到的节点数 / 总节点数
        反映了成员在知识图谱上的知识广度。

        Args:
            matched_nodes: 匹配到的节点 ID 集合

        Returns:
            float: 覆盖率 (0-1)
        """
        total_nodes = len(self._node_map)
        if total_nodes == 0:
            return 0.0

        return len(matched_nodes) / total_nodes

    def _compute_depth(self, matched_nodes: Set[str]) -> float:
        """计算知识深度

        深度 = 匹配节点在知识图谱中的平均层级深度
        反映了成员在知识体系中的深入程度。

        计算方法：
        - 从根节点（无前置知识点的节点）开始定义深度为 0
        - 每个子节点的深度 = 前置节点的最大深度 + 1
        - 返回匹配节点的平均深度

        Args:
            matched_nodes: 匹配到的节点 ID 集合

        Returns:
            float: 平均深度（归一化到 0-1）
        """
        if not matched_nodes or not self._node_map:
            return 0.0

        # 计算每个节点的深度（DFS）
        node_depths = self._compute_node_depths()

        # 计算匹配节点的平均深度
        depths = [node_depths.get(nid, 0) for nid in matched_nodes]
        avg_depth = np.mean(depths) if depths else 0.0

        # 归一化到 0-1（使用最大深度）
        max_depth = max(node_depths.values()) if node_depths else 1
        return min(1.0, avg_depth / max(1, max_depth))

    def _compute_node_depths(self) -> Dict[str, int]:
        """计算知识图谱中所有节点的深度

        Returns:
            Dict[str, int]: 节点 ID 到深度的映射
        """
        # 找到根节点（无前置知识点的节点）
        all_nodes = set(self._node_map.keys())
        all_with_prereqs = set()
        for node in self._node_map.values():
            all_with_prereqs.update(node.prerequisites)

        root_nodes = all_nodes - all_with_prereqs
        if not root_nodes:
            root_nodes = all_nodes  # 如果没有根节点，所有节点都视为根

        # BFS 计算深度
        depths = {nid: 0 for nid in root_nodes}
        queue = list(root_nodes)

        while queue:
            current = queue.pop(0)
            current_depth = depths[current]

            # 向下遍历子节点
            for child_id in self._node_map.get(current, KnowledgeNode()).children:
                if child_id in self._node_map:
                    if child_id not in depths or depths[child_id] < current_depth + 1:
                        depths[child_id] = current_depth + 1
                        queue.append(child_id)

        # 确保所有节点都有深度
        for nid in all_nodes:
            if nid not in depths:
                depths[nid] = 0

        return depths

    def _compute_difficulty(
        self,
        matched_nodes: Set[str],
        match_counts: Dict[str, int],
    ) -> float:
        """计算难度得分

        根据匹配到的知识点的难度等级加权平均。

        Args:
            matched_nodes: 匹配到的节点 ID 集合
            match_counts: 每个节点被匹配的次数

        Returns:
            float: 难度得分 (0-1)
        """
        if not matched_nodes:
            return self._kg_config.default_difficulty if self._kg_config else 0.5

        total_weight = 0.0
        weighted_difficulty = 0.0

        for node_id in matched_nodes:
            node = self._node_map.get(node_id)
            if node is not None:
                weight = match_counts.get(node_id, 1)
                weighted_difficulty += node.difficulty * weight
                total_weight += weight

        if total_weight == 0:
            return 0.5

        return weighted_difficulty / total_weight

    def _compute_cognitive_load(
        self,
        matched_nodes: Set[str],
        match_counts: Dict[str, int],
        messages: List[Dict[str, Any]],
        behavior_profile: Optional[BehaviorProfile] = None,
    ) -> float:
        """计算认知负荷水平

        基于以下因素：
        1. 匹配到的知识点难度（难度越大，负荷越高）
        2. 知识深度（越深，负荷越高）
        3. 消息长度（长消息通常意味着更高的认知处理量）
        4. 行为档案中的响应时间（响应越快可能意味着负荷越低）

        Args:
            matched_nodes: 匹配到的节点 ID 集合
            match_counts: 每个节点被匹配的次数
            messages: 成员的消息列表
            behavior_profile: 成员行为档案（可选）

        Returns:
            float: 认知负荷水平 (0-1)
        """
        # 因素1：难度贡献
        difficulty = self._compute_difficulty(matched_nodes, match_counts)
        load_from_difficulty = difficulty * 0.4

        # 因素2：深度贡献
        depth = self._compute_depth(matched_nodes)
        load_from_depth = depth * 0.3

        # 因素3：消息复杂度贡献
        if messages:
            avg_msg_length = np.mean([len(msg.get("text", "")) for msg in messages])
            load_from_length = min(1.0, avg_msg_length / 200) * 0.2
        else:
            load_from_length = 0.1

        # 因素4：行为贡献（如果提供行为档案）
        load_from_behavior = 0.1
        if behavior_profile:
            # 响应时间越短，可能意味着负荷越低
            response_load = 1.0 - min(1.0, behavior_profile.response_time / 120.0)
            load_from_behavior = response_load * 0.1

        # 综合计算
        total_load = (
            load_from_difficulty
            + load_from_depth
            + load_from_length
            + load_from_behavior
        )

        # 确保在 [0, 1] 范围内
        return float(np.clip(total_load, 0.0, 1.0))

    def _compute_understanding(
        self,
        coverage: float,
        depth: float,
        match_counts: Dict[str, int],
    ) -> float:
        """计算理解水平

        基于以下因素：
        1. 知识覆盖率（覆盖越广，理解越好）
        2. 知识深度（理解越深，掌握越好）
        3. 匹配频次（反复提到某个知识点，可能意味着正在消化）

        Args:
            coverage: 知识覆盖率
            depth: 知识深度
            match_counts: 每个节点被匹配的次数

        Returns:
            float: 理解水平 (0-1)
        """
        # 覆盖率贡献（权重 0.4）
        understanding_from_coverage = coverage * 0.4

        # 深度贡献（权重 0.4）
        understanding_from_depth = depth * 0.4

        # 频次贡献（权重 0.2）
        if match_counts:
            avg_count = np.mean(list(match_counts.values()))
            # 适度的重复有利于理解，过度重复可能意味着不理解
            count_factor = min(1.0, avg_count / 3.0) * (1.0 - abs(avg_count - 2.0) * 0.1)
        else:
            count_factor = 0.0

        understanding = (
            understanding_from_coverage
            + understanding_from_depth
            + count_factor * 0.2
        )

        return float(np.clip(understanding, 0.0, 1.0))

    def _compute_attention(
        self,
        messages: List[Dict[str, Any]],
        behavior_profile: Optional[BehaviorProfile] = None,
    ) -> float:
        """计算注意力集中程度

        基于以下因素：
        1. 消息长度（长度适中可能意味着专注）
        2. 响应一致性（响应时间方差小可能意味着专注）
        3. 主题集中度（消息集中在少数知识点上可能意味着专注）

        Args:
            messages: 成员的消息列表
            behavior_profile: 成员行为档案（可选）

        Returns:
            float: 注意力集中程度 (0-1)
        """
        if not messages:
            return 0.3

        # 因素1：消息长度适宜性
        lengths = [len(msg.get("text", "")) for msg in messages]
        avg_length = np.mean(lengths)

        # 长度在 30-150 字符之间被认为是专注的
        if 30 <= avg_length <= 150:
            length_score = 0.8
        elif avg_length < 10:
            length_score = 0.2  # 太短，可能注意力不集中
        elif avg_length > 300:
            length_score = 0.5  # 太长，可能分散到太多细节
        else:
            length_score = 0.5

        # 因素2：响应时间（如果提供行为档案）
        if behavior_profile and behavior_profile.response_time > 0:
            # 响应时间在 10-60 秒之间可能意味着专注
            rt = behavior_profile.response_time
            if 10 <= rt <= 60:
                response_score = 0.7
            elif rt < 5:
                response_score = 0.3  # 响应太快，可能没有深入思考
            elif rt > 180:
                response_score = 0.2  # 响应太慢，可能走神
            else:
                response_score = 0.5
        else:
            response_score = 0.5

        # 综合评分
        attention = length_score * 0.6 + response_score * 0.4

        return float(np.clip(attention, 0.0, 1.0))

    def _compute_confusion(
        self,
        messages: List[Dict[str, Any]],
        matched_nodes: Set[str],
        match_counts: Dict[str, int],
    ) -> float:
        """计算困惑水平

        基于以下因素：
        1. 提问类消息的比例（问题越多，困惑度越高）
        2. 知识点匹配的多样性（广泛尝试多个知识点可能意味着困惑）
        3. 高低难度节点的混合（同时涉及简单和困难节点可能意味着困惑）

        Args:
            messages: 成员的消息列表
            matched_nodes: 匹配到的节点 ID 集合
            match_counts: 每个节点被匹配的次数

        Returns:
            float: 困惑水平 (0-1)
        """
        if not messages:
            return 0.2

        # 因素1：提问比例
        question_count = 0
        for msg in messages:
            text = msg.get("text", "")
            # 检测问号或其他疑问标记
            if "?" in text or "？" in text or "为什么" in text or "怎么" in text:
                question_count += 1

        question_ratio = question_count / len(messages)
        confusion_from_questions = question_ratio * 0.5

        # 因素2：知识点多样性
        if matched_nodes:
            if len(matched_nodes) == 1:
                diversity = 0.1  # 集中在一个知识点，困惑度低
            elif len(matched_nodes) <= 3:
                diversity = 0.4  # 少数知识点，轻度困惑
            else:
                diversity = 0.7  # 大量知识点，可能困惑
        else:
            diversity = 0.5

        confusion_from_diversity = diversity * 0.3

        # 因素3：难度跨度
        if matched_nodes:
            difficulties = []
            for node_id in matched_nodes:
                node = self._node_map.get(node_id)
                if node:
                    difficulties.append(node.difficulty)

            if difficulties:
                difficulty_range = max(difficulties) - min(difficulties)
                confusion_from_range = difficulty_range * 0.2
            else:
                confusion_from_range = 0.0
        else:
            confusion_from_range = 0.0

        confusion = (
            confusion_from_questions
            + confusion_from_diversity
            + confusion_from_range
        )

        return float(np.clip(confusion, 0.0, 1.0))

    def _compute_flexibility(
        self,
        matched_nodes: Set[str],
        match_counts: Dict[str, int],
    ) -> float:
        """计算认知灵活性

        认知灵活性是指成员在不同知识点之间切换和建立联系的能力。
        高灵活性表现为：
        1. 覆盖多个知识点
        2. 知识点之间在图谱中有连接（互相有关系）
        3. 在不同的难度层面都有覆盖

        Args:
            matched_nodes: 匹配到的节点 ID 集合
            match_counts: 每个节点被匹配的次数

        Returns:
            float: 认知灵活性 (0-1)
        """
        if not matched_nodes or len(matched_nodes) < 2:
            return 0.2  # 只有一个知识点，灵活性低

        # 因素1：节点间连接性
        connections = 0
        total_pairs = 0

        node_list = list(matched_nodes)
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                node_i = self._node_map.get(node_list[i])
                node_j = self._node_map.get(node_list[j])

                if node_i and node_j:
                    total_pairs += 1
                    # 检查是否有直接连接（父子关系或共同前置）
                    if (node_list[j] in node_i.children or
                        node_list[i] in node_j.children or
                        set(node_i.prerequisites) & set(node_j.prerequisites)):
                        connections += 1

        if total_pairs > 0:
            connectivity = connections / total_pairs
        else:
            connectivity = 0.0

        # 因素2：难度分布广度
        if matched_nodes:
            difficulties = []
            for node_id in matched_nodes:
                node = self._node_map.get(node_id)
                if node:
                    difficulties.append(node.difficulty)

            if difficulties:
                difficulty_std = np.std(difficulties)
                diversity = min(1.0, difficulty_std * 2.0)
            else:
                diversity = 0.0
        else:
            diversity = 0.0

        # 综合计算
        flexibility = connectivity * 0.6 + diversity * 0.4

        return float(np.clip(flexibility, 0.0, 1.0))

    def _compute_confidence(
        self,
        matched_nodes: Set[str],
        match_counts: Dict[str, int],
    ) -> float:
        """计算认知状态分析的置信度

        置信度基于：
        1. 匹配到的节点数量（越多越可靠）
        2. 匹配频次（频次越高越可靠）
        3. 知识图谱的完整性（图谱越完整越可靠）

        Args:
            matched_nodes: 匹配到的节点 ID 集合
            match_counts: 每个节点被匹配的次数

        Returns:
            float: 置信度 (0-1)
        """
        # 因素1：匹配节点覆盖率
        total_nodes = len(self._node_map)
        if total_nodes > 0:
            node_coverage = len(matched_nodes) / total_nodes
        else:
            node_coverage = 0.0

        # 因素2：匹配频次
        if match_counts:
            total_matches = sum(match_counts.values())
            frequency_factor = min(1.0, total_matches / 10.0)
        else:
            frequency_factor = 0.0

        # 因素3：图谱完整性
        if self._node_map:
            # 简单的完整性判断：有节点且有连接
            has_edges = any(node.children or node.prerequisites for node in self._node_map.values())
            completeness = 0.8 if has_edges else 0.5
        else:
            completeness = 0.0

        # 综合计算
        confidence = (
            node_coverage * 0.3
            + frequency_factor * 0.4
            + completeness * 0.3
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def _build_embedding(
        self,
        matched_nodes: Set[str],
        match_counts: Dict[str, int],
    ) -> np.ndarray:
        """构建成员认知状态的嵌入向量

        将匹配到的知识图谱节点嵌入聚合为单个认知状态嵌入。

        Args:
            matched_nodes: 匹配到的节点 ID 集合
            match_counts: 每个节点被匹配的次数

        Returns:
            np.ndarray: 认知状态嵌入向量
        """
        if not matched_nodes:
            return np.zeros(self.output_dim)

        # 获取每个节点的嵌入并加权平均
        embeddings = []
        weights = []

        for node_id in matched_nodes:
            if node_id in self._node_embeddings:
                embeddings.append(self._node_embeddings[node_id])
                weights.append(match_counts.get(node_id, 1))

        if not embeddings:
            return np.zeros(self.output_dim)

        # 加权平均
        weights_norm = np.array(weights) / sum(weights)
        embedding = np.average(np.array(embeddings), axis=0, weights=weights_norm)

        # 确保输出维度一致
        if len(embedding) > self.output_dim:
            embedding = embedding[:self.output_dim]
        elif len(embedding) < self.output_dim:
            embedding = np.pad(embedding, (0, self.output_dim - len(embedding)))

        return embedding
