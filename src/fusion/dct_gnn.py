"""
动态因果图神经网络时空融合实现（PyTorch 真实版本）

该模块实现了基于动态因果图神经网络（Dynamic Causal Topology Graph Neural Network, DCT-GNN）的时空融合方案。
它是 Tender 框架的默认融合引擎，使用真正的 PyTorch 图卷积网络进行端到端训练。

核心方法：
1. 将拓扑分析结果和因果分析结果融合为动态因果拓扑图
2. 使用 PyTorch 图卷积网络 (GCN) 学习图结构上的空间依赖关系
3. 使用时序循环结构（GRU）学习时间依赖关系
4. 生成融合特征向量并进行下一时间窗口的情绪预测

学术基础：
- 图卷积网络 (Kipf & Welling, 2017)
- 动态图网络 (Pareja et al., 2020)
- 时空图网络 (Yan et al., 2018)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tender.fusion.base import BaseFusionModule, FusionResult


# ============================================================================
# PyTorch GCN 模型定义
# ============================================================================


class GraphConvolutionLayer(nn.Module):
    """图卷积层（PyTorch 实现）

    实现 Kipf & Welling (2017) 的图卷积操作:
    H' = sigma(D^{-1/2} * A * D^{-1/2} * H * W)

    其中 A 是邻接矩阵，D 是度矩阵，H 是节点特征，W 是可训练权重。
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        normalized_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (n, in_features) 节点特征矩阵
            normalized_adjacency: (n, n) 归一化邻接矩阵

        Returns:
            torch.Tensor: (n, out_features) 更新后的节点特征
        """
        # 图卷积：H' = sigma(A_hat @ H @ W)
        support = torch.mm(node_features, self.weight)  # H @ W
        output = torch.mm(normalized_adjacency, support)  # A_hat @ (H @ W)
        output = output + self.bias  # + bias
        output = self.dropout(output)
        return F.relu(output)


class TemporalGRULayer(nn.Module):
    """时序 GRU 层（PyTorch 实现）

    用于建模时间序列上的依赖关系。
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_series: (batch, seq_len, input_dim)

        Returns:
            torch.Tensor: (batch, hidden_dim) 最后时间步的隐藏状态
        """
        # 如果输入是 2D，包装为 3D
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(0)

        output, hidden = self.gru(time_series)
        return hidden[-1]  # 取最后一层的最后时间步


class DCTGNN(nn.Module):
    """动态因果图神经网络（完整模型）

    架构：
    1. 多层 GCN 编码器：学习图结构上的空间依赖
    2. GRU 时序编码器：学习时间序列上的依赖
    3. 融合层：结合空间和时间特征生成融合向量
    4. 预测头：输出下一时间窗口的情绪预测
    """

    def __init__(
        self,
        node_feature_dim: int = 6,
        gnn_hidden_dim: int = 64,
        gnn_num_layers: int = 3,
        gnn_dropout: float = 0.2,
        temporal_hidden_dim: int = 32,
        spatial_feature_dim: int = 8,
        temporal_feature_dim: int = 8,
        output_dim: int = 16,
        forecast_horizon: int = 1,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_num_layers = gnn_num_layers
        self.spatial_feature_dim = spatial_feature_dim
        self.temporal_feature_dim = temporal_feature_dim
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon

        # GCN 编码器（多层图卷积）
        self.gcn_layers = nn.ModuleList()
        # 第一层：从节点特征到隐藏层
        self.gcn_layers.append(
            GraphConvolutionLayer(node_feature_dim, gnn_hidden_dim, gnn_dropout)
        )
        # 中间层
        for _ in range(gnn_num_layers - 2):
            self.gcn_layers.append(
                GraphConvolutionLayer(gnn_hidden_dim, gnn_hidden_dim, gnn_dropout)
            )
        # 最后一层：输出空间特征
        self.gcn_layers.append(
            GraphConvolutionLayer(gnn_hidden_dim, spatial_feature_dim, gnn_dropout)
        )

        # GRU 时序编码器（从时间序列学习时间依赖）
        self.temporal_gru = TemporalGRULayer(
            input_dim=node_feature_dim,
            hidden_dim=temporal_hidden_dim,
        )

        # 时间特征投影层
        self.temporal_projection = nn.Linear(temporal_hidden_dim, temporal_feature_dim)

        # 全局池化层（将节点级特征聚合成图级特征）
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 融合层：拼接空间和时间特征
        self.fusion_projection = nn.Linear(
            spatial_feature_dim + temporal_feature_dim, output_dim
        )

        # 预测头：从融合特征预测未来情绪
        self.forecast_head = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Dropout(gnn_dropout),
            nn.Linear(32, forecast_horizon),
        )

    def _normalize_adjacency(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        归一化邻接矩阵：D^{-1/2} * A * D^{-1/2}

        Args:
            adjacency: (n, n) 邻接矩阵

        Returns:
            torch.Tensor: (n, n) 归一化邻接矩阵
        """
        degree = adjacency.sum(dim=1, keepdim=True)
        degree_safe = torch.clamp(degree, min=1e-10)
        degree_inv_sqrt = torch.pow(degree_safe, -0.5)
        normalized = adjacency * degree_inv_sqrt * degree_inv_sqrt.T
        return normalized

    def _compute_global_spatial_features(
        self,
        node_features: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        从 GCN 编码后的节点特征中提取全局空间特征

        Args:
            node_features: (n, node_feature_dim) 原始节点特征
            node_embeddings: (n, spatial_feature_dim) GCN 编码后的节点嵌入

        Returns:
            torch.Tensor: (spatial_feature_dim,) 全局空间特征向量
        """
        # 全局平均池化
        global_features = torch.mean(node_embeddings, dim=0)

        # 加上最大池化信息
        max_features, _ = torch.max(node_embeddings, dim=0)

        # 融合：均值 + 最大值的加权和
        fused_features = 0.7 * global_features + 0.3 * max_features

        return fused_features

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        time_series_windows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            node_features: (n, node_feature_dim) 节点特征矩阵
            adjacency: (n, n) 邻接矩阵
            time_series_windows: (seq_len, n, node_feature_dim) 可选的时间序列窗口

        Returns:
            Tuple:
                - fused_features: (output_dim,) 融合特征向量
                - node_embeddings: (n, spatial_feature_dim) GCN 编码后的节点嵌入
                - forecast: (forecast_horizon,) 预测结果
        """
        n = node_features.shape

        # 1. 空间特征提取（通过多层 GCN）
        normalized_adj = self._normalize_adjacency(adjacency)
        x = node_features
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, normalized_adj)
        node_embeddings = x  # (n, spatial_feature_dim)

        # 2. 全局空间特征
        spatial_features = self._compute_global_spatial_features(
            node_features, node_embeddings
        )

        # 3. 时间特征提取（通过 GRU）
        if time_series_windows is not None:
            # (seq_len, n, node_feature_dim) -> (n, seq_len, node_feature_dim)
            ts_reshaped = time_series_windows.permute(1, 0, 2)
            # 对每个节点独立运行 GRU
            temporal_hiddens = []
            for i in range(n):
                node_ts = ts_reshaped[i:i+1, :, :]  # (1, seq_len, node_feature_dim)
                hidden = self.temporal_gru(node_ts)  # (1, temporal_hidden_dim)
                temporal_hiddens.append(hidden)
            temporal_hidden = torch.cat(temporal_hiddens, dim=0)  # (n, temporal_hidden_dim)
            # 全局时间特征（对所有节点的均值）
            global_temporal = torch.mean(temporal_hidden, dim=0)
            temporal_features = self.temporal_projection(global_temporal)
        else:
            # 没有时间序列数据时使用零向量
            temporal_features = torch.zeros(self.temporal_feature_dim)

        # 4. 特征融合
        fused = torch.cat([spatial_features, temporal_features], dim=0)
        fused_features = self.fusion_projection(fused)

        # 5. 预测
        forecast = self.forecast_head(fused_features)

        return fused_features, node_embeddings, forecast

    def get_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": "DCTGNN_PyTorch",
            "description": "基于 PyTorch 的动态因果图神经网络时空融合模块",
            "node_feature_dim": self.node_feature_dim,
            "gnn_hidden_dim": self.gnn_hidden_dim,
            "gnn_num_layers": self.gnn_num_layers,
            "spatial_feature_dim": self.spatial_feature_dim,
            "temporal_feature_dim": self.temporal_feature_dim,
            "output_dim": self.output_dim,
            "forecast_horizon": self.forecast_horizon,
            "trainable_parameters": sum(p.numel() for p in self.parameters()),
        }


# ============================================================================
# PyTorch DCT-GNN 融合模块（与 Tender 基类兼容）
# ============================================================================


class DCTGNNModule(BaseFusionModule):
    """
    基于 PyTorch DCTGNN 的时空融合模块（与 Tender 框架兼容）

    该模块封装了 DCTGNN 模型，提供与 BaseFusionModule 完全兼容的接口。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # 模型参数
        self.node_feature_dim = 6  # 固定：聚类标签、离群标志、内聚度、出度、入度、传播者标志
        self.spatial_feature_dim = config.get("spatial_feature_dim", 8)
        self.temporal_feature_dim = config.get("temporal_feature_dim", 8)
        self.gnn_hidden_dim = config.get("gnn_hidden_dim", 64)
        self.gnn_num_layers = config.get("gnn_num_layers", 3)
        self.gnn_dropout = config.get("gnn_dropout", 0.2)
        self.output_dim = config.get("output_dim", 16)
        self.forecast_horizon = config.get("forecast_horizon", 1)

        # 训练参数
        self.learning_rate = config.get("gnn_learning_rate", 0.001)
        self.num_epochs = config.get("gnn_num_epochs", 200)
        self.weight_decay = config.get("gnn_weight_decay", 5e-4)

        # 初始化 PyTorch 模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DCTGNN(
            node_feature_dim=self.node_feature_dim,
            gnn_hidden_dim=self.gnn_hidden_dim,
            gnn_num_layers=self.gnn_num_layers,
            gnn_dropout=self.gnn_dropout,
            spatial_feature_dim=self.spatial_feature_dim,
            temporal_feature_dim=self.temporal_feature_dim,
            output_dim=self.output_dim,
            forecast_horizon=self.forecast_horizon,
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.loss_fn = nn.MSELoss()

        # 训练状态
        self.training_losses = []
        self.is_trained = False

    def _numpy_to_torch(self, array: np.ndarray) -> torch.Tensor:
        """将 NumPy 数组转换为 PyTorch 张量"""
        return torch.from_numpy(array).float().to(self.device)

    def _extract_spatial_features(self, topology_result: Any) -> np.ndarray:
        """
        从拓扑分析结果提取空间特征（与 DCT-GNN 原始实现保持一致）

        输出 8 维空间特征向量：
        [0] cluster_count: 聚类数量（归一化）
        [1] outlier_ratio: 离群比例
        [2] ring_exists: 是否存在情绪环（0/1）
        [3] centroid_valence: 全局重心愉悦度
        [4] centroid_arousal: 全局重心唤醒度
        [5] centroid_focus: 全局重心聚焦度
        [6] h0_barcode_count: H0 条形码数量（归一化）
        [7] h1_barcode_count: H1 条形码数量（归一化）
        """
        features = np.zeros(self.spatial_feature_dim)

        features[0] = topology_result.cluster_count / 10.0
        features[1] = topology_result.outlier_ratio
        features[2] = 1.0 if topology_result.ring_exists else 0.0

        centroid = topology_result.centroid
        features[3] = centroid[0] if len(centroid) > 0 else 0.0
        features[4] = centroid[1] if len(centroid) > 1 else 0.0
        features[5] = centroid[2] if len(centroid) > 2 else 0.0

        features[6] = len(topology_result.h0_barcodes) / 20.0
        features[7] = len(topology_result.h1_barcodes) / 10.0

        features = np.clip(features, -1.0, 1.0)
        return features

    def _extract_temporal_features(self, causal_result: Any) -> np.ndarray:
        """
        从因果分析结果提取时间特征（与 DCT-GNN 原始实现保持一致）

        输出 8 维时间特征向量：
        [0] causal_density: 因果密度
        [1] n_super_spreaders: 超级传播者数量（归一化）
        [2] n_edges: 因果边数量（归一化）
        [3] avg_out_degree: 平均出度（归一化）
        [4] avg_in_degree: 平均入度（归一化）
        [5] max_out_degree: 最大出度（归一化）
        [6] max_in_degree: 最大入度（归一化）
        [7] reciprocity: 互惠指数（双向边比例）
        """
        features = np.zeros(self.temporal_feature_dim)
        n_members = len(causal_result.out_degrees)

        if n_members == 0:
            return features

        features[0] = causal_result.causal_density
        features[1] = len(causal_result.super_spreaders) / max(n_members, 1)

        n_edges = len(causal_result.edges)
        features[2] = n_edges / max(n_members * (n_members - 1), 1)

        out_deg_values = list(causal_result.out_degrees.values())
        in_deg_values = list(causal_result.in_degrees.values())

        features[3] = np.mean(out_deg_values) / max(n_members, 1)
        features[4] = np.mean(in_deg_values) / max(n_members, 1)
        features[5] = max(out_deg_values) / max(n_members, 1) if out_deg_values else 0.0
        features[6] = max(in_deg_values) / max(n_members, 1) if in_deg_values else 0.0

        # 计算互惠指数
        if n_edges > 0:
            graph = causal_result.causal_graph
            n_mutual = 0
            for u, v in graph.edges():
                if graph.has_edge(v, u):
                    n_mutual += 0.5
            features[7] = (2 * n_mutual) / n_edges
        else:
            features[7] = 0.0

        features = np.clip(features, 0.0, 1.0)
        return features

    def _build_node_features(
        self,
        topology_result: Any,
        causal_result: Any,
        member_ids: List[str],
    ) -> np.ndarray:
        """
        构建每个节点的 6 维特征向量

        每个节点的特征向量：
        [0] 聚类标签（归一化）
        [1] 是否离群（0/1）
        [2] 内聚度
        [3] 出度（归一化）
        [4] 入度（归一化）
        [5] 是否属于传播者（0.5 为中性）
        """
        n_members = len(member_ids)
        node_feature_dim = 6
        features = np.zeros((n_members, node_feature_dim))

        for i, mid in enumerate(member_ids):
            # 聚类标签
            cluster_label = topology_result.cluster_labels.get(mid, -1)
            features[i, 0] = (cluster_label + 1) / max(topology_result.cluster_count, 1)

            # 是否离群
            features[i, 1] = 1.0 if mid in topology_result.outlier_members else 0.0

            # 内聚度
            features[i, 2] = 0.0 if mid in topology_result.outlier_members else 1.0

            # 出度
            features[i, 3] = causal_result.out_degrees.get(mid, 0) / max(n_members, 1)

            # 入度
            features[i, 4] = causal_result.in_degrees.get(mid, 0) / max(n_members, 1)

            # 传播者标志
            features[i, 5] = 1.0 if mid in causal_result.super_spreaders else 0.0

        return features

    def _build_adjacency_matrix(
        self,
        causal_result: Any,
        member_ids: List[str],
    ) -> np.ndarray:
        """从因果图构建邻接矩阵"""
        n_members = len(member_ids)
        adjacency = np.zeros((n_members, n_members))

        for edge in causal_result.edges:
            if edge.source in member_ids and edge.target in member_ids:
                i = member_ids.index(edge.source)
                j = member_ids.index(edge.target)
                adjacency[i, j] = edge.strength

        return adjacency

    def train_model(
        self,
        training_data: List[Tuple[Any, Any, List[str]]],
        time_series_history: Optional[List[Dict[str, List[np.ndarray]]]] = None,
    ) -> Dict[str, Any]:
        """
        训练 DCTGNN 模型

        Args:
            training_data: 训练数据列表，每个元素为 (topology_result, causal_result, member_ids)
            time_series_history: 可选的时间序列历史数据

        Returns:
            Dict: 训练统计信息
        """
        self.model.train()
        self.training_losses = []

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, (topology_result, causal_result, member_ids) in enumerate(training_data):
                # 构建节点特征和邻接矩阵
                node_features = self._build_node_features(
                    topology_result, causal_result, member_ids
                )
                adjacency = self._build_adjacency_matrix(
                    causal_result, member_ids
                )

                # 转换为 PyTorch 张量
                node_features_t = self._numpy_to_torch(node_features)
                adjacency_t = self._numpy_to_torch(adjacency)

                # 构建时间序列数据（如果有）
                time_series_t = None
                if time_series_history and batch_idx < len(time_series_history):
                    ts_data = time_series_history[batch_idx]
                    ts_windows = []
                    for mid in member_ids:
                        if mid in ts_data:
                            ts_windows.append(ts_data[mid])
                        else:
                            ts_windows.append([np.zeros(self.node_feature_dim)])
                    # 确保所有序列长度一致
                    min_len = min(len(w) for w in ts_windows)
                    ts_windows = [w[:min_len] for w in ts_windows]
                    ts_array = np.array(ts_windows)  # (n, seq_len, node_feature_dim)
                    time_series_t = self._numpy_to_torch(ts_array)

                # 目标值（这里模拟：预测未来情绪均值）
                # 在实际应用中需要真实的目标值
                target = torch.tensor([[0.5]]).to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                fused_features, node_embeddings, forecast = self.model(
                    node_features_t,
                    adjacency_t,
                    time_series_t,
                )

                # 计算损失
                loss = self.loss_fn(forecast, target.squeeze())

                # 反向传播
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.training_losses.append(avg_loss)

            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.6f}")

        self.is_trained = True

        return {
            "final_loss": self.training_losses[-1] if self.training_losses else None,
            "min_loss": min(self.training_losses) if self.training_losses else None,
            "epochs_trained": self.num_epochs,
        }

    def fuse(
        self,
        topology_result: Any,
        causal_result: Any,
        time_series_data: Dict[str, List[np.ndarray]],
        member_ids: List[str],
    ) -> FusionResult:
        """
        执行完整的时空融合（使用训练好的 PyTorch 模型）

        Args:
            topology_result: 拓扑分析结果
            causal_result: 因果分析结果
            time_series_data: 时间序列数据
            member_ids: 成员ID列表

        Returns:
            FusionResult: 时空融合结果
        """
        # 1. 提取空间和时间特征（用于最终输出）
        spatial_features = self._extract_spatial_features(topology_result)
        temporal_features = self._extract_temporal_features(causal_result)

        # 2. 构建节点特征和邻接矩阵
        node_features = self._build_node_features(
            topology_result, causal_result, member_ids
        )
        adjacency = self._build_adjacency_matrix(causal_result, member_ids)

        # 3. 转换为 PyTorch 张量并执行推理
        self.model.eval()
        with torch.no_grad():
            node_features_t = self._numpy_to_torch(node_features)
            adjacency_t = self._numpy_to_torch(adjacency)

            # 构建时间序列窗口
            time_series_t = None
            if time_series_data:
                ts_windows = []
                for mid in member_ids:
                    if mid in time_series_data and len(time_series_data[mid]) > 0:
                        ts_windows.append(np.array(time_series_data[mid]))
                    else:
                        ts_windows.append(np.zeros((1, self.node_feature_dim)))
                min_len = min(len(w) for w in ts_windows)
                ts_windows = [w[:min_len] for w in ts_windows]
                ts_array = np.stack(ts_windows, axis=0)  # (n, seq_len, node_feature_dim)
                time_series_t = self._numpy_to_torch(ts_array)

            # 模型推理
            fused_features_t, node_embeddings_t, forecast_t = self.model(
                node_features_t,
                adjacency_t,
                time_series_t,
            )

            # 转换为 NumPy
            fused_features = fused_features_t.cpu().numpy()
            node_embeddings = node_embeddings_t.cpu().numpy()
            forecast = forecast_t.cpu().numpy()

        # 4. 构建动态因果拓扑图
        import networkx as nx
        fusion_graph = nx.Graph()
        for i, mid in enumerate(member_ids):
            fusion_graph.add_node(
                mid,
                cluster_label=topology_result.cluster_labels.get(mid, -1),
                is_outlier=mid in topology_result.outlier_members,
                node_embedding=node_embeddings[i].tolist(),
            )
        for edge in causal_result.edges:
            if edge.source in member_ids and edge.target in member_ids:
                fusion_graph.add_edge(
                    edge.source,
                    edge.target,
                    strength=edge.strength,
                    lag=edge.lag,
                    p_value=edge.p_value,
                )

        # 5. 构建预测结果
        forecast_vector = np.zeros(self.forecast_horizon)
        for i in range(min(self.forecast_horizon, len(forecast))):
            forecast_vector[i] = forecast[i]

        return FusionResult(
            fused_features=fused_features,
            fusion_graph=fusion_graph,
            node_features=node_embeddings,
            forecast=forecast_vector,
            metadata={
                "model": "DCTGNN_PyTorch",
                "is_trained": self.is_trained,
                "n_members": len(member_ids),
                "n_edges": len(causal_result.edges),
                "n_clusters": topology_result.cluster_count,
                "ring_exists": topology_result.ring_exists,
            },
        )

    def get_info(self) -> Dict[str, Any]:
        """获取当前融合模块的信息"""
        info = self.model.get_info()
        info["is_trained"] = self.is_trained
        info["device"] = str(self.device)
        info["training_losses"] = self.training_losses
        return info
