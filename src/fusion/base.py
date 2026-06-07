"""
时空融合模块 - 抽象基类

该模块定义了所有融合实现的统一接口。
核心功能是将空间拓扑分析的结果与时间因果分析的结果融合为：
1. 融合特征向量 F：用于策略匹配的连续特征表示
2. 动态因果拓扑图：可视化融合的图结构

设计动机：
空间拓扑分析给出了情绪在某一时刻的"快照"结构，
时间因果分析给出了情绪在时间维度上的"流动"关系。
融合模块将这两个维度的信息统一为一致表示，
才能用于后续的态势识别和策略推理。

核心输出指标：
- 融合特征向量 F (len = d_space + d_time)：策略匹配的输入
- 动态因果拓扑图 G：预测下一状态的输入
- 时序预测：估计下一窗口的群体情绪状态
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import networkx as nx
import numpy as np

from tender.topology_analysis.base import TopologyResult
from tender.causal_analysis.base import CausalResult


@dataclass
class FusionResult:
    """
    单次融合操作的结果

    包含融合后的特征向量、动态因果拓扑图和时序预测信息。

    Attributes:
        feature_vector: 融合特征向量 F，形状为 (d_space + d_time,)
        fusion_graph: 动态因果拓扑图（有向，节点带有拓扑特征）
        time_series_forecast: 下一窗口的情绪状态预测
            {成员ID: [valence, arousal, focus]}
        forecast_confidence: 预测置信度
        window_start: 当前窗口起始时间戳
        window_end: 当前窗口结束时间戳
        next_window_start: 预测窗口起始时间戳
        next_window_end: 预测窗口结束时间戳
        metadata: 附加元数据
    """
    feature_vector: np.ndarray
    fusion_graph: nx.DiGraph
    time_series_forecast: Dict[str, np.ndarray]
    forecast_confidence: float
    window_start: float = 0.0
    window_end: float = 0.0
    next_window_start: float = 0.0
    next_window_end: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFusionModule(ABC):
    """
    时空融合抽象基类

    定义时空融合的统一接口。所有具体实现必须：
    1. 继承此类
    2. 实现所有抽象方法
    3. 严格保持方法签名一致

    设计原则：
    - 插件式架构：通过配置文件选择具体实现
    - 统一接口：所有实现对外暴露相同的方法签名
    - 可替换性：在不影响上下游模块的前提下替换实现

    核心输入：
    - TopologyResult: 空间拓扑分析的结果
    - CausalResult: 时间因果分析的结果
    - time_series: 过去窗口的情绪时间序列

    核心输出：
    1. 融合特征向量 F
    2. 动态因果拓扑图
    3. 下一窗口的时序预测
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        初始化融合模块

        Args:
            config: 配置字典，包含以下字段：
                - fusion_method: 融合方法（"concat", "attention", "gnn"等）
                - forecast_method: 预测方法（"var", "lstm", "gcn"等）
                - feature_dim: 融合特征向量的目标维度
                - forecast_horizon: 预测时间窗口数
        """
        pass

    @abstractmethod
    def fuse(
        self,
        topology_result: TopologyResult,
        causal_result: CausalResult,
        time_series: Dict[str, List[np.ndarray]],
        member_ids: List[str],
    ) -> FusionResult:
        """
        融合空间拓扑和时间因果分析结果

        这是核心方法，将两个模块的输出统一为一致的表示。

        Args:
            topology_result: 空间拓扑分析结果
            causal_result: 时间因果分析结果
            time_series: 每个成员的情绪时间序列
                格式: {成员ID: [[v1, a1, f1], [v2, a2, f2], ...]}
            member_ids: 成员ID列表

        Returns:
            FusionResult: 融合后的结果
        """
        pass

    @abstractmethod
    def construct_feature_vector(
        self,
        topology_result: TopologyResult,
        causal_result: CausalResult,
        time_series: Dict[str, List[np.ndarray]],
        member_ids: List[str],
    ) -> np.ndarray:
        """
        构建融合特征向量

        将空间特征和时间特征拼接为统一的特征向量。
        空间特征：聚类数、离群比例、环存在标志、重心坐标
        时间特征：因果密度、超级传播者数量、出入度比

        Args:
            topology_result: 空间拓扑分析结果
            causal_result: 时间因果分析结果
            time_series: 情绪时间序列
            member_ids: 成员ID列表

        Returns:
            np.ndarray: 融合特征向量
        """
        pass

    @abstractmethod
    def construct_fusion_graph(
        self,
        topology_result: TopologyResult,
        causal_result: CausalResult,
        member_ids: List[str],
    ) -> nx.DiGraph:
        """
        构建动态因果拓扑图

        将因果网络与拓扑特征结合：
        - 节点：成员，带有拓扑属性（所属聚类、是否离群、情绪向量）
        - 边：因果关系，带有时间属性（滞后阶数、p值、效应量）

        Args:
            topology_result: 空间拓扑分析结果
            causal_result: 时间因果分析结果
            member_ids: 成员ID列表

        Returns:
            nx.DiGraph: 动态因果拓扑图
        """
        pass

    @abstractmethod
    def forecast(
        self,
        time_series: Dict[str, List[np.ndarray]],
        fusion_graph: nx.DiGraph,
        member_ids: List[str],
        forecast_horizon: int,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        预测下一窗口的情绪状态

        使用 VAR 或 GNN 模型，结合融合图的信息进行预测。

        Args:
            time_series: 过去窗口的情绪时间序列
            fusion_graph: 动态因果拓扑图
            member_ids: 成员ID列表
            forecast_horizon: 预测步数

        Returns:
            Tuple[Dict[str, np.ndarray], float]:
                (forecasts, confidence)
                - forecasts: 每个成员的预测情绪向量
                - confidence: 预测置信度
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        获取当前融合模块的信息

        Returns:
            Dict: 信息字典，包含算法名称、参数设置等
        """
        pass

    def _validate_inputs(
        self,
        topology_result: TopologyResult,
        causal_result: CausalResult,
        time_series: Dict[str, List[np.ndarray]],
        member_ids: List[str],
    ) -> None:
        """
        验证输入数据的有效性

        Args:
            topology_result: 空间拓扑分析结果
            causal_result: 时间因果分析结果
            time_series: 情绪时间序列
            member_ids: 成员ID列表

        Raises:
            ValueError: 如果输入无效
        """
        if not member_ids:
            raise ValueError("成员ID列表为空")

        if topology_result is None:
            raise ValueError("拓扑分析结果不能为 None")

        if causal_result is None:
            raise ValueError("因果分析结果不能为 None")

        if not time_series:
            raise ValueError("时间序列字典为空")

        # 检查成员一致性
        topology_members = set(topology_result.cluster_labels.keys())
        causal_members = set(causal_result.out_degrees.keys())
        given_members = set(member_ids)
        ts_members = set(time_series.keys())

        # 检查拓扑结果中的成员
        missing_in_topology = given_members - topology_members
        if missing_in_topology:
            print(f"警告：以下成员在拓扑结果中缺失: {missing_in_topology}")

        # 检查因果结果中的成员
        missing_in_causal = given_members - causal_members
        if missing_in_causal:
            print(f"警告：以下成员在因果结果中缺失: {missing_in_causal}")

    def _compute_spatial_features(
        self,
        topology_result: TopologyResult,
    ) -> np.ndarray:
        """
        提取空间拓扑特征向量

        从拓扑分析结果中提取的特征：
        - cluster_count: 聚类数（归一化）
        - outlier_ratio: 离群比例
        - ring_exists: 环存在标志（0/1）
        - centroid: 全局重心 [v, a, f]
        - centroid_spread: 重心到各点的平均距离

        Returns:
            np.ndarray: 空间特征向量
        """
        # 归一化聚类数（假设最大不超过20个聚类）
        norm_clusters = min(topology_result.cluster_count / 20.0, 1.0)

        # 构建空间特征
        spatial_features = np.array([
            norm_clusters,
            topology_result.outlier_ratio,
            1.0 if topology_result.ring_exists else 0.0,
            topology_result.centroid[0],  # valence
            topology_result.centroid[1],  # arousal
            topology_result.centroid[2],  # focus
        ])

        return spatial_features

    def _compute_temporal_features(
        self,
        causal_result: CausalResult,
        time_series: Dict[str, List[np.ndarray]],
    ) -> np.ndarray:
        """
        提取时间因果特征向量

        从因果分析结果中提取的特征：
        - causal_density: 因果密度
        - n_super_spreaders: 超级传播者数量（归一化）
        - avg_out_degree: 平均出度（归一化）
        - avg_in_degree: 平均入度（归一化）
        - degree_ratio: 平均出入度比值
        - volatility: 情绪波动度（时间序列方差均值）

        Returns:
            np.ndarray: 时间特征向量
        """
        n_members = len(causal_result.out_degrees)

        # 超级传播者数量归一化
        n_super = len(causal_result.super_spreaders) / max(n_members, 1)

        # 平均出度
        out_degrees_list = list(causal_result.out_degrees.values())
        avg_out = np.mean(out_degrees_list) / max(n_members - 1, 1)

        # 平均入度
        in_degrees_list = list(causal_result.in_degrees.values())
        avg_in = np.mean(in_degrees_list) / max(n_members - 1, 1)

        # 出入度比
        degree_ratio = avg_out / max(avg_in, 1e-6)

        # 情绪波动度
        volatility = 0.0
        if time_series:
            all_vals = []
            for mid, series in time_series.items():
                if len(series) > 1:
                    vals = np.array(series)[:, 0]  # valence维度
                    all_vals.append(np.std(vals))
            if all_vals:
                volatility = np.mean(all_vals)

        temporal_features = np.array([
            causal_result.causal_density,
            n_super,
            avg_out,
            avg_in,
            min(degree_ratio, 10.0),  # 限幅
            volatility,
        ])

        return temporal_features
