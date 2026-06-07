"""
空间拓扑分析模块 - 抽象基类

该模块定义了所有拓扑分析的统一接口。
核心功能是将情绪点云（成员数 × 3维）进行拓扑结构分析，
输出聚类数、环存在标志、离群比例和全局重心等关键指标。

学术基础：
持续同调通过构建不同尺度的单纯复形，并跟踪拓扑特征（连通分量、环、空洞等）
在尺度变化过程中的"出生"与"消亡"，从而获得数据的持久拓扑特征。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from tender.emotion_vectorizer.base import EmotionVector


@dataclass
class TopologyResult:
    """
    单次拓扑分析的结果

    包含情绪点云的空间结构信息，这些信息将用于后续的
    时空融合和策略推理。

    Attributes:
        cluster_count: HDBSCAN自动发现的聚类数量（情绪派系数）
        ring_exists: 环存在标志，是否存在情绪循环（H1寿命 > 阈值）
        outlier_ratio: 离群比例，未归属于任何簇的成员占比
        centroid: 全局重心，群体的整体情绪基调 [valence, arousal, focus]
        cluster_labels: 每个成员所属的聚类标签字典（-1表示离群点）
        h0_barcodes: H0条形码列表，每个元素为 (birth, death)
        h1_barcodes: H1条形码列表，每个元素为 (birth, death)
        outlier_members: 离群成员ID列表
        window_start: 时间窗口起始时间戳
        window_end: 时间窗口结束时间戳
        metadata: 附加分析元数据
    """
    cluster_count: int                                      # 聚类数
    ring_exists: bool                                       # 环存在标志
    outlier_ratio: float                                    # 离群比例 (0-1)
    centroid: np.ndarray                                    # 全局重心 [v, a, f]
    cluster_labels: Dict[str, int]                          # 成员ID -> 聚类标签
    h0_barcodes: List[Tuple[float, float]] = field(default_factory=list)   # H0条形码
    h1_barcodes: List[Tuple[float, float]] = field(default_factory=list)   # H1条形码
    outlier_members: List[str] = field(default_factory=list)               # 离群成员列表
    window_start: float = 0.0                                # 窗口起始时间
    window_end: float = 0.0                                  # 窗口结束时间
    metadata: Dict[str, Any] = field(default_factory=dict)   # 附加元数据

    def __post_init__(self):
        """初始化后处理：确保 centroid 是 numpy 数组"""
        if isinstance(self.centroid, list):
            self.centroid = np.array(self.centroid)


class BaseTopologyAnalyzer(ABC):
    """
    空间拓扑分析抽象基类

    定义空间拓扑分析的统一接口。所有具体实现必须：
    1. 继承此类
    2. 实现所有抽象方法
    3. 严格保持方法签名一致

    设计原则：
    - 插件式架构：通过配置文件选择具体实现
    - 统一接口：所有实现对外暴露相同的方法签名
    - 可替换性：在不影响上下游模块的前提下替换实现
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        初始化拓扑分析器

        Args:
            config: 配置字典，包含以下字段：
                - min_cluster_size: HDBSCAN最小聚类大小
                - h1_threshold_ratio: 环阈值比例（相对于窗口时间尺度）
                - min_samples: HDBSCAN最小样本数
                - metric: 距离度量方式（如 "euclidean"）
        """
        pass

    @abstractmethod
    def analyze(
        self,
        emotion_vectors: Dict[str, EmotionVector],
        window_start: float,
        window_end: float
    ) -> TopologyResult:
        """
        对单个时间窗口的情绪向量进行拓扑分析

        这是核心方法，将情绪点云转换为拓扑结构信息。

        Args:
            emotion_vectors: 成员情绪向量字典
                格式: {成员ID: EmotionVector}
            window_start: 时间窗口起始时间戳
            window_end: 时间窗口结束时间戳

        Returns:
            TopologyResult: 拓扑分析结果
        """
        pass

    @abstractmethod
    def compute_persistent_homology(
        self,
        point_cloud: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        计算点云的持续同调

        对情绪点云进行持续同调分析，返回 H0 和 H1 维度的条形码信息。

        Args:
            point_cloud: 点云数据，形状为 (n_samples, 3)

        Returns:
            Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
                (h0_barcodes, h1_barcodes)
                每个条形码为 (birth, death) 元组
        """
        pass

    @abstractmethod
    def compute_clusters(
        self,
        point_cloud: np.ndarray
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        对情绪点云进行聚类分析

        使用 HDBSCAN 进行自动聚类，无需预设聚类数量。

        Args:
            point_cloud: 点云数据，形状为 (n_samples, 3)

        Returns:
            Tuple[np.ndarray, int, np.ndarray]:
                (cluster_labels, n_clusters, centroid)
                - cluster_labels: 每个点的聚类标签（-1表示离群点）
                - n_clusters: 聚类数量（不含离群点）
                - centroid: 全局重心
        """
        pass

    @abstractmethod
    def detect_rings(
        self,
        h1_barcodes: List[Tuple[float, float]],
        threshold: float
    ) -> bool:
        """
        检测是否存在情绪环

        根据 H1 条形码的寿命判断是否存在稳定的环形情绪结构。
        情绪环表示存在 A → B → C → A 的矛盾情绪循环。

        Args:
            h1_barcodes: H1维度条形码列表
            threshold: 寿命阈值，超过该值的环被视为稳定环

        Returns:
            bool: 是否存在稳定的情绪环
        """
        pass

    @abstractmethod
    def identify_outliers(
        self,
        cluster_labels: np.ndarray,
        member_ids: List[str]
    ) -> Tuple[List[str], float]:
        """
        识别离群成员

        找出未归属于任何聚类的成员（离群点），
        这些成员可能面临退群风险或情绪孤立。

        Args:
            cluster_labels: 聚类标签数组
            member_ids: 对应的成员ID列表

        Returns:
            Tuple[List[str], float]:
                (outlier_members, outlier_ratio)
                - outlier_members: 离群成员ID列表
                - outlier_ratio: 离群比例
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        获取当前拓扑分析器的信息

        Returns:
            Dict: 信息字典，包含算法名称、参数设置等
        """
        pass

    def _validate_vectors(
        self,
        emotion_vectors: Dict[str, EmotionVector]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        验证并转换情绪向量为点云数组

        将 EmotionVector 字典转换为可用于拓扑分析的 numpy 数组。

        Args:
            emotion_vectors: 情绪向量字典

        Returns:
            Tuple[np.ndarray, List[str]]:
                (point_cloud, member_ids)
                - point_cloud: 形状 (n_members, 3) 的数组
                - member_ids: 对应的成员ID列表
        """
        if not emotion_vectors:
            raise ValueError("情绪向量字典为空，无法进行拓扑分析")

        member_ids = list(emotion_vectors.keys())
        point_cloud = np.array([
            emotion_vectors[mid].to_array()
            for mid in member_ids
        ])

        return point_cloud, member_ids

    def _compute_centroid(
        self,
        point_cloud: np.ndarray,
        cluster_labels: np.ndarray = None
    ) -> np.ndarray:
        """
        计算全局重心

        如果提供了聚类标签，排除离群点后计算重心。

        Args:
            point_cloud: 点云数据
            cluster_labels: 聚类标签（可选）

        Returns:
            np.ndarray: 重心坐标 [valence, arousal, focus]
        """
        if cluster_labels is not None:
            # 排除离群点（标签为 -1）
            mask = cluster_labels != -1
            if np.any(mask):
                return np.mean(point_cloud[mask], axis=0)

        return np.mean(point_cloud, axis=0)
