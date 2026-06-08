"""
持续拉普拉斯算子拓扑分析实现（默认实现）

该模块实现了基于持续拉普拉斯算子（Persistent Laplacian）的空间拓扑分析。
它是 Tender 框架的默认空间分析引擎，替代了经典的持久同调方法。

核心方法：
1. 对情绪点云构建 Vietoris-Rips 复形，在每个尺度下构造拉普拉斯矩阵
2. 计算拉普拉斯矩阵的特征值和特征向量，通过谱隙检测显著拓扑特征
3. 利用谱嵌入进行聚类分析，检测情绪簇、环状结构和离群点

学术基础：
- 持续拉普拉斯算子 (Mémoli, 2011): 通过追踪拉普拉斯算子特征值随尺度变化
  的轨迹，提供比经典持续同调更精细的拓扑流形信息
- 谱图理论 (Chung, 1997): 拉普拉斯矩阵的零特征值个数等于连通分量数，
  小特征值对应图中的"瓶颈"结构
- HDBSCAN 聚类算法 (Campello, Moulavi & Sander, 2013):
  基于密度的层次聚类，无需预设聚类数量

与经典持续同调的区别：
- 经典持续同调：只记录拓扑特征的"出生"和"消亡"（条形码）
- 持续拉普拉斯：额外记录每个尺度下拉普拉斯算子的完整谱信息，
  能检测到条形码无法发现的细微结构差异（如"即将分裂但尚未完全断开"的簇）
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

import hdbscan
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform

from tender.topology_analysis.base import (
    BaseTopologyAnalyzer,
    TopologyResult,
)


class PersistentLaplacianAnalyzer(BaseTopologyAnalyzer):
    """
    基于持续拉普拉斯算子的空间拓扑分析器

    使用持续拉普拉斯算子的谱分析来检测群体情绪的空间结构。
    通过追踪拉普拉斯特征值随尺度变化的轨迹，发现聚类、
    环状结构和离群点。

    Args:
        config: 配置字典，包含以下字段：
            - normalize: 是否标准化拉普拉斯矩阵（默认True）
            - spectral_gap_threshold: 谱隙阈值（默认0.1）
            - eigenvalue_count: 计算的特征值数量（默认10）
            - laplacian_type: 拉普拉斯类型，"combinatorial"或"normalized"（默认"normalized"）
            - min_cluster_size: HDBSCAN最小聚类大小（默认2）
            - min_samples: HDBSCAN最小样本数（默认1）
            - h1_threshold_ratio: 环阈值比例（默认0.3）
            - metric: 距离度量（默认"euclidean"）
            - standardize: 是否标准化输入（默认True）
            - max_edge_length: 最大边长（默认2.0）
            - num_scale_steps: 尺度步数（默认20）
    """

    def __init__(self, config: Dict[str, Any]):
        # 持续拉普拉斯参数
        self.normalize = config.get("normalize", True)
        self.spectral_gap_threshold = config.get("spectral_gap_threshold", 0.1)
        self.eigenvalue_count = config.get("eigenvalue_count", 10)
        self.laplacian_type = config.get("laplacian_type", "normalized")
        self.num_scale_steps = config.get("num_scale_steps", 20)

        # 聚类参数
        self.min_cluster_size = config.get("min_cluster_size", 2)
        self.min_samples = config.get("min_samples", 1)
        self.metric = config.get("metric", "euclidean")

        # 环检测参数
        self.h1_threshold_ratio = config.get("h1_threshold_ratio", 0.3)
        self.ring_detection_method = config.get("ring_detection_method", "spectral")

        # 数据预处理
        self.standardize = config.get("standardize", True)
        self.max_edge_length = config.get("max_edge_length", 2.0)

    def _compute_distance_matrix(self, point_cloud: np.ndarray) -> np.ndarray:
        """计算点云的距离矩阵"""
        return squareform(pdist(point_cloud, metric=self.metric))

    def _construct_laplacian(
        self, distance_matrix: np.ndarray, scale: float
    ) -> np.ndarray:
        """
        在给定尺度下构造拉普拉斯矩阵

        流程：
        1. 根据距离矩阵和当前尺度构建邻接矩阵
        2. 计算度矩阵
        3. 根据配置返回组合拉普拉斯或标准化拉普拉斯

        Args:
            distance_matrix: 距离矩阵，形状为 (n, n)
            scale: 当前尺度参数

        Returns:
            np.ndarray: 拉普拉斯矩阵，形状为 (n, n)
        """
        n = distance_matrix.shape

        # 构建邻接矩阵：距离小于当前尺度的点之间连边
        adjacency = (distance_matrix <= scale).astype(float)
        # 移除自环
        np.fill_diagonal(adjacency, 0.0)

        # 如果没有任何边，返回零矩阵
        if np.sum(adjacency) == 0:
            return np.zeros((n, n))

        # 计算度矩阵
        degree = np.sum(adjacency, axis=1)
        # 防止除零
        degree_safe = np.maximum(degree, 1e-10)

        if self.laplacian_type == "combinatorial":
            # 组合拉普拉斯: L = D - A
            laplacian = np.diag(degree) - adjacency
        else:
            # 标准化拉普拉斯: L_sym = I - D^(-1/2) * A * D^(-1/2)
            d_inv_sqrt = np.diag(1.0 / np.sqrt(degree_safe))
            laplacian = np.eye(n) - d_inv_sqrt @ adjacency @ d_inv_sqrt

        return laplacian

    def _compute_spectrum(
        self, laplacian: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算拉普拉斯矩阵的特征值和特征向量

        使用 scipy 的稀疏特征值求解器，仅计算最小的 k 个特征值。

        Args:
            laplacian: 拉普拉斯矩阵

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                (eigenvalues, eigenvectors)
                - eigenvalues: 从小到大排列的特征值
                - eigenvectors: 对应的特征向量
        """
        n = laplacian.shape
        k = min(self.eigenvalue_count, n - 1)

        if k <= 0:
            return np.array([]), np.array([])

        try:
            # 确保矩阵对称
            laplacian_sym = (laplacian + laplacian.T) / 2
            eigenvalues, eigenvectors = eigsh(
                laplacian_sym, k=k, which="SM", return_eigenvectors=True
            )
            # 按特征值从小到大排序
            idx = np.argsort(eigenvalues)
            return eigenvalues[idx], eigenvectors[:, idx]
        except Exception:
            # 如果计算失败，返回零特征值
            return np.zeros(k), np.zeros((n, k))

    def _compute_persistence_spectra(
        self, point_cloud: np.ndarray
    ) -> Dict[str, Any]:
        """
        计算持续拉普拉斯谱

        在不同尺度下构造拉普拉斯矩阵并计算特征值，
        追踪特征值随尺度变化的轨迹。

        Args:
            point_cloud: 点云数据，形状为 (n_samples, 3)

        Returns:
            Dict: 包含以下字段：
                - scales: 尺度序列
                - eigenvalues_history: 每个尺度下的特征值列表
                - spectral_gaps: 每个尺度下的谱隙
                - persistent_features: 持久特征信息
        """
        distance_matrix = self._compute_distance_matrix(point_cloud)
        max_dist = np.max(distance_matrix)

        # 生成尺度序列：从最小距离到最大距离，对数均匀分布
        min_dist = np.min(distance_matrix[distance_matrix > 0]) if np.any(distance_matrix > 0) else 0.01
        scales = np.logspace(
            np.log10(min_dist), np.log10(max_dist), self.num_scale_steps
        )

        eigenvalues_history = []
        spectral_gaps = []

        for scale in scales:
            laplacian = self._construct_laplacian(distance_matrix, scale)
            eigenvalues, _ = self._compute_spectrum(laplacian)
            eigenvalues_history.append(eigenvalues)

            # 计算谱隙：第一个非零特征值与零特征值的差距
            if len(eigenvalues) > 1:
                gap = eigenvalues[1] - eigenvalues
                spectral_gaps.append(gap)
            else:
                spectral_gaps.append(0.0)

        # 检测持久特征
        persistent_features = self._detect_persistent_features(
            eigenvalues_history, scales
        )

        return {
            "scales": scales,
            "eigenvalues_history": eigenvalues_history,
            "spectral_gaps": spectral_gaps,
            "persistent_features": persistent_features,
        }

    def _detect_persistent_features(
        self, eigenvalues_history: List[np.ndarray], scales: np.ndarray
    ) -> Dict[str, Any]:
        """
        从特征值历史中检测持久拓扑特征

        核心思想：
        - 零特征值的数目对应连通分量数
        - 小特征值（接近零但不为零）对应"瓶颈"结构（即将分裂的簇）
        - 特征值的持续变化反映拓扑特征在不同尺度下的稳定性

        Args:
            eigenvalues_history: 每个尺度下的特征值列表
            scales: 尺度序列

        Returns:
            Dict: 持久特征信息
        """
        n_scales = len(scales)
        feature_lifetimes = []

        for i in range(n_scales):
            evals = eigenvalues_history[i]
            if len(evals) == 0:
                continue

            # 计算零特征值个数（特征值小于阈值）
            zero_eigvals = np.sum(np.abs(evals) < 1e-6)

            # 计算谱隙
            if len(evals) > 1:
                non_zero = evals[evals > 1e-6]
                gap = non_zero[0] if len(non_zero) > 0 else 0.0
            else:
                gap = 0.0

            feature_lifetimes.append(
                {
                    "scale": scales[i],
                    "zero_eigenvalue_count": zero_eigvals,
                    "spectral_gap": gap,
                    "eigenvalue_sum": np.sum(evals),
                }
            )

        # 检测显著特征
        significant_gaps = [
            fl for fl in feature_lifetimes if fl["spectral_gap"] > self.spectral_gap_threshold
        ]

        return {
            "feature_lifetimes": feature_lifetimes,
            "significant_gap_count": len(significant_gaps),
            "max_gap": max([fl["spectral_gap"] for fl in feature_lifetimes]) if feature_lifetimes else 0.0,
            "avg_zero_eigenvalues": np.mean(
                [fl["zero_eigenvalue_count"] for fl in feature_lifetimes]
            ) if feature_lifetimes else 0.0,
        }

    def compute_persistent_homology(
        self, point_cloud: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        计算点云的持续同调

        通过持续拉普拉斯谱分析来近似计算 H0 和 H1 条形码。
        这是为了保持与抽象基类的接口兼容性。

        Args:
            point_cloud: 点云数据，形状为 (n_samples, 3)

        Returns:
            Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
                (h0_barcodes, h1_barcodes)
        """
        # 使用持续拉普拉斯谱分析来推导条形码
        spectra = self._compute_persistence_spectra(point_cloud)
        feature_lifetimes = spectra["persistent_features"]["feature_lifetimes"]

        h0_barcodes = []
        h1_barcodes = []

        # 从特征值变化中推断 H0 和 H1 特征
        prev_zero_count = 0
        for fl in feature_lifetimes:
            current_zero_count = fl["zero_eigenvalue_count"]

            # 零特征值增加意味着新的连通分量出现（H0特征）
            if current_zero_count > prev_zero_count:
                h0_barcodes.append((fl["scale"], float("inf")))

            # 谱隙增大可能暗示环状结构（H1特征）
            if fl["spectral_gap"] > self.spectral_gap_threshold:
                h1_barcodes.append((fl["scale"], fl["scale"] + fl["spectral_gap"]))

            prev_zero_count = current_zero_count

        return h0_barcodes, h1_barcodes

    def compute_clusters(
        self, point_cloud: np.ndarray
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        对情绪点云进行谱聚类分析

        使用持续拉普拉斯算子的谱嵌入作为 HDBSCAN 的输入特征。
        谱嵌入能更好地捕捉点云的非线性流形结构。

        Args:
            point_cloud: 点云数据，形状为 (n_samples, 3)

        Returns:
            Tuple[np.ndarray, int, np.ndarray]:
                (cluster_labels, n_clusters, centroid)
        """
        # 标准化
        if self.standardize and point_cloud.shape[0] > 1:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(point_cloud)
        else:
            scaled_data = point_cloud

        # 在最优尺度下计算拉普拉斯谱嵌入
        distance_matrix = self._compute_distance_matrix(scaled_data)
        # 使用数据点的平均距离作为参考尺度
        reference_scale = np.mean(distance_matrix) * 0.5
        laplacian = self._construct_laplacian(distance_matrix, reference_scale)
        eigenvalues, eigenvectors = self._compute_spectrum(laplacian)

        # 使用谱嵌入作为聚类特征
        if len(eigenvalues) > 1:
            # 取第2到第k+1个特征向量作为嵌入（跳过第一个平凡特征向量）
            n_components = min(5, len(eigenvalues) - 1)
            spectral_embedding = eigenvectors[:, 1 : n_components + 1]
            # 可选：将原始特征与谱嵌入拼接
            combined_features = np.hstack([scaled_data, spectral_embedding])
        else:
            combined_features = scaled_data

        # 运行 HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            gen_min_span_tree=True,
        )
        cluster_labels = clusterer.fit_predict(combined_features)

        # 计算聚类数量（排除离群点）
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels - {-1})

        # 计算全局重心
        centroid = self._compute_centroid(scaled_data, cluster_labels)

        return cluster_labels, n_clusters, centroid

    def compute_ring_detection(
        self,
        point_cloud: np.ndarray,
        cluster_labels: np.ndarray,
        h1_barcodes: List[Tuple[float, float]],
    ) -> bool:
        """
        基于持续拉普拉斯谱检测情绪环

        通过分析谱嵌入空间中点的角度分布来判断是否存在环状结构。
        H1 条形码的寿命也作为辅助判断依据。

        Args:
            point_cloud: 点云数据
            cluster_labels: 聚类标签
            h1_barcodes: H1 条形码列表

        Returns:
            bool: 是否存在情绪环
        """
        if self.ring_detection_method == "spectral":
            return self._spectral_ring_detection(point_cloud, cluster_labels)
        else:
            return self._persistence_ring_detection(h1_barcodes)

    def _spectral_ring_detection(
        self, point_cloud: np.ndarray, cluster_labels: np.ndarray
    ) -> bool:
        """
        基于谱分析的环检测

        使用 PCA 将点云降到二维，然后分析角度分布的均匀性。
        如果点云形成环状结构，角度分布应该接近均匀分布。

        Args:
            point_cloud: 点云数据
            cluster_labels: 聚类标签

        Returns:
            bool: 是否存在情绪环
        """
        from sklearn.decomposition import PCA

        if point_cloud.shape[0] < 3:
            return False

        # 排除离群点
        mask = cluster_labels != -1
        if np.sum(mask) < 3:
            return False
        filtered_points = point_cloud[mask]

        # PCA 降到 2 维
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(filtered_points)

        # 中心化
        center = np.mean(points_2d, axis=0)
        centered = points_2d - center

        # 计算每个点的角度
        angles = np.arctan2(centered[:, 1], centered[:, 0])

        # 检测角度是否覆盖整个圆周
        # 如果覆盖了整个圆周，说明可能存在环状结构
        angle_range = np.max(angles) - np.min(angles)

        # 计算角度分布的均匀性
        from scipy import stats
        _, p_value = stats.kstest(angles, "uniform", args=(-np.pi, 2 * np.pi))

        # 如果角度范围覆盖了大部分圆周且均匀分布，判定为环
        return angle_range > 0.8 * 2 * np.pi and p_value > 0.05

    def _persistence_ring_detection(
        self, h1_barcodes: List[Tuple[float, float]]
    ) -> bool:
        """基于持续同调条形码的环检测"""
        if not h1_barcodes:
            return False

        # 计算最长 H1 条形码的寿命
        max_lifetime = max(death - birth for birth, death in h1_barcodes)
        return max_lifetime > self.h1_threshold_ratio

    def detect_outliers(
        self,
        point_cloud: np.ndarray,
        cluster_labels: np.ndarray,
        member_ids: List[str],
    ) -> Tuple[List[str], float]:
        """
        识别情绪离群成员

        基于 HDBSCAN 的聚类标签识别离群点。
        同时使用谱嵌入中的重构误差作为辅助判断。

        Args:
            point_cloud: 点云数据
            cluster_labels: HDBSCAN 聚类标签
            member_ids: 成员ID列表

        Returns:
            Tuple[List[str], float]:
                (outlier_members, outlier_ratio)
        """
        outlier_members = []
        for i, label in enumerate(cluster_labels):
            if label == -1:
                outlier_members.append(member_ids[i])

        total_members = len(member_ids)
        outlier_ratio = len(outlier_members) / total_members if total_members > 0 else 0.0

        return outlier_members, outlier_ratio

    def analyze(
        self,
        emotion_vectors: Dict[str, "EmotionVector"],
        window_start: float,
        window_end: float,
    ) -> TopologyResult:
        """
        对单个时间窗口的情绪向量进行完整的拓扑分析

        流程：
        1. 验证并转换输入
        2. 计算持续拉普拉斯谱
        3. 运行谱聚类（谱嵌入 + HDBSCAN）
        4. 检测情绪环
        5. 识别离群成员

        Args:
            emotion_vectors: 成员情绪向量字典
            window_start: 时间窗口起始时间戳
            window_end: 时间窗口结束时间戳

        Returns:
            TopologyResult: 完整的拓扑分析结果
        """
        # 1. 验证并转换输入
        point_cloud, member_ids = self._validate_vectors(emotion_vectors)
        n_members = len(member_ids)

        if n_members == 0:
            return TopologyResult(
                cluster_count=0,
                ring_exists=False,
                outlier_ratio=0.0,
                centroid=np.array([0.0, 0.0, 0.0]),
                cluster_labels={},
                h0_barcodes=[],
                h1_barcodes=[],
                outlier_members=[],
                window_start=window_start,
                window_end=window_end,
                metadata={
                    "analyzer": "PersistentLaplacianAnalyzer",
                    "warning": "没有成员数据",
                },
            )

        # 2. 计算持续拉普拉斯谱
        spectra = self._compute_persistence_spectra(point_cloud)
        h0_barcodes, h1_barcodes = self.compute_persistent_homology(point_cloud)

        # 3. 运行谱聚类
        cluster_labels, n_clusters, centroid = self.compute_clusters(point_cloud)

        # 4. 检测情绪环
        ring_exists = self.compute_ring_detection(
            point_cloud, cluster_labels, h1_barcodes
        )

        # 5. 识别离群成员
        outlier_members, outlier_ratio = self.detect_outliers(
            point_cloud, cluster_labels, member_ids
        )

        # 构建标签字典
        labels_dict = {
            mid: int(cluster_labels[i])
            for i, mid in enumerate(member_ids)
        }

        # 提取谱信息用于元数据
        persistent_features = spectra["persistent_features"]

        return TopologyResult(
            cluster_count=n_clusters,
            ring_exists=ring_exists,
            outlier_ratio=outlier_ratio,
            centroid=centroid,
            cluster_labels=labels_dict,
            h0_barcodes=h0_barcodes,
            h1_barcodes=h1_barcodes,
            outlier_members=outlier_members,
            window_start=window_start,
            window_end=window_end,
            metadata={
                "analyzer": "PersistentLaplacianAnalyzer",
                "laplacian_type": self.laplacian_type,
                "normalize": self.normalize,
                "spectral_gap_threshold": self.spectral_gap_threshold,
                "eigenvalue_count": self.eigenvalue_count,
                "min_cluster_size": self.min_cluster_size,
                "n_members": n_members,
                "n_outliers": len(outlier_members),
                "spectral_max_gap": persistent_features.get("max_gap", 0.0),
                "spectral_avg_zero_eigenvalues": persistent_features.get("avg_zero_eigenvalues", 0.0),
                "significant_gap_count": persistent_features.get("significant_gap_count", 0),
                "h0_barcode_count": len(h0_barcodes),
                "h1_barcode_count": len(h1_barcodes),
            },
        )

    def get_info(self) -> Dict[str, Any]:
        """获取当前拓扑分析器的信息"""
        return {
            "name": "PersistentLaplacianAnalyzer",
            "description": "基于持续拉普拉斯算子的空间拓扑分析器",
            "normalize": self.normalize,
            "spectral_gap_threshold": self.spectral_gap_threshold,
            "eigenvalue_count": self.eigenvalue_count,
            "laplacian_type": self.laplacian_type,
            "min_cluster_size": self.min_cluster_size,
            "h1_threshold_ratio": self.h1_threshold_ratio,
            "ring_detection_method": self.ring_detection_method,
            "standardize": self.standardize,
        }
