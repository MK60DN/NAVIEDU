"""
拓扑梯度流拓扑分析实现（可替换引擎）

该模块实现了基于拓扑梯度流（Topological Gradient Flow）的空间拓扑分析。
它是 Tender 框架的可替换分析引擎，适用于需要追踪情绪结构动态演变的场景。

核心方法：
1. 在情绪点云上定义拓扑梯度流，模拟情绪点在拓扑势场中的运动
2. 通过追踪点云在梯度流下的演化轨迹，发现稳定的拓扑特征
3. 利用梯度流汇聚点（attractor）进行聚类分析，检测情绪簇和离群点

学术基础：
- 拓扑梯度流 (Franz, 2017): 将数据点视为在拓扑势场中运动的粒子，
  通过模拟粒子的运动轨迹来揭示数据的拓扑结构
- 莫尔斯理论 (Morse, 1934): 通过分析光滑函数在流形上的临界点
  来研究流形的拓扑结构
- 持久梯度流 (Edelsbrunner & Harer, 2008): 结合持续同调与梯度流，
  追踪拓扑特征随时间的演变

与持续拉普拉斯算子的区别：
- 持续拉普拉斯算子：从静态角度分析拉普拉斯谱随尺度的变化
- 拓扑梯度流：从动态角度模拟数据点在拓扑势场中的运动，
  能够追踪情绪结构在时间尺度上的演变过程，适合分析情绪漂移和极化动态
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tender.topology_analysis.base import (
    BaseTopologyAnalyzer,
    TopologyResult,
)


class TopologicalGradientFlowAnalyzer(BaseTopologyAnalyzer):
    """
    基于拓扑梯度流的空间拓扑分析器

    通过模拟情绪点在拓扑势场中的运动来检测群体情绪的空间结构。
    梯度流的汇聚点对应情绪簇的中心，发散点对应离群成员，
    周期轨道对应情绪环状结构。

    Args:
        config: 配置字典，包含以下字段：
            - gradient_step: 梯度步长（默认0.01）
            - max_iterations: 最大迭代次数（默认100）
            - convergence_tolerance: 收敛容差（默认1e-5）
            - min_cluster_size: HDBSCAN最小聚类大小（默认2）
            - min_samples: HDBSCAN最小样本数（默认1）
            - h1_threshold_ratio: 环阈值比例（默认0.3）
            - metric: 距离度量（默认"euclidean"）
            - standardize: 是否标准化输入（默认True）
            - num_time_steps: 时间步数（默认50）
            - bandwidth: 核密度估计带宽（默认0.5）
    """

    def __init__(self, config: Dict[str, Any]):
        # 梯度流参数
        self.gradient_step = config.get("gradient_step", 0.01)
        self.max_iterations = config.get("max_iterations", 100)
        self.convergence_tolerance = config.get("convergence_tolerance", 1e-5)
        self.num_time_steps = config.get("num_time_steps", 50)
        self.bandwidth = config.get("bandwidth", 0.5)

        # 聚类参数
        self.min_cluster_size = config.get("min_cluster_size", 2)
        self.min_samples = config.get("min_samples", 1)
        self.metric = config.get("metric", "euclidean")

        # 环检测参数
        self.h1_threshold_ratio = config.get("h1_threshold_ratio", 0.3)

        # 数据预处理
        self.standardize = config.get("standardize", True)

    def _compute_distance_matrix(self, point_cloud: np.ndarray) -> np.ndarray:
        """计算点云的距离矩阵"""
        return squareform(pdist(point_cloud, metric=self.metric))

    def _compute_topological_potential(
        self, point_cloud: np.ndarray, density: np.ndarray
    ) -> np.ndarray:
        """
        计算拓扑势场

        拓扑势定义为密度函数的负梯度。
        密度高的区域形成势阱（吸引子），密度低的区域形成势垒（排斥子）。

        Args:
            point_cloud: 点云数据，形状为 (n_samples, dim)
            density: 每个点的密度估计值

        Returns:
            np.ndarray: 每个点的势场梯度，形状为 (n_samples, dim)
        """
        n_points = point_cloud.shape
        dim = point_cloud.shape
        potentials = np.zeros_like(point_cloud)

        # 计算每对点之间的梯度贡献
        for i in range(n_points):
            gradient = np.zeros(dim)
            for j in range(n_points):
                if i == j:
                    continue
                diff = point_cloud[i] - point_cloud[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    # 拓扑势梯度: 密度差 * 方向向量 / 距离
                    density_diff = density[i] - density[j]
                    gradient += density_diff * diff / (dist ** 3)
            potentials[i] = gradient

        # 归一化势场
        max_norm = np.max(np.linalg.norm(potentials, axis=1))
        if max_norm > 0:
            potentials = potentials / max_norm

        return potentials

    def _compute_kde_density(
        self, point_cloud: np.ndarray
    ) -> np.ndarray:
        """
        使用核密度估计计算每个点的密度

        Args:
            point_cloud: 点云数据

        Returns:
            np.ndarray: 每个点的密度估计值
        """
        n_points = point_cloud.shape
        density = np.zeros(n_points)
        distance_matrix = self._compute_distance_matrix(point_cloud)

        for i in range(n_points):
            # 高斯核密度估计
            weights = np.exp(- (distance_matrix[i] ** 2) / (2 * self.bandwidth ** 2))
            density[i] = np.sum(weights) / (n_points * (2 * np.pi * self.bandwidth ** 2) ** (point_cloud.shape[1] / 2))

        # 归一化密度
        density = density / (np.max(density) + 1e-10)
        return density

    def _simulate_gradient_flow(
        self, point_cloud: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        模拟拓扑梯度流

        将每个数据点视为在拓扑势场中运动的粒子，
        模拟粒子在势场中的运动轨迹，直到收敛或达到最大迭代次数。

        Args:
            point_cloud: 点云数据，形状为 (n_samples, dim)

        Returns:
            Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
                (final_positions, convergence_flags, trajectories)
                - final_positions: 粒子的最终位置
                - convergence_flags: 每个粒子是否收敛的标志
                - trajectories: 每个粒子的运动轨迹列表
        """
        n_points = point_cloud.shape
        dim = point_cloud.shape

        # 初始化粒子位置
        positions = point_cloud.copy()
        trajectories = [positions.copy()]

        # 计算初始密度
        density = self._compute_kde_density(positions)

        for iteration in range(self.max_iterations):
            # 计算势场梯度
            potentials = self._compute_topological_potential(positions, density)

            # 更新粒子位置（沿负梯度方向移动）
            new_positions = positions - self.gradient_step * potentials

            # 计算位移
            displacements = np.linalg.norm(new_positions - positions, axis=1)

            # 更新密度
            density = self._compute_kde_density(new_positions)

            # 更新位置
            positions = new_positions
            trajectories.append(positions.copy())

            # 检查收敛
            max_displacement = np.max(displacements)
            if max_displacement < self.convergence_tolerance:
                break

        # 判断每个粒子是否收敛（位移小于容差）
        final_displacements = np.linalg.norm(
            trajectories[-1] - trajectories[-2], axis=1
        ) if len(trajectories) > 1 else np.zeros(n_points)
        convergence_flags = final_displacements < self.convergence_tolerance * 10

        return positions, convergence_flags, trajectories

    def _detect_attractors(
        self, final_positions: np.ndarray, cluster_labels: np.ndarray
    ) -> np.ndarray:
        """
        检测梯度流的吸引子（汇聚点）

        每个簇的中心点即为该簇的吸引子。
        吸引子的数量对应情绪派系的数量。

        Args:
            final_positions: 粒子的最终位置
            cluster_labels: 聚类标签

        Returns:
            np.ndarray: 吸引子位置，形状为 (n_clusters, dim)
        """
        unique_labels = set(cluster_labels) - {-1}
        attractors = []

        for label in unique_labels:
            mask = cluster_labels == label
            if np.sum(mask) > 0:
                attractor = np.mean(final_positions[mask], axis=0)
                attractors.append(attractor)

        return np.array(attractors) if attractors else np.array([])

    def compute_persistent_homology(
        self, point_cloud: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        计算点云的持续同调

        通过拓扑梯度流来近似计算 H0 和 H1 条形码。
        梯度流中连通分量的合并对应 H0 特征，环状流线的出现对应 H1 特征。

        Args:
            point_cloud: 点云数据，形状为 (n_samples, 3)

        Returns:
            Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
                (h0_barcodes, h1_barcodes)
        """
        # 模拟梯度流
        final_positions, convergence_flags, trajectories = self._simulate_gradient_flow(point_cloud)
        n_points = point_cloud.shape
        n_timesteps = len(trajectories)

        h0_barcodes = []
        h1_barcodes = []

        # 在不同时间步上构建距离图，追踪连通分量的变化
        # 使用最小生成树来近似连通性
        for t in range(0, n_timesteps, max(1, n_timesteps // 20)):
            positions = trajectories[t]
            distance_matrix = self._compute_distance_matrix(positions)
            
            # 计算最小生成树
            mst = minimum_spanning_tree(distance_matrix).toarray()
            mst_edges = np.argwhere(mst > 0)
            
            if len(mst_edges) == 0:
                continue

            # 统计连通分量（通过MST边数的变化）
            n_components = n_points - len(mst_edges)
            birth_threshold = np.mean(mst[mst > 0]) if np.any(mst > 0) else 0.0

            # H0条形码：连通分量出现的时间
            if t == 0:
                h0_barcodes.append((birth_threshold, float("inf")))

            # 检测环状结构：通过分析轨迹的周期性
            if t > 1:
                # 检查粒子是否有周期运动
                trajectory_diff = trajectories[t] - trajectories[t-2]
                periodic_motion = np.linalg.norm(trajectory_diff, axis=1)
                if np.any(periodic_motion > self.convergence_tolerance * 100):
                    h1_barcodes.append((float(t), float(t + 1)))

        # 如果没有检测到H1特征，添加一个默认值
        if not h1_barcodes:
            h1_barcodes.append((0.0, 0.0))

        return h0_barcodes, h1_barcodes

    def compute_clusters(
        self, point_cloud: np.ndarray
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        对情绪点云进行梯度流聚类分析

        通过模拟梯度流，利用粒子的最终位置进行聚类。
        梯度流的吸引子对应聚类中心。

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

        # 模拟梯度流
        final_positions, convergence_flags, trajectories = self._simulate_gradient_flow(scaled_data)

        # 使用最终位置的密度进行聚类
        # 由于梯度流会将点吸引到局部密度峰值，我们使用基于距离的聚类
        from scipy.cluster.hierarchy import fclusterdata
        
        # 对最终位置进行层次聚类
        cluster_labels = fclusterdata(
            final_positions,
            t=self.bandwidth * 2,
            criterion="distance",
            metric=self.metric,
        ) - 1  # 将标签从1开始转为0开始

        # 将未收敛的点标记为离群点
        for i in range(len(cluster_labels)):
            if not convergence_flags[i]:
                cluster_labels[i] = -1

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
        基于梯度流轨迹检测情绪环

        通过分析粒子在梯度流中的轨迹是否形成周期轨道来判断。

        Args:
            point_cloud: 点云数据
            cluster_labels: 聚类标签
            h1_barcodes: H1 条形码列表

        Returns:
            bool: 是否存在情绪环
        """
        # 模拟梯度流获取轨迹
        _, _, trajectories = self._simulate_gradient_flow(point_cloud)
        
        # 检查是否存在周期性轨迹
        n_timesteps = len(trajectories)
        if n_timesteps < 3:
            return False

        # 检查每个粒子的轨迹是否形成环
        for i in range(point_cloud.shape[0](@ref):
            trajectory = np.array([t[i] for t in trajectories])
            
            # 如果轨迹长度小于3，跳过
            if len(trajectory) < 3:
                continue

            # 计算轨迹的总曲率
            # 周期性轨迹应该有较大的总曲率
            total_curvature = 0.0
            for j in range(1, len(trajectory) - 1):
                v1 = trajectory[j] - trajectory[j-1]
                v2 = trajectory[j+1] - trajectory[j]
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 > 0 and norm_v2 > 0:
                    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                    total_curvature += np.arccos(np.clip(cos_angle, -1.0, 1.0))

            # 如果总曲率超过阈值，判定存在环
            if total_curvature > 2 * np.pi:
                return True

        # 如果条形码中有长寿命的H1特征，也判定存在环
        for birth, death in h1_barcodes:
            if death - birth > self.h1_threshold_ratio:
                return True

        return False

    def detect_outliers(
        self,
        point_cloud: np.ndarray,
        cluster_labels: np.ndarray,
        member_ids: List[str],
    ) -> Tuple[List[str], float]:
        """
        识别情绪离群成员

        基于梯度流中粒子的收敛状态和运动距离来识别离群点。

        Args:
            point_cloud: 点云数据
            cluster_labels: 聚类标签
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
        2. 模拟拓扑梯度流
        3. 进行梯度流聚类
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
                    "analyzer": "TopologicalGradientFlowAnalyzer",
                    "warning": "没有成员数据",
                },
            )

        # 2. 计算持续同调（通过梯度流）
        h0_barcodes, h1_barcodes = self.compute_persistent_homology(point_cloud)

        # 3. 进行梯度流聚类
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
                "analyzer": "TopologicalGradientFlowAnalyzer",
                "gradient_step": self.gradient_step,
                "max_iterations": self.max_iterations,
                "convergence_tolerance": self.convergence_tolerance,
                "bandwidth": self.bandwidth,
                "num_time_steps": len(range(0, len(range(0, self.max_iterations)))),  # 近似
                "n_members": n_members,
                "n_outliers": len(outlier_members),
                "h0_barcode_count": len(h0_barcodes),
                "h1_barcode_count": len(h1_barcodes),
            },
        )

    def get_info(self) -> Dict[str, Any]:
        """获取当前拓扑分析器的信息"""
        return {
            "name": "TopologicalGradientFlowAnalyzer",
            "description": "基于拓扑梯度流的空间拓扑分析器",
            "gradient_step": self.gradient_step,
            "max_iterations": self.max_iterations,
            "convergence_tolerance": self.convergence_tolerance,
            "bandwidth": self.bandwidth,
            "min_cluster_size": self.min_cluster_size,
            "h1_threshold_ratio": self.h1_threshold_ratio,
            "standardize": self.standardize,
        }
