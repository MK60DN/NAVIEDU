"""
PC算法+LiNGAM 因果分析实现（可替换引擎）

该模块实现了基于 PC 算法与 LiNGAM 混合方法的时间因果分析。
它是 Tender 框架的可替换因果分析引擎，适用于需要从观测数据中发现因果方向，
且不依赖时间序列结构的场景。

核心方法：
1. 第一阶段 - PC 算法：使用条件独立性检验发现因果骨架（无向图）
2. 第二阶段 - LiNGAM：使用线性非高斯假设确定因果方向
3. 构建有向无环图 (DAG)，识别超级传播者和关键接收者

学术基础：
- PC 算法 (Spirtes, Glymour & Scheines, 2000): 基于条件独立性检验的因果发现算法，
  从观测数据中学习因果结构的骨架（无向边），是约束学习方法的代表
- LiNGAM (Shimizu et al., 2006): 线性非高斯无环模型，利用非高斯分布信息
  来识别因果方向
- 多阶段混合方法：PC 算法负责结构发现，LiNGAM 负责方向确定，
  两者结合比单独使用更准确

与 CCM 和 SCM 的区别：
- CCM：基于流形预测，需要足够长的时间序列
- SCM：基于结构性方程，支持干预模拟和反事实推理
- PC+LiNGAM：基于条件独立性和非高斯性，在观测数据中发现因果方向，
  对时间序列长度要求低于 CCM，对模型假设要求低于 SCM
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from itertools import combinations

from scipy.stats import pearsonr, spearmanr
from scipy.special import gammainc
from sklearn.linear_model import LinearRegression

from tender.causal_analysis.base import (
    BaseCausalAnalyzer,
    CausalResult,
    CausalEdge,
)


class PCLiNGAMAnalyzer(BaseCausalAnalyzer):
    """
    基于 PC 算法 + LiNGAM 的时间因果分析器

    使用两阶段方法从观测数据中学习因果结构：
    第一阶段使用 PC 算法发现因果骨架，
    第二阶段使用 LiNGAM 确定因果方向。

    Args:
        config: 配置字典，包含以下字段：
            - independence_test: 独立性检验方法（默认"fisherz"）
            - alpha: 显著性水平（默认0.05）
            - num_bootstrap: 自助法采样次数（默认100）
            - emotion_dimension: 分析维度（默认"composite"）
            - max_lag: 最大滞后阶数（默认5）
            - seed: 随机种子（默认42）
            - max_conditioning_set_size: 最大条件集大小（默认3）
            - use_bootstrap_ci: 是否使用自助法置信区间（默认True）
    """

    def __init__(self, config: Dict[str, Any]):
        # PC 算法参数
        self.independence_test = config.get("independence_test", "fisherz")
        self.alpha = config.get("alpha", 0.05)
        self.max_conditioning_set_size = config.get("max_conditioning_set_size", 3)

        # LiNGAM 参数
        self.num_bootstrap = config.get("num_bootstrap", 100)
        self.use_bootstrap_ci = config.get("use_bootstrap_ci", False)

        # 通用参数
        self.emotion_dimension = config.get("emotion_dimension", "composite")
        self.max_lag = config.get("max_lag", 5)
        self.seed = config.get("seed", 42)

        # 结果缓存
        self._pc_results: Dict[str, Any] = {}
        self._dag: Optional[np.ndarray] = None

    def _fisher_z_test(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fisher Z 检验：在给定条件集 Z 下检验 X 和 Y 的条件独立性

        零假设：X 和 Y 在给定 Z 下条件独立。
        检验统计量是基于偏相关系数的 Fisher Z 变换。

        Args:
            x: 变量 X，形状为 (n,)
            y: 变量 Y，形状为 (n,)
            z: 条件集 Z，形状为 (n, k) 或 (n,)

        Returns:
            Tuple[float, float]:
                (z_stat, p_value)
                - z_stat: Fisher Z 检验统计量
                - p_value: 显著性 p 值
        """
        n = len(x)

        # 确保 z 是二维数组
        if z.ndim == 1:
            z = z.reshape(-1, 1)

        # 如果条件集为空，计算简单相关系数
        if z.shape[1] == 0:
            r, _ = pearsonr(x, y)
        else:
            # 计算偏相关系数
            # 1. 回归 x ~ z
            model_x = LinearRegression()
            model_x.fit(z, x)
            residuals_x = x - model_x.predict(z)

            # 2. 回归 y ~ z
            model_y = LinearRegression()
            model_y.fit(z, y)
            residuals_y = y - model_y.predict(z)

            # 3. 残差的相关系数
            r, _ = spearmanr(residuals_x, residuals_y)

        # 避免 |r| >= 1
        r = np.clip(r, -0.999, 0.999)

        # Fisher Z 变换
        z_stat = np.sqrt(n - z.shape[1] - 3) * 0.5 * np.log((1 + r) / (1 - r))

        # 计算 p 值（标准正态分布的双侧检验）
        # 使用误差函数近似标准正态分布的 CDF
        from scipy.special import erfc
        p_value = erfc(np.abs(z_stat) / np.sqrt(2))

        return float(z_stat), float(p_value)

    def _perform_kernel_independence_test(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[float, float]:
        """
        基于核方法的条件独立性检验

        使用 Hilbert-Schmidt 独立性准则 (HSIC) 的扩展作为备用方法。
        当 Fisher Z 检验的假设不满足时使用。

        Args:
            x: 变量 X
            y: 变量 Y
            z: 条件集 Z

        Returns:
            Tuple[float, float]:
                (test_stat, p_value)
        """
        from scipy.spatial.distance import cdist

        n = len(x)

        # 确保 z 是二维数组
        if z.ndim == 1:
            z = z.reshape(-1, 1)

        # 对 X 和 Y 进行核主成分分析，去除 Z 的影响
        # 使用高斯核
        def rbf_kernel(a, b, sigma=1.0):
            dist = cdist(a.reshape(-1, 1), b.reshape(-1, 1))
            return np.exp(-dist ** 2 / (2 * sigma ** 2))

        K_x = rbf_kernel(x, x)
        K_y = rbf_kernel(y, y)

        if z.shape[1] > 0:
            K_z = rbf_kernel(z[:, 0], z[:, 0])
            # 中心化核矩阵
            H = np.eye(n) - np.ones((n, n)) / n
            K_x_centered = H @ K_x @ H
            K_y_centered = H @ K_y @ H
            K_z_centered = H @ K_z @ H

            # 计算条件 HSIC
            # HSIC(X, Y | Z) = HSIC(X, Y) - HSIC(X, Z) * HSIC(Z, Y)^{-1} * HSIC(Z, Y)
            # 简化为 HSIC 残差
            hsic_xy = np.trace(K_x_centered @ K_y_centered) / (n - 1) ** 2
            hsic_xz = np.trace(K_x_centered @ K_z_centered) / (n - 1) ** 2
            hsic_yz = np.trace(K_y_centered @ K_z_centered) / (n - 1) ** 2

            # 条件独立性统计量
            test_stat = hsic_xy - hsic_xz * hsic_yz
        else:
            # 无条件的 HSIC
            H = np.eye(n) - np.ones((n, n)) / n
            K_x_centered = H @ K_x @ H
            K_y_centered = H @ K_y @ H
            test_stat = np.trace(K_x_centered @ K_y_centered) / (n - 1) ** 2

        # 计算 p 值（使用 Gamma 近似）
        # 对于 HSIC，零分布可以用 Gamma 分布近似
        mean_approx = 1.0 / (n - 1)  # 近似均值
        var_approx = 2.0 / ((n - 1) ** 2)  # 近似方差

        if var_approx > 0:
            shape = mean_approx ** 2 / var_approx
            scale = var_approx / mean_approx
            p_value = 1.0 - gammainc(shape, test_stat / scale)
        else:
            p_value = 1.0

        return float(test_stat), float(p_value)

    def _perform_independence_test(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[float, float]:
        """
        执行条件独立性检验

        根据配置选择 Fisher Z 检验或核方法检验。

        Args:
            x: 变量 X
            y: 变量 Y
            z: 条件集 Z

        Returns:
            Tuple[float, float]:
                (test_stat, p_value)
        """
        if self.independence_test == "fisherz":
            return self._fisher_z_test(x, y, z)
        else:
            return self._perform_kernel_independence_test(x, y, z)

    def _compute_skeleton(
        self, data_matrix: np.ndarray, feature_names: List[str]
    ) -> np.ndarray:
        """
        使用 PC 算法计算因果骨架（第一阶段）

        流程：
        1. 从完全无向图开始
        2. 对每对变量，检验在给定条件集下的条件独立性
        3. 如果条件独立，删除对应的边
        4. 逐步增大条件集的大小

        Args:
            data_matrix: 数据矩阵，形状为 (T, n_members)
            feature_names: 特征名称列表

        Returns:
            np.ndarray: 邻接矩阵（对称），1 表示存在边
        """
        n_features = data_matrix.shape
        # 初始化完全无向图
        skeleton = np.ones((n_features, n_features)) - np.eye(n_features)

        # 逐步增大条件集大小
        for k in range(self.max_conditioning_set_size + 1):
            # 对每对变量 (i, j)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if skeleton[i, j] == 0:
                        continue  # 边已被删除

                    # 获取当前邻接的变量
                    neighbors = np.where(skeleton[i, :] > 0)
                    # 排除 j 自身
                    neighbors = neighbors[neighbors != j]

                    # 如果邻居数小于 k，无法构建大小为 k 的条件集
                    if len(neighbors) < k:
                        continue

                    # 枚举所有大小为 k 的条件集
                    for cond_set in combinations(neighbors, k):
                        # 提取条件集的列
                        z = data_matrix[:, list(cond_set)]

                        # 执行条件独立性检验
                        _, p_value = self._perform_independence_test(
                            data_matrix[:, i],
                            data_matrix[:, j],
                            z,
                        )

                        # 如果条件独立，删除边
                        if p_value > self.alpha:
                            skeleton[i, j] = 0
                            skeleton[j, i] = 0
                            break

        return skeleton

    def _orient_edges(
        self, data_matrix: np.ndarray, skeleton: np.ndarray
    ) -> np.ndarray:
        """
        使用 LiNGAM 确定因果方向（第二阶段）

        流程：
        1. 对于因果骨架中的每条边，使用非高斯性判断方向
        2. 确保最终结果是无环的 (DAG)

        Args:
            data_matrix: 数据矩阵
            skeleton: 因果骨架（对称邻接矩阵）

        Returns:
            np.ndarray: 有向邻接矩阵，A[i, j] = 1 表示 j -> i
        """
        n_features = data_matrix.shape
        # 初始化为零矩阵
        adjacency = np.zeros((n_features, n_features))

        # 对每条无向边确定方向
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if skeleton[i, j] == 0:
                    continue

                # 使用非高斯性判断方向
                # 拟合两个方向的线性模型，比较残差的非高斯性
                x = data_matrix[:, i]
                y = data_matrix[:, j]

                # 方向 1: i -> j
                model_ij = LinearRegression()
                model_ij.fit(x.reshape(-1, 1), y)
                residuals_ij = y - model_ij.predict(x.reshape(-1, 1))

                # 方向 2: j -> i
                model_ji = LinearRegression()
                model_ji.fit(y.reshape(-1, 1), x)
                residuals_ji = x - model_ji.predict(y.reshape(-1, 1))

                # 使用统计检验判断哪个方向的残差更独立于原因
                # 检验残差与自变量的独立性
                dcorr_ij, p_ij = self._perform_independence_test(
                    x, residuals_ij, np.array([])
                )
                dcorr_ji, p_ji = self._perform_independence_test(
                    y, residuals_ji, np.array([])
                )

                # 如果残差与自变量更独立的那个方向更可能是因果方向
                if dcorr_ij < dcorr_ji:
                    # i -> j 更合理
                    adjacency[j, i] = 1
                elif dcorr_ji < dcorr_ij:
                    # j -> i 更合理
                    adjacency[i, j] = 1
                else:
                    # 无法确定方向，暂时保留为双向
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1

        # 确保无环性（拓扑排序）
        adjacency = self._ensure_dag(adjacency)

        return adjacency

    def _ensure_dag(self, adjacency: np.ndarray) -> np.ndarray:
        """
        确保邻接矩阵是无环的 (Directed Acyclic Graph)

        使用拓扑排序检测环，如果存在环则移除最弱的边。

        Args:
            adjacency: 有向邻接矩阵

        Returns:
            np.ndarray: 无环的有向邻接矩阵
        """
        n = adjacency.shape
        dag = adjacency.copy()

        # 使用 Kahn 算法进行拓扑排序
        def has_cycle(graph):
            in_degree = np.sum(graph, axis=1)
            queue = [i for i in range(n) if in_degree[i] == 0]
            visited = 0

            while queue:
                node = queue.pop(0)
                visited += 1
                for j in range(n):
                    if graph[node, j] > 0:
                        in_degree[j] -= 1
                        if in_degree[j] == 0:
                            queue.append(j)

            return visited != n

        # 检测环
        max_iterations = 10
        iteration = 0

        while has_cycle(dag) and iteration < max_iterations:
            # 找到环中最弱的边并移除
            # 这里简化为移除值最小的边
            min_val = np.inf
            min_i, min_j = -1, -1

            for i in range(n):
                for j in range(n):
                    if dag[i, j] > 0 and dag[i, j] < min_val:
                        min_val = dag[i, j]
                        min_i, min_j = i, j

            if min_i >= 0 and min_j >= 0:
                dag[min_i, min_j] = 0

            iteration += 1

        return dag

    def _extract_emotion_time_series(
        self,
        time_series_data: Dict[str, List[np.ndarray]],
        member_ids: List[str],
    ) -> np.ndarray:
        """
        从时间序列数据中提取特定维度的情绪矩阵

        Args:
            time_series_data: 时间序列数据字典
            member_ids: 成员ID列表

        Returns:
            np.ndarray: 情绪数据矩阵，形状为 (T, n_members)
        """
        # 确定最小时间长度
        min_len = min(
            [len(ts) for mid, ts in time_series_data.items() if mid in member_ids],
            default=0,
        )

        if min_len < 3:
            return np.array([])

        # 构建数据矩阵
        data_matrix = np.zeros((min_len, len(member_ids)))

        for j, mid in enumerate(member_ids):
            if mid in time_series_data:
                ts = np.array(time_series_data[mid][:min_len])

                if len(ts.shape) == 2 and ts.shape[1] >= 3:
                    if self.emotion_dimension == "composite":
                        data_matrix[:, j] = np.linalg.norm(ts, axis=1)
                    elif self.emotion_dimension == "valence":
                        data_matrix[:, j] = ts[:, 0]
                    elif self.emotion_dimension == "arousal":
                        data_matrix[:, j] = ts[:, 1]
                    elif self.emotion_dimension == "focus":
                        data_matrix[:, j] = ts[:, 2]
                    else:
                        data_matrix[:, j] = np.linalg.norm(ts, axis=1)
                elif len(ts.shape) == 1:
                    data_matrix[:, j] = ts[:min_len]

        return data_matrix

    def analyze(
        self,
        time_series_data: Dict[str, List[np.ndarray]],
        member_ids: List[str],
        window_start: float,
        window_end: float,
    ) -> CausalResult:
        """
        对群体的情绪时间序列进行完整的因果分析

        流程：
        1. 提取情绪时间序列矩阵
        2. 第一阶段：PC 算法发现因果骨架
        3. 第二阶段：LiNGAM 确定因果方向
        4. 构建有向无环图 (DAG)
        5. 计算网络统计量并识别超级传播者

        Args:
            time_series_data: 时间序列数据字典
            member_ids: 成员ID列表
            window_start: 窗口起始时间戳
            window_end: 窗口结束时间戳

        Returns:
            CausalResult: 因果分析结果
        """
        n_members = len(member_ids)

        if n_members < 2:
            import networkx as nx
            return CausalResult(
                causal_graph=nx.DiGraph(),
                edges=[],
                out_degrees={mid: 0 for mid in member_ids},
                in_degrees={mid: 0 for mid in member_ids},
                super_spreaders=[],
                causal_density=0.0,
            )

        # 1. 提取情绪数据矩阵
        data_matrix = self._extract_emotion_time_series(time_series_data, member_ids)

        if data_matrix.shape[1] < 2 or data_matrix.shape[0] < 3:
            import networkx as nx
            return CausalResult(
                causal_graph=nx.DiGraph(),
                edges=[],
                out_degrees={mid: 0 for mid in member_ids},
                in_degrees={mid: 0 for mid in member_ids},
                super_spreaders=[],
                causal_density=0.0,
            )

        # 2. 第一阶段：PC 算法发现因果骨架
        skeleton = self._compute_skeleton(data_matrix, member_ids)

        # 3. 第二阶段：LiNGAM 确定因果方向
        adjacency_matrix = self._orient_edges(data_matrix, skeleton)

        # 存储 DAG 以供后续查询
        self._dag = adjacency_matrix

        # 4. 构建边列表和出入度
        edges = []
        out_degrees = defaultdict(int)
        in_degrees = defaultdict(int)

        for i in range(n_members):
            for j in range(n_members):
                if i == j:
                    continue
                if adjacency_matrix[i, j] > 0:  # j -> i
                    # 计算因果强度
                    # 使用偏相关系数作为强度的度量
                    z_others = np.delete(data_matrix, [i, j], axis=1)
                    _, p_value = self._perform_independence_test(
                        data_matrix[:, j],
                        data_matrix[:, i],
                        z_others,
                    )

                    strength = float(adjacency_matrix[i, j])

                    edges.append(CausalEdge(
                        source=member_ids[j],
                        target=member_ids[i],
                        strength=strength,
                        lag=1,
                        p_value=float(p_value),
                        method="pc_lingsam",
                    ))
                    out_degrees[member_ids[j]] += 1
                    in_degrees[member_ids[i]] += 1

        # 5. 构建因果网络图
        import networkx as nx
        causal_graph = nx.DiGraph()

        for mid in member_ids:
            causal_graph.add_node(mid)

        for edge in edges:
            causal_graph.add_edge(edge.source, edge.target)

        # 6. 计算因果密度
        total_possible_edges = n_members * (n_members - 1)
        causal_density = len(edges) / total_possible_edges if total_possible_edges > 0 else 0.0

        # 7. 识别超级传播者
        out_degree_values = list(out_degrees.values())
        if out_degree_values:
            mean_out = np.mean(out_degree_values)
            std_out = np.std(out_degree_values)
            super_spreader_threshold = mean_out + std_out
            super_spreaders = [
                mid for mid in member_ids
                if out_degrees.get(mid, 0) > super_spreader_threshold
            ]
        else:
            super_spreaders = []

        # 存储结果以供后续查询
        self._pc_results = {
            "skeleton": skeleton,
            "adjacency_matrix": adjacency_matrix,
            "n_features": n_members,
            "n_edges": len(edges),
            "causal_density": causal_density,
        }

        return CausalResult(
            causal_graph=causal_graph,
            edges=edges,
            out_degrees=dict(out_degrees),
            in_degrees=dict(in_degrees),
            super_spreaders=super_spreaders,
            causal_density=causal_density,
            metadata={
                "method": "pc_lingsam",
                "independence_test": self.independence_test,
                "alpha": self.alpha,
                "max_conditioning_set_size": self.max_conditioning_set_size,
                "n_skeleton_edges": int(np.sum(skeleton) / 2),
                "n_directed_edges": len(edges),
            },
        )

    def get_skeleton(self) -> Optional[np.ndarray]:
        """
        获取 PC 算法发现的因果骨架

        Returns:
            Optional[np.ndarray]: 骨架邻接矩阵（对称），如果未运行则返回 None
        """
        return self._pc_results.get("skeleton")

    def get_dag(self) -> Optional[np.ndarray]:
        """
        获取最终的有向无环图

        Returns:
            Optional[np.ndarray]: DAG 邻接矩阵，如果未运行则返回 None
        """
        return self._dag

    def get_info(self) -> Dict[str, Any]:
        """获取当前因果分析器的信息"""
        return {
            "name": "PCLiNGAMAnalyzer",
            "description": "基于 PC 算法 + LiNGAM 的因果分析器，两阶段混合方法",
            "independence_test": self.independence_test,
            "alpha": self.alpha,
            "max_conditioning_set_size": self.max_conditioning_set_size,
            "emotion_dimension": self.emotion_dimension,
            "max_lag": self.max_lag,
        }
