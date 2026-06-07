"""
收敛交叉映射因果分析实现（默认实现）

该模块实现了基于收敛交叉映射（Convergent Cross Mapping, CCM）的时间因果分析。
它是 Tender 框架的默认因果分析引擎，替代了经典的格兰杰因果检验。

核心方法：
1. 为每个成员的每个情绪维度构建延迟嵌入（Takens 嵌入）流形
2. 判断成员 i 是否是成员 j 的格兰杰原因：如果 i 影响了 j，
   那么从 j 的流形应该能可靠地估计 i 的状态
3. 通过库长度收敛来检验因果关系的显著性
4. 构建有向因果网络，识别超级传播者和关键接收者

学术基础：
- 收敛交叉映射 (Sugihara et al., 2012): 基于动力系统理论的因果推断方法
  核心思想：如果 X 是 Y 的原因，那么 Y 的时间序列包含了 X 的信息，
  因此从 Y 的流形可以预测 X 的状态
- Takens 嵌入定理 (Takens, 1981): 通过时间延迟嵌入重构动力系统的相空间
- 收敛性检验：当库长度增加时，预测技能必须收敛（提高）

与格兰杰因果检验的区别：
- 格兰杰因果：基于线性回归模型的预测能力，假设线性关系
- 收敛交叉映射：基于流形预测，适用于非线性系统，不要求线性假设
  能检测到格兰杰因果无法发现的非线性因果关系
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

from tender.causal_analysis.base import (
    BaseCausalAnalyzer,
    CausalResult,
    CausalEdge,
)


class ConvergentCrossMappingAnalyzer(BaseCausalAnalyzer):
    """
    基于收敛交叉映射的时间因果分析器

    使用收敛交叉映射方法检测群体情绪在成员之间的非线性因果关系。
    通过重构每个成员情绪时间序列的延迟嵌入流形，
    判断情绪影响的方向、强度和显著性。

    Args:
        config: 配置字典，包含以下字段：
            - embedding_dimension: 嵌入维度 E（默认5）
            - tau: 时间延迟 τ（默认1）
            - lib_size_ratio: 库大小比例（默认0.8）
            - significance_level: 显著性水平（默认0.05）
            - emotion_dimension: 分析维度（默认"composite"）
            - max_lag: 最大滞后阶数（默认5）
            - num_lib_sizes: 库大小采样数量（默认10）
            - seed: 随机种子（默认42）
    """

    def __init__(self, config: Dict[str, Any]):
        # CCM 核心参数
        self.embedding_dimension = config.get("embedding_dimension", 5)   # E
        self.tau = config.get("tau", 1)                                   # τ
        self.lib_size_ratio = config.get("lib_size_ratio", 0.8)
        self.significance_level = config.get("significance_level", 0.05)
        self.emotion_dimension = config.get("emotion_dimension", "composite")
        self.max_lag = config.get("max_lag", 5)
        self.num_lib_sizes = config.get("num_lib_sizes", 10)
        self.seed = config.get("seed", 42)

        # 结果缓存
        self._ccm_results: Dict[Tuple[str, str, str], Dict] = {}

    def _build_takens_manifold(
        self, time_series: np.ndarray
    ) -> np.ndarray:
        """
        构建 Takens 延迟嵌入流形

        给定时间序列 x = [x_1, x_2, ..., x_L]，
        构建 E 维延迟嵌入向量：
        X(t) = [x_t, x_{t-τ}, x_{t-2τ}, ..., x_{t-(E-1)τ}]

        Args:
            time_series: 一维时间序列，形状为 (L,)

        Returns:
            np.ndarray: 流形点集，形状为 (L - (E-1)*τ, E)
        """
        L = len(time_series)
        E = self.embedding_dimension
        tau = self.tau

        # 检查数据长度是否足够
        min_length = (E - 1) * tau + 1
        if L < min_length:
            return np.array([])

        # 构建延迟嵌入矩阵
        n_points = L - (E - 1) * tau
        manifold = np.zeros((n_points, E))

        for i in range(E):
            manifold[:, i] = time_series[i * tau : i * tau + n_points]

        return manifold

    def _find_nearest_neighbors(
        self, target_point: np.ndarray, manifold: np.ndarray, exclude_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        在流形中查找目标点的最近邻（排除自身）

        Args:
            target_point: 目标点，形状为 (E,)
            manifold: 流形点集，形状为 (n, E)
            exclude_idx: 需要排除的索引（避免自预测）

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                (neighbor_indices, neighbor_distances)
                返回最近邻的索引和距离
        """
        n = manifold.shape

        # 创建排除掩码
        mask = np.ones(n, dtype=bool)
        if exclude_idx < n:
            mask[exclude_idx] = False

        # 计算所有点到目标点的距离
        distances = cdist([target_point], manifold[mask])

        # 取 E + 1 个最近邻（E 维流形需要 E+1 个点来生成单纯形）
        n_neighbors = min(self.embedding_dimension + 1, len(distances))
        nearest_indices = np.argsort(distances)[:n_neighbors]

        # 将掩码索引映射回原始索引
        original_indices = np.where(mask)[nearest_indices]

        return original_indices, distances[nearest_indices]

    def _simplex_projection(
        self, manifold_x: np.ndarray, manifold_y: np.ndarray, target_idx: int
    ) -> float:
        """
        使用单纯形投影预测目标点的值

        在流形 Y 中找到目标点的最近邻，
        然后用这些最近邻在流形 X 中对应的值来预测。

        Args:
            manifold_x: 被预测变量的流形（原因变量）
            manifold_y: 预测变量的流形（效果变量）
            target_idx: 目标点在流形中的索引

        Returns:
            float: 预测值
        """
        if target_idx >= manifold_y.shape:
            return np.nan

        target_point = manifold_y[target_idx]

        # 找到最近邻
        neighbor_indices, distances = self._find_nearest_neighbors(
            target_point, manifold_y, target_idx
        )

        if len(neighbor_indices) == 0:
            return np.nan

        # 计算权重：距离的倒数归一化
        epsilon = 1e-10
        weights = 1.0 / (distances + epsilon)
        weights = weights / np.sum(weights)

        # 使用最近邻在流形 X 中的对应值进行加权预测
        predicted_values = manifold_x[neighbor_indices, 0].flatten()
        prediction = np.sum(weights * predicted_values)

        return prediction

    def _compute_ccm_causality(
        self, time_series_x: np.ndarray, time_series_y: np.ndarray
    ) -> Dict[str, Any]:
        """
        计算 X -> Y 的 CCM 因果强度

        从 Y 的流形预测 X 的状态。如果预测技能随库大小收敛并改善，
        则说明 X 是 Y 的原因。

        Args:
            time_series_x: 潜在原因变量的时间序列
            time_series_y: 潜在结果变量的时间序列

        Returns:
            Dict:
                - rho_lib: 不同库大小下的预测相关系数列表
                - lib_sizes: 对应库大小列表
                - convergence_score: 收敛分数（正值表示因果关系显著）
                - best_rho: 最大库大小下的相关系数
                - p_value: 显著性 p 值（使用随机洗牌的零假设）
        """
        # 构建流形
        manifold_x = self._build_takens_manifold(time_series_x)
        manifold_y = self._build_takens_manifold(time_series_y)

        if manifold_x.shape == 0 or manifold_y.shape == 0:
            return {
                "rho_lib": [],
                "lib_sizes": [],
                "convergence_score": 0.0,
                "best_rho": 0.0,
                "p_value": 1.0,
            }

        # 两个流形长度应一致
        n_points = min(manifold_x.shape, manifold_y.shape)
        manifold_x = manifold_x[:n_points]
        manifold_y = manifold_y[:n_points]

        if n_points < 3:
            return {
                "rho_lib": [],
                "lib_sizes": [],
                "convergence_score": 0.0,
                "best_rho": 0.0,
                "p_value": 1.0,
            }

        # 采样不同的库大小
        lib_sizes = np.linspace(
            max(3, n_points // 10), n_points, self.num_lib_sizes
        ).astype(int)
        lib_sizes = np.unique(lib_sizes)

        rho_lib = []
        convergence_slopes = []

        for lib_size in lib_sizes:
            # 随机选择库点
            np.random.seed(self.seed)
            lib_indices = np.random.choice(
                n_points, size=lib_size, replace=False
            )
            lib_indices.sort()

            # 从库中构建子流形
            lib_manifold_x = manifold_x[lib_indices]
            lib_manifold_y = manifold_y[lib_indices]

            # 对每个库点进行交叉映射预测
            predictions = []
            observations = []

            for j, idx in enumerate(lib_indices):
                pred = self._simplex_projection(
                    manifold_x, lib_manifold_y, j
                )
                if not np.isnan(pred):
                    predictions.append(pred)
                    observations.append(manifold_x[idx, 0])

            # 计算预测技能（皮尔逊相关系数）
            if len(predictions) > 2:
                correlation = np.corrcoef(predictions, observations)[0, 1]
                rho = correlation if not np.isnan(correlation) else 0.0
            else:
                rho = 0.0

            rho_lib.append(rho)

            # 计算收敛斜率
            if len(rho_lib) >= 2:
                slope = rho_lib[-1] - rho_lib[-2]
                convergence_slopes.append(slope)

        # 收敛分数：如果预测技能随库大小增加而提高（正斜率），
        # 则说明存在因果关系
        convergence_score = np.mean(convergence_slopes) if convergence_slopes else 0.0

        # 最大库大小的相关系数
        best_rho = rho_lib[-1] if rho_lib else 0.0

        # 计算 p 值（通过洗牌零假设）
        p_value = self._compute_ccm_p_value(
            time_series_x, time_series_y, best_rho
        )

        return {
            "rho_lib": rho_lib,
            "lib_sizes": lib_sizes.tolist(),
            "convergence_score": convergence_score,
            "best_rho": best_rho,
            "p_value": p_value,
        }

    def _compute_ccm_p_value(
        self, time_series_x: np.ndarray, time_series_y: np.ndarray, observed_rho: float
    ) -> float:
        """
        通过随机洗牌计算 CCM 结果的 p 值

        零假设：X 和 Y 之间不存在因果关系。
        通过重复洗牌 Y 的时间序列并重新计算 CCM，
        统计观察到的 rho 值在随机分布中的位置。

        Args:
            time_series_x: 原因变量时间序列
            time_series_y: 结果变量时间序列
            observed_rho: 观察到的相关系数

        Returns:
            float: p 值
        """
        n_shuffles = 20  # 为了提高性能，使用较少的洗牌次数
        shuffled_rhos = []

        for _ in range(n_shuffles):
            # 洗牌 Y 的时间序列
            np.random.seed(None)  # 使用不同的随机种子
            shuffled_y = np.random.permutation(time_series_y)

            # 重新计算 CCM
            result = self._compute_ccm_causality(time_series_x, shuffled_y)
            shuffled_rhos.append(result["best_rho"])

        # 计算 p 值
        shuffled_rhos = np.array(shuffled_rhos)
        p_value = np.mean(shuffled_rhos >= observed_rho)

        return float(p_value)

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
        1. 将多个情绪维度合并为复合维度（如果需要）
        2. 为每对成员计算 CCM 因果关系
        3. 构建有向因果网络
        4. 计算因果网络统计量
        5. 识别超级传播者和关键接收者

        Args:
            time_series_data: 时间序列数据字典
                {member_id: [vector_array, ...]}
            member_ids: 成员ID列表
            window_start: 窗口起始时间戳
            window_end: 窗口结束时间戳

        Returns:
            CausalResult: 因果分析结果
        """
        n_members = len(member_ids)

        if n_members < 2:
            # 只有一个或零个成员，无法进行因果分析
            import networkx as nx
            return CausalResult(
                causal_graph=nx.DiGraph(),
                edges=[],
                out_degrees={mid: 0 for mid in member_ids},
                in_degrees={mid: 0 for mid in member_ids},
                super_spreaders=[],
                causal_density=0.0,
            )

        # 1. 提取每个成员的复合时间序列
        member_ts = {}
        for mid in member_ids:
            if mid in time_series_data and len(time_series_data[mid]) > 0:
                vectors = np.array(time_series_data[mid])
                member_ts[mid] = vectors

        # 如果数据不足，返回空结果
        if len(member_ts) < 2:
            import networkx as nx
            return CausalResult(
                causal_graph=nx.DiGraph(),
                edges=[],
                out_degrees={mid: 0 for mid in member_ids},
                in_degrees={mid: 0 for mid in member_ids},
                super_spreaders=[],
                causal_density=0.0,
            )

        # 2. 计算每对成员的因果关系
        edges = []
        out_degrees = defaultdict(int)
        in_degrees = defaultdict(int)

        for i in range(len(member_ids)):
            for j in range(len(member_ids)):
                if i == j:
                    continue

                mid_i = member_ids[i]
                mid_j = member_ids[j]

                if mid_i not in member_ts or mid_j not in member_ts:
                    continue

                # 提取时间序列
                ts_i = member_ts[mid_i]
                ts_j = member_ts[mid_j]

                # 确保长度一致
                min_len = min(len(ts_i), len(ts_j))
                if min_len < 3:
                    continue

                ts_i = ts_i[:min_len]
                ts_j = ts_j[:min_len]

                # 如果要分析复合维度，取范数
                if self.emotion_dimension == "composite":
                    ts_1d_i = np.linalg.norm(ts_i, axis=1)
                    ts_1d_j = np.linalg.norm(ts_j, axis=1)
                elif self.emotion_dimension == "valence":
                    ts_1d_i = ts_i[:, 0]
                    ts_1d_j = ts_j[:, 0]
                elif self.emotion_dimension == "arousal":
                    ts_1d_i = ts_i[:, 1]
                    ts_1d_j = ts_j[:, 1]
                elif self.emotion_dimension == "focus":
                    ts_1d_i = ts_i[:, 2]
                    ts_1d_j = ts_j[:, 2]
                else:
                    ts_1d_i = np.linalg.norm(ts_i, axis=1)
                    ts_1d_j = np.linalg.norm(ts_j, axis=1)

                # 计算 X_j -> X_i：j 是否影响 i
                result_ji = self._compute_ccm_causality(ts_1d_j, ts_1d_i)

                # 计算 X_i -> X_j：i 是否影响 j
                result_ij = self._compute_ccm_causality(ts_1d_i, ts_1d_j)

                # 如果 j -> i 显著，添加边 j -> i
                if (
                    result_ji["best_rho"] > 0.3
                    and result_ji["p_value"] < self.significance_level
                    and result_ji["convergence_score"] > 0.01
                ):
                    edges.append(CausalEdge(
                        source=mid_j,
                        target=mid_i,
                        strength=float(result_ji["best_rho"]),
                        lag=1,
                        p_value=float(result_ji["p_value"]),
                        method="ccm",
                    ))
                    out_degrees[mid_j] += 1
                    in_degrees[mid_i] += 1

                # 如果 i -> j 显著，添加边 i -> j
                if (
                    result_ij["best_rho"] > 0.3
                    and result_ij["p_value"] < self.significance_level
                    and result_ij["convergence_score"] > 0.01
                ):
                    edges.append(CausalEdge(
                        source=mid_i,
                        target=mid_j,
                        strength=float(result_ij["best_rho"]),
                        lag=1,
                        p_value=float(result_ij["p_value"]),
                        method="ccm",
                    ))
                    out_degrees[mid_i] += 1
                    in_degrees[mid_j] += 1

        # 3. 构建因果网络图
        import networkx as nx
        causal_graph = nx.DiGraph()

        for mid in member_ids:
            causal_graph.add_node(mid)

        for edge in edges:
            causal_graph.add_edge(edge.source, edge.target)

        # 4. 计算因果密度
        total_possible_edges = n_members * (n_members - 1)
        causal_density = len(edges) / total_possible_edges if total_possible_edges > 0 else 0.0

        # 5. 识别超级传播者（出度 > 平均 + 1 标准差）
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

        return CausalResult(
            causal_graph=causal_graph,
            edges=edges,
            out_degrees=dict(out_degrees),
            in_degrees=dict(in_degrees),
            super_spreaders=super_spreaders,
            causal_density=causal_density,
        )

    def get_info(self) -> Dict[str, Any]:
        """获取当前因果分析器的信息"""
        return {
            "name": "ConvergentCrossMappingAnalyzer",
            "description": "基于收敛交叉映射的时间因果分析器",
            "embedding_dimension": self.embedding_dimension,
            "tau": self.tau,
            "lib_size_ratio": self.lib_size_ratio,
            "significance_level": self.significance_level,
            "emotion_dimension": self.emotion_dimension,
            "max_lag": self.max_lag,
        }
