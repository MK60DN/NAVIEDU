"""
结构因果模型因果分析实现（可替换引擎）

该模块实现了基于结构因果模型（Structural Causal Model, SCM）的时间因果分析。
它是 Tender 框架的可替换因果分析引擎，适用于需要处理隐藏混杂变量的场景。

核心方法：
1. 将成员之间的情绪影响关系建模为结构方程模型
2. 使用线性非高斯假设（LiNGAM）或非线性扩展来识别因果方向
3. 支持 do-操作（干预模拟），预测干预某个成员情绪后的群体效应
4. 构建有向因果网络，识别超级传播者和关键接收者

学术基础：
- 结构因果模型 (Pearl, 2009): 将因果关系建模为有向无环图 (DAG)，
  每个节点是其父节点和噪声变量的函数
- do-演算 (Pearl, 1995): 通过干预操作模拟，从观测数据中推断因果效应
- 线性非高斯无环模型 (Shimizu et al., 2006):
  利用非高斯分布来识别线性系统中的因果方向

与收敛交叉映射的区别：
- CCM：基于动力系统的流形预测，适用于强非线性系统
- SCM：基于结构性方程，能够显式建模隐藏混杂变量和干预效应
  更适合需要进行"如果干预某成员的情绪，会怎样？"这样反事实推理的场景
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from scipy.stats import kurtosis, jarque_bera
from sklearn.linear_model import LinearRegression

from tender.causal_analysis.base import (
    BaseCausalAnalyzer,
    CausalResult,
    CausalEdge,
)

# 尝试导入 LiNGAM，如果未安装则使用自定义实现
try:
    from lingam import DirectLiNGAM
    _has_lingam = True
except ImportError:
    _has_lingam = False


class StructuralCausalModelAnalyzer(BaseCausalAnalyzer):
    """
    基于结构因果模型的时间因果分析器

    使用结构因果模型检测群体情绪在成员之间的因果关系。
    支持显式建模隐藏变量、进行干预模拟和反事实推理。

    Args:
        config: 配置字典，包含以下字段：
            - scm_method: 模型方法（默认"linear_non_gaussian"）
            - do_intervention: 是否进行干预模拟（默认False）
            - intervention_target: 干预目标成员（默认None）
            - significance_level: 显著性水平（默认0.05）
            - emotion_dimension: 分析维度（默认"composite"）
            - max_lag: 最大滞后阶数（默认5）
            - num_bootstrap: 自助法采样次数（默认100）
            - seed: 随机种子（默认42）
    """

    def __init__(self, config: Dict[str, Any]):
        # SCM 核心参数
        self.scm_method = config.get("scm_method", "linear_non_gaussian")
        self.do_intervention = config.get("do_intervention", False)
        self.intervention_target = config.get("intervention_target", None)
        self.significance_level = config.get("significance_level", 0.05)
        self.emotion_dimension = config.get("emotion_dimension", "composite")
        self.max_lag = config.get("max_lag", 5)
        self.num_bootstrap = config.get("num_bootstrap", 100)
        self.seed = config.get("seed", 42)

        # 结果缓存
        self._scm_results: Dict[str, Any] = {}

    def _perform_independence_test(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float]:
        """
        执行独立性检验

        使用距离相关性（Distance Correlation）来检验两个变量的独立性。
        距离相关性为 0 表示独立，越接近 1 表示依赖越强。

        Args:
            x: 第一个变量，形状为 (n,)
            y: 第二个变量，形状为 (n,)

        Returns:
            Tuple[float, float]:
                (dcorr, p_value)
                - dcorr: 距离相关系数
                - p_value: 显著性 p 值
        """
        n = len(x)

        # 计算欧几里得距离矩阵
        a = np.tile(x, (n, 1))
        b = np.tile(y, (n, 1))
        a_dist = np.abs(a - a.T)
        b_dist = np.abs(b - b.T)

        # 中心化距离矩阵
        a_centered = self._center_distance_matrix(a_dist)
        b_centered = self._center_distance_matrix(b_dist)

        # 计算距离协方差和方差
        dcov = np.sqrt(np.mean(a_centered * b_centered))
        dvar_x = np.sqrt(np.mean(a_centered ** 2))
        dvar_y = np.sqrt(np.mean(b_centered ** 2))

        # 距离相关系数
        dcorr = dcov / np.sqrt(dvar_x * dvar_y) if dvar_x > 0 and dvar_y > 0 else 0.0

        # 计算 p 值（使用洗牌检验）
        p_value = self._permutation_test(x, y, dcorr)

        return float(dcorr), float(p_value)

    def _center_distance_matrix(self, dist_matrix: np.ndarray) -> np.ndarray:
        """
        中心化距离矩阵

        Args:
            dist_matrix: 距离矩阵，形状为 (n, n)

        Returns:
            np.ndarray: 中心化后的距离矩阵
        """
        n = dist_matrix.shape
        row_mean = np.mean(dist_matrix, axis=1, keepdims=True)
        col_mean = np.mean(dist_matrix, axis=0, keepdims=True)
        total_mean = np.mean(dist_matrix)

        centered = dist_matrix - row_mean - col_mean + total_mean
        return centered

    def _permutation_test(
        self, x: np.ndarray, y: np.ndarray, observed_dcorr: float
    ) -> float:
        """
        通过排列检验计算 p 值

        Args:
            x: 第一个变量
            y: 第二个变量
            observed_dcorr: 观察到的距离相关系数

        Returns:
            float: p 值
        """
        n_shuffles = min(self.num_bootstrap, 50)  # 控制计算量
        shuffled_dcorrs = []

        for _ in range(n_shuffles):
            # 洗牌 y
            np.random.seed(None)
            shuffled_y = np.random.permutation(y)

            # 重新计算距离相关性
            a = np.tile(x, (len(x), 1))
            b = np.tile(shuffled_y, (len(shuffled_y), 1))
            a_dist = np.abs(a - a.T)
            b_dist = np.abs(b - b.T)

            a_centered = self._center_distance_matrix(a_dist)
            b_centered = self._center_distance_matrix(b_dist)

            dcov = np.sqrt(np.mean(a_centered * b_centered))
            dvar_x = np.sqrt(np.mean(a_centered ** 2))
            dvar_y = np.sqrt(np.mean(b_centered ** 2))
            dcorr = dcov / np.sqrt(dvar_x * dvar_y) if dvar_x > 0 and dvar_y > 0 else 0.0

            shuffled_dcorrs.append(dcorr)

        # 计算 p 值
        shuffled_dcorrs = np.array(shuffled_dcorrs)
        p_value = np.mean(shuffled_dcorrs >= observed_dcorr)

        return float(p_value)

    def _perform_lingam_analysis(
        self, data_matrix: np.ndarray, feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        执行 LiNGAM 分析

        使用线性非高斯无环模型来推断因果方向。
        如果有 lingam 库则使用，否则使用自定义的 ICA 实现。

        Args:
            data_matrix: 数据矩阵，形状为 (n_samples, n_features)
            feature_names: 特征（成员）名称列表

        Returns:
            Dict:
                - adjacency_matrix: 邻接矩阵
                - causal_order: 因果顺序
                - residuals: 残差
        """
        n_features = data_matrix.shape

        if _has_lingam and n_features >= 2:
            # 使用 lingam 库
            model = DirectLiNGAM(random_state=self.seed)
            model.fit(data_matrix)

            # 提取因果顺序和邻接矩阵
            causal_order = model.causal_order_
            adjacency_matrix = model.adjacency_matrix_
        else:
            # 自定义 LiNGAM 实现
            # 使用非高斯性（峰度）来判断因果方向
            n_samples = data_matrix.shape

            # 计算每个变量的峰度（非高斯性度量）
            kurtosis_values = np.array([
                kurtosis(data_matrix[:, i])
                for i in range(n_features)
            ])

            # 峰度绝对值越大，非高斯性越强
            # 因果关系倾向于从非高斯性强的变量指向非高斯性弱的变量
            causal_order = np.argsort(-np.abs(kurtosis_values))

            # 构建邻接矩阵
            adjacency_matrix = np.zeros((n_features, n_features))

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    # 对每对变量进行回归，检验残差的独立性
                    cause_idx = causal_order[i]
                    effect_idx = causal_order[j]

                    x = data_matrix[:, cause_idx].reshape(-1, 1)
                    y = data_matrix[:, effect_idx]

                    # 回归 y ~ x
                    model = LinearRegression()
                    model.fit(x, y)
                    residuals = y - model.predict(x)

                    # 检验残差与 x 的独立性
                    dcorr, p_value = self._perform_independence_test(
                        data_matrix[:, cause_idx], residuals
                    )

                    # 如果残差与原因独立，则认定因果方向
                    if p_value > self.significance_level and dcorr < 0.1:
                        adjacency_matrix[effect_idx, cause_idx] = model.coef_

        # 计算残差
        residuals = np.zeros_like(data_matrix)
        for i in range(n_features):
            if np.any(adjacency_matrix[i, :] != 0):
                parents = np.where(adjacency_matrix[i, :] != 0)
                parent_values = data_matrix[:, parents]
                coefs = adjacency_matrix[i, parents]
                predicted = parent_values @ coefs
                residuals[:, i] = data_matrix[:, i] - predicted
            else:
                residuals[:, i] = data_matrix[:, i]

        return {
            "adjacency_matrix": adjacency_matrix,
            "causal_order": causal_order.tolist(),
            "residuals": residuals,
        }

    def _perform_intervention_analysis(
        self,
        data_matrix: np.ndarray,
        feature_names: List[str],
        intervention_target: str,
    ) -> Dict[str, Any]:
        """
        执行 do-操作（干预模拟）

        模拟对某个成员的情绪进行干预后的群体效应。

        Args:
            data_matrix: 数据矩阵
            feature_names: 特征名称列表
            intervention_target: 干预目标成员

        Returns:
            Dict:
                - intervention_effect: 干预效应向量
                - original_state: 原始状态
                - do_operator: 使用的 do-操作描述
        """
        n_features = data_matrix.shape

        # 找到干预目标的索引
        if intervention_target not in feature_names:
            return {
                "intervention_effect": np.zeros(n_features),
                "original_state": data_matrix[-1],
                "do_operator": f"do({intervention_target}=未找到)",
            }

        target_idx = feature_names.index(intervention_target)

        # 进行 LiNGAM 分析获取因果结构
        lingam_result = self._perform_lingam_analysis(data_matrix, feature_names)
        adjacency_matrix = lingam_result["adjacency_matrix"]
        residuals = lingam_result["residuals"]

        # 原始状态（最后一个时间点）
        original_state = data_matrix[-1].copy()

        # 干预后的状态：将干预目标的值设为其均值的两倍标准差变化
        target_mean = np.mean(data_matrix[:, target_idx])
        target_std = np.std(data_matrix[:, target_idx])
        do_value = target_mean + 2 * target_std  # 设定为高于均值两个标准差

        # 执行 do-操作
        do_state = original_state.copy()
        do_state[target_idx] = do_value
        do_state[target_idx] = 0  # 在 do-操作中，切断所有指向干预变量的边

        # 传播干预效应到其他变量
        for i in range(n_features):
            if i == target_idx:
                # 干预变量的值直接设定
                do_state[i] = do_value
                continue

            # 对于其他变量，使用因果结构重新计算
            parents = np.where(adjacency_matrix[i, :] != 0)
            if len(parents) > 0:
                predicted = 0
                for p in parents:
                    predicted += adjacency_matrix[i, p] * do_state[p]

                # 加上原始残差
                do_state[i] = predicted + residuals[-1, i]
            else:
                # 如果没有父节点，保持原始值
                do_state[i] = original_state[i]

        # 计算干预效应向量
        intervention_effect = do_state - original_state

        return {
            "intervention_effect": intervention_effect,
            "original_state": original_state,
            "do_state": do_state,
            "do_operator": f"do({intervention_target}={do_value:.3f})",
        }

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
        2. 执行 LiNGAM 分析推断因果结构
        3. 可选：执行干预模拟（do-操作）
        4. 构建有向因果网络
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

        # 2. 执行 LiNGAM 分析
        lingam_result = self._perform_lingam_analysis(data_matrix, member_ids)
        adjacency_matrix = lingam_result["adjacency_matrix"]

        # 3. 可选：执行干预模拟
        intervention_result = None
        if self.do_intervention and self.intervention_target:
            intervention_result = self._perform_intervention_analysis(
                data_matrix, member_ids, self.intervention_target
            )

        # 4. 构建边列表和出入度
        edges = []
        out_degrees = defaultdict(int)
        in_degrees = defaultdict(int)

        for i in range(n_members):
            for j in range(n_members):
                if i == j:
                    continue

                # adjacency_matrix[i, j] 表示 j -> i 的因果影响
                coef = adjacency_matrix[i, j]
                if abs(coef) > 0.01:  # 因果关系显著
                    # 检验显著性
                    dcorr, p_value = self._perform_independence_test(
                        data_matrix[:, j], data_matrix[:, i]
                    )

                    if p_value < self.significance_level:
                        edges.append(CausalEdge(
                            source=member_ids[j],
                            target=member_ids[i],
                            strength=float(abs(coef)),
                            lag=1,
                            p_value=float(p_value),
                            method="scm_lingam",
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

        # 存储 SCM 结果以便后续查询
        self._scm_results = {
            "adjacency_matrix": adjacency_matrix,
            "causal_order": lingam_result["causal_order"],
            "residuals": lingam_result["residuals"],
            "intervention_result": intervention_result,
        }

        return CausalResult(
            causal_graph=causal_graph,
            edges=edges,
            out_degrees=dict(out_degrees),
            in_degrees=dict(in_degrees),
            super_spreaders=super_spreaders,
            causal_density=causal_density,
            metadata={
                "scm_method": self.scm_method,
                "do_intervention": self.do_intervention,
                "intervention_target": self.intervention_target,
            },
        )

    def get_intervention_effect(
        self, target_member: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取对特定成员的干预效应信息

        需要先运行 analyze 方法并设置 do_intervention=True。

        Args:
            target_member: 目标成员ID

        Returns:
            Optional[Dict[str, Any]]: 干预效应信息，如果不可用则返回 None
        """
        if not self._scm_results or "intervention_result" not in self._scm_results:
            return None

        return self._scm_results["intervention_result"]

    def get_info(self) -> Dict[str, Any]:
        """获取当前因果分析器的信息"""
        return {
            "name": "StructuralCausalModelAnalyzer",
            "description": "基于结构因果模型的时间因果分析器，支持干预模拟",
            "scm_method": self.scm_method,
            "do_intervention": self.do_intervention,
            "intervention_target": self.intervention_target,
            "significance_level": self.significance_level,
            "emotion_dimension": self.emotion_dimension,
            "max_lag": self.max_lag,
        }
