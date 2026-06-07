"""
时间因果分析模块 - 抽象基类

该模块定义了所有因果分析实现的统一接口。
核心功能是分析不同成员情绪在时间序列上的影响关系，
构建有向因果网络，识别超级传播者，并计算因果密度等关键指标。

学术基础：
格兰杰因果检验 (Granger, 1969) 是一种基于时间先行性和预测能力的因果关系定义。
其核心思想：如果加入 X 的过去值能显著提高对 Y 的预测精度，则称 X 是 Y 的格兰杰原因。
这本质上是一种统计预测意义上的因果关系，而非基于干预的哲学因果关系。

关键概念：
- 出度 (out-degree)：该成员影响了多少人（传播者角色）
- 入度 (in-degree)：该成员被多少人影响（接收者角色）
- 超级传播者：出度排名前 10% 的成员
- 因果密度：实际因果边数 / 最大可能边数，衡量群体的情绪传染程度
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import networkx as nx
import numpy as np
from tender.emotion_vectorizer.base import EmotionVector


@dataclass
class CausalEdge:
    """
    因果边数据结构

    表示从源成员到目标成员的一条显著因果影响关系。

    Attributes:
        source: 原因成员ID
        target: 结果成员ID
        p_value: 格兰杰因果检验的 p 值
        f_statistic: F 统计量
        lag_order: 最优滞后阶数
        effect_size: 效应量（系数估计值）
    """
    source: str
    target: str
    p_value: float
    f_statistic: float
    lag_order: int
    effect_size: float


@dataclass
class CausalResult:
    """
    单次因果分析的结果

    包含完整的因果网络信息、节点度分布和关键指标。

    Attributes:
        causal_graph: NetworkX 有向图，节点=成员，边=显著因果关系
        edges: 因果边列表
        out_degrees: 每个成员的出度字典 {成员ID: out_degree}
        in_degrees: 每个成员的入度字典 {成员ID: in_degree}
        super_spreaders: 超级传播者列表（出度 Top 10%）
        causal_density: 因果密度（边数 / 最大可能边数）
        window_start: 时间窗口起始时间戳
        window_end: 时间窗口结束时间戳
        metadata: 附加分析元数据
    """
    causal_graph: nx.DiGraph
    edges: List[CausalEdge]
    out_degrees: Dict[str, int]
    in_degrees: Dict[str, int]
    super_spreaders: List[str]
    causal_density: float
    window_start: float = 0.0
    window_end: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCausalAnalyzer(ABC):
    """
    时间因果分析抽象基类

    定义时间因果分析的统一接口。所有具体实现必须：
    1. 继承此类
    2. 实现所有抽象方法
    3. 严格保持方法签名一致

    设计原则：
    - 插件式架构：通过配置文件选择具体实现
    - 统一接口：所有实现对外暴露相同的方法签名
    - 可替换性：在不影响上下游模块的前提下替换实现

    核心输出指标：
    - 有向因果网络：显示情绪的传播路径
    - 节点出入度：量化每个成员的情绪影响力
    - 超级传播者：高影响力成员列表
    - 因果密度：群体情绪传染程度的度量
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        初始化因果分析器

        Args:
            config: 配置字典，包含以下字段：
                - significance_level: 显著性水平（默认 0.05）
                - max_lag: 最大滞后阶数
                - window_size_for_causality: 因果计算的窗口数
                - min_samples: 最小样本数要求
        """
        pass

    @abstractmethod
    def analyze(
        self,
        emotion_time_series: Dict[str, List[np.ndarray]],
        member_ids: List[str],
        window_start: float,
        window_end: float
    ) -> CausalResult:
        """
        对成员情绪时间序列进行因果分析

        这是核心方法，将时间序列数据转换为因果网络结构。

        Args:
            emotion_time_series: 每个成员的情绪时间序列
                格式: {成员ID: [[v1, a1, f1], [v2, a2, f2], ...]}
            member_ids: 成员ID列表
            window_start: 时间窗口起始时间戳
            window_end: 时间窗口结束时间戳

        Returns:
            CausalResult: 完整的因果分析结果
        """
        pass

    @abstractmethod
    def test_pair(
        self,
        time_series_x: np.ndarray,
        time_series_y: np.ndarray,
        member_x: str,
        member_y: str
    ) -> Optional[CausalEdge]:
        """
        对一对成员进行因果检验

        检验成员 X 的情绪是否是成员 Y 情绪的格兰杰原因。

        Args:
            time_series_x: 成员 X 的情绪时间序列
            time_series_y: 成员 Y 的情绪时间序列
            member_x: 成员 X 的 ID
            member_y: 成员 Y 的 ID

        Returns:
            Optional[CausalEdge]: 如果因果关系显著则返回边，否则返回 None
        """
        pass

    @abstractmethod
    def identify_super_spreaders(
        self,
        out_degrees: Dict[str, int]
    ) -> List[str]:
        """
        识别超级传播者

        找出出度排名前 10% 的成员，这些成员是群体情绪的关键影响节点。

        Args:
            out_degrees: 出度字典

        Returns:
            List[str]: 超级传播者 ID 列表
        """
        pass

    @abstractmethod
    def compute_causal_density(
        self,
        n_edges: int,
        n_members: int
    ) -> float:
        """
        计算因果密度

        因果密度 = 实际因果边数 / 最大可能边数
        该指标衡量群体情绪的传染程度。

        Args:
            n_edges: 实际显著的因果边数
            n_members: 成员总数

        Returns:
            float: 因果密度 (0-1)
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        获取当前因果分析器的信息

        Returns:
            Dict: 信息字典，包含算法名称、参数设置等
        """
        pass

    def _validate_time_series(
        self,
        emotion_time_series: Dict[str, List[np.ndarray]],
        member_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        验证并转换情绪时间序列

        检查时间序列的完整性和一致性。

        Args:
            emotion_time_series: 情绪时间序列字典
            member_ids: 成员ID列表

        Returns:
            Dict[str, np.ndarray]: 验证后的时间序列
        """
        if not emotion_time_series:
            raise ValueError("情绪时间序列字典为空")

        # 检查所有成员都存在
        for mid in member_ids:
            if mid not in emotion_time_series:
                raise ValueError(f"成员 {mid} 的时间序列缺失")

        # 检查序列长度一致性
        lengths = [len(emotion_time_series[mid]) for mid in member_ids]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"成员的时间序列长度不一致: {dict(zip(member_ids, lengths))}"
            )

        # 将列表转换为 numpy 数组
        return {
            mid: np.array(series) if isinstance(series, list) else series
            for mid, series in emotion_time_series.items()
        }

    def _build_causal_graph(
        self,
        edges: List[CausalEdge],
        member_ids: List[str]
    ) -> nx.DiGraph:
        """
        根据因果边构建有向因果网络

        Args:
            edges: 因果边列表
            member_ids: 所有成员ID列表

        Returns:
            nx.DiGraph: 有向图，边属性包含 p 值、F 统计量等
        """
        graph = nx.DiGraph()

        # 添加所有节点
        for mid in member_ids:
            graph.add_node(mid)

        # 添加因果边
        for edge in edges:
            graph.add_edge(
                edge.source,
                edge.target,
                p_value=edge.p_value,
                f_statistic=edge.f_statistic,
                lag_order=edge.lag_order,
                effect_size=edge.effect_size,
            )

        return graph

    def _compute_node_degrees(
        self,
        graph: nx.DiGraph,
        member_ids: List[str]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        计算每个成员的出度和入度

        Args:
            graph: 有向因果网络
            member_ids: 所有成员ID列表

        Returns:
            Tuple[Dict[str, int], Dict[str, int]]:
                (out_degrees, in_degrees)
        """
        out_degrees = {}
        in_degrees = {}

        for mid in member_ids:
            # 出度：该成员影响了多少人
            out_degrees[mid] = graph.out_degree(mid) if mid in graph else 0
            # 入度：该成员被多少人影响
            in_degrees[mid] = graph.in_degree(mid) if mid in graph else 0

        return out_degrees, in_degrees
