"""
管道编排器 - 核心调度模块

该模块将所有的分析阶段编排为一个完整的处理管道。
它是整个框架的入口点，负责：
1. 加载配置并初始化所有模块
2. 依次执行各分析阶段
3. 管理分析历史状态
4. 提供统一的高层API
"""

import time
from typing import Dict, List, Optional, Any

import numpy as np

from tender.emotion_vectorizer.base import EmotionVector
from tender.emotion_vectorizer.config import DEFAULT_CONFIG as VECTORIZER_DEFAULT
from tender.topology_analysis.config import DEFAULT_CONFIG as TOPOLOGY_DEFAULT
from tender.causal_analysis.config import DEFAULT_CONFIG as CAUSAL_DEFAULT
from tender.fusion.config import DEFAULT_CONFIG as FUSION_DEFAULT
from tender.strategy.base import StrategyDecision


class TenderPipeline:
    """
    Tender 框架的主管道编排器

    协调所有分析模块的执行顺序，维护分析历史状态。
    所有模块均支持通过配置动态加载可替换实现。

    Args:
        config: 完整配置字典（推荐使用 load_config 加载）
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._init_modules()
        self._history: Dict[str, Any] = {
            "time_series": {},        # 情绪时间序列历史
            "topology_results": [],   # 拓扑分析历史
            "causal_results": [],     # 因果分析历史
            "fusion_results": [],     # 融合结果历史
            "decisions": [],          # 策略决策历史
        }

    def _init_modules(self):
        """根据配置动态初始化各分析模块"""
        # 1. 情绪向量化器
        v_config = VECTORIZER_DEFAULT.copy()
        v_config.update(self.config.get("emotion_vectorizer", {}))
        engine = v_config.get("engine", "neuro_symbolic")
        from tender.emotion_vectorizer import ENGINE_MAP as V_ENGINE_MAP
        vectorizer_cls_name = V_ENGINE_MAP.get(engine, "NeuroSymbolicVectorizer")
        import importlib
        module = importlib.import_module("tender.emotion_vectorizer")
        vectorizer_cls = getattr(module, vectorizer_cls_name)
        self.vectorizer = vectorizer_cls(v_config)
        self._v_config = v_config

        # 2. 拓扑分析器
        t_config = TOPOLOGY_DEFAULT.copy()
        t_config.update(self.config.get("topology_analysis", {}))
        engine = t_config.get("engine", "persistent_laplacian")
        from tender.topology_analysis import ENGINE_MAP as T_ENGINE_MAP
        analyzer_cls_name = T_ENGINE_MAP.get(engine, "PersistentLaplacianAnalyzer")
        module = importlib.import_module("tender.topology_analysis")
        analyzer_cls = getattr(module, analyzer_cls_name)
        self.topology_analyzer = analyzer_cls(t_config)
        self._t_config = t_config

        # 3. 因果分析器
        c_config = CAUSAL_DEFAULT.copy()
        c_config.update(self.config.get("causal_analysis", {}))
        engine = c_config.get("engine", "convergent_cross_mapping")
        from tender.causal_analysis import ENGINE_MAP as C_ENGINE_MAP
        analyzer_cls_name = C_ENGINE_MAP.get(engine, "ConvergentCrossMappingAnalyzer")
        module = importlib.import_module("tender.causal_analysis")
        analyzer_cls = getattr(module, analyzer_cls_name)
        self.causal_analyzer = analyzer_cls(c_config)
        self._c_config = c_config

        # 4. 融合模块
        f_config = FUSION_DEFAULT.copy()
        f_config.update(self.config.get("fusion", {}))
        engine = f_config.get("engine", "dct_gnn")
        from tender.fusion import ENGINE_MAP as F_ENGINE_MAP
        fusion_cls_name = F_ENGINE_MAP.get(engine, "DCTGNN")
        module = importlib.import_module("tender.fusion")
        fusion_cls = getattr(module, fusion_cls_name)
        self.fusion_module = fusion_cls(f_config)
        self._f_config = f_config

        # 5. 策略引擎（支持动态加载）
        s_config = self.config.get("strategy", {})
        engine = s_config.get("engine", "causal_rl")
        from tender.strategy.config import ENGINE_MAP as S_ENGINE_MAP
        strategy_cls_name = S_ENGINE_MAP.get(engine, "CausalRLEngine")
        import importlib
        module = importlib.import_module("tender.strategy")
        strategy_cls = getattr(module, strategy_cls_name)
        self.strategy_engine = strategy_cls(s_config)
        self._s_config = s_config

    def analyze_window(
        self,
        member_messages: Dict[str, List[Dict[str, Any]]],
        window_start: float,
        window_end: float,
    ) -> StrategyDecision:
        """
        分析单个时间窗口并返回策略决策

        完整流程：
        1. 情绪向量化：将原始文本转换为三维情绪向量
        2. 空间拓扑分析：检测聚类、环和离群点
        3. 时间因果分析：检测情绪影响关系
        4. 时空融合：构建融合特征向量并预测
        5. 策略推理：评估风险并进行干预
        6. [可选] 共识化过滤：根据共识度优化策略推荐

        Args:
            member_messages: 成员消息字典
            window_start: 窗口起始时间戳
            window_end: 窗口结束时间戳

        Returns:
            StrategyDecision: 策略决策结果
        """
        # 1. 情绪向量化
        vectorization_result = self.vectorizer.vectorize(
            member_messages, window_start, window_end
        )

        # 更新时间序列历史
        for member_id, vector in vectorization_result.vectors.items():
            if member_id not in self._history["time_series"]:
                self._history["time_series"][member_id] = []
            self._history["time_series"][member_id].append(vector.to_array())

        # 2. 空间拓扑分析
        topology_result = self.topology_analyzer.analyze(
            vectorization_result.vectors, window_start, window_end
        )
        self._history["topology_results"].append(topology_result)

        # 3. 时间因果分析（需要足够的时间点）
        member_ids = list(vectorization_result.vectors.keys())
        causal_result = None
        ts = self._history["time_series"]
        if all(len(ts.get(mid, [])) >= 3 for mid in member_ids):
            causal_result = self.causal_analyzer.analyze(
                {
                    mid: ts[mid][-10:]  # 取最近10个窗口
                    for mid in member_ids
                },
                member_ids,
                window_start,
                window_end,
            )
            self._history["causal_results"].append(causal_result)

        # 4. 时空融合
        fusion_result = self.fusion_module.fuse(
            topology_result,
            causal_result or self._create_empty_causal(member_ids),
            {
                mid: ts[mid][-5:]  # 取最近5个窗口
                for mid in member_ids
            },
            member_ids,
        )
        self._history["fusion_results"].append(fusion_result)

        # 5. 策略推理
        decision = self.strategy_engine.assess_risk(fusion_result)

        # 6. [可选] 共识化过滤层
        # 当配置中 enable_reconciliation 为 true 时，对策略结果进行共识化优化
        if self.config.get("strategy", {}).get("enable_reconciliation", False):
            from tender.strategy.reconciliation_layer import ReconciliationLayer
            rec_layer = ReconciliationLayer(self.config.get("strategy", {}))
            decision = rec_layer.refine_strategies(decision, fusion_result)

        self._history["decisions"].append(decision)

        return decision

    def _create_empty_causal(self, member_ids):
        """创建空因果结果（当数据不足时使用）"""
        from tender.causal_analysis.base import CausalResult
        import networkx as nx
        return CausalResult(
            causal_graph=nx.DiGraph(),
            edges=[],
            out_degrees={mid: 0 for mid in member_ids},
            in_degrees={mid: 0 for mid in member_ids},
            super_spreaders=[],
            causal_density=0.0,
        )

    def get_history(self) -> Dict[str, Any]:
        """获取完整的分析历史"""
        return self._history

    def get_info(self) -> Dict[str, Any]:
        """获取管道信息"""
        return {
            "vectorizer": self.vectorizer.get_info(),
            "topology_analyzer": self.topology_analyzer.get_info(),
            "causal_analyzer": self.causal_analyzer.get_info(),
            "fusion_module": self.fusion_module.get_info(),
            "strategy_engine": self.strategy_engine.get_info(),
            "history_length": len(self._history.get("decisions", [])),
        }
