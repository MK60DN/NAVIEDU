"""
Tender v2.0 — Embracing Heterogeneity
基础管道示例

本示例演示如何使用 Tender 管道对一个时间窗口的群聊消息进行完整的
情绪-认知分析，从向量化到策略输出。
"""

import json
import os
from pathlib import Path

from tender.pipeline import TenderPipeline, PipelineResult, load_config


def main() -> None:
    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    if not os.path.exists(config_path):
        # 使用默认配置
        config = {
            "emotion_vectorizer": {"engine": "neuro_symbolic", "model_name": "gpt-4o-mini"},
            "topology_analysis": {"engine": "persistent_laplacian"},
            "causal_analysis": {"engine": "convergent_cross_mapping"},
            "cognition": {"engine": "hybrid_state"},
            "fusion": {"engine": "dct_gnn"},
            "synergy": {"engine": "layered_reasoning"},
            "strategy": {"engine": "causal_rl", "heterogeneity_coordination": {"enabled": True}},
            "heterogeneity": {"enabled": True},
            "mismatch": {"enabled": True},
            "general": {"log_level": "INFO", "random_seed": 42},
        }
    else:
        config = load_config(config_path)

    # =========================================================================
    # 2. 准备模拟数据
    # =========================================================================
    # 模拟一个时间窗口内的群聊消息
    messages = {
        "alice": [
            {
                "content": "我终于明白傅立叶变换了，太爽了！",
                "timestamp": 100.0,
                "user_id": "alice",
            },
            {
                "content": "这个性质真的优雅，难怪叫现代数学的赞歌。",
                "timestamp": 105.0,
                "user_id": "alice",
            },
        ],
        "bob": [
            {
                "content": "老师，能不能再讲一遍，我有点懵...",
                "timestamp": 102.0,
                "user_id": "bob",
            },
            {
                "content": "那个频率域的概念我还是不太理解",
                "timestamp": 107.0,
                "user_id": "bob",
            },
        ],
        "charlie": [
            {
                "content": "我查一下资料，感觉这个和之前讲的拉普拉斯变换有关系？",
                "timestamp": 103.0,
                "user_id": "charlie",
            },
        ],
        "diana": [
            {
                "content": "昨天的作业第三题有人会做吗？",
                "timestamp": 101.0,
                "user_id": "diana",
            },
        ],
        "eve": [
            {
                "content": "有没有录播回放？我想再看一遍概念部分。",
                "timestamp": 106.0,
                "user_id": "eve",
            },
        ],
        # 模拟一个"无法融入"的成员
        "frank": [
            {
                "content": "比赛开始了！马上要团战了！有没有人一起？",
                "timestamp": 104.0,
                "user_id": "frank",
            },
            {
                "content": "你们学什么傅立叶啊，来打游戏啊！",
                "timestamp": 108.0,
                "user_id": "frank",
            },
        ],
        # 模拟一个潜水成员
        "grace": [
            {
                "content": "好的",
                "timestamp": 100.0,
                "user_id": "grace",
            },
        ],
        "hector": [
            {
                "content": "我觉得这个例子不太对，因为边界条件没有考虑进去。",
                "timestamp": 106.0,
                "user_id": "hector",
            },
        ],
    }

    # =========================================================================
    # 3. 初始化并运行管道
    # =========================================================================
    print("=" * 60)
    print("Tender v2.0 — Embracing Heterogeneity")
    print("基础管道分析示例")
    print("=" * 60)

    pipeline = TenderPipeline(config)
    result: PipelineResult = pipeline.analyze_window(messages)

    # =========================================================================
    # 4. 输出结果
    # =========================================================================
    print("\n" + "=" * 60)
    print("分析结果摘要")
    print("=" * 60)

    # 4.1 拓扑分析结果
    print(f"\n📊 拓扑分析:")
    print(f"   - 情绪簇数量: {result.topology_result.n_clusters}")
    print(f"   - 是否存在环状结构: {result.topology_result.ring_exists}")
    print(f"   - 离群点比例: {result.topology_result.outlier_ratio:.2%}")
    print(f"   - 全局重心: {result.topology_result.global_centroid}")

    # 4.2 因果分析结果
    print(f"\n🔗 因果分析:")
    if result.causal_result.super_spreaders:
        print(f"   - 超级传播者: {', '.join(result.causal_result.super_spreaders[:3])}")
    print(f"   - 因果边数量: {len(result.causal_result.causal_edges)}")
    print(f"   - 因果网络密度: {result.causal_result.causal_density:.3f}")

    # 4.3 认知分析结果
    print(f"\n🧠 认知分析:")
    for member_id, state in result.cognition_states.items():
        print(f"   - {member_id}: 负荷={state.cognitive_load:.2f}, "
              f"理解={state.understanding_level:.2f}, "
              f"困惑={state.confusion_level:.2f}, "
              f"阶段={state.cognitive_phase.value}")

    # 4.4 情绪-认知协同
    print(f"\n🤝 情绪-认知协同:")
    print(f"   - 协同度: {result.synergy_result.synergy_score:.2f}")
    print(f"   - 协同模式: {result.synergy_result.synergy_mode.value}")
    print(f"   - 主导维度: {result.synergy_result.dominant_dimension}")
    print(f"   - 建议: {result.synergy_result.recommendation}")

    # 4.5 异质性分析 (🆕)
    print(f"\n🔬 群体异质性分析:")
    if hasattr(result, 'heterogeneity_metrics') and result.heterogeneity_metrics is not None:
        h = result.heterogeneity_metrics
        print(f"   - 拓扑丰富度: {h.topological_richness:.2f}")
        print(f"   - 因果碎片化: {h.causal_fragmentation:.2f}")
        print(f"   - 时间异步度: {h.temporal_asynchrony:.2f}")
        print(f"   - 参与度基尼系数: {h.participation_gini:.2f}")
        if hasattr(h, 'outlier_types') and h.outlier_types:
            print(f"   - 离群者分类:")
            for mid, otype in h.outlier_types.items():
                print(f"       {mid}: {otype.value}")

    # 4.6 不匹配检测 (🆕)
    print(f"\n📏 个人-群体不匹配检测:")
    if hasattr(result, 'mismatch_metrics') and result.mismatch_metrics is not None:
        for member_id, mm in result.mismatch_metrics.items():
            status = "✅ 健康独立" if mm.personal_self_consistency > 0.7 and mm.dynamic_distance > 0.7 else \
                     "⚠️ 需要关注" if mm.personal_self_consistency < 0.5 and mm.dynamic_distance > 0.7 else \
                     "🟢 正常融入" if mm.dynamic_distance < 0.4 else \
                     "🔍 观察中"
            print(f"   - {member_id}: 拓扑距离={mm.structural_distance:.2f}, "
                  f"动态距离={mm.dynamic_distance:.2f}, "
                  f"自洽性={mm.personal_self_consistency:.2f} "
                  f"→ {status}")

    # 4.7 融合结果
    print(f"\n🔮 融合与预测:")
    print(f"   - 群体健康度: {result.fusion_result.health_index:.2f}")
    if result.fusion_result.forecast is not None:
        print(f"   - 下一窗口预测: {result.fusion_result.forecast}")

    # 4.8 最终策略
    print(f"\n🎯 异质性协调后的最终策略:")
    for i, strategy in enumerate(result.final_strategies):
        print(f"\n   策略 {i+1}:")
        print(f"   - 风险等级: {strategy.risk_level.value}")
        print(f"   - 目标成员数: {len(strategy.target_members)}")
        print(f"   - 推荐动作: {strategy.action}")
        print(f"   - 置信度: {strategy.confidence:.2f}")
        if strategy.rationale:
            print(f"   - 决策理由: {strategy.rationale}")

    # =========================================================================
    # 5. 保存结果（可选）
    # =========================================================================
    output_dir = Path("output/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "basic_pipeline_result.json"

    # 只保存可 JSON 序列化的部分
    export_data = {
        "topology": {
            "n_clusters": result.topology_result.n_clusters,
            "ring_exists": result.topology_result.ring_exists,
            "outlier_ratio": result.topology_result.outlier_ratio,
            "cluster_labels": result.topology_result.cluster_labels.tolist(),
        },
        "causal": {
            "n_edges": len(result.causal_result.causal_edges),
            "super_spreaders": result.causal_result.super_spreaders,
            "density": result.causal_result.causal_density,
        },
        "cognition": {
            mid: {
                "cognitive_load": state.cognitive_load,
                "understanding_level": state.understanding_level,
                "confusion_level": state.confusion_level,
                "cognitive_phase": state.cognitive_phase.value,
            }
            for mid, state in result.cognition_states.items()
        },
        "synergy": {
            "score": result.synergy_result.synergy_score,
            "mode": result.synergy_result.synergy_mode.value,
            "dominant_dimension": result.synergy_result.dominant_dimension,
            "recommendation": result.synergy_result.recommendation,
        },
        "heterogeneity": {
            "topological_richness": result.heterogeneity_metrics.topological_richness,
            "causal_fragmentation": result.heterogeneity_metrics.causal_fragmentation,
            "outlier_types": {k: v.value for k, v in result.heterogeneity_metrics.outlier_types.items()},
        } if hasattr(result, 'heterogeneity_metrics') and result.heterogeneity_metrics is not None else {},
        "strategies": [
            {
                "risk_level": s.risk_level.value,
                "action": s.action,
                "n_targets": len(s.target_members),
                "confidence": s.confidence,
                "rationale": s.rationale,
            }
            for s in result.final_strategies
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
