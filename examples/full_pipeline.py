"""
Tender v2.0 — Embracing Heterogeneity
完整全链路管道示例

本示例演示如何使用 Tender 管道对一个时间窗口的群聊消息进行完整的
全链路分析，从数据加载、情绪向量化、拓扑分析、因果分析、认知分析、
融合、协同分析、异质性分析、不匹配检测到最终的异质性协调策略输出。

本示例整合了 basic_pipeline.py、heterogeneity_analysis.py、
mismatch_detection.py 和 voluntary_isolate.py 的所有功能。
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live

from tender.pipeline import TenderPipeline, PipelineResult, load_config
from tender.heterogeneity.topology_analysis import TopologicalDisconnectAnalyzer, LoopDetector
from tender.heterogeneity.causal_analysis import CausalFragmentationAnalyzer, PowerCentralityAnalyzer
from tender.heterogeneity.behavior_analysis import (
    TemporalAsynchronyAnalyzer,
    LinguisticDivergenceAnalyzer,
    ParticipationGiniAnalyzer,
)
from tender.heterogeneity.isolate_analyzer import IsolateAnalyzer
from tender.heterogeneity.base import HeterogeneityMetrics
from tender.mismatch.topological_mismatch_detector import TopologicalMismatchDetector
from tender.mismatch.dynamic_mismatch_detector import DynamicMismatchDetector
from tender.mismatch.personal_independence import PersonalIndependenceModel
from tender.mismatch.base import MismatchMetrics

console = Console()


def generate_complex_mock_data() -> Dict[str, List[Dict]]:
    """
    生成复杂的模拟数据，模拟一个在线学习社区的真实讨论场景。

    成员角色分布：
    - 活跃学习者 (alice, bob, charlie)：积极发言、互相讨论、理解水平高
    - 困惑学习者 (diana, eve)：频繁提问、认知负荷高、理解水平低
    - 重点突破者 (frank)：曾经困惑但最近突然理解了，情绪转变明显
    - 知识分享者 (grace)：喜欢总结知识点、分享资料、影响力高
    - 批判性思考者 (hector)：经常提出不同意见、喜欢辩论、结构独特
    - 自愿隔离者 (ivy)：自洽性强、只发简短确认、不参与讨论
    - 夜间活跃者 (jack)：只在深夜活动、时间异步、但质量高
    - 潜水旁观者 (kate)：几乎不发言、偶尔点个赞或发表情
    - 情绪带动者 (leo)：情绪波动大、感染他人、影响群体氛围
    """
    np.random.seed(42)

    n_windows = 5  # 5 个时间窗口
    members = {
        "alice": {"type": "active_learner", "n_msgs_per_window": (3, 5)},
        "bob": {"type": "active_learner", "n_msgs_per_window": (2, 4)},
        "charlie": {"type": "active_learner", "n_msgs_per_window": (2, 4)},
        "diana": {"type": "confused_learner", "n_msgs_per_window": (4, 6)},
        "eve": {"type": "confused_learner", "n_msgs_per_window": (3, 5)},
        "frank": {"type": "breakthrough", "n_msgs_per_window": (2, 3)},
        "grace": {"type": "knowledge_sharer", "n_msgs_per_window": (3, 5)},
        "hector": {"type": "critical_thinker", "n_msgs_per_window": (2, 4)},
        "ivy": {"type": "voluntary_isolate", "n_msgs_per_window": (0, 2)},
        "jack": {"type": "night_owl", "n_msgs_per_window": (2, 4)},
        "kate": {"type": "lurker", "n_msgs_per_window": (0, 1)},
        "leo": {"type": "emotional_driver", "n_msgs_per_window": (3, 5)},
    }

    # 定义各类型成员的消息模板
    message_templates = {
        "active_learner": [
            "这个知识点我理解了！和大家分享一下我的理解：...",
            "@其他人 你们的看法是什么？我觉得这个部分很关键。",
            "例子中的公式推导非常清晰，建议大家都看一下。",
            "我来补充一个应用场景，帮助理解抽象概念。",
            "课后习题第二题用这个方法可以解出来。",
        ],
        "confused_learner": [
            "有没有人能解释一下这个定理？我看了三遍还是不懂...",
            "我是不是错过了什么前置知识？感觉很吃力。",
            "为什么这里要引入这个新概念？和前面讲的有关系吗？",
            "这个推导的第三步是怎么来的？求大神指点。",
            "好焦虑啊，感觉进度太快跟不上了。",
        ],
        "breakthrough": [
            "突然就理解了！原来是这个意思！太开心了！",
            "刚才还在迷茫，现在豁然开朗，感谢 @grace 的分享。",
            "顿悟时刻！这个理论真的太优雅了！",
            "原来之前看不懂是因为忽略了边界条件，现在终于通了。",
        ],
        "knowledge_sharer": [
            "我整理了一份学习笔记，大家自取：[链接]",
            "这个工具可以自动化分析过程，推荐给大家。",
            "我来总结一下今天的内容重点...",
            "分享一个我看到的相关论文，写得很好：",
        ],
        "critical_thinker": [
            "我持不同意见。这个推导的前提假设是有问题的。",
            "大家有没有考虑过另一种可能性？比如...",
            "这个观点有一定的局限性，特别是在这种情况下...",
            "我不同意 @hector 的说法，我们有证据表明...",  # 自我辩论
        ],
        "voluntary_isolate": [
            "已阅。",
            "收到。",
            "好的。",
        ],
        "night_owl": [
            "深夜学习效率真高，现在思路特别清晰。",
            "半夜来复盘一下今天的知识点...",
            "安静的环境最适合深入思考。",
        ],
        "lurker": [
            "👍",
            "收到。",
            "好的谢谢。",
        ],
        "emotional_driver": [
            "太棒了！今天的内容太精彩了！！",
            "好难啊... 我觉得自己好笨... 😭",  # 情绪波动大
            "大家一起加油！我们一定可以的！💪",
            "我有点崩溃了，今天的作业也太多了吧...",
            "太开心了！终于做出来了！！！",
        ],
    }

    messages = {}
    for member_id, config in members.items():
        mtype = config["type"]
        n_range = config["n_msgs_per_window"]
        templates = message_templates[mtype]

        msgs = []
        for w in range(n_windows):
            base_time = w * 60.0  # 每个窗口 60 分钟
            n_msgs = np.random.randint(n_range[0], n_range[1] + 1)

            for _ in range(n_msgs):
                content = templates[np.random.randint(0, len(templates))]

                # 时间偏移：夜间活跃者偏移到深夜
                if mtype == "night_owl":
                    timestamp = base_time + 720.0 + np.random.uniform(0, 60)
                else:
                    timestamp = base_time + np.random.uniform(0, 55)

                msgs.append({
                    "content": content,
                    "timestamp": timestamp,
                    "user_id": member_id,
                })

        messages[member_id] = msgs

    return messages


def main() -> None:
    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
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

    # =========================================================================
    # 2. 准备数据
    # =========================================================================
    console.print(Panel.fit(
        "[bold blue]Tender v2.0 — Embracing Heterogeneity[/bold blue]\n"
        "[bold cyan]完整全链路分析管道[/bold cyan]\n"
        "[dim]从情绪向量化到异质性协调的完整流程[/dim]",
        border_style="blue",
    ))

    messages = generate_complex_mock_data()
    member_ids = list(messages.keys())

    console.print(f"\n[bold green]✅ 数据准备完成[/bold green]")
    console.print(f"   成员数: {len(member_ids)}")
    console.print(f"   消息总数: {sum(len(v) for v in messages.values())}")
    console.print(f"   成员类型分布:")
    type_counts = {}
    for mid, msgs in messages.items():
        # 通过消息数量和行为模式推断类型
        n_msgs = len(msgs)
        if n_msgs <= 1:
            t = "lurker"
        elif n_msgs <= 3:
            t = "observer"
        elif n_msgs <= 6:
            t = "participant"
        else:
            t = "active_member"
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, count in type_counts.items():
        console.print(f"     - {t}: {count}人")

    # =========================================================================
    # 3. 运行完整管道
    # =========================================================================
    console.print("\n[yellow]🔄 运行全链路分析管道...[/yellow]")

    with console.status("[bold green]分析中...") as status:
        pipeline = TenderPipeline(config)

        status.update("[bold green]步骤 1/12: 情绪向量化...")
        result = pipeline.analyze_window(messages)

        status.update("[bold green]步骤 2/12: 拓扑分析...")
        # 拓扑分析已在 analyze_window 中完成

        status.update("[bold green]步骤 3/12: 因果分析...")
        # 因果分析已在 analyze_window 中完成

        status.update("[bold green]步骤 4/12: 认知分析...")
        # 认知分析已在 analyze_window 中完成

        status.update("[bold green]步骤 5/12: 时空融合...")

        status.update("[bold green]步骤 6/12: 协同分析...")

        status.update("[bold green]步骤 7/12: 异质性分析...")
        # 初始化异质性分析器
        hetero_config = config.get("heterogeneity", {})
        disconnect_analyzer = TopologicalDisconnectAnalyzer(hetero_config)
        loop_detector = LoopDetector(hetero_config)
        frag_analyzer = CausalFragmentationAnalyzer(hetero_config)
        power_analyzer = PowerCentralityAnalyzer(hetero_config)
        async_analyzer = TemporalAsynchronyAnalyzer(hetero_config)
        ling_analyzer = LinguisticDivergenceAnalyzer(hetero_config)
        gini_analyzer = ParticipationGiniAnalyzer(hetero_config)
        isolate_analyzer = IsolateAnalyzer(hetero_config)

        disconnect_scores = {}
        for mid in member_ids:
            score = disconnect_analyzer.compute_disconnect(
                member_id=mid,
                topology_result=result.topology_result,
                causal_result=result.causal_result,
                cognition_states=result.cognition_states,
            )
            disconnect_scores[mid] = score

        loops = loop_detector.detect_loops(result.topology_result)
        fragment_metrics = frag_analyzer.compute_fragmentation(result.causal_result)
        centrality_metrics = power_analyzer.compute_centrality(result.causal_result, result.topology_result)
        asynchrony_scores = async_analyzer.compute_asynchrony(messages)
        divergence_scores = ling_analyzer.compute_divergence(messages)
        message_counts = {mid: len(msgs) for mid, msgs in messages.items()}
        gini = gini_analyzer.compute_gini(message_counts)

        outlier_types = {}
        outlier_threshold = hetero_config.get("topology_disconnect", {}).get("outlier_threshold", 0.6)
        for mid in member_ids:
            if disconnect_scores[mid].total > outlier_threshold:
                isolate_type = isolate_analyzer.classify(
                    disconnect_score=disconnect_scores[mid],
                    personal_profile=result.personal_profiles.get(mid),
                    group_profile=result.topology_result,
                )
                outlier_types[mid] = isolate_type

        heterogeneity_metrics = HeterogeneityMetrics(
            topological_richness=loop_detector.compute_richness(result.topology_result),
            loop_strength=loops[0].persistence if loops else 0.0,
            causal_fragmentation=fragment_metrics.fragmentation_index,
            component_separation=fragment_metrics.largest_component_ratio,
            temporal_asynchrony=asynchrony_scores.overall_asynchrony,
            linguistic_divergence=divergence_scores.overall_divergence,
            participation_gini=gini,
            cluster_ids=list(set(result.topology_result.cluster_labels.tolist())),
            cluster_members={},
            outlier_types={
                k: v if hasattr(v, 'value') else v
                for k, v in outlier_types.items()
            },
        )

        status.update("[bold green]步骤 8/12: 不匹配检测...")
        mismatch_config = config.get("mismatch", {})
        topo_detector = TopologicalMismatchDetector(mismatch_config)
        dyna_detector = DynamicMismatchDetector(mismatch_config)
        independence_model = PersonalIndependenceModel(mismatch_config)

        personal_point_clouds = {}
        personal_time_series = {}
        for mid in member_ids:
            if mid in result.emotion_vectors and result.emotion_vectors[mid]:
                arr = np.array([v.to_array() for v in result.emotion_vectors[mid]])
                personal_point_clouds[mid] = arr
                personal_time_series[mid] = arr
            else:
                personal_point_clouds[mid] = np.random.randn(1, 3)
                personal_time_series[mid] = np.random.randn(1, 3)

        mismatch_results = {}
        for mid in member_ids:
            topo_dist = topo_detector.compute_distance(
                personal_point_cloud=personal_point_clouds[mid],
                group_point_cloud=result.topology_result.point_cloud,
            )
            dyna_dist = dyna_detector.compute_distance(
                personal_ts=personal_time_series[mid],
                group_ts=result.topology_result.trajectory,
            )
            self_cons = independence_model.compute_self_consistency(
                time_series=personal_time_series[mid],
            )
            mismatch_results[mid] = MismatchMetrics(
                structural_distance=topo_dist,
                dynamic_distance=dyna_dist,
                personal_self_consistency=self_cons,
            )

        status.update("[bold green]步骤 9/12: 策略推理...")

        status.update("[bold green]步骤 10/12: 异质性协调...")

        status.update("[bold green]步骤 11/12: 生成报告...")

        status.update("[bold green]步骤 12/12: 保存结果...")

    console.print("[bold green]✅ 全链路分析完成！[/bold green]")

    # =========================================================================
    # 4. 输出综合报告
    # =========================================================================
    console.print("\n[bold]=" * 70)
    console.print("[bold underline cyan]📊 全链路分析综合报告[/bold underline cyan]")
    console.print("[bold]=" * 70)

    # --- 4.1 拓扑分析报告 ---
    console.print(f"\n[bold]1️⃣  空间拓扑分析[/bold]")
    console.print(f"   • 情绪簇数量: {result.topology_result.n_clusters}")
    console.print(f"   • 环状结构存在: {result.topology_result.ring_exists}")
    console.print(f"   • 离群点比例: {result.topology_result.outlier_ratio:.2%}")
    console.print(f"   • 全局重心: {result.topology_result.global_centroid}")

    # --- 4.2 因果分析报告 ---
    console.print(f"\n[bold]2️⃣  时间因果分析[/bold]")
    console.print(f"   • 因果边数量: {len(result.causal_result.causal_edges)}")
    console.print(f"   • 因果网络密度: {result.causal_result.causal_density:.3f}")
    if result.causal_result.super_spreaders:
        console.print(f"   • 超级传播者: {', '.join(result.causal_result.super_spreaders[:3])}")

    # --- 4.3 认知分析报告 ---
    console.print(f"\n[bold]3️⃣  认知状态分析[/bold]")
    cogn_table = Table(title="成员认知状态")
    cogn_table.add_column("成员", style="cyan")
    cogn_table.add_column("认知负荷", justify="right")
    cogn_table.add_column("理解水平", justify="right")
    cogn_table.add_column("困惑水平", justify="right")
    cogn_table.add_column("认知阶段")
    for mid in member_ids:
        if mid in result.cognition_states:
            s = result.cognition_states[mid]
            cogn_table.add_row(
                mid,
                f"{s.cognitive_load:.2f}",
                f"{s.understanding_level:.2f}",
                f"{s.confusion_level:.2f}",
                s.cognitive_phase.value if hasattr(s.cognitive_phase, 'value') else str(s.cognitive_phase),
            )
    console.print(cogn_table)

    # --- 4.4 异质性报告 ---
    console.print(f"\n[bold]4️⃣  异质性分析[/bold]")
    hetero_table = Table(title="群体异质性指标")
    hetero_table.add_column("维度", style="cyan")
    hetero_table.add_column("得分", justify="right")
    hetero_table.add_column("解释")
    hetero_table.add_row("拓扑丰富度", f"{heterogeneity_metrics.topological_richness:.3f}", "结构多样性")
    hetero_table.add_row("环状矛盾强度", f"{heterogeneity_metrics.loop_strength:.3f}", "矛盾持久性")
    hetero_table.add_row("因果碎片化", f"{heterogeneity_metrics.causal_fragmentation:.3f}", "网络分裂度")
    hetero_table.add_row("时间异步度", f"{heterogeneity_metrics.temporal_asynchrony:.3f}", "活动时间错位")
    hetero_table.add_row("语言离散度", f"{heterogeneity_metrics.linguistic_divergence:.3f}", "语言风格差异")
    hetero_table.add_row("参与度基尼", f"{heterogeneity_metrics.participation_gini:.3f}", "参与不均衡度")
    console.print(hetero_table)

    # 离群者类型
    if heterogeneity_metrics.outlier_types:
        console.print(f"\n[bold]🏷️  离群者类型:[/bold]")
        isolate_table = Table(title="离群者分类")
        isolate_table.add_column("成员", style="cyan")
        isolate_table.add_column("脱离度", justify="right")
        isolate_table.add_column("类型", style="bold")
        isolate_table.add_column("建议")
        for mid, otype in heterogeneity_metrics.outlier_types.items():
            otype_str = otype.value if hasattr(otype, 'value') else str(otype)
            suggestions = {
                "VOLUNTARY_ISOLATE": "尊重独立性",
                "INVOLUNTARY_OUTCAST": "软性连接",
                "OPINION_LEADER_DEVIANT": "保留视角",
                "PURE_DIVERSE": "拥抱多样性",
            }
            suggestion = suggestions.get(otype_str, "观察")
            isolate_table.add_row(
                mid,
                f"{disconnect_scores.get(mid, None).total:.3f}" if mid in disconnect_scores else "N/A",
                otype_str,
                suggestion,
            )
        console.print(isolate_table)

    # --- 4.5 不匹配报告 ---
    console.print(f"\n[bold]5️⃣  不匹配检测[/bold]")
    mismatch_table = Table(title="个人-群体不匹配评估")
    mismatch_table.add_column("成员", style="cyan")
    mismatch_table.add_column("拓扑距离", justify="right")
    mismatch_table.add_column("动态距离", justify="right")
    mismatch_table.add_column("自洽性", justify="right")
    mismatch_table.add_column("诊断", style="bold")
    mismatch_table.add_column("建议")

    for mid in member_ids:
        mm = mismatch_results[mid]
        if mm.structural_distance < 0.4 and mm.dynamic_distance < 0.4 and mm.personal_self_consistency > 0.7:
            diagnosis = "[green]高度融入者[/green]"
            action = "维持现状"
        elif mm.structural_distance > 0.6 and mm.dynamic_distance > 0.6 and mm.personal_self_consistency > 0.7:
            diagnosis = "[cyan]自愿独行者[/cyan]"
            action = "尊重独立"
        elif mm.structural_distance > 0.6 and mm.dynamic_distance > 0.6 and mm.personal_self_consistency < 0.5:
            diagnosis = "[red]需要关注者[/red]"
            action = "个性化支持"
        elif mm.structural_distance < 0.5 and mm.dynamic_distance < 0.5 and mm.personal_self_consistency < 0.5:
            diagnosis = "[yellow]迷失追随者[/yellow]"
            action = "建立自我认知"
        elif mm.structural_distance > 0.5 and mm.dynamic_distance < 0.4 and mm.personal_self_consistency > 0.6:
            diagnosis = "[magenta]结构独特者[/magenta]"
            action = "鼓励创新"
        else:
            diagnosis = "[white]一般状态[/white]"
            action = "常规观察"

        mismatch_table.add_row(
            mid,
            f"{mm.structural_distance:.3f}",
            f"{mm.dynamic_distance:.3f}",
            f"{mm.personal_self_consistency:.3f}",
            diagnosis,
            action,
        )
    console.print(mismatch_table)

    # --- 4.6 协同报告 ---
    console.print(f"\n[bold]6️⃣  情绪-认知协同[/bold]")
    console.print(f"   • 协同度: {result.synergy_result.synergy_score:.3f}")
    console.print(f"   • 协同模式: {result.synergy_result.synergy_mode.value if hasattr(result.synergy_result.synergy_mode, 'value') else result.synergy_result.synergy_mode}")
    console.print(f"   • 主导维度: {result.synergy_result.dominant_dimension}")
    console.print(f"   • 情绪适应度: {result.synergy_result.adaptation_score:.3f}")
    console.print(f"   • 建议: {result.synergy_result.recommendation}")

    # --- 4.7 融合与预测 ---
    console.print(f"\n[bold]7️⃣  融合与预测[/bold]")
    console.print(f"   • 群体健康度: {result.fusion_result.health_index:.3f}")

    # --- 4.8 最终策略 ---
    console.print(f"\n[bold]8️⃣  异质性协调后的策略[/bold]")
    for i, s in enumerate(result.final_strategies):
        console.print(f"\n   ┌── [bold cyan]策略 {i+1}[/bold cyan] ──")
        console.print(f"   │ 风险等级: {s.risk_level.value if hasattr(s.risk_level, 'value') else s.risk_level}")
        console.print(f"   │ 目标成员: {', '.join(s.target_members[:5])}{'...' if len(s.target_members) > 5 else ''}")
        console.print(f"   │ 推荐动作: {s.action}")
        console.print(f"   │ 置信度: {s.confidence:.2f}")
        if s.rationale:
            console.print(f"   │ 决策理由: {s.rationale}")
        console.print(f"   └──{'─' * 40}")

    # =========================================================================
    # 5. 关键洞察
    # =========================================================================
    console.print(f"\n[bold]🔑 关键洞察[/bold]")

    # 统计
    total = len(member_ids)
    need_intervention = sum(
        1 for mm in mismatch_results.values()
        if (mm.structural_distance > 0.6 or mm.dynamic_distance > 0.6) and mm.personal_self_consistency < 0.5
    )
    independent_healthy = sum(
        1 for mm in mismatch_results.values()
        if mm.dynamic_distance > 0.6 and mm.personal_self_consistency > 0.7
    )
    well_integrated = sum(
        1 for mm in mismatch_results.values()
        if mm.dynamic_distance < 0.4 and mm.personal_self_consistency > 0.7
    )

    console.print(f"   • [green]高度融入者: {well_integrated}/{total}[/green]")
    console.print(f"   • [cyan]独立健康者: {independent_healthy}/{total}[/cyan]")
    console.print(f"   • [red]需要关注者: {need_intervention}/{total}[/red]")
    console.print(f"   • [yellow]其他: {total - well_integrated - independent_healthy - need_intervention}/{total}[/yellow]")

    if need_intervention == 0:
        console.print("\n[bold green]✅ 群体状态整体健康，无需紧急干预。[/bold green]")
    if independent_healthy > 0:
        console.print(f"\n[bold cyan]💡 {independent_healthy} 名成员是独立健康的自愿独行者，已赋予其独立空间。[/bold cyan]")

    # =========================================================================
    # 6. 保存完整的分析报告
    # =========================================================================
    output_dir = Path("output/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "full_pipeline_result.json"

    # 构建可序列化的输出
    export_data = {
        "summary": {
            "n_members": len(member_ids),
            "n_messages": sum(len(v) for v in messages.values()),
            "window_analyzed": "single_window",
        },
        "topology": {
            "n_clusters": result.topology_result.n_clusters,
            "ring_exists": result.topology_result.ring_exists,
            "outlier_ratio": result.topology_result.outlier_ratio,
            "global_centroid": result.topology_result.global_centroid.tolist(),
            "cluster_labels": result.topology_result.cluster_labels.tolist(),
        },
        "causal": {
            "n_edges": len(result.causal_result.causal_edges),
            "super_spreaders": result.causal_result.super_spreaders,
            "density": result.causal_result.causal_density,
        },
        "cognition": {
            mid: {
                "cognitive_load": s.cognitive_load,
                "understanding_level": s.understanding_level,
                "confusion_level": s.confusion_level,
                "cognitive_phase": s.cognitive_phase.value if hasattr(s.cognitive_phase, 'value') else str(s.cognitive_phase),
            }
            for mid, s in result.cognition_states.items()
        },
        "heterogeneity": {
            "topological_richness": heterogeneity_metrics.topological_richness,
            "loop_strength": heterogeneity_metrics.loop_strength,
            "causal_fragmentation": heterogeneity_metrics.causal_fragmentation,
            "component_separation": heterogeneity_metrics.component_separation,
            "temporal_asynchrony": heterogeneity_metrics.temporal_asynchrony,
            "linguistic_divergence": heterogeneity_metrics.linguistic_divergence,
            "participation_gini": heterogeneity_metrics.participation_gini,
            "outlier_types": {
                k: v.value if hasattr(v, 'value') else str(v)
                for k, v in heterogeneity_metrics.outlier_types.items()
            },
        },
        "mismatch": {
            mid: {
                "structural_distance": mm.structural_distance,
                "dynamic_distance": mm.dynamic_distance,
                "self_consistency": mm.personal_self_consistency,
                "needs_intervention": (mm.structural_distance > 0.6 or mm.dynamic_distance > 0.6) and mm.personal_self_consistency < 0.5,
            }
            for mid, mm in mismatch_results.items()
        },
        "synergy": {
            "score": result.synergy_result.synergy_score,
            "mode": result.synergy_result.synergy_mode.value if hasattr(result.synergy_result.synergy_mode, 'value') else str(result.synergy_result.synergy_mode),
            "dominant_dimension": result.synergy_result.dominant_dimension,
            "adaptation_score": result.synergy_result.adaptation_score,
            "recommendation": result.synergy_result.recommendation,
        },
        "strategies": [
            {
                "risk_level": s.risk_level.value if hasattr(s.risk_level, 'value') else str(s.risk_level),
                "action": s.action,
                "target_members": s.target_members,
                "confidence": s.confidence,
                "rationale": s.rationale,
            }
            for s in result.final_strategies
        ],
        "key_insights": {
            "well_integrated": well_integrated,
            "independent_healthy": independent_healthy,
            "need_intervention": need_intervention,
            "total_members": total,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    console.print(f"\n[bold green]💾 完整分析报告已保存到 {output_path}[/bold green]")
    console.print("\n[bold cyan]✨ 全链路分析完成！[/bold cyan]")
    console.print("[dim]拥抱异质性，尊重每一个独立的 Being。[/dim]")


if __name__ == "__main__":
    main()
