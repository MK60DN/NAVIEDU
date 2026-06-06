"""
Tender v2.0 — Embracing Heterogeneity
自愿隔离者案例分析

本示例演示 Tender 如何处理一个“自愿隔离者”场景——一个行为模式与群体
完全不同、但个人状态良好（高自洽性）的成员。系统应当尊重其独立性，
而不是强行要求融入。

场景设定：
- Bob 是一个“自愿隔离者”：他在学习群中从不参与讨论，但会独立完成作业，
  情绪状态稳定，自洽性高。
- 其他成员（Alice, Charlie, Diana, Eve）是积极讨论的学习者。
- 系统应当识别 Bob 是“自愿隔离者”，并给出“尊重选择”的策略建议。
"""

import os
import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tender.pipeline import TenderPipeline, load_config
from tender.mismatch.topological_mismatch_detector import TopologicalMismatchDetector
from tender.mismatch.dynamic_mismatch_detector import DynamicMismatchDetector
from tender.mismatch.personal_independence import PersonalIndependenceModel
from tender.heterogeneity.topology_analysis import TopologicalDisconnectAnalyzer
from tender.heterogeneity.isolate_analyzer import IsolateAnalyzer

console = Console()


def main() -> None:
    console.print(Panel.fit(
        "[bold blue]Tender v2.0 — 自愿隔离者案例分析[/bold blue]\n"
        "[dim]场景：一个自洽的独立学习者 vs. 积极互动的群体[/dim]",
        border_style="blue",
    ))

    # =========================================================================
    # 1. 构建模拟数据
    # =========================================================================
    console.print("\n[bold green]📦 构建模拟数据...[/bold green]")

    messages = {
        # 积极互动的学习者（Alice, Charlie, Diana, Eve）
        "alice": [
            {"content": "这个知识点太棒了！我来总结一下...", "timestamp": 100.0, "user_id": "alice"},
            {"content": "大家看这个例子，完美诠释了刚才的概念。", "timestamp": 105.0, "user_id": "alice"},
            {"content": "@charlie 你说得对，我再补充一点。", "timestamp": 110.0, "user_id": "alice"},
            {"content": "有没有人想一起做课后练习？", "timestamp": 115.0, "user_id": "alice"},
        ],
        "charlie": [
            {"content": "我有一个问题，这个定理怎么证明？", "timestamp": 102.0, "user_id": "charlie"},
            {"content": "我查到了！这个定理的证明在这里...", "timestamp": 108.0, "user_id": "charlie"},
            {"content": "@alice 你说的对，我理解了，谢谢！", "timestamp": 112.0, "user_id": "charlie"},
        ],
        "diana": [
            {"content": "有没有录播回放？我想再看一遍。", "timestamp": 101.0, "user_id": "diana"},
            {"content": "第三题有人会吗？我卡了好久了。", "timestamp": 107.0, "user_id": "diana"},
            {"content": "终于做出来了！谢谢 @charlie 的提示！", "timestamp": 113.0, "user_id": "diana"},
        ],
        "eve": [
            {"content": "我建议我们把重点整理成笔记。", "timestamp": 103.0, "user_id": "eve"},
            {"content": "这个工具真方便，自动化了我的分析流程。", "timestamp": 109.0, "user_id": "eve"},
            {"content": "明天的课别忘了预习第四章。", "timestamp": 114.0, "user_id": "eve"},
        ],
        # Bob：自愿隔离者（独自完成学习，不参与互动）
        "bob": [
            {"content": "已阅。", "timestamp": 100.0, "user_id": "bob"},
            {"content": "作业已完成。", "timestamp": 120.0, "user_id": "bob"},
            # Bob 的消息只有简短的确认，没有互动内容
        ],
    }

    console.print(f"   成员数: {len(messages)}")
    console.print(f"   消息总数: {sum(len(v) for v in messages.values())}")
    console.print("   [dim]Bob: 只有 2 条消息，均为简短的确认[/dim]")

    # =========================================================================
    # 2. 加载配置并运行管道
    # =========================================================================
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = load_config(config_path) if os.path.exists(config_path) else {
        "emotion_vectorizer": {"engine": "neuro_symbolic"},
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

    console.print("\n[yellow]🔄 运行基础分析管道...[/yellow]")
    pipeline = TenderPipeline(config)
    result = pipeline.analyze_window(messages)
    console.print("[bold green]✅ 管道分析完成[/bold green]")

    # =========================================================================
    # 3. 不匹配与自洽性分析
    # =========================================================================
    mismatch_config = config.get("mismatch", {})
    hetero_config = config.get("heterogeneity", {})

    topo_detector = TopologicalMismatchDetector(mismatch_config)
    dyna_detector = DynamicMismatchDetector(mismatch_config)
    independence_model = PersonalIndependenceModel(mismatch_config)
    disconnect_analyzer = TopologicalDisconnectAnalyzer(hetero_config)
    isolate_analyzer = IsolateAnalyzer(hetero_config)

    console.print("\n[bold]=" * 60)
    console.print("[bold underline cyan]📊 成员分析报告[/bold underline cyan]")
    console.print("[bold]=" * 60)

    member_ids = list(messages.keys())
    results = {}

    for member_id in member_ids:
        # 拓扑不匹配距离
        if member_id in result.emotion_vectors and result.emotion_vectors[member_id]:
            personal_pc = np.array([v.to_array() for v in result.emotion_vectors[member_id]])
            personal_ts = personal_pc
        else:
            personal_pc = np.random.randn(1, 3)
            personal_ts = np.random.randn(1, 3)

        topo_dist = topo_detector.compute_distance(
            personal_point_cloud=personal_pc,
            group_point_cloud=result.topology_result.point_cloud,
        )
        dyna_dist = dyna_detector.compute_distance(
            personal_ts=personal_ts,
            group_ts=result.topology_result.trajectory,
        )
        self_cons = independence_model.compute_self_consistency(time_series=personal_ts)
        disconnect = disconnect_analyzer.compute_disconnect(
            member_id=member_id,
            topology_result=result.topology_result,
            causal_result=result.causal_result,
            cognition_states=result.cognition_states,
        )

        results[member_id] = {
            "topo_dist": topo_dist,
            "dyna_dist": dyna_dist,
            "self_cons": self_cons,
            "disconnect": disconnect,
        }

    # 用表格展示
    table = Table(title="成员不匹配与自洽性")
    table.add_column("成员", style="cyan")
    table.add_column("拓扑距离", justify="right")
    table.add_column("动态距离", justify="right")
    table.add_column("自洽性", justify="right")
    table.add_column("脱离度", justify="right")
    table.add_column("诊断")
    table.add_column("建议动作")

    for member_id in member_ids:
        r = results[member_id]
        topo_dist = r["topo_dist"]
        dyna_dist = r["dyna_dist"]
        self_cons = r["self_cons"]
        disconnect = r["disconnect"]

        # 诊断
        if topo_dist > 0.6 and dyna_dist > 0.6 and self_cons > 0.7:
            diagnosis = "[cyan]自愿独行者[/cyan]"
            action = "[green]尊重独立性，不干预[/green]"
        elif topo_dist > 0.6 and dyna_dist > 0.6 and self_cons < 0.5:
            diagnosis = "[red]需要关注者[/red]"
            action = "[yellow]个性化支持[/yellow]"
        elif topo_dist < 0.4 and dyna_dist < 0.4 and self_cons > 0.7:
            diagnosis = "[green]高度融入者[/green]"
            action = "[dim]维持现状[/dim]"
        else:
            diagnosis = "[white]一般状态[/white]"
            action = "[dim]常规观察[/dim]"

        table.add_row(
            member_id,
            f"{topo_dist:.3f}",
            f"{dyna_dist:.3f}",
            f"{self_cons:.3f}",
            f"{disconnect.total:.3f}",
            diagnosis,
            action,
        )

    console.print(table)

    # =========================================================================
    # 4. 重点分析 Bob
    # =========================================================================
    console.print("\n[bold]=" * 60)
    console.print("[bold underline yellow]🔍 重点成员分析: Bob[/bold underline yellow]")
    console.print("[bold]=" * 60)

    bob = results["bob"]

    console.print(f"\n   📐 拓扑不匹配距离: {bob['topo_dist']:.4f}")
    console.print(f"   📊 动态不匹配距离: {bob['dyna_dist']:.4f}")
    console.print(f"   🔄 自洽性: {bob['self_cons']:.4f}")
    console.print(f"   🚪 脱离度: {bob['disconnect'].total:.4f}")

    # 离群者类型分类
    isolate_type = isolate_analyzer.classify(
        disconnect_score=bob["disconnect"],
        personal_profile=result.personal_profiles.get("bob"),
        group_profile=result.topology_result,
    )
    console.print(f"\n   🏷️  离群者类型: [bold cyan]{isolate_type.value}[/bold cyan]")

    # 综合诊断
    console.print(f"\n   🧠 [bold]综合诊断:[/bold]")

    if isolate_type.value == "VOLUNTARY_ISOLATE":
        console.print(Panel.fit(
            "[bold cyan]诊断结论: Bob 是自愿隔离者[/bold cyan]\n\n"
            "Bob 的个人状态分析：\n"
            "• 自洽性高：他的行为和情绪模式是稳定的、可预测的\n"
            "• 脱离度高：他几乎不参与群体互动，但他自己并不痛苦\n"
            "• 不匹配度高：他的学习节奏和互动模式与群体完全不同\n\n"
            "[bold green]推荐策略：尊重独立性[/bold green]\n\n"
            "Bob 不需要“被拉回”群体。他是一个自洽的独立学习者。\n"
            "强行要求他参与互动反而会降低他的学习体验。\n"
            "系统建议：赋予“自由潜水员”角色，仅在他主动求助时提供支持。",
            border_style="green",
        ))
    else:
        console.print(Panel.fit(
            f"[bold yellow]Bob 的类型为 {isolate_type.value}，需要进一步分析。[/bold yellow]",
            border_style="yellow",
        ))

    # =========================================================================
    # 5. 策略输出
    # =========================================================================
    console.print("\n[bold]=" * 60)
    console.print("[bold underline cyan]🎯 最终策略建议[/bold underline cyan]")
    console.print("[bold]=" * 60)

    strategy_table = Table(title="异质性协调后的策略")
    strategy_table.add_column("策略", style="cyan")
    strategy_table.add_column("目标成员")
    strategy_table.add_column("动作", style="bold")
    strategy_table.add_column("置信度", justify="right")
    strategy_table.add_column("理由")

    for i, strategy in enumerate(result.final_strategies):
        target_str = ", ".join(strategy.target_members[:3])
        if len(strategy.target_members) > 3:
            target_str += f" ... (+{len(strategy.target_members) - 3})"
        strategy_table.add_row(
            f"策略 {i+1}",
            target_str,
            strategy.action,
            f"{strategy.confidence:.2f}",
            strategy.rationale[:60] + "..." if strategy.rationale else "",
        )

    console.print(strategy_table)

    # =========================================================================
    # 6. 保存结果
    # =========================================================================
    output_dir = Path("output/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "voluntary_isolate_result.json"

    export_data = {
        "scenario": "自愿隔离者案例分析",
        "members": {
            mid: {
                "topological_distance": r["topo_dist"],
                "dynamic_distance": r["dyna_dist"],
                "self_consistency": r["self_cons"],
                "disconnect_score": r["disconnect"].total,
                "diagnosis": "自愿独行者" if r["topo_dist"] > 0.6 and r["dyna_dist"] > 0.6 and r["self_cons"] > 0.7 else
                             "需要关注者" if r["topo_dist"] > 0.6 and r["dyna_dist"] > 0.6 and r["self_cons"] < 0.5 else
                             "高度融入者" if r["topo_dist"] < 0.4 and r["dyna_dist"] < 0.4 and r["self_cons"] > 0.7 else
                             "一般状态",
            }
            for mid, r in results.items()
        },
        "isolate_analysis": {
            "bob": {
                "isolate_type": isolate_type.value,
                "recommendation": "尊重独立性，赋予自由潜水员角色",
                "intervention_needed": False,
            }
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
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    console.print(f"\n[bold green]💾 分析报告已保存到 {output_path}[/bold green]")
    console.print("\n[bold cyan]✨ 核心洞察：[/bold cyan] 一个自洽的个体，即使与群体完全不匹配，也属于“健康”状态。")
    console.print("[dim]拥抱异质性，尊重每一个独立的 Being。[/dim]")


if __name__ == "__main__":
    main()
