"""
Tender v2.0 — Embracing Heterogeneity
个人-群体不匹配检测示例

本示例演示如何使用 mismatch/ 模块检测个人与群体之间的“不匹配”程度，
以及如何基于自洽性和不匹配度做出干预决策。

核心哲学：只有当个人同时处于“低自洽性”和“高不匹配”时才需要干预。
一个自洽的个体，即使与群体完全不匹配，也属于“健康”状态。
"""

import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from tender.pipeline import TenderPipeline, load_config
from tender.mismatch.topological_mismatch_detector import TopologicalMismatchDetector
from tender.mismatch.dynamic_mismatch_detector import DynamicMismatchDetector
from tender.mismatch.personal_independence import PersonalIndependenceModel
from tender.mismatch.base import MismatchMetrics

console = Console()


def generate_mock_data() -> Dict[str, List[Dict]]:
    """
    生成包含多种不匹配类型的模拟数据。

    成员类型：
    - 高度融入者 (user_1, user_2)：情绪节奏与群体一致
    - 自愿独行者 (user_3)：自洽性高，但与群体节奏不同
    - 迷失追随者 (user_4)：融入但无自我，自洽性低
    - 需要关注者 (user_5)：不匹配且不自洽
    - 批判性创新者 (user_6)：结构独特但自洽
    """
    np.random.seed(42)
    
    # 生成群体重心轨迹（sin 波，模拟讲座中的情绪起伏）
    t = np.linspace(0, 4*np.pi, 100)
    group_trajectory = np.column_stack([
        0.3 * np.sin(t),          # valence 波动
        0.5 + 0.2 * np.sin(t/2),  # arousal 缓慢变化
        0.6 + 0.1 * np.cos(t/3),  # focus 缓慢变化
    ])
    
    # 构建每个成员的消息
    messages = {}
    
    # user_1 和 user_2：高度融入者（与群体轨迹接近）
    for mid in ["user_1", "user_2"]:
        msgs = []
        for i in range(20):
            idx = (i * 5) % 100
            noise = np.random.normal(0, 0.05)
            msgs.append({
                "content": f"这个部分讲得好，我觉得 {['很有道理', '非常清晰', '值得思考'][i % 3]}。",
                "timestamp": float(i * 10 + np.random.uniform(0, 5)),
                "user_id": mid,
            })
        messages[mid] = msgs
    
    # user_3：自愿独行者（节奏快一倍）
    msgs = []
    for i in range(30):
        msgs.append({
            "content": "快进快进，这些我都懂了，能不能讲深一点？",
            "timestamp": float(i * 5 + np.random.uniform(0, 3)),
            "user_id": "user_3",
        })
    messages["user_3"] = msgs
    
    # user_4：迷失追随者（完全跟着群体走，但无自我观点）
    msgs = []
    for i in range(10):
        msgs.append({
            "content": f"同意 {'上面' if i % 2 == 0 else '前面'} 说的，我也觉得是这样。",
            "timestamp": float(i * 12 + np.random.uniform(0, 3)),
            "user_id": "user_4",
        })
    messages["user_4"] = msgs
    
    # user_5：需要关注者（节奏缓慢，情绪低落）
    msgs = []
    for i in range(5):
        msgs.append({
            "content": "我好像完全跟不上... 有没有人能帮我解释一下基础概念？",
            "timestamp": float(i * 20 + np.random.uniform(5, 10)),
            "user_id": "user_5",
        })
    messages["user_5"] = msgs
    
    # user_6：批判性创新者（结构独特但自洽——总是反向思考）
    msgs = []
    for i in range(15):
        msgs.append({
            "content": f"我持保留意见，{'这个结论的前提是...' if i % 2 == 0 else '有没有考虑过另一种可能性...'}",
            "timestamp": float(i * 8 + np.random.uniform(0, 4)),
            "user_id": "user_6",
        })
    messages["user_6"] = msgs
    
    return messages


def main() -> None:
    console.print(Panel.fit(
        "[bold blue]Tender v2.0 — 个人-群体不匹配检测示例[/bold blue]\n"
        "[dim]允许不融入：只有当不自洽时才需要关注[/dim]",
        border_style="blue",
    ))
    
    # =========================================================================
    # 1. 加载配置并运行基础管道
    # =========================================================================
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = load_config(config_path) if os.path.exists(config_path) else {}
    
    # 强制启用不匹配检测
    if "mismatch" not in config:
        config["mismatch"] = {"enabled": True}
    else:
        config["mismatch"]["enabled"] = True
    
    pipeline = TenderPipeline(config)
    
    # 生成模拟数据
    messages = generate_mock_data()
    member_ids = list(messages.keys())
    
    console.print(f"\n[bold green]✅ 数据准备完成[/bold green]")
    console.print(f"   成员 ({len(member_ids)}): {', '.join(member_ids)}")
    
    # 运行管道
    console.print("\n[yellow]🔄 运行基础分析管道...[/yellow]")
    result = pipeline.analyze_window(messages)
    console.print("[bold green]✅ 管道分析完成[/bold green]")
    
    # =========================================================================
    # 2. 初始化不匹配检测器
    # =========================================================================
    mismatch_config = config.get("mismatch", {})
    
    topo_detector = TopologicalMismatchDetector(mismatch_config)
    dyna_detector = DynamicMismatchDetector(mismatch_config)
    independence_model = PersonalIndependenceModel(mismatch_config)
    
    # =========================================================================
    # 3. 构建个人点云（从情绪向量构建）
    # =========================================================================
    personal_point_clouds = {}
    personal_time_series = {}
    
    for member_id in member_ids:
        if member_id in result.emotion_vectors:
            vectors = result.emotion_vectors[member_id]
            if vectors:
                personal_point_clouds[member_id] = np.array([
                    v.to_array() for v in vectors
                ])
                personal_time_series[member_id] = np.array([
                    v.to_array() for v in vectors
                ])
            else:
                personal_point_clouds[member_id] = np.random.randn(1, 3)
                personal_time_series[member_id] = np.random.randn(1, 3)
    
    # =========================================================================
    # 4. 执行不匹配检测
    # =========================================================================
    console.print("\n[bold]=" * 60)
    console.print("[bold underline cyan]📏 不匹配检测报告[/bold underline cyan]")
    console.print("[bold]=" * 60)
    
    mismatch_results = {}
    
    for member_id in member_ids:
        console.print(f"\n[bold]{'─' * 50}[/bold]")
        console.print(f"[bold cyan]▶ 成员: {member_id}[/bold cyan]")
        
        # 4.1 拓扑不匹配
        if member_id in personal_point_clouds:
            topo_dist = topo_detector.compute_distance(
                personal_point_cloud=personal_point_clouds[member_id],
                group_point_cloud=result.topology_result.point_cloud,
            )
        else:
            topo_dist = 0.5  # 默认值
        
        console.print(f"   📐 拓扑不匹配距离: [bold]{topo_dist:.4f}[/bold] "
                      f"{'[red]高[/red]' if topo_dist > 0.7 else '[green]低[/green]' if topo_dist < 0.3 else '[yellow]中[/yellow]'}")
        
        # 4.2 动态不匹配
        if member_id in personal_time_series:
            dyna_dist = dyna_detector.compute_distance(
                personal_ts=personal_time_series[member_id],
                group_ts=result.topology_result.trajectory,
            )
        else:
            dyna_dist = 0.5
        
        console.print(f"   📊 动态不匹配距离: [bold]{dyna_dist:.4f}[/bold] "
                      f"{'[red]高[/red]' if dyna_dist > 0.7 else '[green]低[/green]' if dyna_dist < 0.3 else '[yellow]中[/yellow]'}")
        
        # 4.3 自洽性
        if member_id in personal_time_series:
            self_cons = independence_model.compute_self_consistency(
                time_series=personal_time_series[member_id],
            )
        else:
            self_cons = 0.5
        
        console.print(f"   🔄 自洽性: [bold]{self_cons:.4f}[/bold] "
                      f"{'[green]高[/green]' if self_cons > 0.7 else '[red]低[/red]' if self_cons < 0.4 else '[yellow]中[/yellow]'}")
        
        # 4.4 决策
        console.print(f"   🧠 诊断: ", end="")
        
        if topo_dist < 0.4 and dyna_dist < 0.4 and self_cons > 0.7:
            console.print("[bold green]高度融入者[/bold green] — 与群体完美同步，且自我一致")
            action = "维持现状"
        elif topo_dist > 0.7 and dyna_dist > 0.7 and self_cons > 0.7:
            console.print("[bold cyan]自愿独行者[/bold cyan] 🆕 — 自洽且不与群体同步，尊重选择")
            action = "赋予独立空间"
        elif topo_dist < 0.5 and dyna_dist < 0.5 and self_cons < 0.5:
            console.print("[bold yellow]迷失追随者[/bold yellow] — 虽然融入了，但缺乏自我")
            action = "帮助建立自我认知"
        elif topo_dist > 0.7 and dyna_dist > 0.7 and self_cons < 0.5:
            console.print("[bold red]需要关注者[/bold red] ⚠️ — 不匹配且不自洽，建议干预")
            action = "个性化支持"
        elif topo_dist > 0.6 and dyna_dist < 0.4 and self_cons > 0.6:
            console.print("[bold magenta]结构独特者[/bold magenta] — 拓扑结构独特但节奏同步，可能为创新者")
            action = "观察，鼓励创新"
        else:
            console.print("[bold]混合型[/bold] — 状态复杂，需综合判断")
            action = "持续观察"
        
        console.print(f"   💡 建议: [italic]{action}[/italic]")
        
        mismatch_results[member_id] = MismatchMetrics(
            structural_distance=topo_dist,
            dynamic_distance=dyna_dist,
            personal_self_consistency=self_cons,
        )
    
    # =========================================================================
    # 5. 综合报告
    # =========================================================================
    console.print("\n[bold]=" * 60)
    console.print("[bold underline cyan]📋 综合不匹配报告[/bold underline cyan]")
    console.print("[bold]=" * 60)
    
    table = Table(title="不匹配评估矩阵")
    table.add_column("成员", style="cyan")
    table.add_column("拓扑距离", justify="right")
    table.add_column("动态距离", justify="right")
    table.add_column("自洽性", justify="right")
    table.add_column("诊断", style="bold")
    table.add_column("建议动作")
    
    for member_id in member_ids:
        mm = mismatch_results[member_id]
        
        # 诊断
        if mm.structural_distance < 0.4 and mm.dynamic_distance < 0.4 and mm.personal_self_consistency > 0.7:
            diagnosis = "[green]高度融入[/green]"
            action = "维持现状"
        elif mm.structural_distance > 0.7 and mm.dynamic_distance > 0.7 and mm.personal_self_consistency > 0.7:
            diagnosis = "[cyan]自愿独行者[/cyan]"
            action = "尊重独立"
        elif mm.structural_distance < 0.5 and mm.dynamic_distance < 0.5 and mm.personal_self_consistency < 0.5:
            diagnosis = "[yellow]迷失追随者[/yellow]"
            action = "建立自我"
        elif mm.structural_distance > 0.7 and mm.dynamic_distance > 0.7 and mm.personal_self_consistency < 0.5:
            diagnosis = "[red]需要关注[/red]"
            action = "干预支持"
        elif mm.structural_distance > 0.6 and mm.dynamic_distance < 0.4 and mm.personal_self_consistency > 0.6:
            diagnosis = "[magenta]创新者[/magenta]"
            action = "鼓励创新"
        else:
            diagnosis = "[white]混合型[/white]"
            action = "观察"
        
        table.add_row(
            member_id,
            f"{mm.structural_distance:.3f}",
            f"{mm.dynamic_distance:.3f}",
            f"{mm.personal_self_consistency:.3f}",
            diagnosis,
            action,
        )
    
    console.print(table)
    
    # =========================================================================
    # 6. 关键结论
    # =========================================================================
    console.print("\n[bold]🔑 关键结论[/bold]")
    
    # 统计各类成员的分布
    categories = {
        "高度融入者": [],
        "自愿独行者": [],
        "迷失追随者": [],
        "需要关注者": [],
        "创新者": [],
        "混合型": [],
    }
    
    for member_id in member_ids:
        mm = mismatch_results[member_id]
        if mm.structural_distance < 0.4 and mm.dynamic_distance < 0.4 and mm.personal_self_consistency > 0.7:
            categories["高度融入者"].append(member_id)
        elif mm.structural_distance > 0.7 and mm.dynamic_distance > 0.7 and mm.personal_self_consistency > 0.7:
            categories["自愿独行者"].append(member_id)
        elif mm.structural_distance < 0.5 and mm.dynamic_distance < 0.5 and mm.personal_self_consistency < 0.5:
            categories["迷失追随者"].append(member_id)
        elif mm.structural_distance > 0.7 and mm.dynamic_distance > 0.7 and mm.personal_self_consistency < 0.5:
            categories["需要关注者"].append(member_id)
        elif mm.structural_distance > 0.6 and mm.dynamic_distance < 0.4 and mm.personal_self_consistency > 0.6:
            categories["创新者"].append(member_id)
        else:
            categories["混合型"].append(member_id)
    
    for cat, members in categories.items():
        if members:
            console.print(f"   • {cat}: {', '.join(members)} ({len(members)}人)")
    
    # 统计需要干预的成员
    intervention_needed = [
        member_id for member_id in member_ids
        if (mismatch_results[member_id].structural_distance > 0.7 or
            mismatch_results[member_id].dynamic_distance > 0.7) and
           mismatch_results[member_id].personal_self_consistency < 0.5
    ]
    
    independent_healthy = [
        member_id for member_id in member_ids
        if mismatch_results[member_id].dynamic_distance > 0.7 and
           mismatch_results[member_id].personal_self_consistency > 0.7
    ]
    
    console.print(f"\n[bold red]⚠️  需要关注的成员: {len(intervention_needed)}人[/bold red]" if intervention_needed else "\n[green]✅ 没有成员需要紧急关注[/green]")
    console.print(f"[bold green]✅ 独立健康的成员: {len(independent_healthy)}人[/bold green]" if independent_healthy else "[dim]无自愿独行者[/dim]")
    
    # =========================================================================
    # 7. 保存结果
    # =========================================================================
    output_dir = Path("output/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mismatch_detection.json"
    
    export_data = {
        "member_ids": member_ids,
        "mismatch_results": {
            mid: {
                "structural_distance": mm.structural_distance,
                "dynamic_distance": mm.dynamic_distance,
                "self_consistency": mm.personal_self_consistency,
                "intervention_needed": (mm.structural_distance > 0.7 or mm.dynamic_distance > 0.7) and mm.personal_self_consistency < 0.5,
                "diagnosis": "高度融入者" if mm.structural_distance < 0.4 and mm.dynamic_distance < 0.4 and mm.personal_self_consistency > 0.7 else
                             "自愿独行者" if mm.structural_distance > 0.7 and mm.dynamic_distance > 0.7 and mm.personal_self_consistency > 0.7 else
                             "迷失追随者" if mm.structural_distance < 0.5 and mm.dynamic_distance < 0.5 and mm.personal_self_consistency < 0.5 else
                             "需要关注者" if mm.structural_distance > 0.7 and mm.dynamic_distance > 0.7 and mm.personal_self_consistency < 0.5 else
                             "创新者" if mm.structural_distance > 0.6 and mm.dynamic_distance < 0.4 and mm.personal_self_consistency > 0.6 else
                             "混合型",
            }
            for mid, mm in mismatch_results.items()
        },
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[bold green]💾 不匹配检测报告已保存到 {output_path}[/bold green]")
    console.print("\n[bold cyan]✨ 分析完成！[/bold cyan] 允许不融入，但关注不自洽。")


if __name__ == "__main__":
    main()
