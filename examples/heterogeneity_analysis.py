"""
Tender v2.0 — Embracing Heterogeneity
异质性分析示例

本示例演示如何使用 heterogeneity/ 模块对群体进行深度的异质性分析，
包括拓扑脱离度计算、环状矛盾检测、因果碎片化分析和离群者类型分类。
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
from tender.heterogeneity.topology_analysis import TopologicalDisconnectAnalyzer, LoopDetector
from tender.heterogeneity.causal_analysis import CausalFragmentationAnalyzer, PowerCentralityAnalyzer
from tender.heterogeneity.behavior_analysis import (
    TemporalAsynchronyAnalyzer,
    LinguisticDivergenceAnalyzer,
    ParticipationGiniAnalyzer,
)
from tender.heterogeneity.isolate_analyzer import IsolateAnalyzer
from tender.heterogeneity.base import HeterogeneityMetrics


console = Console()


def generate_mock_data() -> Dict[str, List[Dict]]:
    """
    生成模拟数据，包含多种异质性类型。
    
    成员分布：
    - 氛围组 (alice, bob, charlie): 高理解水平，积极讨论
    - 焦虑组 (diana, eve): 高困惑水平，提问密集
    - 独行者 (frank): 自洽性强，但与学习主题脱离（游戏爱好者）
    - 潜水员 (grace): 几乎不发言
    - 批判者 (hector): 与主流意见不同，但积极参与
    - 夜猫子 (ivy, jack): 在深夜活跃（与其他成员时间错位）
    """
    # 模拟 10 个成员、5 个时间窗口的数据
    np.random.seed(42)
    n_members = 10
    n_windows = 5
    member_ids = [f"user_{i}" for i in range(n_members)]
    
    # 定义成员类型
    member_types = {
        "user_0": "positive",     # alice
        "user_1": "positive",     # bob
        "user_2": "positive",     # charlie
        "user_3": "anxious",      # diana
        "user_4": "anxious",      # eve
        "user_5": "isolated",     # frank
        "user_6": "lurker",       # grace
        "user_7": "critical",     # hector
        "user_8": "night_owl",    # ivy
        "user_9": "night_owl",    # jack
    }
    
    messages = {}
    for mid in member_ids:
        mtype = member_types[mid]
        msgs = []
        for t in range(n_windows):
            base_time = t * 60.0
            
            if mtype == "positive":
                for _ in range(np.random.randint(2, 5)):
                    msgs.append({
                        "content": "这个理论很有意思，我来补充一个例子...",
                        "timestamp": base_time + np.random.uniform(0, 50),
                        "user_id": mid,
                    })
            elif mtype == "anxious":
                for _ in range(np.random.randint(3, 6)):
                    msgs.append({
                        "content": "我是不是漏掉了什么前置知识？这个部分完全听不懂...",
                        "timestamp": base_time + np.random.uniform(0, 50),
                        "user_id": mid,
                    })
            elif mtype == "isolated":
                for _ in range(np.random.randint(1, 2)):
                    msgs.append({
                        "content": "来打游戏啊！新赛季开始了！",
                        "timestamp": base_time + np.random.uniform(5, 45),
                        "user_id": mid,
                    })
            elif mtype == "lurker":
                # 几乎不发言
                if t == 0:
                    msgs.append({
                        "content": "收到",
                        "timestamp": base_time + np.random.uniform(0, 10),
                        "user_id": mid,
                    })
            elif mtype == "critical":
                for _ in range(np.random.randint(2, 4)):
                    msgs.append({
                        "content": "我觉得这个推导有个漏洞，你们有没有考虑边界条件？",
                        "timestamp": base_time + np.random.uniform(10, 40),
                        "user_id": mid,
                    })
            elif mtype == "night_owl":
                for _ in range(np.random.randint(2, 4)):
                    msgs.append({
                        "content": "深夜学习效率真高，没人打扰。",
                        # 时间偏移到深夜（基础时间 + 720 分钟）
                        "timestamp": base_time + 720.0 + np.random.uniform(0, 60),
                        "user_id": mid,
                    })
        messages[mid] = msgs
    
    return messages


def main() -> None:
    console.print(Panel.fit(
        "[bold blue]Tender v2.0 — 群体异质性分析示例[/bold blue]\n"
        "[dim]拥抱异质性：不是所有齿轮都必须咬合[/dim]",
        border_style="blue",
    ))
    
    # =========================================================================
    # 1. 加载配置并运行基础管道
    # =========================================================================
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = load_config(config_path) if os.path.exists(config_path) else {}
    
    # 强制启用异质性分析
    if "heterogeneity" not in config:
        config["heterogeneity"] = {"enabled": True}
    else:
        config["heterogeneity"]["enabled"] = True
    
    pipeline = TenderPipeline(config)
    
    # 生成模拟数据
    messages = generate_mock_data()
    console.print("\n[bold green]✅ 数据准备完成[/bold green]")
    console.print(f"   成员数: {len(messages)}")
    console.print(f"   消息总数: {sum(len(v) for v in messages.values())}")
    
    # 运行管道
    console.print("\n[yellow]🔄 运行基础分析管道...[/yellow]")
    result = pipeline.analyze_window(messages)
    console.print("[bold green]✅ 管道分析完成[/bold green]")
    
    # =========================================================================
    # 2. 初始化异质性分析器
    # =========================================================================
    hetero_config = config.get("heterogeneity", {})
    
    disconnect_analyzer = TopologicalDisconnectAnalyzer(hetero_config)
    loop_detector = LoopDetector(hetero_config)
    frag_analyzer = CausalFragmentationAnalyzer(hetero_config)
    power_analyzer = PowerCentralityAnalyzer(hetero_config)
    async_analyzer = TemporalAsynchronyAnalyzer(hetero_config)
    ling_analyzer = LinguisticDivergenceAnalyzer(hetero_config)
    gini_analyzer = ParticipationGiniAnalyzer(hetero_config)
    isolate_analyzer = IsolateAnalyzer(hetero_config)
    
    # =========================================================================
    # 3. 逐个维度分析
    # =========================================================================
    console.print("\n[bold]=" * 50)
    console.print("[bold underline cyan]📊 异质性分析报告[/bold underline cyan]")
    console.print("[bold]=" * 50)
    
    # --- 3.1 拓扑脱离分析 ---
    console.print("\n[bold]1️⃣  拓扑脱离分析[/bold]")
    console.print("-" * 40)
    
    disconnect_scores = {}
    outlier_threshold = hetero_config.get("topology_disconnect", {}).get("outlier_threshold", 0.7)
    
    for member_id in messages.keys():
        score = disconnect_analyzer.compute_disconnect(
            member_id=member_id,
            topology_result=result.topology_result,
            causal_result=result.causal_result,
            cognition_states=result.cognition_states,
        )
        disconnect_scores[member_id] = score
    
    # 按脱离度排序
    sorted_scores = sorted(
        disconnect_scores.items(),
        key=lambda x: x[1].total,
        reverse=True,
    )
    
    # 用 rich 表格展示
    table = Table(title="脱离度排名")
    table.add_column("成员", style="cyan")
    table.add_column("总脱离度", justify="right")
    table.add_column("空间脱离", justify="right")
    table.add_column("因果脱离", justify="right")
    table.add_column("认知脱离", justify="right")
    table.add_column("状态", style="bold")
    
    for member_id, score in sorted_scores:
        status = "🔴 高度脱离" if score.total > 0.8 else \
                 "🟠 中度脱离" if score.total > 0.6 else \
                 "🟡 轻度脱离" if score.total > 0.4 else \
                 "🟢 正常融入"
        table.add_row(
            member_id,
            f"{score.total:.3f}",
            f"{score.space_disconnect:.3f}",
            f"{score.causal_disconnect:.3f}",
            f"{score.cognition_disconnect:.3f}",
            status,
        )
    console.print(table)
    
    # --- 3.2 环状矛盾检测 ---
    console.print("\n[bold]2️⃣  环状矛盾检测[/bold]")
    console.print("-" * 40)
    
    loops = loop_detector.detect_loops(result.topology_result)
    if loops:
        console.print(f"[bold red]⚠️  检测到 {len(loops)} 个环状结构:[/bold red]")
        for i, loop in enumerate(loops):
            # 将环成员 ID 转为有意义的标签
            member_labels = ", ".join(loop.member_ids[:5])
            if len(loop.member_ids) > 5:
                member_labels += f"... (+{len(loop.member_ids) - 5})"
            console.print(f"   环 {i+1}: 持久性={loop.persistence:.3f}, 成员=[{member_labels}]")
    else:
        console.print("[green]未检测到明显的环状矛盾结构。[/green]")
    
    # --- 3.3 因果网络碎片化分析 ---
    console.print("\n[bold]3️⃣  因果网络碎片化分析[/bold]")
    console.print("-" * 40)
    
    fragment_metrics = frag_analyzer.compute_fragmentation(result.causal_result)
    console.print(f"   因果网络碎片化指数: [bold]{fragment_metrics.fragmentation_index:.3f}[/bold] ")
    console.print(f"   强连通组件数: {fragment_metrics.n_components}")
    console.print(f"   最大组件占比: {fragment_metrics.largest_component_ratio:.2%}")
    
    # --- 3.4 影响力集中度分析 ---
    console.print("\n[bold]4️⃣  影响力集中度分析[/bold]")
    console.print("-" * 40)
    
    centrality_metrics = power_analyzer.compute_centrality(
        result.causal_result,
        result.topology_result
    )
    console.print(f"   影响力基尼系数: [bold]{centrality_metrics.gini_coefficient:.3f}[/bold] ")
    console.print(f"   前 20% 成员控制的影响力: {centrality_metrics.top20_percent_influence:.2%}")
    if centrality_metrics.high_influence_members:
        console.print(f"   高影响力成员: {', '.join(centrality_metrics.high_influence_members[:3])}")
    
    # --- 3.5 时间异步分析 ---
    console.print("\n[bold]5️⃣  时间异步分析[/bold]")
    console.print("-" * 40)
    
    asynchrony_scores = async_analyzer.compute_asynchrony(messages)
    console.print(f"   群体时间异步度: [bold]{asynchrony_scores.overall_asynchrony:.3f}[/bold] ")
    console.print(f"   活跃时间标准差: {asynchrony_scores.active_time_std:.2f} 小时")
    if asynchrony_scores.out_of_sync_members:
        console.print(f"   异步成员: {', '.join(asynchrony_scores.out_of_sync_members)}")
    
    # --- 3.6 语言离散度分析 ---
    console.print("\n[bold]6️⃣  语言离散度分析[/bold]")
    console.print("-" * 40)
    
    divergence_scores = ling_analyzer.compute_divergence(messages)
    console.print(f"   语言离散度: [bold]{divergence_scores.overall_divergence:.3f}[/bold] ")
    console.print(f"   词汇多样性指数: {divergence_scores.vocabulary_diversity:.3f}")
    if divergence_scores.divergent_members:
        console.print(f"   语言离散成员: {', '.join(divergence_scores.divergent_members[:3])}")
    
    # --- 3.7 参与度基尼系数 ---
    console.print("\n[bold]7️⃣  参与度不均分析[/bold]")
    console.print("-" * 40)
    
    message_counts = {mid: len(msgs) for mid, msgs in messages.items()}
    gini = gini_analyzer.compute_gini(message_counts)
    console.print(f"   参与度基尼系数: [bold]{gini:.3f}[/bold] ")
    if gini > 0.6:
        console.print("   ⚠️  参与度严重不均，少数成员主导讨论")
    elif gini > 0.4:
        console.print("   🟡 参与度存在一定不均衡")
    else:
        console.print("   ✅ 参与度分布较为均衡")
    
    # =========================================================================
    # 4. 离群者类型分类
    # =========================================================================
    console.print("\n[bold]=" * 50)
    console.print("[bold underline cyan]🏷️  离群者类型分类[/bold underline cyan]")
    console.print("[bold]=" * 50)
    
    type_table = Table(title="离群者类型")
    type_table.add_column("成员", style="cyan")
    type_table.add_column("脱离度", justify="right")
    type_table.add_column("分类结果", style="bold")
    type_table.add_column("干预建议")
    
    for member_id, score in sorted_scores:
        if score.total < outlier_threshold:
            continue
        
        isolate_type = isolate_analyzer.classify(
            disconnect_score=score,
            personal_profile=result.personal_profiles.get(member_id),
            group_profile=result.topology_result,
        )
        
        suggestions = {
            "VOLUNTARY_ISOLATE": "赋予独立空间，尊重选择",
            "INVOLUNTARY_OUTCAST": "软性牵线搭桥，提供安全连接",
            "OPINION_LEADER_DEVIANT": "保留视角，鼓励建设性冲突",
            "PURE_DIVERSE": "拥抱多样性，勿强制改变",
        }
        suggestion = suggestions.get(isolate_type.value, "观察")
        
        type_table.add_row(
            member_id,
            f"{score.total:.3f}",
            f"[bold]{isolate_type.value}[/bold]",
            suggestion,
        )
    
    console.print(type_table)
    
    # =========================================================================
    # 5. 生成综合异质性报告
    # =========================================================================
    console.print("\n[bold]=" * 50)
    console.print("[bold underline cyan]📋 综合异质性报告[/bold underline cyan]")
    console.print("[bold]=" * 50)
    
    hetero_metrics = HeterogeneityMetrics(
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
            mid: isolate_analyzer.classify(
                disconnect_scores[mid],
                result.personal_profiles.get(mid),
                result.topology_result,
            )
            for mid in messages.keys()
            if disconnect_scores[mid].total > outlier_threshold
        },
    )
    
    # 输出综合评分
    console.print("")
    summary_table = Table(title="群体异质性画像")
    summary_table.add_column("维度", style="cyan")
    summary_table.add_column("得分", justify="right")
    summary_table.add_column("解释")
    
    summary_table.add_row(
        "拓扑丰富度",
        f"{hetero_metrics.topological_richness:.3f}",
        "结构多样性：越高表示群体结构越丰富"
    )
    summary_table.add_row(
        "环状矛盾强度",
        f"{hetero_metrics.loop_strength:.3f}",
        "矛盾持久性：越高表示存在难以调和的矛盾"
    )
    summary_table.add_row(
        "因果碎片化",
        f"{hetero_metrics.causal_fragmentation:.3f}",
        "网络分裂度：越高表示群体越分裂"
    )
    summary_table.add_row(
        "时间异步度",
        f"{hetero_metrics.temporal_asynchrony:.3f}",
        "活动时间错位：越高表示成员活跃时间越分散"
    )
    summary_table.add_row(
        "语言离散度",
        f"{hetero_metrics.linguistic_divergence:.3f}",
        "语言风格差异：越高表示语言越多元"
    )
    summary_table.add_row(
        "参与度基尼",
        f"{hetero_metrics.participation_gini:.3f}",
        "参与不均衡度：越高表示发言越集中在少数人"
    )
    console.print(summary_table)
    
    # =========================================================================
    # 6. 保存结果
    # =========================================================================
    output_dir = Path("output/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "heterogeneity_analysis.json"
    
    export_data = {
        "heterogeneity_metrics": {
            "topological_richness": hetero_metrics.topological_richness,
            "loop_strength": hetero_metrics.loop_strength,
            "causal_fragmentation": hetero_metrics.causal_fragmentation,
            "component_separation": hetero_metrics.component_separation,
            "temporal_asynchrony": hetero_metrics.temporal_asynchrony,
            "linguistic_divergence": hetero_metrics.linguistic_divergence,
            "participation_gini": hetero_metrics.participation_gini,
        },
        "disconnect_scores": {
            mid: {
                "total": score.total,
                "space": score.space_disconnect,
                "causal": score.causal_disconnect,
                "cognition": score.cognition_disconnect,
            }
            for mid, score in sorted_scores
        },
        "outlier_types": {
            mid: str(otype.value) if hasattr(otype, 'value') else str(otype)
            for mid, otype in hetero_metrics.outlier_types.items()
        },
        "loops": [
            {
                "persistence": loop.persistence,
                "member_count": len(loop.member_ids),
                "member_ids": loop.member_ids,
            }
            for loop in loops
        ],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[bold green]💾 异质性分析报告已保存到 {output_path}[/bold green]")
    console.print("\n[bold cyan]✨ 分析完成！[/bold cyan] 拥抱异质性，尊重每一个独立的 Being。")


if __name__ == "__main__":
    main()
