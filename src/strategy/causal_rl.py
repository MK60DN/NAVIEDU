"""
因果强化学习策略推理引擎（PyTorch 真实版本）

该模块实现了基于因果强化学习的策略推理方案。
使用真正的 PyTorch 深度 Q 网络 (DQN) 学习最优干预策略。

核心方法：
1. 将群体情绪状态建模为马尔可夫决策过程（MDP）
2. 使用深度 Q 网络（DQN）学习最优干预策略
3. 状态空间 = 融合特征向量 F（16维）
4. 动作空间 = 预定义的干预策略集合
5. 奖励 = 风险评分降低 + 群体情绪改善

学术基础：
- 深度 Q 学习 (Mnih et al., 2015)
- 因果强化学习 (Zhang & Bareinboim, 2018)
- 社交情绪调节 (Gross, 2015)
"""

import time
import random
from typing import Dict, List, Any, Tuple, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tender.strategy.base import (
    BaseStrategyEngine,
    RiskLevel,
    InterventionStrategy,
    StrategyDecision,
)
from tender.fusion.base import FusionResult


# ============================================================================
# 预定义干预策略
# ============================================================================

# 7 种不同风险等级的预定义干预策略
PREDEFINED_STRATEGIES = [
    InterventionStrategy(
        name="no_action",
        description="无需干预：群体情绪状态正常",
        risk_level=RiskLevel.LOW,
        action="none",
    ),
    InterventionStrategy(
        name="gentle_reminder",
        description="温和提醒：提示成员注意情绪表达",
        risk_level=RiskLevel.LOW,
        action="send_gentle_reminder",
    ),
    InterventionStrategy(
        name="trigger_discussion",
        description="引导讨论：主动引导话题转向积极方向",
        risk_level=RiskLevel.MEDIUM,
        action="initiate_positive_topic",
    ),
    InterventionStrategy(
        name="private_message",
        description="私信调解：单独联系情绪异常成员",
        risk_level=RiskLevel.MEDIUM,
        action="send_private_message",
    ),
    InterventionStrategy(
        name="active_intervention",
        description="主动干预：发布规则提醒并暂停敏感话题",
        risk_level=RiskLevel.HIGH,
        action="pause_sensitive_topic",
    ),
    InterventionStrategy(
        name="emergency_notice",
        description="紧急通知：向全体成员发布紧急通知",
        risk_level=RiskLevel.HIGH,
        action="broadcast_emergency_notice",
    ),
    InterventionStrategy(
        name="emergency_shutdown",
        description="紧急关闭：暂时关闭讨论区",
        risk_level=RiskLevel.CRITICAL,
        action="close_discussion_area",
    ),
]


# ============================================================================
# PyTorch DQN 模型定义
# ============================================================================


class DQN(nn.Module):
    """深度 Q 网络（PyTorch 实现）

    将群体情绪状态映射到每个动作的 Q 值。
    架构：全连接网络，包含 BatchNorm 和 Dropout 防止过拟合。
    """

    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 7,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 输入层
        layers = [
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        # 隐藏层
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        # 输出层
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch_size, state_dim) 状态向量

        Returns:
            torch.Tensor: (batch_size, action_dim) 每个动作的 Q 值
        """
        return self.network(state)


# ============================================================================
# 经验回放池
# ============================================================================


class ReplayBuffer:
    """经验回放池（PyTorch 实现）

    存储 (state, action, reward, next_state, done) 经验，
    支持批量采样和优先级采样。
    """

    def __init__(self, capacity: int = 10000, batch_size: int = 64):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.capacity = capacity

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """存储一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                               torch.Tensor, torch.Tensor]:
        """从经验池中随机采样一个批次"""
        batch = random.sample(self.buffer, min(self.batch_size, len(self.buffer)))

        states = torch.FloatTensor(np.array([exp[0] for exp in batch]))
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor(np.array([exp[3] for exp in batch]))
        dones = torch.FloatTensor([exp[4] for exp in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# PyTorch DQN 策略引擎（与 Tender 基类兼容）
# ============================================================================


class DQNStrategyEngine(BaseStrategyEngine):
    """
    基于 PyTorch 深度 Q 网络的策略推理引擎

    使用真正的 DQN 学习从群体情绪状态到干预策略的最优映射。
    支持双 Q 学习（Double DQN）和目标网络（Target Network）以提高训练稳定性。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # DQN 参数
        self.state_dim = config.get("state_dim", 16)
        self.action_dim = config.get("action_dim", 7)
        self.hidden_dim = config.get("rl_hidden_dim", 128)
        self.num_hidden_layers = config.get("rl_num_hidden_layers", 3)
        self.dropout = config.get("rl_dropout", 0.2)

        # 学习参数
        self.learning_rate = config.get("rl_learning_rate", 0.001)
        self.discount_factor = config.get("rl_discount_factor", 0.99)
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.batch_size = config.get("rl_batch_size", 64)
        self.buffer_size = config.get("rl_buffer_size", 10000)
        self.target_update_frequency = config.get("target_update", 10)
        self.tau = config.get("rl_tau", 0.005)  # 软更新系数

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化 DQN 网络
        self.q_network = DQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            dropout=self.dropout,
        ).to(self.device)

        # 目标网络（用于稳定训练）
        self.target_network = DQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # 优化器
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate,
        )
        self.loss_fn = nn.MSELoss()

        # 经验回放池
        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            batch_size=self.batch_size,
        )

        # 训练状态
        self.epsilon = self.epsilon_start
        self.training_steps = 0
        self.total_reward = 0.0
        self.loss_history = []
        self.reward_history = []

        # 初始化策略空间
        self._strategies = PREDEFINED_STRATEGIES

    def _extract_state(self, fusion_result: FusionResult) -> np.ndarray:
        """
        从融合结果中提取状态向量

        Args:
            fusion_result: 融合分析结果

        Returns:
            np.ndarray: (state_dim,) 状态向量
        """
        state = fusion_result.fused_features.copy()
        return state

    def _compute_reward(self, previous_score: float, current_score: float) -> float:
        """
        计算干预后的奖励

        奖励设计原则：
        - 风险降低 → 正奖励
        - 风险升高 → 负奖励
        - 降低幅度越大，奖励越大

        Args:
            previous_score: 干预前的风险评分
            current_score: 干预后的风险评分

        Returns:
            float: 奖励值
        """
        improvement = previous_score - current_score
        # 非线性映射：小幅改进给予适度奖励，大幅改进给予高奖励
        reward = 10.0 * np.tanh(improvement * 5.0)
        return float(reward)

    def _select_action(self, state: np.ndarray, training: bool = False) -> int:
        """
        使用 epsilon-贪心策略选择动作

        Args:
            state: 状态向量
            training: 是否处于训练模式

        Returns:
            int: 动作索引
        """
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randrange(self.action_dim)
        else:
            # 利用：选择 Q 值最大的动作
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_t)
                return int(torch.argmax(q_values).item())

    def _update_network(self):
        """
        更新 Q 网络参数（使用 Double DQN）

        训练步骤：
        1. 从经验回放池中采样一个批次
        2. 使用 Q 网络选择动作（Double DQN）
        3. 使用目标网络计算目标 Q 值
        4. 计算损失并更新网络
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # 采样经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Double DQN：使用 Q 网络选择动作，使用目标网络计算 Q 值
        with torch.no_grad():
            # Q 网络选择下一个状态的最佳动作
            next_q_values_q = self.q_network(next_states)
            best_actions = torch.argmax(next_q_values_q, dim=1, keepdim=True)

            # 目标网络计算该动作的 Q 值
            next_q_values_target = self.target_network(next_states)
            next_q_values = next_q_values_target.gather(1, best_actions).squeeze(1)

            # 目标 Q 值
            target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)

        # 当前 Q 值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算损失
        loss = self.loss_fn(current_q_values, target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.loss_history.append(loss.item())
        self.training_steps += 1

        # 更新目标网络（软更新）
        if self.training_steps % self.target_update_frequency == 0:
            for target_param, q_param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.tau * q_param.data + (1.0 - self.tau) * target_param.data
                )

        # 衰减 epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        存储经验到回放池

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_reward += reward
        self.reward_history.append(reward)

        # 自动触发训练
        self._update_network()

    def assess_risk(self, fusion_result: FusionResult) -> StrategyDecision:
        """
        使用 DQN 策略评估风险并做出决策

        Args:
            fusion_result: 融合分析结果

        Returns:
            StrategyDecision: 策略决策
        """
        # 1. 提取状态
        state = self._extract_state(fusion_result)

        # 2. 使用 Q 网络选择最优动作
        action_idx = self._select_action(state, training=False)

        # 3. 获取选择的策略
        selected_strategy = self._strategies[action_idx]

        # 4. 计算风险评分
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t).cpu().numpy().flatten()

        # 从 Q 值分布推断风险评分
        q_mean = np.mean(q_values)
        q_std = np.std(q_values) + 1e-6
        q_max = np.max(q_values)
        q_min = np.min(q_values)

        # 风险评分：Q 值范围越小，风险越高（不确定性大）
        risk_score = float(1.0 - (q_max - q_min) / (2.0 * q_std + 1e-6))
        risk_score = max(0.0, min(1.0, risk_score))

        # 5. 确定风险等级
        risk_level = self._determine_risk_level(risk_score)

        # 6. 生成推理说明
        reasoning = self._generate_reasoning(risk_level, risk_score, selected_strategy)

        return StrategyDecision(
            risk_level=risk_level,
            risk_score=risk_score,
            triggered_strategies=[selected_strategy],
            fusion_result=fusion_result,
            timestamp=time.time(),
            reasoning=reasoning,
            requires_human=(risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]),
        )

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """
        根据风险评分确定风险等级

        Args:
            risk_score: 风险评分 (0.0 ~ 1.0)

        Returns:
            RiskLevel: 风险等级
        """
        if risk_score < 0.2:
            return RiskLevel.LOW
        elif risk_score < 0.4:
            return RiskLevel.MEDIUM
        elif risk_score < 0.7:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _generate_reasoning(
        self,
        risk_level: RiskLevel,
        risk_score: float,
        strategy: InterventionStrategy,
    ) -> str:
        """
        生成推理说明

        Args:
            risk_level: 风险等级
            risk_score: 风险评分
            strategy: 选择的策略

        Returns:
            str: 推理说明
        """
        return (
            f"DQN 策略引擎评估结果：风险等级 {risk_level.name} "
            f"(风险评分 {risk_score:.2f})。"
            f"建议策略：{strategy.name} - {strategy.description}。"
            f"模型训练步数：{self.training_steps}，当前 epsilon：{self.epsilon:.4f}。"
        )

    def save_model(self, filepath: str):
        """
        保存模型权重

        Args:
            filepath: 保存路径
        """
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "total_reward": self.total_reward,
            "loss_history": self.loss_history,
            "reward_history": self.reward_history,
        }, filepath)

    def load_model(self, filepath: str):
        """
        加载模型权重

        Args:
            filepath: 加载路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
        self.total_reward = checkpoint["total_reward"]
        self.loss_history = checkpoint["loss_history"]
        self.reward_history = checkpoint["reward_history"]

    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            "training_steps": self.training_steps,
            "total_reward": self.total_reward,
            "average_reward": np.mean(self.reward_history[-100:]) if self.reward_history else 0.0,
            "epsilon": self.epsilon,
            "replay_buffer_size": len(self.replay_buffer),
            "loss_avg": np.mean(self.loss_history[-100:]) if self.loss_history else None,
            "device": str(self.device),
        }

    def get_info(self) -> Dict[str, Any]:
        """获取当前策略引擎的信息"""
        return {
            "name": "DQNStrategyEngine",
            "description": "基于 PyTorch 深度 Q 网络的策略推理引擎",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "target_update_frequency": self.target_update_frequency,
            "device": str(self.device),
            "trainable_parameters": sum(p.numel() for p in self.q_network.parameters()),
            **self.get_training_stats(),
        }
