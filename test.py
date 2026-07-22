import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from itertools import combinations
import math

# ======================== 参数设置 ========================
NUM_BEAMS = 4
NUM_SPOTS = 12
SLOT_TIME = 0.01          # 10ms
MAX_DELAY = 400           # ms
PACKET_SIZE = 10          # kbit
BASE_CAPACITY = 10        # Mbps (每个波位基础容量)

# 预生成所有波位组合 C(12,4)=495
ALL_ACTIONS = list(combinations(range(NUM_SPOTS), NUM_BEAMS))
ACTION_DIM = len(ALL_ACTIONS)

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)

# 空间业务分布（离散系数约0.5）
base_arrival_rates = np.random.uniform(5, 20, NUM_SPOTS)

# 时间变化因子（9:00-14:00 波峰在11:00）
def time_factor(t):
    hour = 9 + t * SLOT_TIME / 3600
    if 9 <= hour <= 11:
        return 1.0 + 0.5 * (hour - 9) / 2
    elif 11 < hour <= 14:
        return 1.0 + 0.5 * (14 - hour) / 3
    else:
        return 1.0

# 波位间距离矩阵（用于干扰简化）
spot_distances = np.random.rand(NUM_SPOTS, NUM_SPOTS)
spot_distances = (spot_distances + spot_distances.T) / 2
np.fill_diagonal(spot_distances, 0)

# ======================== 环境类 ========================
class BeamHoppingEnv:
    def __init__(self, train_mode=True):
        self.train_mode = train_mode
        self.reset()

    def reset(self):
        self.real_queue = np.zeros(NUM_SPOTS, dtype=int)
        self.nonreal_queue = np.zeros(NUM_SPOTS, dtype=int)
        self.served_real = np.zeros(NUM_SPOTS, dtype=int)
        self.served_nonreal = np.zeros(NUM_SPOTS, dtype=int)
        self.total_real_arrived = np.zeros(NUM_SPOTS, dtype=int)
        self.total_nonreal_arrived = np.zeros(NUM_SPOTS, dtype=int)
        self.slot_index = 0
        self.history = []
        return self._get_state()

    def _get_state(self):
        max_q = 200  # 归一化上限
        real_norm = np.clip(self.real_queue / max_q, 0, 1)
        nonreal_norm = np.clip(self.nonreal_queue / max_q, 0, 1)
        satisfaction = np.zeros(NUM_SPOTS)
        for i in range(NUM_SPOTS):
            total = self.total_real_arrived[i] + self.total_nonreal_arrived[i]
            if total > 0:
                satisfaction[i] = (self.served_real[i] + self.served_nonreal[i]) / total
            else:
                satisfaction[i] = 0.5
        state = np.concatenate([real_norm, nonreal_norm, satisfaction])
        return state.astype(np.float32)

    def step(self, action_vec):
        self.slot_index += 1
        # 1. 业务到达（实时与非实时各半）
        tf = time_factor(self.slot_index)
        real_arrivals = np.random.poisson(base_arrival_rates * tf * 0.5)
        nonreal_arrivals = np.random.poisson(base_arrival_rates * tf * 0.5)
        self.real_queue += real_arrivals
        self.nonreal_queue += nonreal_arrivals
        self.total_real_arrived += real_arrivals
        self.total_nonreal_arrived += nonreal_arrivals

        # 2. 容量计算（考虑同频干扰）
        capacities = np.ones(NUM_SPOTS) * BASE_CAPACITY  # Mbps
        active = np.where(action_vec == 1)[0]
        for i in active:
            for j in active:
                if i != j and spot_distances[i, j] < 0.3:
                    capacities[i] *= 0.7  # 干扰衰减

        # 3. 服务数据包（先实时后非实时）
        served_real_count = np.zeros(NUM_SPOTS, dtype=int)
        served_nonreal_count = np.zeros(NUM_SPOTS, dtype=int)
        for i in active:
            # 修正：容量(Mbps) -> kbit/s -> kbit/时隙 -> 包数
            max_serve = int(capacities[i] * 1000 * SLOT_TIME / PACKET_SIZE)
            if max_serve <= 0:
                max_serve = 1  # 确保至少服务1个包（若容量极小）
            # 服务实时
            serve = min(self.real_queue[i], max_serve)
            self.real_queue[i] -= serve
            served_real_count[i] = serve
            self.served_real[i] += serve
            # 剩余容量服务非实时
            remain = max_serve - serve
            if remain > 0 and self.nonreal_queue[i] > 0:
                serve2 = min(self.nonreal_queue[i], remain)
                self.nonreal_queue[i] -= serve2
                served_nonreal_count[i] = serve2
                self.served_nonreal[i] += serve2

        # 4. 超时丢弃（时延 > MAX_DELAY）
        # 这里简化：丢弃队列中部分包（按比例），但保留一些以维持学习信号
        drop_ratio = 0.01  # 1% 丢弃率
        drop_real = int(drop_ratio * self.real_queue.sum())
        drop_nonreal = int(drop_ratio * self.nonreal_queue.sum())
        self.real_queue = np.maximum(self.real_queue - drop_real, 0)
        self.nonreal_queue = np.maximum(self.nonreal_queue - drop_nonreal, 0)

        # 5. 计算指标与奖励
        avg_delay = self.real_queue.mean() * SLOT_TIME * 1000  # ms
        r1 = -avg_delay
        r2 = served_nonreal_count.sum()  # 本时隙非实时吞吐量（包数）
        # 满意度
        satisfaction = np.zeros(NUM_SPOTS)
        for i in range(NUM_SPOTS):
            total = self.total_real_arrived[i] + self.total_nonreal_arrived[i]
            if total > 0:
                satisfaction[i] = (self.served_real[i] + self.served_nonreal[i]) / total
            else:
                satisfaction[i] = 0.5
        r3 = satisfaction.sum()
        # 奖励加权（调高吞吐量和满意度的权重）
        reward = 0.2 * r1 + 0.4 * r2 + 0.4 * r3

        self.history.append({
            'delay': avg_delay,
            'throughput': served_nonreal_count.sum(),
            'satisfaction': satisfaction.mean(),
            'reward': reward
        })

        next_state = self._get_state()
        done = False
        return next_state, reward, done, {}

    def get_period_stats(self):
        if not self.history:
            return 0, 0, 0
        delays = [h['delay'] for h in self.history]
        throughputs = [h['throughput'] for h in self.history]
        satisfactions = [h['satisfaction'] for h in self.history]
        return np.mean(delays), np.mean(throughputs), np.mean(satisfactions)

# ======================== DQN网络 ========================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ======================== 多智能体（MoE） ========================
class MultiObjectiveDQN:
    def __init__(self, state_dim, lr=1e-5, gamma=0.9, epsilon=0.5, epsilon_min=0.01):
        self.action_dim = ACTION_DIM
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.999

        self.q_net1 = DQN(state_dim, ACTION_DIM)
        self.q_net2 = DQN(state_dim, ACTION_DIM)
        self.q_net3 = DQN(state_dim, ACTION_DIM)
        self.target_net1 = DQN(state_dim, ACTION_DIM)
        self.target_net2 = DQN(state_dim, ACTION_DIM)
        self.target_net3 = DQN(state_dim, ACTION_DIM)
        self.update_target(tau=1.0)

        self.optimizer1 = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(self.q_net2.parameters(), lr=lr)
        self.optimizer3 = optim.Adam(self.q_net3.parameters(), lr=lr)

        self.memory1 = deque(maxlen=3000)
        self.memory2 = deque(maxlen=3000)
        self.memory3 = deque(maxlen=3000)

    def update_target(self, tau=1.0):
        if tau == 1.0:
            self.target_net1.load_state_dict(self.q_net1.state_dict())
            self.target_net2.load_state_dict(self.q_net2.state_dict())
            self.target_net3.load_state_dict(self.q_net3.state_dict())

    def act(self, state, eval_mode=False):
        if not eval_mode and np.random.random() < self.epsilon:
            idx = np.random.randint(ACTION_DIM)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q1 = self.q_net1(state_t).squeeze(0)
                q2 = self.q_net2(state_t).squeeze(0)
                q3 = self.q_net3(state_t).squeeze(0)
                # L2归一化
                q1 = q1 / (torch.norm(q1, p=2) + 1e-8)
                q2 = q2 / (torch.norm(q2, p=2) + 1e-8)
                q3 = q3 / (torch.norm(q3, p=2) + 1e-8)
                q_total = q1 + q2 + q3
            idx = torch.argmax(q_total).item()
        action_vec = np.zeros(NUM_SPOTS)
        action_vec[list(ALL_ACTIONS[idx])] = 1
        return action_vec, idx

    def remember(self, state, action_idx, reward, next_state, done, target_idx):
        if target_idx == 1:
            self.memory1.append((state, action_idx, reward, next_state, done))
        elif target_idx == 2:
            self.memory2.append((state, action_idx, reward, next_state, done))
        else:
            self.memory3.append((state, action_idx, reward, next_state, done))

    def replay(self, batch_size, target_idx):
        if target_idx == 1:
            memory = self.memory1
            q_net = self.q_net1
            target_net = self.target_net1
            optimizer = self.optimizer1
        elif target_idx == 2:
            memory = self.memory2
            q_net = self.q_net2
            target_net = self.target_net2
            optimizer = self.optimizer2
        else:
            memory = self.memory3
            q_net = self.q_net3
            target_net = self.target_net3
            optimizer = self.optimizer3

        if len(memory) < batch_size:
            return
        batch = random.sample(memory, batch_size)
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions = torch.LongTensor(np.array([b[1] for b in batch]))
        rewards = torch.FloatTensor(np.array([b[2] for b in batch]))
        next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
        dones = torch.BoolTensor(np.array([b[4] for b in batch]))

        current_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = target_net(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q * (~dones)
        loss = nn.MSELoss()(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ======================== 训练主程序 ========================
state_dim = NUM_SPOTS * 3
agent = MultiObjectiveDQN(state_dim)
env = BeamHoppingEnv(train_mode=True)

LOOPS = 450
TIME_SLOTS = 200
BATCH_SIZE = 8
TARGET_UPDATE = 10

all_episode_stats = []

for episode in range(LOOPS):
    state = env.reset()
    for t in range(TIME_SLOTS):
        action_vec, action_idx = agent.act(state)
        next_state, reward, done, _ = env.step(action_vec)
        agent.remember(state, action_idx, reward, next_state, done, 1)
        agent.remember(state, action_idx, reward, next_state, done, 2)
        agent.remember(state, action_idx, reward, next_state, done, 3)

        agent.replay(BATCH_SIZE, 1)
        agent.replay(BATCH_SIZE, 2)
        agent.replay(BATCH_SIZE, 3)

        state = next_state
        if t % TARGET_UPDATE == 0:
            agent.update_target(tau=1.0)

    avg_delay, avg_throughput, avg_satisfaction = env.get_period_stats()
    all_episode_stats.append((avg_delay, avg_throughput, avg_satisfaction))
    agent.decay_epsilon()
    print(f"Episode {episode+1}/{LOOPS}, Delay={avg_delay:.2f}ms, Throughput={avg_throughput:.2f}, Satisfaction={avg_satisfaction:.4f}")

# ======================== 输出训练周期数据 ========================
print("\n=== 训练周期数据（每个周期平均时延、吞吐量、满意度） ===")
for i, (d, t, s) in enumerate(all_episode_stats):
    print(f"周期{i+1}: ({d:.3f}, {t:.3f}, {s:.3f})")

# ======================== 时变环境评估 ========================
print("\n=== 时变环境性能对比（按时间顺序的时隙数据） ===")
eval_env = BeamHoppingEnv(train_mode=False)
state = eval_env.reset()
time_varying_data = []
for t in range(200):
    action_vec, _ = agent.act(state, eval_mode=True)
    next_state, reward, done, _ = eval_env.step(action_vec)
    delay = eval_env.real_queue.mean() * SLOT_TIME * 1000
    throughput = eval_env.history[-1]['throughput'] if eval_env.history else 0
    satisfaction = eval_env.history[-1]['satisfaction'] if eval_env.history else 0
    time_varying_data.append((delay, throughput, satisfaction))
    state = next_state

for i, (d, t, s) in enumerate(time_varying_data[:20]):
    print(f"时隙{i+1}: ({d:.3f}, {t:.3f}, {s:.3f})")
print("... (完整数据在 time_varying_data 中)")