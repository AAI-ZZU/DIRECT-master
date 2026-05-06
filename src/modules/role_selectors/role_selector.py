import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th

from components.epsilon_schedules import DecayThenFlatSchedule


class Role_Selector(nn.Module):
    def __init__(self, args):
        super(Role_Selector, self).__init__()
        # args.role_dim=input_shape
        self.args = args
        # self.device = th.device('cuda:1')
        self.device = args.device
        self.n_agents = args.n_agents

        self.selector_net = nn.Sequential(nn.Linear(self.args.intention_hidden_dim, 128), nn.ReLU(), nn.Linear(128, self.args.role_num)).to(self.device)
        # self.selector_net = nn.Sequential(
        #   #  输入层
            # nn.Linear(self.args.intention_hidden_dim, 128),
            # nn.LayerNorm(128),
            # nn.ReLU(),
            # ResidualBlock(128, 128 * 2),  # 128*2是残差块内部的扩展维度;一个或多个残差块，以增加深度和稳定性
            # nn.Linear(128, self.args.role_num)  # 输出层
        # ).to(self.device)

        self.epsilon_selector = EpsilonGreedyRoleSelector(args)

    def forward(self, inputs, role_repr, test_mode, t):
        role_logits = self.selector_net(inputs)

        if not test_mode:
            role_prob = F.gumbel_softmax(role_logits, hard=True, dim=-1)
            role_id = th.argmax(role_prob, dim=-1)
        else:
            role_id = self.epsilon_selector.select_role(role_logits, t, test_mode)

        select_role_repr = role_repr[role_id]
        return role_id, select_role_repr


class Role_Encode(nn.Module):
    def __init__(self, args):
        super(Role_Encode, self).__init__()
        self.args = args
        # self.device = th.device('cuda:1')
        self.device = args.device
        self.n_agents = args.n_agents

        self.alpha = 0.99
        initial_encodings = self._create_orthogonal_init(self.args.role_num, self.args.intention_hidden_dim).to(self.device)
        # self.register_buffer('role_encodings', initial_encodings)
        self.role_encodings = nn.Parameter(initial_encodings)

    def _create_orthogonal_init(self, num_roles: int, role_dim: int) -> th.Tensor:
        """创建一个随机正交的初始角色编码矩阵"""
        # if role_dim < num_roles:
        #     raise ValueError("Role dimension must be >= number of roles for orthogonal init.")
        random_matrix = th.randn(role_dim, num_roles)
        q, _ = th.linalg.qr(random_matrix)
        return q.T  # (num_roles, role_dim)

    def get_all_encodings(self):
        return self.role_encodings

    def update_roles(self, beliefs, role_ids):
        with th.no_grad():
            # 创建一个用于存储本次更新后角色编码的副本
            new_role_encodings = self.role_encodings.clone()

            # --- 步骤1: 并行聚合【被选择】角色的信念 ---

            # 为了高效聚合，需要两个张量：
            # 1. 一个用于累加信念的张量 (sum_beliefs)
            # 2. 一个用于统计每个角色被选了多少次的张量 (counts)
            # 形状都为 (num_roles, dim)
            sum_beliefs = th.zeros_like(self.role_encodings, device=self.device)
            counts = th.zeros(self.args.role_num, 1, device=self.device)

            # 将role_ids和beliefs的batch维度展平，以便使用scatter_add_
            flat_role_ids = role_ids.view(-1)
            flat_beliefs = beliefs.view(-1, self.args.intention_hidden_dim)

            # scatter_add_ 是一个高效的并行聚合操作
            # 它会根据flat_role_ids中的索引，将flat_beliefs中的值累加到sum_beliefs的相应行
            sum_beliefs.scatter_add_(0, flat_role_ids.unsqueeze(1).expand(-1, self.args.intention_hidden_dim), flat_beliefs)

            # 同样的方法，统计每个角色的选择次数
            ones = th.ones_like(flat_role_ids, dtype=th.float32).unsqueeze(1)
            counts.scatter_add_(0, flat_role_ids.unsqueeze(1), ones)

            # 找出那些被选择过的角色 (counts > 0)
            selected_mask = (counts > 0).squeeze()

            # --- 步骤2: 更新【被选择】的角色编码 ---

            if selected_mask.any():
                # 计算平均信念 (只对被选择的角色)
                avg_beliefs = sum_beliefs[selected_mask] / counts[selected_mask]

                # 使用EMA平滑更新
                new_role_encodings[selected_mask] = self.alpha * self.role_encodings[selected_mask] + \
                                                    (1 - self.alpha) * avg_beliefs

            # --- 步骤3: 更新【未被选择】的角色编码 ---

            unselected_mask = ~selected_mask

            if unselected_mask.any() and selected_mask.any():  # 必须同时存在被选和未被选的角色
                # 计算当前所有角色编码的平均值
                average_role_vector = th.mean(self.role_encodings, dim=0)

                # 让未被选择的角色，缓慢地向“平均水平”靠拢
                new_role_encodings[unselected_mask] = self.alpha * self.role_encodings[unselected_mask] + \
                                                      (1 - self.alpha) * average_role_vector

            # --- 步骤4: 将本次更新应用到类的状态中 ---
            self.role_encodings.data = new_role_encodings.data


class RoleToActionGate(nn.Module):
    def __init__(self, args):
        super(RoleToActionGate, self).__init__()
        self.args = args
        # self.device = th.device('cuda:1')
        self.device = args.device
        self.n_agents = args.n_agents

        # self.affinity_net = nn.Sequential(nn.Linear(self.args.intention_hidden_dim*2, self.args.role_hidden_dim),
        #                                   nn.ReLU(),
        #                                   nn.Linear(self.args.role_hidden_dim, self.args.role_hidden_dim),
        #                                   nn.ReLU(),
        #                                   nn.Linear(self.args.role_hidden_dim, 1))
        self.affinity_net = nn.Sequential(nn.LayerNorm(self.args.intention_hidden_dim * 2),
                                          nn.Linear(self.args.intention_hidden_dim * 2, self.args.role_hidden_dim),
                                          nn.LeakyReLU(0.1),
                                          nn.Linear(self.args.role_hidden_dim, 1))

        self.affinity_net[-1].bias.data.fill_(0.1)

    def forward(self, role_embeddings, action_embeddings):
        roles_expand = role_embeddings.unsqueeze(1).expand(-1, action_embeddings.shape[0], -1)
        actions_expand = (action_embeddings.unsqueeze(0).expand(self.args.role_num, -1, -1))

        combined = th.cat([roles_expand, actions_expand], dim=-1)
        affinity_logits = self.affinity_net(combined).squeeze(-1)

        # 将连续分数转换为“软”概率 (0到1之间)
        # Softmax用于“多选一”，而Sigmoid用于对每个选项进行独立的“是/否”判断，更适合这里。
        # action_probs = th.sigmoid(affinity_logits)

        # “硬化”决策，得到0/1的离散结果
        # 这是前向传播时实际使用的、符合您要求的0/1向量
        # hard_action_mask = (action_probs > 0.5).float()

        # 直通估计器 (STE): 连接前向和反向传播的桥梁
        # 在反向传播时，梯度会“跳过”硬化操作，直接作用在“软”概率上
        # 这是STE的标准实现: (硬结果 - 软结果).detach() + 软结果
        # .detach()会阻断梯度，所以梯度只会通过右侧的action_probs回传
        # ste_action_mask = (hard_action_mask - action_probs).detach() + action_probs

        return affinity_logits

class EpsilonGreedyRoleSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_role(self, agent_inputs, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)  # eval 方法用于根据当前的训练步数或时间步长来计算当前的学习率

        if test_mode:
            # Greedy action selection only
            self.epsilon = getattr(self.args, "test_noise", 0.0)

        # mask actions that are excluded from selection

        _, max_indices = agent_inputs.max(dim=2)
        # 创建一个与 max_indices 形状相同的随机张量，以 epsilon 的概率设置为 1
        random_mask = (th.rand(max_indices.shape) < self.epsilon).to(max_indices.device)
        # 如果随机数小于 epsilon，则随机选择一个战术；否则选择Q值最大的战术
        random_roles = th.randint(0, agent_inputs.size(2), max_indices.shape, device=max_indices.device)
        chosen_roles = th.where(random_mask, random_roles, max_indices)

        return chosen_roles


class ResidualBlock(nn.Module):
    """
    一个包含层归一化、Dropout和残差连接的健壮网络块。
    采用 Pre-LN 结构，稳定性更好。
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim) # 输出维度需与输入一致以进行残差连接
        self.ln = nn.LayerNorm(input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Pre-LN 结构: 先归一化，再进行计算
        residual = x
        x = self.ln(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x + residual # 残差连接
