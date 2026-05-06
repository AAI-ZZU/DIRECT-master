import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm


class RECL_MIX(nn.Module):
    def __init__(self, args):
        super(RECL_MIX, self).__init__()
        self.args = args
        self.N = args.n_agents
        self.state_shape = args.state_shape
        self.mix_input_dim = args.state_shape + args.n_agents * args.att_out_dim
        self.batch_size = args.batch_size
        self.qmix_hidden_dim = args.qmix_hidden_dim
        self.hyper_hidden_dim = args.hypernet_embed
        self.hyper_layers_num = args.hyper_layers_num
        self.role_gru_hidden = None

        self.state_fc = nn.Linear(args.state_shape, args.state_shape)
        self.state_gru = nn.GRUCell(args.state_shape, args.n_agents * args.state_embed_dim)
        self.state_gru_hidden = None
        self.dim_q = args.state_embed_dim
        self.attention_net = MultiHeadAttention(args.n_heads, args.att_dim, args.att_out_dim, args.soft_temperature,
                                                self.dim_q, self.args.intention_hidden_dim, self.args.intention_hidden_dim)

        """
        w1:(N, qmix_hidden_dim)
        b1:(1, qmix_hidden_dim)
        w2:(qmix_hidden_dim, 1)
        b2:(1, 1)

        """
        if self.hyper_layers_num == 2:
            print("hyper_layers_num=2")
            self.hyper_w1 = nn.Sequential(nn.Linear(self.mix_input_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, self.N * self.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(self.mix_input_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, self.qmix_hidden_dim * 1))
        elif self.hyper_layers_num == 1:
            print("hyper_layers_num=1")
            self.hyper_w1 = nn.Linear(self.mix_input_dim, self.N * self.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.mix_input_dim, self.qmix_hidden_dim * 1)
        else:
            print("wrong!!!")

        self.hyper_b1 = nn.Linear(self.mix_input_dim, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.mix_input_dim, self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, 1))

    def role_gru_forward(self, role_embeddings):
        # role_embeddings.shape = (batch_size, N*role_embedding_dim)
        self.role_gru_hidden = self.role_gru(role_embeddings, self.role_gru_hidden)
        output = th.sigmoid(self.role_gru_hidden)
        return output

    def forward(self, q, s, att):
        # q.shape(batch_size, max_episode_len, N)
        # s.shape(batch_size, max_episode_len,state_shape)

        q = q.view(-1, 1, self.N)  # (batch_size * max_episode_len, 1, N)
        s = s.reshape(-1, self.state_shape)  # (batch_size * max_episode_len, state_shape)
        att = att.reshape(-1, att.shape[2])
        state = th.cat([s, att], dim=-1)

        w1 = th.abs(self.hyper_w1(state))  # (batch_size * max_episode_len, N * qmix_hidden_dim)
        b1 = self.hyper_b1(state)  # (batch_size * max_episode_len, qmix_hidden_dim)
        w1 = w1.view(-1, self.N, self.qmix_hidden_dim)  # (batch_size * max_episode_len, N,  qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        # th.bmm: 3 dimensional tensor multiplication
        q_hidden = F.elu(th.bmm(q, w1) + b1)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        w2 = th.abs(self.hyper_w2(state))  # (batch_size * max_episode_len, qmix_hidden_dim * 1)
        b2 = self.hyper_b2(state)  # (batch_size * max_episode_len,1)
        w2 = w2.view(-1, self.qmix_hidden_dim, 1)  # (b\atch_size * max_episode_len, qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)  # (batch_size * max_episode_len, 1， 1)

        q_total = th.bmm(q_hidden, w2) + b2  # (batch_size * max_episode_len, 1， 1)
        q_total = q_total.view(self.batch_size, -1, 1)  # (batch_size, max_episode_len, 1)
        return q_total


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, att_dim, att_out_dim, soft_temperature, dim_q, dim_k, dim_v):
        super(MultiHeadAttention, self).__init__()
        assert (att_dim % n_heads) == 0, "n_heads must divide att_dim"
        self.att_dim = att_dim
        self.att_out_dim = att_out_dim
        self.head_att_dim = att_dim // n_heads
        self.n_heads = n_heads
        self.temperature = self.head_att_dim ** 0.5 / soft_temperature

        self.fc_q = nn.Linear(dim_q, self.att_dim, bias=False)
        self.fc_k = nn.Linear(dim_k, self.att_dim, bias=False)
        self.fc_v = nn.Linear(dim_v, self.att_dim)
        self.fc_final = nn.Linear(self.att_dim, self.att_out_dim)

    def forward(self, q, k, v):
        # q.shape = (batch, N, dim)
        batch_size = q.shape[0]
        # shape = (batch*N, att_dim)->(batch, N, heads, head_att_dim)->(batch, heads, N, head_att_dim)
        q = self.fc_q(q.view(-1, q.shape[2])).view(batch_size, -1, self.n_heads, self.head_att_dim).transpose(1, 2)
        # shape = (batch*N, att_dim)->(batch, N, heads, head_att_dim)->(batch, heads, head_att_dim, N)
        k_T = self.fc_k(k.view(-1, k.shape[2])).view(batch_size, -1, self.n_heads, self.head_att_dim).permute(0, 2, 3, 1)
        v = self.fc_v(v.view(-1, v.shape[2])).view(batch_size, -1, self.n_heads, self.head_att_dim).transpose(1, 2)
        alpha = F.softmax(th.matmul(q/self.temperature, k_T), dim=-1)  # shape = (batch, heads, N, N)
        # shape = (batch, heads, N, head_att_dim)->(batch, N, heads, head_att_dim)->(batch, N, att_dim)
        result = th.matmul(alpha, v).transpose(1, 2).reshape(batch_size, -1, self.att_dim)
        result = self.fc_final(result)  # shape = (batch, N, att_out_dim)
        return result


class Mixer(nn.Module):
    def __init__(self, args, abs=True):
        super(Mixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape))

        self.abs = abs  # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")

        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))

        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.embed_dim, 1))

        if getattr(args, "use_orthogonal", False):
            for m in self.modules():
                orthogonal_init_(m)

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()

        qvals = qvals.reshape(b * t, 1, self.n_agents)

        states = states.reshape(-1, self.state_dim)
        # First layer
        w1 = self.hyper_w1(states).view(-1, self.n_agents, self.embed_dim)  # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)

        # Second layer
        w2 = self.hyper_w2(states).view(-1, self.embed_dim, 1)  # b * t, emb, 1
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)

        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1)  # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2  # b * t, 1, 1

        return y.view(b, t, -1)

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)
