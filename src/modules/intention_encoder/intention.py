import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from timeit import timeit_method


class NextIntentionPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()

        input_dim = args.rnn_hidden_dim + args.n_actions
        self.args = args

        self.network = nn.Sequential(
            nn.Linear(input_dim, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.entity_dim)
        )
        self.mu = nn.Linear(args.entity_dim, args.n_actions * args.horizon)
        self.var = nn.Linear(args.entity_dim, args.n_actions * args.horizon)

    def forward(self, hidden_state, action_logits):
        combined_features = torch.cat([hidden_state, action_logits], dim=-1)
        next_intention = self.network(combined_features)
        mu = self.mu(next_intention)
        std = F.softplus(self.var(next_intention)) + 1e-6
        eps = torch.randn_like(std)
        z = mu + eps * std
        logvar = torch.log(std * std + 1e-12)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        kl_loss = kl_loss.mean()
        return z, kl_loss

    def init(self):
        return self.network[0].weight.new_zeros(self.args.n_agents, self.args.n_actions)


class AggIntent(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.att = nn.Linear(args.action_latent_dim * args.horizon, args.action_latent_dim * args.horizon)
        self.mix = nn.Parameter(torch.tensor(0.3))
        self.q_proj = nn.Linear(args.action_latent_dim * args.horizon, args.action_latent_dim * args.horizon)
        self.k_proj = nn.Linear(args.action_latent_dim * args.horizon, args.action_latent_dim * args.horizon)
        self.v_proj = nn.Linear(args.action_latent_dim * args.horizon, args.action_latent_dim * args.horizon)

    def forward(self, intention, intent2act, time_memory, t, gamma=0.6):
        B, N, _, D = intent2act.shape

        delta_t = (t - time_memory).clamp(min=0).to(torch.float32)
        decay = gamma ** delta_t

        new_q = self.q_proj(intention) # [B, N, 1, D]
        q = new_q.unsqueeze(2)  # [B, N, 1, D]
        k = self.k_proj(intent2act)  # [B, N, N, D]
        v = self.v_proj(intent2act)
        scores = torch.matmul(q, k.transpose(-1, -2)).squeeze(2) / math.sqrt(D)
        scores = (scores - scores.mean(dim=-1, keepdim=True)) / (scores.std(dim=-1, keepdim=True) + 1e-6)
        attn = F.softmax(scores, dim=-1) * decay
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        weighted = torch.matmul(attn.unsqueeze(2), v).squeeze(2)
        fused = self.att(weighted)
        # fused = self.mix * new_q + (1 - self.mix) * fused

        return fused


class IntentToActionMapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    # @timeit_method
    def forward(self, logits, action_embed_matrix, tau=1.0):
        if logits.dim()==3:
            B, N, _ = logits.shape
            action_logits = logits.view(B, N, self.args.horizon, self.args.n_actions)
        elif logits.dim()==4:
            B, N, _, _ = logits.shape
            action_logits = logits.view(B, N, N, self.args.horizon, self.args.n_actions)

        action_logits = action_logits.reshape(-1, self.args.n_actions)

        probs = F.softmax(action_logits / tau, dim=-1)  # [B, Seq, N_actions]

        e_soft = torch.matmul(probs, action_embed_matrix)  # [B, Seq, D_latent]

        if logits.dim()==3:
            e_soft = e_soft.reshape(B, N, -1)
            return e_soft
        elif logits.dim()==4:
            e_soft = e_soft.reshape(B, N, N, -1)
            return e_soft

