import copy

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY

# from timeit import timeit_method
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np
from modules.action_encoders.obs_reward_encoder import ObsRewardEncoder as action_encoder
from modules.intention_encoder.intention import NextIntentionPredictor, IntentToActionMapper, AggIntent

from components.epsilon_schedules import DecayThenFlatSchedule
import torch.nn as nn


# This multi-agent controller shares parameters between agents
class TMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(TMAC, self).__init__(scheme, groups, args)
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.args = args
        self.obs_dim = int(np.prod(args.obs_shape))
        args.input_shape, args.obs_shape = self._get_input_shape(scheme)
        self.hidden_states = None

        self.device = args.device
        self.intention_module = NextIntentionPredictor(args).to(args.device)
        self.intent2act = IntentToActionMapper(args).to(args.device)
        self.agg_intent = AggIntent(args).to(args.device)

        self.action_encoder = action_encoder(args).to(args.device)
        action_repr = th.empty(self.n_actions, self.args.action_latent_dim, device=args.device)
        nn.init.xavier_uniform_(action_repr)
        self.action_repr = action_repr

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, drop_prob=1):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals, drop_inputs, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        # return chosen_actions, drop_inputs
        return chosen_actions, th.squeeze(drop_inputs)[bs]

    # @timeit_method
    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()

        agent_inputs, drop_inputs = self._build_inputs(ep_batch, t)
        agent_inputs = agent_inputs.view(-1, self.n_agents, self.obs_dim + self.n_actions + self.n_agents)
        if drop_inputs.shape[0] < 10:
            drop_inputs = self.generate_drop(drop_inputs, 0.6)

        intention, kl = self.intention_module(self.hidden_states, self.mem_intent)
        self.update_memory(t, drop_inputs, intention)
        intent_act = self.intent2act(intention, self.action_repr)
        imt_intent = self.intent2act(self.intent_memory, self.action_repr)

        fused_intent = self.agg_intent(intent_act, imt_intent, self.time_memory, t)
        new_agent_inputs = th.cat((agent_inputs, fused_intent), -1)
        agent_outs, self.hidden_states = self.agent(new_agent_inputs, self.hidden_states)

        self.mem_intent = agent_outs.clone().detach()

        return agent_outs, drop_inputs.view(ep_batch.batch_size, self.n_agents, -1), kl

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1).to(self.device)  # bav
        self.intention_hidden = self.role_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1).to(self.device)
        self.mem_intent = self.intention_module.init().unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        intent_memory = th.empty(batch_size, self.n_agents, self.n_agents, self.args.n_actions * self.args.horizon, device=self.device)
        nn.init.xavier_uniform_(intent_memory.view(-1, self.args.n_actions * self.args.horizon))
        self.intent_memory = intent_memory.view(batch_size, self.n_agents, self.n_agents, self.args.n_actions * self.args.horizon)
        # self.intent_memory = th.zeros(batch_size, self.n_agents, self.n_agents, self.args.horizon*self.args.n_actions, device=self.device)  # [B, N, N, D]
        self.time_memory = th.full((batch_size, self.n_agents, self.n_agents), fill_value=-1, device=self.device, dtype=th.int16)  # [B, N, N]

    def generate_drop(self, drop_inputs, drop_prob):
        B, N, _ = drop_inputs.shape

        rand_tensor = th.rand(B, N, N, device=self.device)
        drop_mask = rand_tensor < drop_prob
        diag_mask = th.eye(N, dtype=th.bool, device=self.device).unsqueeze(0)

        drop_mask.masked_fill_(diag_mask, False)
        return drop_mask.int()

    def update_memory(self, t, drop_inputs, intent_t):
        B, N, _ = drop_inputs.shape

        recv_mask = (drop_inputs == 0)  # [B,N,N]
        eye = th.eye(N, dtype=th.bool, device=self.args.device).unsqueeze(0)  # [1, N, N]
        recv_mask = recv_mask & (~eye)

        # intent_t: [B, N(send), D] -> [B, 1, N(send), D] -> [B, N(recv), N(send), D]
        intent_full = intent_t.unsqueeze(1).expand(-1, N, -1, -1)

        self.intent_memory = th.where(recv_mask.unsqueeze(-1), intent_full, self.intent_memory)

        t_tensor = th.full_like(self.time_memory, fill_value=int(t))
        self.time_memory = th.where(recv_mask, t_tensor, self.time_memory)

    def action_encoder_params(self):
        return list(self.action_encoder.parameters())

    def action_repr_forward(self, ep_batch, t):
        return self.action_encoder.predict(ep_batch["obs"][:, t], ep_batch["actions_onehot"][:, t])

    def update_action_repr(self):
        action_repr = self.action_encoder()
        self.action_repr = action_repr.detach().clone()

    def parameters(self):
        params = list(self.agent.parameters())
        params += list(self.role_agent.parameters())
        params += list(self.intention_module.parameters())
        # params += list(self.intent2act.parameters())
        params += list(self.agg_intent.parameters())
        return params

    def intent2act_params(self):
        return list(self.intent2act.parameters())

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.role_agent.load_state_dict(other_mac.role_agent.state_dict())
        self.intention_module.load_state_dict(other_mac.intention_module.state_dict())
        self.intent2act.load_state_dict(other_mac.intent2act.state_dict())
        self.action_encoder.load_state_dict(other_mac.action_encoder.state_dict())
        self.action_repr = copy.deepcopy(other_mac.action_repr)

    def new_load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.role_agent.load_state_dict(other_mac.role_agent.state_dict())
        self.intention_module.load_state_dict(other_mac.intention_module.state_dict())
        self.intent2act.load_state_dict(other_mac.intent2act.state_dict())
        self.agg_intent.load_state_dict(other_mac.agg_intent.state_dict())
        self.action_encoder.load_state_dict(other_mac.action_encoder.state_dict())
        self.action_repr = copy.deepcopy(other_mac.action_repr)

    def cuda(self):
        self.agent.cuda()
        self.role_agent.cuda()
        self.intention_module.cuda()
        self.intent2act.cuda()
        self.agg_intent.cuda()
        self.action_encoder.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        if bs < 10:
            drop_inputs = th.zeros(bs, self.n_agents, self.n_agents, dtype=th.int16).to(self.device)
            # drop_inputs = None
        else:
            drop_inputs = batch["drop_inputs"][:, t]

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

        return inputs, drop_inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        obs_shape = scheme["obs"]["vshape"]

        return input_shape, obs_shape


class PlaceholderModule(nn.Module):
    def __init__(self, horizon, dim):
        super().__init__()
        self.embedding = nn.Parameter(th.empty(horizon, dim))
        nn.init.xavier_uniform_(self.embedding)

    def forward(self):
        return self.embedding
