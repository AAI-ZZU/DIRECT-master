from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMAC, self).__init__(scheme, groups, args)
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals,drop_inputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions, th.squeeze(drop_inputs)[bs]

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()
            
        agent_inputs ,drop_inputs= self._build_inputs(ep_batch, t)
        agent_inputs = agent_inputs.view(-1, self.n_agents, agent_inputs.shape[-1])
        if drop_inputs is None:
            drop_inputs = th.zeros(agent_inputs.shape[0], self.n_agents, self.n_agents).cuda()
            # if test_mode:
            for i in range(0, self.n_agents):
                for j in range(0, self.n_agents):
                    if (np.random.random() > 0.4) and i != j:
                        drop_inputs[:, i, j] = 1
        
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, drop_inputs,test_mode)

        return agent_outs,drop_inputs.view(ep_batch.batch_size, self.n_agents, -1)
    
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
            drop_inputs = None
        else:
            drop_inputs = batch["drop_inputs"][:, t]

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

        return inputs, drop_inputs