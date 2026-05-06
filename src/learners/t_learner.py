import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.rode_qmix import QMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
import torch.nn.functional as F


class TLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.params = list(mac.parameters())

        self.mixer = QMixer(args).to(args.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(args.device)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        self.state_dim = args.state_shape
        self.n_agents = args.n_agents

        self.n_actions = args.n_actions
        self.obs_dim = int(np.prod(args.obs_shape))

        self.action_updated = True
        self.action_encoder_params = list(self.mac.action_encoder_params())
        self.action_encoder_optimiser = RMSprop(params=self.action_encoder_params, lr=args.lr,
                                                alpha=args.optim_alpha, eps=args.optim_eps)

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        kl_ls = th.tensor(0.0, requires_grad=True).to(self.args.device)

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, _, kl = self.mac.forward(batch, t)
            mac_out.append(agent_outs)
            kl_ls += kl

        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        kl_ls /= (batch.batch_size * self.n_agents)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs, _, _ = self.target_mac.forward(batch, t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = th.argmax(mac_out_detach, dim=3, keepdim=True)
            target_max_qvals = th.gather(target_mac_out[:, 1:], 3, cur_max_actions[:, 1:]).squeeze(3)

        # ----new loss
        q_total_eval = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        q_total_target = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
        targets = batch["reward"][:, :-1] + self.args.gamma * (1 - terminated) * q_total_target
        td_error = (q_total_eval - targets.detach())  # targets.detach() to cut the backward
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / (1 - terminated).sum()

        ANNEAL_START_STEP = 150000
        ANNEAL_END_STEP = 300000
        progress = max(0.0, min(1.0, (t_env - ANNEAL_START_STEP) / (ANNEAL_END_STEP - ANNEAL_START_STEP)))
        start_beta = 0.1
        end_beta = 5

        beta = start_beta + (end_beta - start_beta) * progress
        tot_loss = loss + beta * kl_ls

        # Optimise
        self.optimiser.zero_grad()
        tot_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        pred_obs_loss = None
        pred_r_loss = None
        pred_grad_norm = None
        if self.action_updated:
            # train action encoder
            no_pred = []
            r_pred = []
            for t in range(batch.max_seq_length):
                no_preds, r_preds = self.mac.action_repr_forward(batch, t=t)
                no_pred.append(no_preds)
                r_pred.append(r_preds)
            no_pred = th.stack(no_pred, dim=1)[:, :-1]  # Concat over time
            r_pred = th.stack(r_pred, dim=1)[:, :-1]
            no = batch["obs"][:, 1:].detach().clone()
            repeated_rewards = batch["reward"][:, :-1].detach().clone().unsqueeze(2).repeat(1, 1, self.n_agents, 1)

            pred_obs_loss = th.sqrt(((no_pred - no) ** 2).sum(dim=-1)).mean()
            pred_r_loss = ((r_pred - repeated_rewards) ** 2).mean()

            pred_loss = pred_obs_loss + 10 * pred_r_loss
            self.action_encoder_optimiser.zero_grad()
            pred_loss.backward()
            pred_grad_norm = th.nn.utils.clip_grad_norm_(self.action_encoder_params, self.args.grad_norm_clip)
            self.action_encoder_optimiser.step()

            if t_env > self.args.action_spaces_update:
                self.mac.update_action_repr()
                self.action_updated = False
                self.params += list(self.mac.intent2act_params())
                self._update_targets()
                self.last_target_update_episode = episode_num

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets(self.action_updated)
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("kl_ls", kl_ls.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            # self.logger.log_stat("alpha", alpha, t_env)
            if pred_obs_loss is not None:
                self.logger.log_stat("pred_obs_loss", pred_obs_loss.item(), t_env)
                self.logger.log_stat("pred_r_loss", pred_r_loss.item(), t_env)
                self.logger.log_stat("action_encoder_grad_norm", pred_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                        / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                         / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self, action_updated=True):
        if action_updated:
            self.target_mac.load_state(self.mac)
        else:
            self.target_mac.new_load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_mac.action_updated = self.action_updated
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
