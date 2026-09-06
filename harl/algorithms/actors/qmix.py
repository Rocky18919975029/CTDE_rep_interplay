# harl/algorithms/actors/qmix.py

"""QMIX algorithm."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from harl.models.value_function_models.qmix_agent_q import AgentQNet
from harl.models.value_function_models.qmix_mixer import QMixer
from harl.utils.envs_tools import check
from harl.utils.models_tools import update_linear_schedule, get_grad_norm


class QMIX:
    """QMIX learner."""

    def __init__(
        self,
        args,
        obs_space,
        share_obs_space,
        act_space,
        device=torch.device("cpu"),
    ):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.num_agents = args["num_agents"]
        self.gamma = args.get("gamma", 0.99)
        self.lr = args.get("lr", 5e-4)
        self.opti_eps = args.get("opti_eps", 1e-5)
        self.weight_decay = args.get("weight_decay", 0.0)

        self.use_max_grad_norm = args.get("use_max_grad_norm", True)
        self.max_grad_norm = args.get("max_grad_norm", 10.0)

        self.target_update_interval = args.get("target_update_interval", 200)
        self.use_double_q = args.get("use_double_q", True)

        self.agent_q = AgentQNet(args, obs_space, act_space, device=device)
        self.mixer = QMixer(args, share_obs_space, device=device)

        self.target_agent_q = copy.deepcopy(self.agent_q)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimizer = torch.optim.Adam(
            list(self.agent_q.parameters()) + list(self.mixer.parameters()),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

        self.train_step = 0

    def prep_training(self):
        self.agent_q.train()
        self.mixer.train()

    def prep_rollout(self):
        self.agent_q.eval()
        self.mixer.eval()

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    @torch.no_grad()
    def get_actions(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        epsilon=0.0,
    ):
        """Select actions for all agents.

        Returns:
            actions: [batch, n_agents, 1]
            rnn_states: [batch, n_agents, recurrent_n, hidden_dim]
        """
        actions_collector = []
        rnn_states_collector = []

        for agent_id in range(self.num_agents):
            agent_available_actions = (
                available_actions[:, agent_id]
                if available_actions is not None
                else None
            )

            actions, new_rnn_states = self.agent_q.act(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                masks[:, agent_id],
                agent_available_actions,
                epsilon=epsilon,
            )

            actions_collector.append(actions)
            rnn_states_collector.append(new_rnn_states)

        actions = torch.stack(actions_collector, dim=1)
        rnn_states = torch.stack(rnn_states_collector, dim=1)

        return actions, rnn_states

    def update(self, batch):
        obs = check(batch["obs"]).to(**self.tpdv)
        share_obs = check(batch["share_obs"]).to(**self.tpdv)
        actions = check(batch["actions"]).to(device=self.device).long()
        rewards = check(batch["rewards"]).to(**self.tpdv)
        dones = check(batch["dones"]).to(**self.tpdv)
        next_obs = check(batch["next_obs"]).to(**self.tpdv)
        next_share_obs = check(batch["next_share_obs"]).to(**self.tpdv)

        available_actions = batch.get("available_actions", None)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        next_available_actions = batch.get("next_available_actions", None)
        if next_available_actions is not None:
            next_available_actions = check(next_available_actions).to(**self.tpdv)

        batch_size = obs.shape[0]

        rnn_states = batch.get("rnn_states", None)
        if rnn_states is None:
            rnn_states = torch.zeros(
                batch_size,
                self.num_agents,
                self.args.get("recurrent_n", 1),
                self.args["hidden_sizes"][-1],
                device=self.device,
            )
        else:
            rnn_states = check(rnn_states).to(**self.tpdv)

        next_rnn_states = batch.get("next_rnn_states", None)
        if next_rnn_states is None:
            next_rnn_states = torch.zeros_like(rnn_states)
        else:
            next_rnn_states = check(next_rnn_states).to(**self.tpdv)

        masks = batch.get("masks", None)
        if masks is None:
            masks = torch.ones(batch_size, self.num_agents, 1, device=self.device)
        else:
            masks = check(masks).to(**self.tpdv)

        next_masks = batch.get("next_masks", None)
        if next_masks is None:
            next_masks = torch.ones_like(masks)
        else:
            next_masks = check(next_masks).to(**self.tpdv)

        chosen_agent_qs = []

        for agent_id in range(self.num_agents):
            q_values, _ = self.agent_q(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                masks[:, agent_id],
                available_actions[:, agent_id] if available_actions is not None else None,
            )

            agent_actions = actions[:, agent_id].long()
            chosen_q = q_values.gather(dim=-1, index=agent_actions)
            chosen_agent_qs.append(chosen_q)

        chosen_agent_qs = torch.cat(chosen_agent_qs, dim=-1)
        q_tot = self.mixer(chosen_agent_qs, share_obs)

        with torch.no_grad():
            target_agent_qs = []

            for agent_id in range(self.num_agents):
                target_q_values, _ = self.target_agent_q(
                    next_obs[:, agent_id],
                    next_rnn_states[:, agent_id],
                    next_masks[:, agent_id],
                    next_available_actions[:, agent_id]
                    if next_available_actions is not None
                    else None,
                )

                if self.use_double_q:
                    online_next_q_values, _ = self.agent_q(
                        next_obs[:, agent_id],
                        next_rnn_states[:, agent_id],
                        next_masks[:, agent_id],
                        next_available_actions[:, agent_id]
                        if next_available_actions is not None
                        else None,
                    )

                    next_actions = online_next_q_values.argmax(dim=-1, keepdim=True)
                    target_chosen_q = target_q_values.gather(
                        dim=-1,
                        index=next_actions,
                    )
                else:
                    target_chosen_q = target_q_values.max(dim=-1, keepdim=True)[0]

                target_agent_qs.append(target_chosen_q)

            target_agent_qs = torch.cat(target_agent_qs, dim=-1)
            target_q_tot = self.target_mixer(target_agent_qs, next_share_obs)

            reward = self._common_reward(rewards)
            done = self._episode_done(dones)

            td_target = reward + self.gamma * (1.0 - done) * target_q_tot

        td_error = q_tot - td_target
        loss = F.mse_loss(q_tot, td_target)

        self.optimizer.zero_grad()
        loss.backward()

        parameters = list(self.agent_q.parameters()) + list(self.mixer.parameters())

        if self.use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                parameters,
                self.max_grad_norm,
            )
        else:
            grad_norm = get_grad_norm(parameters)

        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update_interval == 0:
            self.update_targets()

        return {
            "qmix_loss": loss.item(),
            "q_tot": q_tot.mean().item(),
            "target_q_tot": target_q_tot.mean().item(),
            "td_error_abs": td_error.abs().mean().item(),
            "qmix_grad_norm": (
                grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            ),
        }

    def update_targets(self):
        self.target_agent_q.load_state_dict(self.agent_q.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _common_reward(self, rewards):
        """Convert reward tensor to [batch, 1]."""
        if rewards.dim() == 3:
            return rewards[:, 0]
        return rewards

    def _episode_done(self, dones):
        """Convert done tensor to episode-level done [batch, 1].

        For synchronous MARL envs like GRF, all agents usually terminate together.
        Using min means the target is truncated only when the whole env is done.
        """
        if dones.dim() == 3:
            return dones.min(dim=1)[0]
        return dones

    def save(self):
        return {
            "agent_q": self.agent_q.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_agent_q": self.target_agent_q.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
        }

    def restore(self, checkpoint):
        self.agent_q.load_state_dict(checkpoint["agent_q"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_agent_q.load_state_dict(checkpoint["target_agent_q"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_step = checkpoint.get("train_step", 0)