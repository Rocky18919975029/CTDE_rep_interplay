# harl/runners/on_policy_ha_runner.py

"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner
from harl.utils.decomposition_experiment import (
    ACTOR_ALIGNMENT_MODES,
    CRITIC_ALIGNMENT_MODES,
    normalized_alignment_loss,
    unique_trainable_parameters,
)


class OnPolicyHARunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""

    def train(self):
        """Train the model."""
        actor_train_infos = []

       
        if self.args["algo"] in ["magpo", "sable", "mat"]:
           
            if self.value_normalizer is not None:
                advantages = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
            else:
                advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

           
            if self.state_type == "EP":
                adv_to_use = advantages.copy()
                adv_to_use[self.actor_buffer[0].active_masks[:-1] == 0.0] = np.nan
                mean_adv = np.nanmean(adv_to_use)
                std_adv = np.nanstd(adv_to_use)
                adv_to_use = (adv_to_use - mean_adv) / (std_adv + 1e-5)
            elif self.state_type == "FP":
                adv_to_use = np.zeros_like(advantages)
                for i in range(self.num_agents):
                    adv_copy = advantages[:, :, i].copy()
                    adv_copy[self.actor_buffer[i].active_masks[:-1] == 0.0] = np.nan
                    adv_to_use[:, :, i] = (advantages[:, :, i] - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5)

            magpo_train_info = {
                "guider_policy_loss": 0, "guider_kl_loss": 0,
                "learner_bc_loss": 0, "learner_aux_loss": 0,
                "guider_grad_norm": 0, "learner_grad_norm": 0,
                "guider_entropy": 0 
            }

            ppo_epoch = self.algo_args["algo"]["ppo_epoch"]
            num_mini_batch = self.algo_args["algo"]["actor_num_mini_batch"]
            
           
            episode_length, n_rollout_threads = self.actor_buffer[0].actions.shape[0:2]
            batch_size = n_rollout_threads * episode_length
            mini_batch_size = batch_size // num_mini_batch

            obs_all = np.stack([self.actor_buffer[i].obs[:-1] for i in range(self.num_agents)], axis=2)
            actions_all = np.stack([self.actor_buffer[i].actions for i in range(self.num_agents)], axis=2)
            action_log_probs_all = np.stack([self.actor_buffer[i].action_log_probs for i in range(self.num_agents)], axis=2)
            masks_all = np.stack([self.actor_buffer[i].masks[:-1] for i in range(self.num_agents)], axis=2)
            active_masks_all = np.stack([self.actor_buffer[i].active_masks[:-1] for i in range(self.num_agents)], axis=2)
            rnn_states_all = np.stack([self.actor_buffer[i].rnn_states[:-1] for i in range(self.num_agents)], axis=2)
            
            if self.actor_buffer[0].available_actions is not None:
                avail_actions_all = np.stack([self.actor_buffer[i].available_actions[:-1] for i in range(self.num_agents)], axis=2)
            else:
                avail_actions_all = None

            share_obs = self.critic_buffer.share_obs[:-1]
            
            obs_flat = obs_all.reshape(-1, self.num_agents, *obs_all.shape[3:])
            actions_flat = actions_all.reshape(-1, self.num_agents, *actions_all.shape[3:])
            log_probs_flat = action_log_probs_all.reshape(-1, self.num_agents, *action_log_probs_all.shape[3:])
            masks_flat = masks_all.reshape(-1, self.num_agents, 1)
            active_masks_flat = active_masks_all.reshape(-1, self.num_agents, 1)
            rnn_states_flat = rnn_states_all.reshape(-1, self.num_agents, *rnn_states_all.shape[3:])
            share_obs_flat = share_obs.reshape(-1, *share_obs.shape[2:])
            adv_flat = adv_to_use.reshape(-1, self.num_agents, 1) if self.state_type == "FP" else adv_to_use.reshape(-1, 1, 1).repeat(self.num_agents, axis=1)
            
            if avail_actions_all is not None:
                avail_actions_flat = avail_actions_all.reshape(-1, self.num_agents, *avail_actions_all.shape[3:])
            else:
                avail_actions_flat = None

            if not np.all(active_masks_flat == 0.0):
                for _ in range(ppo_epoch):
                    rand_indices = torch.randperm(batch_size).numpy()
                    
                    for i in range(num_mini_batch):
                        idx = rand_indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                        
                        sample = (
                            obs_flat[idx], share_obs_flat[idx], rnn_states_flat[idx],
                            actions_flat[idx], masks_flat[idx], active_masks_flat[idx],
                            log_probs_flat[idx], adv_flat[idx], 
                            avail_actions_flat[idx] if avail_actions_flat is not None else None,
                        )
                        
                        
                        if self.args["algo"] == "magpo":
                            (
                                g_policy_loss, g_kl_loss, l_bc_loss, l_aux_loss, g_grad_norm, l_grad_norm
                            ) = self.actor[0].update(sample, self.guider, self.guider_optimizer)
                            
                            magpo_train_info["guider_policy_loss"] += g_policy_loss
                            magpo_train_info["guider_kl_loss"] += g_kl_loss
                            magpo_train_info["learner_bc_loss"] += l_bc_loss
                            magpo_train_info["learner_aux_loss"] += l_aux_loss
                            magpo_train_info["guider_grad_norm"] += g_grad_norm
                            magpo_train_info["learner_grad_norm"] += l_grad_norm
                        
                        elif self.args["algo"] in ["sable", "mat"]:
                           
                            (
                                g_policy_loss, g_entropy, g_grad_norm
                            ) = self.actor[0].update_ctce(sample, self.guider, self.guider_optimizer)
                            
                            magpo_train_info["guider_policy_loss"] += g_policy_loss
                            magpo_train_info["guider_entropy"] += g_entropy
                            magpo_train_info["guider_grad_norm"] += g_grad_norm

                num_updates = ppo_epoch * num_mini_batch
                for k in magpo_train_info.keys():
                    magpo_train_info[k] /= num_updates
            
            
            actor_train_infos = [magpo_train_info for _ in range(self.num_agents)]

        
        else:
           
            factor = np.ones(
                (
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                ),
                dtype=np.float32,
            )

            
            if self.value_normalizer is not None:
                advantages = self.critic_buffer.returns[
                    :-1
                ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
            else:
                advantages = (
                    self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
                )

           
            if self.state_type == "FP":
                active_masks_collector = [
                    self.actor_buffer[i].active_masks for i in range(self.num_agents)
                ]
                active_masks_array = np.stack(active_masks_collector, axis=2)
                advantages_copy = advantages.copy()
                advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
                mean_advantages = np.nanmean(advantages_copy)
                std_advantages = np.nanstd(advantages_copy)
                advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

            if self.fixed_order:
                agent_order = list(range(self.num_agents))
            else:
                agent_order = list(torch.randperm(self.num_agents).numpy())
                
            for agent_id in agent_order:
                self.actor_buffer[agent_id].update_factor(
                    factor
                ) 

                
                available_actions = (
                    None
                    if self.actor_buffer[agent_id].available_actions is None
                    else self.actor_buffer[agent_id]
                    .available_actions[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
                )

               
                old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                    self.actor_buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                    self.actor_buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                    self.actor_buffer[agent_id].actions.reshape(
                        -1, *self.actor_buffer[agent_id].actions.shape[2:]
                    ),
                    self.actor_buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.actor_buffer[agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
                )

               
                if self.state_type == "EP":
                    adv_to_use = advantages.copy()
                elif self.state_type == "FP":
                    adv_to_use = advantages[:, :, agent_id].copy()
                    
                
                if self.state_type == "EP":
                    adv_copy_for_norm = adv_to_use.copy()
                    adv_copy_for_norm[self.actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                    mean_adv = np.nanmean(adv_copy_for_norm)
                    std_adv = np.nanstd(adv_copy_for_norm)
                    adv_to_use = (adv_to_use - mean_adv) / (std_adv + 1e-5)

                
                actor_train_info = {
                    "policy_loss": 0, "dist_entropy": 0, 
                    "actor_grad_norm": 0, "ratio": 0, "aux_loss": 0,
                    "action_pred_loss": 0, "alignment_loss": 0
                }

               
                if not np.all(self.actor_buffer[agent_id].active_masks[:-1] == 0.0):
                    ppo_epoch = self.actor[agent_id].ppo_epoch
                    actor_num_mini_batch = self.actor[agent_id].actor_num_mini_batch
                    
                    use_aux_loss = self.algo_args["algo"].get("use_aux_loss", False)
                    use_prob_aux_loss = self.algo_args["algo"].get("use_prob_aux_loss", False)
                    use_actor_alignment = (
                        getattr(self, "alignment_mode", "separate")
                        in ACTOR_ALIGNMENT_MODES
                    )
                    need_target_embed = (
                        use_aux_loss or use_prob_aux_loss or use_actor_alignment
                    )
                    
                    use_action_pred = self.algo_args["algo"].get("use_action_pred", False)

                    if use_action_pred:
                        all_actions = [self.actor_buffer[i].actions for i in range(self.num_agents)]
                        joint_actions_input = np.concatenate(all_actions, axis=-1)
                    else:
                        joint_actions_input = None

                    for _ in range(ppo_epoch):
                       
                        share_obs_to_pass = self.critic_buffer.share_obs if need_target_embed else None

                        prl_target = (
                            "current"
                            if use_actor_alignment
                            else self.algo_args["algo"].get("prl_target", "next")
                        )
                        assert prl_target in ["next", "current"], f"Unsupported prl_target: {prl_target}"
                        
                        if self.actor[agent_id].use_recurrent_policy:
                            data_generator = self.actor_buffer[agent_id].recurrent_generator_actor(
                                adv_to_use, actor_num_mini_batch, self.actor[agent_id].data_chunk_length,
                                share_obs=share_obs_to_pass, joint_actions=joint_actions_input,
                                prl_target=prl_target
                            )
                        elif self.actor[agent_id].use_naive_recurrent_policy:
                            data_generator = self.actor_buffer[agent_id].naive_recurrent_generator_actor(
                                adv_to_use, actor_num_mini_batch,
                                share_obs=share_obs_to_pass,
                                joint_actions=joint_actions_input,
                                prl_target=prl_target 
                            )
                        else:
                            data_generator = self.actor_buffer[agent_id].feed_forward_generator_actor(
                                adv_to_use, actor_num_mini_batch,
                                share_obs=share_obs_to_pass,
                                joint_actions=joint_actions_input,
                                prl_target=prl_target 
                            )

                        for sample in data_generator:
                            target_embedding = None
                            target_joint_actions = None
                            
                            current_sample = list(sample)
                            
                            if use_action_pred:
                                target_joint_actions = torch.tensor(current_sample.pop()).to(self.device)
                            
                            if need_target_embed:
                                target_share_obs_batch = current_sample.pop()
                                
                                if target_share_obs_batch is not None:
                                    with torch.no_grad():
                                        b_size = target_share_obs_batch.shape[0]
                                        rec_n = self.critic.critic.recurrent_n
                                        rnn_hid_size = self.critic.critic.hidden_sizes[-1]
                                        
                                        dummy_rnn = np.zeros((b_size, rec_n, rnn_hid_size), dtype=np.float32)
                                        dummy_masks = np.ones((b_size, 1), dtype=np.float32)
                                        
                                        target_embedding = self.critic.critic.get_embedding(
                                            target_share_obs_batch, dummy_rnn, dummy_masks
                                        )
                            
                            actor_sample = tuple(current_sample)
                            
                            legacy_target_embedding = (
                                target_embedding
                                if use_aux_loss or use_prob_aux_loss
                                else None
                            )
                            alignment_target = (
                                target_embedding if use_actor_alignment else None
                            )
                            policy_loss, dist_entropy, actor_grad_norm, imp_weights, aux_loss, action_pred_loss, alignment_loss = self.actor[agent_id].update(
                                actor_sample,
                                legacy_target_embedding,
                                target_joint_actions,
                                alignment_target=alignment_target,
                            )

                            actor_train_info["policy_loss"] += policy_loss.item()
                            actor_train_info["dist_entropy"] += dist_entropy.item()
                            actor_train_info["actor_grad_norm"] += actor_grad_norm
                            actor_train_info["ratio"] += imp_weights.mean()
                            actor_train_info["aux_loss"] += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
                            actor_train_info["action_pred_loss"] += action_pred_loss.item() if isinstance(action_pred_loss, torch.Tensor) else action_pred_loss
                            actor_train_info["alignment_loss"] += alignment_loss.item()

                    num_updates = ppo_epoch * actor_num_mini_batch
                    for k in actor_train_info.keys():
                        actor_train_info[k] /= num_updates

               
                new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                    self.actor_buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                    self.actor_buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                    self.actor_buffer[agent_id].actions.reshape(
                        -1, *self.actor_buffer[agent_id].actions.shape[2:]
                    ),
                    self.actor_buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.actor_buffer[agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
                )

                
                factor = factor * _t2n(
                    getattr(torch, self.action_aggregation)(
                        torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                    ).reshape(
                        self.algo_args["train"]["episode_length"],
                        self.algo_args["train"]["n_rollout_threads"],
                        1,
                    )
                )
                actor_train_infos.append(actor_train_info)

        
        critic_alignment_actors = (
            self.actor
            if getattr(self, "alignment_mode", "separate")
            in CRITIC_ALIGNMENT_MODES
            else None
        )
        critic_train_info = self.critic.train(
            self.critic_buffer,
            self.value_normalizer,
            alignment_actors=critic_alignment_actors,
        )

        if getattr(self, "alignment_mode", "separate") == "no_stop":
            critic_train_info["no_stop_alignment_loss"] = (
                self._joint_no_stop_alignment_update()
            )

        return actor_train_infos, critic_train_info

    def _joint_no_stop_alignment_update(self):
        """Run the optional symmetric no-stop-gradient diagnostic update."""
        share_obs = self.critic_buffer.share_obs[:-1].reshape(
            -1, *self.critic_buffer.share_obs.shape[2:]
        )
        batch_size = share_obs.shape[0]
        mini_batches = self.algo_args["experiment"].get(
            "no_stop_num_mini_batch", 1
        )
        epochs = self.algo_args["experiment"].get("no_stop_epochs", 1)
        if mini_batches < 1 or epochs < 1:
            raise ValueError("no-stop epochs and mini-batches must be positive")
        normalize = self.algo_args["experiment"].get("alignment_normalize", True)
        coefficient = self.algo_args["experiment"].get("alignment_coef", 1.0)
        losses = []

        for _ in range(epochs):
            permutation = torch.randperm(batch_size).numpy()
            for indices in np.array_split(permutation, mini_batches):
                if len(indices) == 0:
                    continue
                obs_batch = share_obs[indices]
                self.critic.critic_optimizer.zero_grad()
                for actor in self.actor:
                    actor.actor_optimizer.zero_grad()

                critic_features = self.critic.critic.get_embedding(
                    obs_batch, None, None
                )
                alignment_losses = [
                    normalized_alignment_loss(
                        actor.actor.get_encoder_features(obs_batch),
                        critic_features,
                        normalize=normalize,
                    )
                    for actor in self.actor
                ]
                loss = torch.stack(alignment_losses).mean() * coefficient
                loss.backward()

                if self.algo_args["algo"]["use_max_grad_norm"]:
                    torch.nn.utils.clip_grad_norm_(
                        unique_trainable_parameters(
                            [self.critic.critic]
                            + [actor.actor for actor in self.actor]
                        ),
                        self.algo_args["algo"]["max_grad_norm"],
                    )
                self.critic.critic_optimizer.step()
                for actor in self.actor:
                    actor.actor_optimizer.step()
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else 0.0
