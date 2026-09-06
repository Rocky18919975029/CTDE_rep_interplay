

"""Runner for on-policy MA algorithms."""
import numpy as np
import torch
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyMARunner(OnPolicyBaseRunner):
    """Runner for on-policy MA algorithms."""

    def train(self):
        """Training procedure for MAPPO, TAPPO, and MAPPO-PRL."""
        actor_train_infos =[]

       
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

       
        use_action_pred = self.algo_args["algo"].get("use_action_pred", False)
        use_aux_loss = self.algo_args["algo"].get("use_aux_loss", False)
        use_prob_aux_loss = self.algo_args["algo"].get("use_prob_aux_loss", False)
        need_target_embed = use_aux_loss or use_prob_aux_loss
        
        if use_action_pred:
            all_actions =[self.actor_buffer[i].actions for i in range(self.num_agents)]
            joint_actions_input = np.concatenate(all_actions, axis=-1)
        else:
            joint_actions_input = None

        if self.share_param:
           
            if self.state_type == "EP":
                advantages_ori_list = []
                advantages_copy_list =[]
                for agent_id in range(self.num_agents):
                    advantages_ori_list.append(advantages.copy())
                    adv_copy = advantages.copy()
                    adv_copy[self.actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                    advantages_copy_list.append(adv_copy)
                advantages_ori_tensor = np.array(advantages_ori_list)
                advantages_copy_tensor = np.array(advantages_copy_list)
                mean_advantages = np.nanmean(advantages_copy_tensor)
                std_advantages = np.nanstd(advantages_copy_tensor)
                normalized_advantages = (advantages_ori_tensor - mean_advantages) / (std_advantages + 1e-5)
                advantages_list = [normalized_advantages[i] for i in range(self.num_agents)]
            elif self.state_type == "FP":
                advantages_list = [advantages[:, :, i] for i in range(self.num_agents)]

            train_info = {
                "policy_loss": 0, "dist_entropy": 0, 
                "actor_grad_norm": 0, "ratio": 0, 
                "action_pred_loss": 0, "aux_loss": 0
            }
            
            actor_num_mini_batch = self.actor[0].actor_num_mini_batch
            ppo_epoch = self.actor[0].ppo_epoch

            for _ in range(ppo_epoch):
                data_generators =[]
                share_obs_to_pass = self.critic_buffer.share_obs if need_target_embed else None
                
                for agent_id in range(self.num_agents):
                    if self.actor[0].use_recurrent_policy:
                        data_generator = self.actor_buffer[agent_id].recurrent_generator_actor(
                            advantages_list[agent_id], actor_num_mini_batch, self.actor[0].data_chunk_length,
                            share_obs=share_obs_to_pass, joint_actions=joint_actions_input
                        )
                    elif self.actor[0].use_naive_recurrent_policy:
                        data_generator = self.actor_buffer[agent_id].naive_recurrent_generator_actor(
                            advantages_list[agent_id], actor_num_mini_batch,
                            share_obs=share_obs_to_pass, joint_actions=joint_actions_input
                        )
                    else:
                        data_generator = self.actor_buffer[agent_id].feed_forward_generator_actor(
                            advantages_list[agent_id], actor_num_mini_batch,
                            share_obs=share_obs_to_pass, joint_actions=joint_actions_input
                        )
                    data_generators.append(data_generator)

                for _ in range(actor_num_mini_batch):
                    
                    base_len = 8
                    has_share = 1 if need_target_embed else 0
                    has_joint = 1 if use_action_pred else 0
                    tuple_len = base_len + has_share + has_joint
                    
                    batches = [[] for _ in range(tuple_len)]
                    
                    for generator in data_generators:
                        sample = list(next(generator))
                        for i in range(tuple_len):
                            batches[i].append(sample[i])
                            
                   
                    for i in range(7):
                        batches[i] = np.concatenate(batches[i], axis=0)
                    
                   
                    if batches[7][0] is None:
                        batches[7] = None
                    else:
                        batches[7] = np.concatenate(batches[7], axis=0)
                        
                    target_embedding = None
                    target_joint_actions = None
                    current_idx = 8
                    
                   
                    if need_target_embed:
                        next_share_obs_batch = np.concatenate(batches[current_idx], axis=0)
                        current_idx += 1
                        
                        with torch.no_grad():
                            b_size = next_share_obs_batch.shape[0]
                            rec_n = self.critic.critic.recurrent_n
                            rnn_hid_size = self.critic.critic.hidden_sizes[-1]
                            
                            dummy_rnn = np.zeros((b_size, rec_n, rnn_hid_size), dtype=np.float32)
                            dummy_masks = np.ones((b_size, 1), dtype=np.float32)
                            
                            target_embedding = self.critic.critic.get_embedding(
                                next_share_obs_batch, dummy_rnn, dummy_masks
                            )
                            
                   
                    if use_action_pred:
                        batches_joint = np.concatenate(batches[current_idx], axis=0)
                        target_joint_actions = torch.tensor(batches_joint).to(self.device)

                    
                    sample_to_update = tuple(batches[:8])

                    policy_loss, dist_entropy, actor_grad_norm, imp_weights, action_pred_loss, aux_loss = self.actor[0].update(
                        sample_to_update, target_embedding, target_joint_actions
                    )

                    train_info["policy_loss"] += policy_loss.item()
                    train_info["dist_entropy"] += dist_entropy.item()
                    train_info["actor_grad_norm"] += actor_grad_norm
                    train_info["ratio"] += imp_weights.mean()
                    train_info["action_pred_loss"] += action_pred_loss.item() if isinstance(action_pred_loss, torch.Tensor) else action_pred_loss
                    train_info["aux_loss"] += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

            num_updates = ppo_epoch * actor_num_mini_batch
            for k in train_info.keys():
                train_info[k] /= num_updates

            for _ in range(self.num_agents):
                actor_train_infos.append(train_info)

        else:
           
            for agent_id in range(self.num_agents):
                train_info = {
                    "policy_loss": 0, "dist_entropy": 0, 
                    "actor_grad_norm": 0, "ratio": 0, 
                    "action_pred_loss": 0, "aux_loss": 0
                }
                
                if np.all(self.actor_buffer[agent_id].active_masks[:-1] == 0.0):
                    actor_train_infos.append(train_info)
                    continue

                if self.state_type == "EP":
                    adv_to_use = advantages.copy()
                    adv_copy = adv_to_use.copy()
                    adv_copy[self.actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                    adv_to_use = (adv_to_use - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5)
                elif self.state_type == "FP":
                    adv_to_use = advantages[:, :, agent_id].copy()
                    
                actor_num_mini_batch = self.actor[agent_id].actor_num_mini_batch
                ppo_epoch = self.actor[agent_id].ppo_epoch
                share_obs_to_pass = self.critic_buffer.share_obs if need_target_embed else None

                for _ in range(ppo_epoch):
                    if self.actor[agent_id].use_recurrent_policy:
                        data_generator = self.actor_buffer[agent_id].recurrent_generator_actor(
                            adv_to_use, actor_num_mini_batch, self.actor[agent_id].data_chunk_length,
                            share_obs=share_obs_to_pass, joint_actions=joint_actions_input
                        )
                    elif self.actor[agent_id].use_naive_recurrent_policy:
                        data_generator = self.actor_buffer[agent_id].naive_recurrent_generator_actor(
                            adv_to_use, actor_num_mini_batch,
                            share_obs=share_obs_to_pass, joint_actions=joint_actions_input
                        )
                    else:
                        data_generator = self.actor_buffer[agent_id].feed_forward_generator_actor(
                            adv_to_use, actor_num_mini_batch,
                            share_obs=share_obs_to_pass, joint_actions=joint_actions_input
                        )

                    for sample in data_generator:
                        current_sample = list(sample)
                        target_joint_actions = None
                        target_embedding = None
                        
                        if use_action_pred:
                            target_joint_actions = torch.tensor(current_sample.pop()).to(self.device)
                            
                        if need_target_embed:
                            next_share_obs_batch = current_sample.pop()
                            if next_share_obs_batch is not None:
                                with torch.no_grad():
                                    b_size = next_share_obs_batch.shape[0]
                                    rec_n = self.critic.critic.recurrent_n
                                    rnn_hid_size = self.critic.critic.hidden_sizes[-1]
                                    
                                    dummy_rnn = np.zeros((b_size, rec_n, rnn_hid_size), dtype=np.float32)
                                    dummy_masks = np.ones((b_size, 1), dtype=np.float32)
                                    
                                    target_embedding = self.critic.critic.get_embedding(
                                        next_share_obs_batch, dummy_rnn, dummy_masks
                                    )
                            
                        policy_loss, dist_entropy, actor_grad_norm, imp_weights, action_pred_loss, aux_loss = self.actor[agent_id].update(
                            tuple(current_sample), target_embedding, target_joint_actions
                        )

                        train_info["policy_loss"] += policy_loss.item()
                        train_info["dist_entropy"] += dist_entropy.item()
                        train_info["actor_grad_norm"] += actor_grad_norm
                        train_info["ratio"] += imp_weights.mean()
                        train_info["action_pred_loss"] += action_pred_loss.item() if isinstance(action_pred_loss, torch.Tensor) else action_pred_loss
                        train_info["aux_loss"] += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

                num_updates = ppo_epoch * actor_num_mini_batch
                for k in train_info.keys():
                    train_info[k] /= num_updates
                actor_train_infos.append(train_info)

        
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info