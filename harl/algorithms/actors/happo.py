

"""HAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase


class HAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize HAPPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(HAPPO, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        
       
        self.aux_loss_coef = args.get("aux_loss_coef", 1.0)

   
    def update(self, sample, target_embedding=None, target_joint_actions=None):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
            target_embedding: (torch.Tensor, optional) Target embedding from Critic for aux loss.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
            aux_loss: (torch.Tensor) auxiliary prediction loss.
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        
       
        if target_embedding is not None:
            target_embedding = check(target_embedding).to(**self.tpdv).detach()
            if self.args.get("use_target_norm", True):
                target_embedding = F.layer_norm(target_embedding,[target_embedding.shape[-1]])


        
        action_log_probs, dist_entropy, _, predicted_embedding, predicted_joint_actions = self.actor.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            return_aux=True, 
            return_action_pred=True 
        )

        # actor update
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

       
        aux_loss = torch.tensor(0.0).to(**self.tpdv)
        if target_embedding is not None:
           
            if self.args.get("use_prob_aux_loss", False):
               
                mu, logvar = predicted_embedding
                var = torch.exp(logvar)
                
                nll_loss = 0.5 * ((target_embedding - mu) ** 2) / var + 0.5 * logvar
                
                if self.use_policy_active_masks:
                    nll_loss = nll_loss.mean(dim=-1, keepdim=True)
                    aux_loss = (nll_loss * active_masks_batch).sum() / active_masks_batch.sum()
                else:
                    aux_loss = nll_loss.mean()
                    
            
            else:
                
                use_huber = self.args.get("use_huber_aux_loss", False)
                
                if self.args.get("strict_mask_aux_loss", True) and self.use_policy_active_masks:
                    unreduced_aux_mse = F.mse_loss(predicted_embedding, target_embedding, reduction='none')
                    unreduced_aux_mse = unreduced_aux_mse.mean(dim=-1, keepdim=True)
                    aux_loss = (unreduced_aux_mse * active_masks_batch).sum() / active_masks_batch.sum()
                else:
                    if use_huber:
                        aux_loss = F.smooth_l1_loss(predicted_embedding, target_embedding)
                    else:
                        aux_loss = F.mse_loss(predicted_embedding, target_embedding)

        
        action_pred_loss = torch.tensor(0.0).to(**self.tpdv)
        if target_joint_actions is not None:
           
            if self.use_policy_active_masks:
                unreduced_mse = F.mse_loss(predicted_joint_actions, target_joint_actions.detach(), reduction='none')
                unreduced_mse = unreduced_mse.mean(dim=-1, keepdim=True)
                action_pred_loss = (unreduced_mse * active_masks_batch).sum() / active_masks_batch.sum()
            else:
                action_pred_loss = F.mse_loss(predicted_joint_actions, target_joint_actions.detach())

        action_pred_coef = self.args.get("action_pred_coef", 1.0)
        
        loss = policy_action_loss - dist_entropy * self.entropy_coef + aux_loss * self.aux_loss_coef + action_pred_loss * action_pred_coef

        self.actor_optimizer.zero_grad()

        loss.backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        
        if getattr(self.args, "use_aux_clip", False) and aux_loss is not None and aux_loss.item() > 0:
            aux_loss_scaled = aux_loss * self.aux_loss_coef
            aux_loss_scaled.backward(retain_graph=True)
            max_aux_grad_norm = getattr(self.args, "max_aux_grad_norm", self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_aux_grad_norm)

        self.actor_optimizer.step()

        return policy_action_loss, dist_entropy, actor_grad_norm, imp_weights, aux_loss, action_pred_loss

    
    def train(self, actor_buffer, advantages, state_type):
        """
        [NOTE]: This method is kept for compatibility. 
        However, to pass 'target_embedding', we highly recommend 
        writing the mini-batch loop directly in the Runner 
        and calling `self.update(sample, target_embedding)`.
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0
        train_info["aux_loss"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
               
                policy_loss, dist_entropy, actor_grad_norm, imp_weights, aux_loss, action_pred_loss = self.update(
                    sample, target_embedding=None, target_joint_actions=None
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
                train_info["aux_loss"] += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
                
                train_info["action_pred_loss"] = train_info.get("action_pred_loss", 0) + (action_pred_loss.item() if isinstance(action_pred_loss, torch.Tensor) else action_pred_loss)

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info