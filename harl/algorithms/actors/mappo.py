
"""MAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase


class MAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize MAPPO algorithm."""
        super(MAPPO, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        
       
        self.aux_loss_coef = args.get("aux_loss_coef", 1.0)

    
    def update(self, sample, target_embedding=None, target_joint_actions=None):
        """Update actor network."""
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

       
        if target_embedding is not None:
            target_embedding = check(target_embedding).to(**self.tpdv).detach()
            if self.args.get("use_target_norm", False):
                target_embedding = F.layer_norm(target_embedding,[target_embedding.shape[-1]])

        use_action_pred = self.args.get("use_action_pred", False)
        use_aux = target_embedding is not None
        
        
        eval_out = self.actor.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            return_aux=use_aux,
            return_action_pred=use_action_pred
        )
        
        action_log_probs = eval_out[0]
        dist_entropy = eval_out[1]
        
        
        predicted_embedding = eval_out[3] if len(eval_out) > 3 else None
        predicted_joint_actions = eval_out[4] if len(eval_out) > 4 else None

        
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

       
        aux_loss = torch.tensor(0.0).to(**self.tpdv)
        if target_embedding is not None and predicted_embedding is not None:
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
                if self.args.get("strict_mask_aux_loss", False) and self.use_policy_active_masks:
                    unreduced_aux_mse = F.mse_loss(predicted_embedding, target_embedding, reduction='none')
                    unreduced_aux_mse = unreduced_aux_mse.mean(dim=-1, keepdim=True)
                    aux_loss = (unreduced_aux_mse * active_masks_batch).sum() / active_masks_batch.sum()
                else:
                    aux_loss = F.mse_loss(predicted_embedding, target_embedding)

       
        action_pred_loss = torch.tensor(0.0).to(**self.tpdv)
        if use_action_pred and target_joint_actions is not None and predicted_joint_actions is not None:
            if self.use_policy_active_masks:
                unreduced_mse = F.mse_loss(predicted_joint_actions, target_joint_actions.detach(), reduction='none')
                unreduced_mse = unreduced_mse.mean(dim=-1, keepdim=True)
                action_pred_loss = (unreduced_mse * active_masks_batch).sum() / active_masks_batch.sum()
            else:
                action_pred_loss = F.mse_loss(predicted_joint_actions, target_joint_actions.detach())

        action_pred_coef = self.args.get("action_pred_coef", 1.0)
        
       
        policy_loss = policy_action_loss - dist_entropy * self.entropy_coef + action_pred_loss * action_pred_coef + aux_loss * self.aux_loss_coef

        self.actor_optimizer.zero_grad()
        policy_loss.backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        
        return policy_loss, dist_entropy, actor_grad_norm, imp_weights, action_pred_loss, aux_loss

    def train(self, actor_buffer, advantages, state_type):
        pass 

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        pass