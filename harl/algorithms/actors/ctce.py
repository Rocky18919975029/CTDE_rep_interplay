# harl/algorithms/actors/ctce.py
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase

class CTCE(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(CTCE, self).__init__(args, obs_space, act_space, device)
        self.num_agents = args["num_agents"]
        self.clip_param = args["clip_param"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        self.use_policy_active_masks = args.get("use_policy_active_masks", True)

    def update_ctce(self, sample, guider, guider_optimizer):
        """Pure PPO update for Sable/MAT Guider."""
        (
            _, share_obs_batch, _, actions_batch, _, active_masks_batch,
            old_guider_log_probs_batch, adv_targ, available_actions_batch,
        ) = sample
        
        old_guider_log_probs_batch = check(old_guider_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        
        # 1. 算出当前 Guider 的分布
        guider_log_probs, guider_dist_entropy, _ = guider.evaluate_actions(
            share_obs_batch, actions_batch, available_actions_batch, active_masks_batch
        )
        
        # 2. 算 PPO Loss
        ratio = torch.exp(guider_log_probs - old_guider_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        
        if self.use_policy_active_masks:
            policy_loss = (-torch.min(surr1, surr2).sum(dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
            entropy = (guider_dist_entropy * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()
            entropy = guider_dist_entropy.mean()
            
        loss = policy_loss - entropy * self.entropy_coef
        
        # 3. 优化网络
        guider_optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(guider.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_grad_norm(guider.parameters())
        guider_optimizer.step()
        
        return policy_loss.item(), entropy.item(), grad_norm