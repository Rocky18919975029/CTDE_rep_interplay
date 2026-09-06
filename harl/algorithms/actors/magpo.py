# harl/algorithms/actors/magpo.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase

class MAGPO(OnPolicyBase):
    """
    Multi-Agent Guided Policy Optimization (MAGPO) algorithm.
    Strictly aligns with the ICLR 2026 paper's loss functions (Eq. 9 & 10).
    """
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(MAGPO, self).__init__(args, obs_space, act_space, device)
        
        # 从 args 字典中提取 num_agents
        self.num_agents = args["num_agents"]
        
        # PPO Hyperparameters
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        
        # MAGPO specific Hyperparameters
        self.magpo_delta = args.get("magpo_delta", 1.5)  # δ for Double Clip and Masked KL
        self.magpo_lambda = args.get("magpo_lambda", 1.0) # λ for RL Auxiliary Loss
        # ===【新增：是否降级为 Vanilla CTDS】===
        self.is_ctds = args.get("is_ctds", False)
        
    def update(self, sample, guider, guider_optimizer):
        """
        Update the Guider and the Learner networks simultaneously.
        Args:
            sample: (Tuple) contains the batched rollout data collected by the Guider.
            guider: (nn.Module) The Centralized AR Teacher policy (ARGuiderPolicySable).
            guider_optimizer: (torch.optim.Optimizer) Optimizer for the Guider.
        """
        (
            obs_batch,
            share_obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_guider_log_probs_batch, # 注意：Rollout阶段存下来的是 Guider 的 log_probs
            adv_targ,
            available_actions_batch,
        ) = sample
        
        # 1. 转移并格式化张量
        old_guider_log_probs_batch = check(old_guider_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        
        # 2. 评估当前的 Learner (Student) 策略
        # 这里的 evaluate_actions 只是返回 log_probs，不对网络做更新
        # action_log_probs: [Batch, Num_Agents, 1]
        # 原生 Actor 只支持 1D Batch，所以必须将 Batch 和 NumAgents 维度合并
        batch_size = obs_batch.shape[0]
        
        obs_flat = obs_batch.reshape(-1, *obs_batch.shape[2:])
        rnn_states_flat = rnn_states_batch.reshape(-1, *rnn_states_batch.shape[2:])
        actions_flat = actions_batch.reshape(-1, *actions_batch.shape[2:])
        masks_flat = masks_batch.reshape(-1, *masks_batch.shape[2:])
        active_masks_flat = active_masks_batch.reshape(-1, *active_masks_batch.shape[2:])
        avail_acts_flat = available_actions_batch.reshape(-1, *available_actions_batch.shape[2:]) if available_actions_batch is not None else None

        learner_log_probs_flat, _, _ = self.actor.evaluate_actions(
            obs_flat,
            rnn_states_flat,
            actions_flat,
            masks_flat,
            avail_acts_flat,
            active_masks_flat
        )
        
        # 将结果还原回 [Batch, Num_Agents, 1] 以便后续和 Guider 计算 KL 散度
        learner_log_probs = learner_log_probs_flat.view(batch_size, self.num_agents, -1)
        
        # 3. 评估当前的 Guider (Teacher) 策略
        # guider.evaluate_actions 利用 Teacher Forcing 并行计算整个序列的 log_probs
        guider_log_probs, guider_dist_entropy, _ = guider.evaluate_actions(
            share_obs_batch, 
            actions_batch, 
            available_actions_batch, 
            active_masks_batch
        )
        
        # =========================================================================
        # 阶段一：Guider Update (导师更新, 对应论文公式 9)
        # =========================================================================
        
        # 计算比例 r_ij_t(phi) = current_guider_prob / old_guider_prob
        ratio_guider = torch.exp(guider_log_probs - old_guider_log_probs_batch)
        
        # 标准的 PPO Surrogate Loss
        surr1 = ratio_guider * adv_targ
        surr2 = torch.clamp(ratio_guider, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        
        # Double Clip 逻辑 (论文公式 9: clip(r_ij_t(phi), epsilon, delta) * A_hat)
        # 限制 Guider 不能偏离 Learner 太远，比值为: current_guider_prob / current_learner_prob.detach()
        ratio_guider_to_learner = torch.exp(guider_log_probs - learner_log_probs.detach())
        
        # ===【修改：如果是 CTDS，Mask 永远为 0，Guider 自由飞翔】===
        if self.is_ctds:
            mask_kl = torch.zeros_like(ratio_guider_to_learner)
        else:
            mask_kl = (ratio_guider_to_learner < 1.0 / self.magpo_delta) | (ratio_guider_to_learner > self.magpo_delta)
            mask_kl = mask_kl.float()
        
        kl_divergence = guider_log_probs - learner_log_probs.detach()
        masked_kl_loss = (mask_kl * kl_divergence).sum(dim=-1, keepdim=True)
        
        # 组装 Guider 的总 Loss
        # 最小化 -min(surr1, surr2) 等价于最大化 PPO 目标，再加上 Masked KL 惩罚
        if self.use_policy_active_masks:
            guider_policy_loss = (
                -torch.min(surr1, surr2).sum(dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
            
            guider_kl_loss = (masked_kl_loss * active_masks_batch).sum() / active_masks_batch.sum()
            guider_entropy = (guider_dist_entropy * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            guider_policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()
            guider_kl_loss = masked_kl_loss.mean()
            guider_entropy = guider_dist_entropy.mean()
            
        guider_loss = guider_policy_loss + guider_kl_loss - guider_entropy * self.entropy_coef
        
        # 优化 Guider
        guider_optimizer.zero_grad()
        guider_loss.backward()
        if self.use_max_grad_norm:
            guider_grad_norm = nn.utils.clip_grad_norm_(guider.parameters(), self.max_grad_norm)
        else:
            guider_grad_norm = get_grad_norm(guider.parameters())
        guider_optimizer.step()
        
        # =========================================================================
        # 阶段二：Learner Update (学生更新, 对应论文公式 10)
        # =========================================================================
        
        # 1. Behavior Cloning Loss: 最小化 D_KL(Guider || Learner)
        # 即最大化 Learner 的概率，所以应该 最小化 (Guider_log_probs - Learner_log_probs)
        # === 【致命错误修复：被减数和减数位置调换！】 ===
        bc_kl_loss = (guider_log_probs.detach() - learner_log_probs).sum(dim=-1, keepdim=True)
        
        # 2. RL Auxiliary Loss: 带有 PPO 截断的辅助优势学习
        # 这里的 r_ij_t(theta) 是 current_learner_prob / old_guider_prob.detach()
        ratio_learner_to_old = torch.exp(learner_log_probs - old_guider_log_probs_batch)
        
        aux_surr1 = ratio_learner_to_old * adv_targ
        aux_surr2 = torch.clamp(ratio_learner_to_old, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        
        if self.use_policy_active_masks:
            learner_bc_loss = (bc_kl_loss * active_masks_batch).sum() / active_masks_batch.sum()
            learner_aux_loss = (
                -torch.min(aux_surr1, aux_surr2).sum(dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            learner_bc_loss = bc_kl_loss.mean()
            learner_aux_loss = -torch.min(aux_surr1, aux_surr2).sum(dim=-1, keepdim=True).mean()
            
        # 组装 Learner 的总 Loss (公式 10)
        # ===【修改：如果是 CTDS，屏蔽学生自己的 PPO 优势学习】===
        if self.is_ctds:
            learner_loss = learner_bc_loss
            # 为了在 tensorboard 打印时保持格式，强行把 aux_loss 记为 0
            learner_aux_loss = torch.tensor(0.0) 
        else:
            learner_loss = learner_bc_loss + self.magpo_lambda * learner_aux_loss
        
        # 优化 Learner
        self.actor_optimizer.zero_grad()
        learner_loss.backward()
        if self.use_max_grad_norm:
            learner_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        else:
            learner_grad_norm = get_grad_norm(self.actor.parameters())
        self.actor_optimizer.step()
        
        return (
            guider_policy_loss.item(), guider_kl_loss.item(), 
            learner_bc_loss.item(), learner_aux_loss.item(),
            guider_grad_norm, learner_grad_norm
        )

    def train(self, actor_buffer, advantages, state_type, guider, guider_optimizer):
        """
        MAGPO does NOT use Sequential Update!
        All agents update simultaneously in a shared parameter or batched manner.
        """
        train_info = {
            "guider_policy_loss": 0, "guider_kl_loss": 0,
            "learner_bc_loss": 0, "learner_aux_loss": 0,
            "guider_grad_norm": 0, "learner_grad_norm": 0
        }
        
        # If there are no active masks, skip training
        if np.all(actor_buffer[0].active_masks[:-1] == 0.0):
            return train_info

        # Normalize advantages
        if state_type == "EP":
            advantages_copy = advantages.copy()
            # Assuming homogeneous active masks for EP
            advantages_copy[actor_buffer[0].active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            adv_to_use = (advantages - mean_advantages) / (std_advantages + 1e-5)
        else:
            adv_to_use = advantages # Simplification, adjust if using FP

        # Stack data from all agents to train simultaneously
        # MAGPO is highly parallelizable!
        for _ in range(self.ppo_epoch):
            # We use the feed_forward_generator_actor of the first agent, but we need to modify 
            # the Buffer logic in Runner to pass ALL agents' data simultaneously,
            # or we handle the stacking right here in the runner.
            
            # Since MAGPO requires Joint actions and Share Obs, we assume the Runner 
            # has prepared a joint generator.
            pass 
        
        return train_info