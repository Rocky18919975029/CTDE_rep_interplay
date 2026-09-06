# harl/models/policy_models/stochastic_policy.py

import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.mlp import DeepResNetBase 
from harl.models.base.rnn import RNNLayer
from harl.models.base.act import ACTLayer
from harl.utils.envs_tools import get_shape_from_obs_space
import torch.nn.functional as F

class StochasticPolicy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model."""
        super(StochasticPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        

        obs_shape = get_shape_from_obs_space(obs_space)
        self.use_deep_resnet = args.get("use_deep_resnet", False)
        
        # === 【细化开关】：拆分为两个独立的控制变量 ===
        self.use_aux_loss = args.get("use_aux_loss", False)
        # === 【新增：概率变分预测开关】 ===
        self.use_prob_aux_loss = args.get("use_prob_aux_loss", False)
        self.condition_on_pred = args.get("condition_on_pred", False)
        
        if len(obs_shape) == 3:
            self.base = CNNBase(args, obs_shape)
        else:
            if self.use_deep_resnet:
                self.base = DeepResNetBase(args, obs_shape)
            else:
                self.base = MLPBase(args, obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        if self.use_aux_loss:
            self.target_embed_dim = self.hidden_sizes[-1]
            self.aux_head = nn.Sequential(
                nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1]),
                nn.ReLU(),
                nn.Linear(self.hidden_sizes[-1], self.target_embed_dim)
            )
            
            # 只有在明确要求 condition 时，才改变 ACTLayer 的输入维度
            if self.condition_on_pred:
                combined_dim = self.hidden_sizes[-1] + self.target_embed_dim
            else:
                combined_dim = self.hidden_sizes[-1]
                
        # === 【新增：初始化概率变分预测头】 ===
        elif self.use_prob_aux_loss:
            self.target_embed_dim = self.hidden_sizes[-1]
            # 预测均值 mu
            self.prob_aux_head_mu = nn.Sequential(
                nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1]),
                nn.ReLU(),
                nn.Linear(self.hidden_sizes[-1], self.target_embed_dim)
            )
            # 预测对数方差 log_var (使用 log 方差防止出现负数方差，保证数值稳定性)
            self.prob_aux_head_logvar = nn.Sequential(
                nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1]),
                nn.ReLU(),
                nn.Linear(self.hidden_sizes[-1], self.target_embed_dim)
            )
            
            # 如果条件化，我们拼接预测的期望均值 mu 作为策略输入条件
            if self.condition_on_pred:
                combined_dim = self.hidden_sizes[-1] + self.target_embed_dim
            else:
                combined_dim = self.hidden_sizes[-1]
        else:
            combined_dim = self.hidden_sizes[-1]
        
        self.use_action_pred = args.get("use_action_pred", False)
        if self.use_action_pred:
            self.total_action_dim = args["total_action_dim"]
            self.action_pred_head = nn.Sequential(
                nn.Linear(self.hidden_sizes[-1], self.hidden_sizes[-1]),
                nn.ReLU(),
                nn.Linear(self.hidden_sizes[-1], self.total_action_dim)
            )

        self.act = ACTLayer(
            action_space,
            combined_dim,  # 根据开关决定是拼接维度还是原始维度
            self.initialization_method,
            self.gain,
            args,
        )

        self.to(device)

    def get_encoder_features(self, obs):
        """Return the pre-head encoder representation used for alignment.

        The decomposition experiment is deliberately feed-forward, so this is
        exactly comparable to the critic-side ``base`` representation.
        """
        obs = check(obs).to(**self.tpdv)
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            raise RuntimeError(
                "Direct encoder alignment currently requires feed-forward policies"
            )
        return self.base(obs)

    def forward(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs."""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_aux_loss:
            predicted_embedding = self.aux_head(actor_features)
            # 只有在明确要求 condition 时，才进行特征拼接
            if self.condition_on_pred:
                # === 【终极防爆修复】：对齐量纲，防止爆炸 ===
                detached_pred = predicted_embedding.detach()
                # 强制归一化，使其标准差为 1，均值为 0，且无学习参数
                normalized_pred = F.layer_norm(detached_pred, [detached_pred.shape[-1]])
                combined_features = torch.cat([actor_features, normalized_pred], dim=-1)
            else:
                combined_features = actor_features
                
        # === 【新增：前向传播中的概率变分特征拼接】 ===
        elif self.use_prob_aux_loss:
            mu = self.prob_aux_head_mu(actor_features)
            # 前向采样阶段仅需拼接预期均值(mu)
            if self.condition_on_pred:
                detached_mu = mu.detach()
                normalized_mu = F.layer_norm(detached_mu, [detached_mu.shape[-1]])
                combined_features = torch.cat([actor_features, normalized_mu], dim=-1)
            else:
                combined_features = actor_features
                
        else:
            combined_features = actor_features

        actions, action_log_probs = self.act(
            combined_features, available_actions, deterministic
        )

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, 
        return_aux=False, return_action_pred=False  # === 【修复 1：增加参数】 ===
    ):
        """Compute action log probability, distribution entropy, action distribution, and auxiliary embedding."""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        predicted_embedding = None
        if self.use_aux_loss:
            predicted_embedding = self.aux_head(actor_features)
            # 同样，只在 condition 时拼接特征去算动作的对数概率
            if self.condition_on_pred:
                # === 【终极防爆修复】：保持前向与评估逻辑一致 ===
                detached_pred = predicted_embedding.detach()
                normalized_pred = F.layer_norm(detached_pred, [detached_pred.shape[-1]])
                combined_features = torch.cat([actor_features, normalized_pred], dim=-1)
            else:
                combined_features = actor_features
                
        # === 【新增：评估动作时的概率变分预测与特征拼接】 ===
        elif self.use_prob_aux_loss:
            mu = self.prob_aux_head_mu(actor_features)
            logvar = self.prob_aux_head_logvar(actor_features)
            
            # 截断 logvar 防止方差过小或过大导致数值溢出（极度重要）
            logvar = torch.clamp(logvar, min=-10.0, max=2.0)
            
            # 将 mu 和 logvar 打包为元组返回，后续在 happo.py 拆包计算 NLL
            predicted_embedding = (mu, logvar)
            
            if self.condition_on_pred:
                detached_mu = mu.detach()
                normalized_mu = F.layer_norm(detached_mu, [detached_mu.shape[-1]])
                combined_features = torch.cat([actor_features, normalized_mu], dim=-1)
            else:
                combined_features = actor_features
                
        else:
            combined_features = actor_features

        # === 【修复 2：删除了错误的拼接，仅做单纯的预测】 ===
        predicted_joint_actions = None
        if self.use_action_pred:
            predicted_joint_actions = self.action_pred_head(actor_features)

        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
            combined_features,
            action,
            available_actions,
            active_masks=active_masks if self.use_policy_active_masks else None,
        )

        # === 【修复 3：动态返回逻辑，完美对齐 Base Actor 的解包】 ===
        if return_aux and return_action_pred:
            return action_log_probs, dist_entropy, action_distribution, predicted_embedding, predicted_joint_actions
        elif return_aux:
            return action_log_probs, dist_entropy, action_distribution, predicted_embedding, None
        elif return_action_pred:
            return action_log_probs, dist_entropy, action_distribution, None, predicted_joint_actions
        else:
            return action_log_probs, dist_entropy, action_distribution
