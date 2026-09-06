# harl/models/value_function_models/v_net.py

import torch
import torch.nn as nn
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_init_method


class VNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(VNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = get_init_method(self.initialization_method)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_sizes[-1], 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """Compute actions from the given inputs.
        Args:
            cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states

    def get_embedding(self, cent_obs, rnn_states, masks):
        """Get the state embedding from the critic network for the auxiliary task.
        Args:
            cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        Returns:
            critic_features: (torch.Tensor) extracted state embedding before value projection.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        # rnn_states = check(rnn_states).to(**self.tpdv)
        # masks = check(masks).to(**self.tpdv)

        # 仅提取基础网络 (CNN/MLP/DeepResNet) 的纯空间特征 (Spatial Features)
        critic_features = self.base(cent_obs)
        
        # =========================================================================
        # === 【核心修复：阻断特征污染】 ===
        # 在计算 Aux Loss 时，Actor 是用自己带有记忆 (RNN) 的特征，
        # 去预测 Critic 视角的全局特征。
        # 
        # 如果在这里让 Critic_features 经过 RNN，由于我们在 Runner 中提供的是 
        # 全 0 的假记忆 (dummy_rnn_states)，Critic 会输出一个被严重污染的 Target。
        # 这将导致 Actor 学到错误的表征，一旦开启 Condition 就会导致策略崩溃。
        #
        # 因此，这里强制跳过 Critic 的 RNN 层，让目标特征回归为纯粹的、无状态的“全局快照”。
        # =========================================================================
        
        # if self.use_naive_recurrent_policy or self.use_recurrent_policy:
        #     critic_features, _ = self.rnn(critic_features, rnn_states, masks)
            
        return critic_features