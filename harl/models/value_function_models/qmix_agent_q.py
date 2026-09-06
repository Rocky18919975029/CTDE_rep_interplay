"""Agent Q-network for QMIX."""

import torch
import torch.nn as nn

from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_init_method


class AgentQNet(nn.Module):
    """Local agent Q-network.

    It maps each agent's local observation to Q-values over discrete actions.
    """

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(AgentQNet, self).__init__()

        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.use_naive_recurrent_policy = args.get("use_naive_recurrent_policy", False)
        self.use_recurrent_policy = args.get("use_recurrent_policy", False)
        self.recurrent_n = args.get("recurrent_n", 1)
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        # QMIX only supports discrete actions.
        # For gym.spaces.Discrete, the number of Q outputs must be act_space.n.
        if hasattr(act_space, "n"):
            self.act_dim = act_space.n
        else:
            raise ValueError(
                f"QMIX only supports Discrete action spaces, but got {act_space}"
            )

        init_method = get_init_method(self.initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.q_out = init_(nn.Linear(self.hidden_sizes[-1], self.act_dim))

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None):
        """Compute Q-values.

        Args:
            obs: local observations.
            rnn_states: recurrent states.
            masks: masks for RNN reset.
            available_actions: optional binary action mask.

        Returns:
            q_values: [batch, act_dim]
            rnn_states: updated recurrent states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            features, rnn_states = self.rnn(features, rnn_states, masks)

        q_values = self.q_out(features)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            q_values = q_values.masked_fill(available_actions <= 0.0, -1e10)

        return q_values, rnn_states

    @torch.no_grad()
    def act(self, obs, rnn_states, masks, available_actions=None, epsilon=0.0):
        """Epsilon-greedy action selection.

        Returns:
            actions: [batch, 1]
            rnn_states: updated recurrent states.
        """
        q_values, rnn_states = self.forward(
            obs, rnn_states, masks, available_actions
        )

        greedy_actions = q_values.argmax(dim=-1, keepdim=True)

        if epsilon <= 0.0:
            return greedy_actions, rnn_states

        batch_size = q_values.shape[0]

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv).float()
            random_actions = torch.multinomial(available_actions, 1)
        else:
            random_actions = torch.randint(
                low=0,
                high=self.act_dim,
                size=(batch_size, 1),
                device=q_values.device,
            )

        choose_random = torch.rand(batch_size, 1, device=q_values.device) < epsilon
        actions = torch.where(choose_random, random_actions, greedy_actions)

        return actions, rnn_states