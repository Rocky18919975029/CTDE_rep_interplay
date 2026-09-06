

"""Replay buffer for QMIX."""

import numpy as np



class QMIXBuffer:
    """Simple transition-level replay buffer for non-recurrent QMIX."""

    def __init__(self, args, obs_space, share_obs_space, act_space, num_agents):
        self.buffer_size = args.get("buffer_size", 50000)
        self.batch_size = args.get("batch_size", 256)
        self.num_agents = num_agents

        self.current_size = 0
        self.ptr = 0

        self.obs_shape = self._get_obs_shape(obs_space[0] if isinstance(obs_space, list) else obs_space)
        self.share_obs_shape = self._get_obs_shape(share_obs_space)

        self.obs = np.zeros(
            (self.buffer_size, self.num_agents, *self.obs_shape),
            dtype=np.float32,
        )
        self.share_obs = np.zeros(
            (self.buffer_size, *self.share_obs_shape),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (self.buffer_size, self.num_agents, 1),
            dtype=np.int64,
        )
        self.rewards = np.zeros(
            (self.buffer_size, self.num_agents, 1),
            dtype=np.float32,
        )
        self.dones = np.zeros(
            (self.buffer_size, self.num_agents, 1),
            dtype=np.float32,
        )
        self.next_obs = np.zeros_like(self.obs)
        self.next_share_obs = np.zeros_like(self.share_obs)

        self.available_actions = None
        self.next_available_actions = None
        self.use_available_actions = False

    def insert(
        self,
        obs,
        share_obs,
        actions,
        rewards,
        dones,
        next_obs,
        next_share_obs,
        available_actions=None,
        next_available_actions=None,
    ):
        """Insert one vectorized transition batch."""
        n_threads = obs.shape[0]

        share_obs = self._normalize_share_obs(share_obs)
        next_share_obs = self._normalize_share_obs(next_share_obs)

        if actions.ndim == 2:
            actions = np.expand_dims(actions, axis=-1)

        if rewards.ndim == 2:
            rewards = np.expand_dims(rewards, axis=-1)

        if dones.ndim == 2:
            dones = np.expand_dims(dones, axis=-1)

        has_available_actions = (
            available_actions is not None and next_available_actions is not None
        )

        if has_available_actions and not self.use_available_actions:
            self._init_available_actions(available_actions.shape[-1])

        for i in range(n_threads):
            idx = self.ptr

            self.obs[idx] = obs[i]
            self.share_obs[idx] = share_obs[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.dones[idx] = dones[i]
            self.next_obs[idx] = next_obs[i]
            self.next_share_obs[idx] = next_share_obs[i]

            if self.use_available_actions and has_available_actions:
                self.available_actions[idx] = available_actions[i]
                self.next_available_actions[idx] = next_available_actions[i]

            self.ptr = (self.ptr + 1) % self.buffer_size
            self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        """Sample a random transition batch."""
        if self.current_size < self.batch_size:
            raise ValueError(
                f"Not enough samples in QMIXBuffer: "
                f"{self.current_size} < {self.batch_size}"
            )

        indices = np.random.randint(0, self.current_size, size=self.batch_size)

        batch = {
            "obs": self.obs[indices],
            "share_obs": self.share_obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "next_obs": self.next_obs[indices],
            "next_share_obs": self.next_share_obs[indices],
        }

        if self.use_available_actions:
            batch["available_actions"] = self.available_actions[indices]
            batch["next_available_actions"] = self.next_available_actions[indices]

        return batch

    def ready(self):
        return self.current_size >= self.batch_size

    def _normalize_share_obs(self, share_obs):
        """Convert share_obs to [n_threads, state_dim]."""
        if share_obs.ndim >= 3:
            return share_obs[:, 0]
        return share_obs

    def _init_available_actions(self, act_dim):
        self.use_available_actions = True

        self.available_actions = np.zeros(
            (self.buffer_size, self.num_agents, act_dim),
            dtype=np.float32,
        )
        self.next_available_actions = np.zeros_like(self.available_actions)

   

    @staticmethod
    def _get_obs_shape(space):
        """
        返回单个 agent obs 或 centralized obs 的 flat shape（tuple），
        支持：
        - int
        - tuple/list
        - SMAC EP 嵌套 list 描述
        - Box / ndarray
        关键点：
        不递归展开多层嵌套，避免内存爆炸。
        """
       
        if isinstance(space, (list, tuple, np.ndarray)):
           
            first = space[0]
           
            if isinstance(first, int):
                return (first,)
           
            if isinstance(first, (list, tuple, np.ndarray)):
                
                return tuple([int(np.prod(first))])
            
            raise ValueError(f"Unsupported element type in observation space list: {type(first)}")

        
        if hasattr(space, "shape"):
            return tuple(space.shape)

       
        if isinstance(space, int):
            return (space,)

        raise ValueError(f"Unsupported observation space: {space}")