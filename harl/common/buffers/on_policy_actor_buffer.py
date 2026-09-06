

"""On-policy buffer for actor."""

import torch
import numpy as np
from harl.utils.trans_tools import _flatten, _sa_cast
from harl.utils.envs_tools import get_shape_from_obs_space, get_shape_from_act_space


class OnPolicyActorBuffer:
    """On-policy buffer for actor data storage."""

    def __init__(self, args, obs_space, act_space):
        """Initialize on-policy actor buffer.
        Args:
            args: (dict) arguments
            obs_space: (gym.Space or list) observation space
            act_space: (gym.Space) action space
        """
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.hidden_sizes = args["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = args["recurrent_n"]

        obs_shape = get_shape_from_obs_space(obs_space)

        if isinstance(obs_shape[-1], list):
            obs_shape = obs_shape[:1]

       
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape),
            dtype=np.float32,
        )

        
        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        
        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (self.episode_length + 1, self.n_rollout_threads, act_space.n),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

       
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32
        )

       
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32
        )

        
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        
        self.active_masks = np.ones_like(self.masks)

        self.factor = None

        self.step = 0

    def update_factor(self, factor):
        """Save factor for this actor."""
        self.factor = factor.copy()

    def insert(
        self,
        obs,
        rnn_states,
        actions,
        action_log_probs,
        masks,
        active_masks=None,
        available_actions=None,
    ):
        """Insert data into actor buffer."""
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """After an update, copy the data at the last step to the first position of the buffer."""
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def feed_forward_generator_actor(
        self, advantages, actor_num_mini_batch=None, mini_batch_size=None, share_obs=None, joint_actions=None, prl_target="next",  
    ):
        """Training data generator for actor that uses MLP network."""

       
        episode_length, n_rollout_threads = self.actions.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        if mini_batch_size is None:
            assert batch_size >= actor_num_mini_batch, (
                f"The number of processes ({n_rollout_threads}) "
                f"* the number of steps ({episode_length}) = {n_rollout_threads * episode_length}"
                f" is required to be greater than or equal to the number of actor mini batches ({actor_num_mini_batch})."
            )
            mini_batch_size = batch_size // actor_num_mini_batch

       
        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(actor_num_mini_batch)
        ]

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])  
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(
                -1, self.available_actions.shape[-1]
            )
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        if self.factor is not None:
            factor = self.factor.reshape(-1, self.factor.shape[-1])
        advantages = advantages.reshape(-1, 1)
        
       
        if share_obs is not None:
            if prl_target == "next":
               
                target_share_obs = share_obs[1:]
            elif prl_target == "current":
                
                target_share_obs = share_obs[:-1]
            else:
                raise ValueError(f"Unsupported prl_target: {prl_target}")

            target_share_obs = target_share_obs.reshape(-1, *share_obs.shape[2:])
        
        if joint_actions is not None:
            
            flat_joint_actions = joint_actions.reshape(-1, joint_actions.shape[-1])
        for indices in sampler:
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield_tuple = (
                obs_batch, rnn_states_batch, actions_batch, masks_batch, 
                active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch
            )
            
            if self.factor is not None:
                yield_tuple += (factor[indices],)
                
            
            if share_obs is not None:
                yield_tuple += (target_share_obs[indices],)

           
            if joint_actions is not None:
                yield_tuple += (flat_joint_actions[indices],)

            yield yield_tuple

    def naive_recurrent_generator_actor(self, advantages, actor_num_mini_batch, share_obs=None, joint_actions=None, prl_target="next"):
        """Training data generator for actor that uses RNN network.
        This generator does not split the trajectories into chunks, 
        and therefore maybe less efficient than the recurrent_generator_actor in training.
        """

        
        n_rollout_threads = self.actions.shape[1]
        assert n_rollout_threads >= actor_num_mini_batch, (
            f"The number of processes ({n_rollout_threads}) "
            f"has to be greater than or equal to the number of "
            f"mini batches ({actor_num_mini_batch})."
        )
        num_envs_per_batch = n_rollout_threads // actor_num_mini_batch

       
        perm = torch.randperm(n_rollout_threads).numpy()

        T, N = self.episode_length, num_envs_per_batch

        
        for batch_id in range(actor_num_mini_batch):
            start_id = batch_id * num_envs_per_batch
            ids = perm[start_id : start_id + num_envs_per_batch]
            obs_batch = _flatten(T, N, self.obs[:-1, ids])
            actions_batch = _flatten(T, N, self.actions[:, ids])
            masks_batch = _flatten(T, N, self.masks[:-1, ids])
            active_masks_batch = _flatten(T, N, self.active_masks[:-1, ids])
            old_action_log_probs_batch = _flatten(T, N, self.action_log_probs[:, ids])
            adv_targ = _flatten(T, N, advantages[:, ids])
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, self.available_actions[:-1, ids])
            else:
                available_actions_batch = None
            rnn_states_batch = self.rnn_states[0, ids]
            
            yield_tuple = (
                obs_batch, rnn_states_batch, actions_batch, masks_batch, 
                active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch
            )

            if self.factor is not None:
                factor_batch = _flatten(T, N, self.factor[:, ids])
                yield_tuple += (factor_batch,)

            if share_obs is not None:
               
                if prl_target == "next":
                    target_share_obs = share_obs[1:]
                elif prl_target == "current":
                    target_share_obs = share_obs[:-1]
                else:
                    raise ValueError(f"Unsupported prl_target: {prl_target}")

                target_share_obs_batch = _flatten(T, N, target_share_obs[:, ids])
                yield_tuple += (target_share_obs_batch,)
                
           
            if joint_actions is not None:
               
                joint_actions_batch = _flatten(T, N, joint_actions[:, ids])
                yield_tuple += (joint_actions_batch,)

            yield yield_tuple

    def recurrent_generator_actor(self, advantages, actor_num_mini_batch, data_chunk_length, share_obs=None, joint_actions=None, prl_target="next"):
        """Training data generator for actor that uses RNN network.
        This generator splits the trajectories into chunks of length data_chunk_length, 
        and therefore maybe more efficient than the naive_recurrent_generator_actor in training.
        """

       
        episode_length, n_rollout_threads = self.actions.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // actor_num_mini_batch

        assert episode_length % data_chunk_length == 0, (
            f"episode length ({episode_length}) must be a multiple of data chunk length ({data_chunk_length})."
        )
        assert data_chunks >= 2, "need larger batch size"

        
        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(actor_num_mini_batch)
        ]

        if len(self.obs.shape) > 3:
            obs = self.obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
        else:
            obs = _sa_cast(self.obs[:-1])
        actions = _sa_cast(self.actions)
        action_log_probs = _sa_cast(self.action_log_probs)
        advantages = _sa_cast(advantages)
        masks = _sa_cast(self.masks[:-1])
        active_masks = _sa_cast(self.active_masks[:-1])
        if self.factor is not None:
            factor = _sa_cast(self.factor)
        rnn_states = (
            self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        )
        if self.available_actions is not None:
            available_actions = _sa_cast(self.available_actions[:-1])

        if share_obs is not None:
           
            if prl_target == "next":
                target_share_obs_raw = share_obs[1:]
            elif prl_target == "current":
                target_share_obs_raw = share_obs[:-1]
            else:
                raise ValueError(f"Unsupported prl_target: {prl_target}")

            if len(share_obs.shape) > 3:
                target_share_obs = target_share_obs_raw.transpose(1, 0, 2, 3, 4).reshape(
                    -1, *share_obs.shape[2:]
                )
            else:
                target_share_obs = _sa_cast(target_share_obs_raw)
                
        
        if joint_actions is not None:
            joint_actions = _sa_cast(joint_actions)

       
        for indices in sampler:
            obs_batch = []
            rnn_states_batch = []
            actions_batch = []
            available_actions_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            factor_batch = []
            
            target_share_obs_batch = [] if share_obs is not None else None
            
            joint_actions_batch = [] if joint_actions is not None else None

            for index in indices:
                ind = index * data_chunk_length
                obs_batch.append(obs[ind : ind + data_chunk_length])
                actions_batch.append(actions[ind : ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind : ind + data_chunk_length])
                adv_targ.append(advantages[ind : ind + data_chunk_length])
                rnn_states_batch.append(rnn_states[ind])  
                if self.factor is not None:
                    factor_batch.append(factor[ind : ind + data_chunk_length])
                    
                if share_obs is not None:
                    target_share_obs_batch.append(target_share_obs[ind : ind + data_chunk_length])
                    
                
                if joint_actions is not None:
                    joint_actions_batch.append(joint_actions[ind : ind + data_chunk_length])
            
            L, N = data_chunk_length, mini_batch_size
            
            obs_batch = np.stack(obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            if self.factor is not None:
                factor_batch = np.stack(factor_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)
            
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])
            
            if share_obs is not None:
                target_share_obs_batch = np.stack(target_share_obs_batch, axis=1)
                
            
            if joint_actions is not None:
                joint_actions_batch = np.stack(joint_actions_batch, axis=1)

           
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(L, N, factor_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)
            
            if share_obs is not None:
                target_share_obs_batch = _flatten(L, N, target_share_obs_batch)
                
            
            if joint_actions is not None:
                joint_actions_batch = _flatten(L, N, joint_actions_batch)

            yield_tuple = (
                obs_batch, rnn_states_batch, actions_batch, masks_batch, 
                active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch
            )
            
            if self.factor is not None:
                yield_tuple += (factor_batch,)
                
            if share_obs is not None:
                yield_tuple += (target_share_obs_batch,)
                
            
            if joint_actions is not None:
                yield_tuple += (joint_actions_batch,)

            yield yield_tuple