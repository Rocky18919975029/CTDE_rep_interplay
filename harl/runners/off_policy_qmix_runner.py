# harl/runners/off_policy_qmix_runner.py

"""Off-policy runner for QMIX."""

import os
import time
import numpy as np
import torch
import setproctitle
from tqdm import tqdm

from harl.algorithms.actors.qmix import QMIX
from harl.common.buffers.qmix_buffer import QMIXBuffer
from harl.envs import LOGGER_REGISTRY
from harl.utils.configs_tools import init_dir, save_config
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.trans_tools import _t2n


class OffPolicyQMIXRunner:
    """Independent runner for QMIX."""

    def __init__(self, args, algo_args, env_args):
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.state_type = env_args.get("state_type", "EP")
        if self.state_type != "EP":
            raise NotImplementedError(
                "This QMIX runner currently supports EP centralized state only."
            )

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])

        if not algo_args["render"]["use_render"]:
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir)

        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        if algo_args["render"]["use_render"]:
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
            self.eval_envs = None
        else:
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )

        self.num_agents = get_num_agents(args["env"], env_args, self.envs)

        print("share_observation_space:", self.envs.share_observation_space)
        print("observation_space:", self.envs.observation_space)
        print("action_space:", self.envs.action_space)
        print("num_agents:", self.num_agents)

        merged_args = {**algo_args["model"], **algo_args["algo"], **algo_args["train"]}
        merged_args["num_agents"] = self.num_agents

        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"].get("recurrent_n", 1)

        self.policy = QMIX(
            merged_args,
            obs_space=self.envs.observation_space[0],
            share_obs_space=self.envs.share_observation_space[0],
            act_space=self.envs.action_space[0],
            device=self.device,
        )

        if not algo_args["render"]["use_render"]:
            self.buffer = QMIXBuffer(
                {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                obs_space=self.envs.observation_space,
                share_obs_space=self.envs.share_observation_space[0],
                act_space=self.envs.action_space,
                num_agents=self.num_agents,
            )

            self.logger = LOGGER_REGISTRY[args["env"]](
                args,
                algo_args,
                env_args,
                self.num_agents,
                self.writter,
                self.run_dir,
            )

        self.start_episode = 1
        if algo_args["train"]["model_dir"] is not None:
            self.restore()

    def run(self):
        if self.algo_args["render"]["use_render"]:
            self.render()
            return

        print("start running QMIX")

        obs, share_obs, available_actions = self.envs.reset()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        self.logger.init(episodes)

        pbar = tqdm(total=episodes, initial=self.start_episode - 1, desc="QMIX Training")
        train_start_time = time.time()

        rnn_states = np.zeros(
            (
                self.algo_args["train"]["n_rollout_threads"],
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        masks = np.ones(
            (
                self.algo_args["train"]["n_rollout_threads"],
                self.num_agents,
                1,
            ),
            dtype=np.float32,
        )

        # === added logger state: track finished training episode rewards ===
        train_episode_rewards = np.zeros(
            (self.algo_args["train"]["n_rollout_threads"], 1),
            dtype=np.float32,
        )

        for episode in range(self.start_episode, episodes + 1):
            if self.algo_args["train"].get("use_linear_lr_decay", False):
                self.policy.lr_decay(episode, episodes)

            self.logger.episode_init(episode)

            epsilon = self._get_epsilon(episode, episodes)

            for step in range(self.algo_args["train"]["episode_length"]):
                self.policy.prep_rollout()

                actions, new_rnn_states = self.collect(
                    obs,
                    rnn_states,
                    masks,
                    available_actions,
                    epsilon,
                )

                (
                    next_obs,
                    next_share_obs,
                    rewards,
                    dones,
                    infos,
                    next_available_actions,
                ) = self.envs.step(actions)

                current_steps = (
                    episode
                    * self.algo_args["train"]["episode_length"]
                    * self.algo_args["train"]["n_rollout_threads"]
                    + step * self.algo_args["train"]["n_rollout_threads"]
                )

                # === added train reward logger ===
                dones_env = np.all(dones, axis=1)

                # GRF/HARL reward shape is usually [n_threads, n_agents, 1].
                # EP setting uses shared reward, so agent 0 is enough.
                if rewards.ndim == 3:
                    train_episode_rewards += rewards[:, 0]
                elif rewards.ndim == 2:
                    train_episode_rewards += rewards[:, 0:1]
                else:
                    train_episode_rewards += rewards.reshape(-1, 1)

                if np.any(dones_env):
                    finished_rewards = train_episode_rewards[dones_env]

                    self.writter.add_scalar(
                        "train_episode_rewards",
                        float(np.mean(finished_rewards)),
                        current_steps,
                    )
                    self.writter.add_scalar(
                        "train_episode_rewards_max",
                        float(np.max(finished_rewards)),
                        current_steps,
                    )

                    train_episode_rewards[dones_env] = 0.0

                buffer_actions = actions
                if buffer_actions.ndim == 2:
                    buffer_actions = np.expand_dims(buffer_actions, axis=-1)

                has_available_actions = (
                    available_actions is not None and available_actions[0] is not None
                )
                has_next_available_actions = (
                    next_available_actions is not None
                    and next_available_actions[0] is not None
                )

                self.buffer.insert(
                    obs=obs,
                    share_obs=share_obs,
                    actions=buffer_actions,
                    rewards=rewards,
                    dones=dones,
                    next_obs=next_obs,
                    next_share_obs=next_share_obs,
                    available_actions=available_actions
                    if has_available_actions and has_next_available_actions
                    else None,
                    next_available_actions=next_available_actions
                    if has_available_actions and has_next_available_actions
                    else None,
                )

                new_rnn_states[dones_env == True] = np.zeros(
                    (
                        (dones_env == True).sum(),
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )

                masks = np.ones(
                    (
                        self.algo_args["train"]["n_rollout_threads"],
                        self.num_agents,
                        1,
                    ),
                    dtype=np.float32,
                )
                masks[dones_env == True] = np.zeros(
                    ((dones_env == True).sum(), self.num_agents, 1),
                    dtype=np.float32,
                )

                obs = next_obs
                share_obs = next_share_obs
                available_actions = next_available_actions
                rnn_states = new_rnn_states

                if self.buffer.ready():
                    self.policy.prep_training()
                    for _ in range(self.algo_args["train"].get("train_interval", 1)):
                        train_info = self.policy.update(self.buffer.sample())
                        self._log_train_info(train_info, episode, step)

            # === added scalar logger for epsilon ===
            episode_steps = (
                episode
                * self.algo_args["train"]["episode_length"]
                * self.algo_args["train"]["n_rollout_threads"]
            )
            self.writter.add_scalar("epsilon", epsilon, episode_steps)

            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.eval()
                self.save(episode)

            pbar.set_postfix({"epsilon": f"{epsilon:.3f}"})
            pbar.update(1)

        pbar.close()

        total_time_sec = time.time() - train_start_time
        time_log_path = os.path.join(self.run_dir, "training_time.txt")
        with open(time_log_path, "w") as f:
            f.write(f"Start Episode: {self.start_episode}\n")
            f.write(f"End Episode: {episodes}\n")
            f.write(f"Total Training Time (Seconds): {total_time_sec:.2f}\n")

    @torch.no_grad()
    def collect(self, obs, rnn_states, masks, available_actions, epsilon):
        """Collect epsilon-greedy actions.

        Returns:
            actions_np: [n_threads, n_agents]
            new_rnn_states_np: [n_threads, n_agents, recurrent_n, hidden_dim]
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        rnn_t = torch.tensor(rnn_states, dtype=torch.float32, device=self.device)
        masks_t = torch.tensor(masks, dtype=torch.float32, device=self.device)

        if available_actions is not None and available_actions[0] is not None:
            avail_t = torch.tensor(
                available_actions,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            avail_t = None

        actions, new_rnn_states = self.policy.get_actions(
            obs_t,
            rnn_t,
            masks_t,
            avail_t,
            epsilon=epsilon,
        )

        actions_np = _t2n(actions)
        if actions_np.shape[-1] == 1:
            actions_np = actions_np.squeeze(-1)

        return actions_np, _t2n(new_rnn_states)

    @torch.no_grad()
    def eval(self):
        self.policy.prep_rollout()
        self.logger.eval_init()

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.algo_args["eval"]["n_eval_rollout_threads"],
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (
                self.algo_args["eval"]["n_eval_rollout_threads"],
                self.num_agents,
                1,
            ),
            dtype=np.float32,
        )

        eval_episode = 0

        while True:
            eval_actions, eval_rnn_states = self.collect(
                eval_obs,
                eval_rnn_states,
                eval_masks,
                eval_available_actions,
                epsilon=0.0,
            )

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)

            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (
                    self.algo_args["eval"]["n_eval_rollout_threads"],
                    self.num_agents,
                    1,
                ),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1),
                dtype=np.float32,
            )

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(eval_episode)
                break

    def _get_epsilon(self, episode, episodes):
        eps_start = self.algo_args["algo"].get("epsilon_start", 1.0)
        eps_finish = self.algo_args["algo"].get("epsilon_finish", 0.05)
        eps_anneal_time = self.algo_args["algo"].get("epsilon_anneal_time", episodes)

        progress = min(float(episode) / float(eps_anneal_time), 1.0)
        return eps_start + progress * (eps_finish - eps_start)

    def _log_train_info(self, train_info, episode, step):
        current_steps = (
            episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
            + step * self.algo_args["train"]["n_rollout_threads"]
        )

        for k, v in train_info.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            elif isinstance(v, np.ndarray):
                v = float(np.mean(v))
            elif isinstance(v, (np.integer, np.floating)):
                v = float(v)

            if isinstance(v, (int, float)):
                self.writter.add_scalar(k, v, current_steps)

    def save(self, episode=None):
        checkpoint = {
            "episode": episode,
            "policy": self.policy.save(),
        }

        latest_path = os.path.join(self.save_dir, "latest.pt")
        torch.save(checkpoint, latest_path)

        save_backup_interval = self.algo_args["train"].get("save_backup_interval", 100)
        if episode is not None and episode % save_backup_interval == 0:
            backup_path = os.path.join(self.save_dir, f"backup_ep{episode}.pt")
            torch.save(checkpoint, backup_path)

    def restore(self):
        model_dir = str(self.algo_args["train"]["model_dir"])
        latest_path = os.path.join(model_dir, "latest.pt")

        checkpoint = torch.load(latest_path, map_location=self.device)
        self.start_episode = checkpoint.get("episode", 0) + 1
        self.policy.restore(checkpoint["policy"])

    def render(self):
        raise NotImplementedError("QMIX render is not implemented yet.")

    def close(self):
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.logger.close()