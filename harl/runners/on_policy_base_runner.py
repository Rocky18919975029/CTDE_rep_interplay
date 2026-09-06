

"""Base runner for on-policy algorithms."""

import time
import os
import numpy as np
import torch
import setproctitle
from tqdm import tqdm  
from harl.common.valuenorm import ValueNorm
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics.v_critic import VCritic
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config
from harl.envs import LOGGER_REGISTRY


class OnPolicyBaseRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OnPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]
        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        if not self.algo_args["render"]["use_render"]: 
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

      
        if self.algo_args["render"]["use_render"]: 
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
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

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

       
        from harl.utils.envs_tools import get_shape_from_act_space
        total_action_dim = sum([get_shape_from_act_space(self.envs.action_space[i]) for i in range(self.num_agents)])
        algo_args["algo"]["total_action_dim"] = total_action_dim
        
       
        algo_args["algo"]["num_agents"] = self.num_agents

        
        if self.share_param:
            self.actor = []
            agent = ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"]},
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
            self.actor.append(agent)
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id]
                    == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
        else:
            self.actor = []
            for agent_id in range(self.num_agents):
                agent = ALGO_REGISTRY[args["algo"]](
                    {**algo_args["model"], **algo_args["algo"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                self.actor.append(agent)

        if self.algo_args["render"]["use_render"] is False:  
            self.actor_buffer = []
            for agent_id in range(self.num_agents):
                ac_bu = OnPolicyActorBuffer(
                    {**algo_args["train"], **algo_args["model"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                )
                self.actor_buffer.append(ac_bu)

            share_observation_space = self.envs.share_observation_space[0]
            self.critic = VCritic(
                {**algo_args["model"], **algo_args["algo"]},
                share_observation_space,
                device=self.device,
            )
            if self.state_type == "EP":
                self.critic_buffer = OnPolicyCriticBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                )
            elif self.state_type == "FP":
                self.critic_buffer = OnPolicyCriticBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                    self.num_agents,
                )
            else:
                raise NotImplementedError

           

            self.use_guider = algo_args["algo"].get("use_guider", False)

            if self.use_guider:
                
                guider_type = algo_args["algo"].get("guider_type", "sable").lower()
                
                if guider_type == "mat":
                    from harl.models.policy_models.ar_guider_mat import ARGuiderPolicyMAT
                    GuiderClass = ARGuiderPolicyMAT
                else:
                    from harl.models.policy_models.ar_guider_sable import ARGuiderPolicySable
                    GuiderClass = ARGuiderPolicySable
                
               
                self.guider = GuiderClass(
                    {**algo_args["model"], **algo_args["algo"]},
                    share_obs_space=self.envs.share_observation_space[0],
                    action_space=self.envs.action_space[0], 
                    num_agents=self.num_agents,
                    device=self.device,
                )
                
                
                self.guider_optimizer = torch.optim.Adam(
                    self.guider.parameters(),
                    lr=algo_args["model"]["lr"],
                    eps=algo_args["model"]["opti_eps"],
                    weight_decay=algo_args["model"]["weight_decay"],
                )


            if self.algo_args["train"]["use_valuenorm"] is True:
                self.value_normalizer = ValueNorm(1, device=self.device)
            else:
                self.value_normalizer = None

            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )
            
        
        self.start_episode = 1
        if self.algo_args["train"]["model_dir"] is not None:  # restore model
            self.restore()

    def run(self):
        """Run the training (or rendering) pipeline."""
  
        if self.algo_args["render"]["use_render"] is True:
            self.render()
            return
        print("start running")
        self.warmup()

        
        import time
        train_start_time = time.time()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        self.logger.init(episodes)  

        
        pbar = tqdm(total=episodes, initial=self.start_episode - 1, desc="Training")

        for episode in range(self.start_episode, episodes + 1):
            if self.algo_args["train"][
                "use_linear_lr_decay"
            ]:  
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes)
                self.critic.lr_decay(episode, episodes)
                
                if getattr(self, "use_guider", False):
                    from harl.utils.models_tools import update_linear_schedule
                    update_linear_schedule(self.guider_optimizer, episode, episodes, self.algo_args["model"]["lr"])

            self.logger.episode_init(episode) 

            self.prep_rollout() 
            for step in range(self.algo_args["train"]["episode_length"]):
                
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)
               
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                self.logger.per_step(data)  
                self.insert(data) 

            
            self.compute()
            self.prep_training()  

            actor_train_infos, critic_train_info = self.train()

           
            aux_losses = []
            action_pred_losses = []
            
            for agent_id in range(self.num_agents):
               
                if "aux_loss" in actor_train_infos[agent_id]:
                    aux_losses.append(actor_train_infos[agent_id]["aux_loss"])
                
                if "action_pred_loss" in actor_train_infos[agent_id]:
                    action_pred_losses.append(actor_train_infos[agent_id]["action_pred_loss"])
            
            pbar_postfix = {}
           
            current_steps = (
                episode 
                * self.algo_args["train"]["episode_length"] 
                * self.algo_args["train"]["n_rollout_threads"]
            )

           
            if len(aux_losses) > 0 and np.mean(aux_losses) != 0.0:
                avg_aux_loss = np.mean(aux_losses)
                pbar_postfix["Aux Loss"] = f"{avg_aux_loss:.4f}"
                self.writter.add_scalar("avg_aux_loss", avg_aux_loss, current_steps)
                
            
            if len(action_pred_losses) > 0 and np.mean(action_pred_losses) != 0.0:
                avg_act_pred_loss = np.mean(action_pred_losses)
                pbar_postfix["Act Pred"] = f"{avg_act_pred_loss:.4f}"
                self.writter.add_scalar("avg_action_pred_loss", avg_act_pred_loss, current_steps)

           
            if pbar_postfix:
                pbar.set_postfix(pbar_postfix)

           
            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )

           
            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
            
            
            save_backup_interval = self.algo_args["train"].get("save_backup_interval", 100)
            if (episode % self.algo_args["train"]["eval_interval"] == 0) or \
               (episode % save_backup_interval == 0):
                self.save(episode)

            self.after_update()
            
           
            pbar.update(1)

        pbar.close()

        
        train_end_time = time.time()
        total_time_sec = train_end_time - train_start_time
        
       
        hours, rem = divmod(total_time_sec, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        
        print(f"\n🎉 Training Pipeline Finished!")
        print(f"⏱️  Total Training Time: {time_str}")
        
        
        time_log_path = os.path.join(self.run_dir, "training_time.txt")
        with open(time_log_path, "w") as f:
            f.write(f"Start Episode: {self.start_episode}\n")
            f.write(f"End Episode: {episodes}\n")
            f.write(f"Total Training Time (Seconds): {total_time_sec:.2f}\n")
            f.write(f"Total Training Time (Formatted): {time_str}\n")

    def warmup(self):
        """Warm up the replay buffer."""
       
        obs, share_obs, available_actions = self.envs.reset()
       
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()
        if self.state_type == "EP":
            self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()
        elif self.state_type == "FP":
            self.critic_buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        """Collect actions and values from actors and critics."""
        if self.use_guider:
           
            current_share_obs = self.critic_buffer.share_obs[step]
            if self.state_type == "FP":
                current_share_obs = np.concatenate(current_share_obs)

           
            available_actions = None
            if self.actor_buffer[0].available_actions is not None:
                avail_acts = [self.actor_buffer[i].available_actions[step] for i in range(self.num_agents)]
                available_actions = np.stack(avail_acts, axis=1)

           
            actions_pt, action_log_probs_pt = self.guider.get_actions(
                current_share_obs, available_actions, deterministic=False
            )
            
           
            actions = _t2n(actions_pt)
            action_log_probs = _t2n(action_log_probs_pt)
            
            
            rnn_states = np.zeros(
                (self.algo_args["train"]["n_rollout_threads"], self.num_agents, self.recurrent_n, self.rnn_hidden_size),
                dtype=np.float32
            )
            
        else:
            action_collector = []
            action_log_prob_collector = []
            rnn_state_collector = []
            for agent_id in range(self.num_agents):
                action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(
                    self.actor_buffer[agent_id].obs[step],
                    self.actor_buffer[agent_id].rnn_states[step],
                    self.actor_buffer[agent_id].masks[step],
                    self.actor_buffer[agent_id].available_actions[step]
                    if self.actor_buffer[agent_id].available_actions is not None
                    else None,
                )
                action_collector.append(_t2n(action))
                action_log_prob_collector.append(_t2n(action_log_prob))
                rnn_state_collector.append(_t2n(rnn_state))
            actions = np.array(action_collector).transpose(1, 0, 2)
            action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
            rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)

        if self.state_type == "EP":
            value, rnn_state_critic = self.critic.get_values(
                self.critic_buffer.share_obs[step],
                self.critic_buffer.rnn_states_critic[step],
                self.critic_buffer.masks[step],
            )
            values = _t2n(value)
            rnn_states_critic = _t2n(rnn_state_critic)
        elif self.state_type == "FP":
            value, rnn_state_critic = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[step]),
                np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                np.concatenate(self.critic_buffer.masks[step]),
            )
            values = np.array(
                np.split(_t2n(value), self.algo_args["train"]["n_rollout_threads"])
            )
            rnn_states_critic = np.array(
                np.split(
                    _t2n(rnn_state_critic), self.algo_args["train"]["n_rollout_threads"]
                )
            )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        """Insert data into buffer."""
        (
            obs, share_obs, rewards, dones, infos, available_actions,
            values, actions, action_log_probs, rnn_states, rnn_states_critic,
        ) = data

        dones_env = np.all(dones, axis=1)
        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_n, self.rnn_hidden_size,), dtype=np.float32,
        )

        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size), dtype=np.float32,
            )
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.num_agents, self.recurrent_n, self.rnn_hidden_size,), dtype=np.float32,
            )

        masks = np.ones((self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1), dtype=np.float32,)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1), dtype=np.float32,)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if self.state_type == "EP":
            bad_masks = np.array(
                [[0.0] if "bad_transition" in info[0].keys() and info[0]["bad_transition"] == True else [1.0] for info in infos]
            )
        elif self.state_type == "FP":
            bad_masks = np.array(
                [[[0.0] if "bad_transition" in info[agent_id].keys() and info[agent_id]["bad_transition"] == True else [1.0] for agent_id in range(self.num_agents)] for info in infos]
            )

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id] if available_actions[0] is not None else None,
            )

        if self.state_type == "EP":
            self.critic_buffer.insert(
                share_obs[:, 0], rnn_states_critic, values, rewards[:, 0], masks[:, 0], bad_masks,
            )
        elif self.state_type == "FP":
            self.critic_buffer.insert(
                share_obs, rnn_states_critic, values, rewards, masks, bad_masks
            )

    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages."""
        if self.state_type == "EP":
            next_value, _ = self.critic.get_values(
                self.critic_buffer.share_obs[-1],
                self.critic_buffer.rnn_states_critic[-1],
                self.critic_buffer.masks[-1],
            )
            next_value = _t2n(next_value)
        elif self.state_type == "FP":
            next_value, _ = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[-1]),
                np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                np.concatenate(self.critic_buffer.masks[-1]),
            )
            next_value = np.array(np.split(_t2n(next_value), self.algo_args["train"]["n_rollout_threads"]))
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)

    def train(self):
        """Train the model."""
        raise NotImplementedError

    def after_update(self):
        """Do the necessary data operations after an update."""
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.critic_buffer.after_update()

    @torch.no_grad()
    def eval(self):
        """Evaluate the model."""
        self.logger.eval_init() 
        eval_episode = 0

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
            (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )

        while True:
           
            if self.args["algo"] in ["mat", "sable"]:
                current_share_obs = eval_share_obs
                if self.state_type == "FP":
                    current_share_obs = np.concatenate(current_share_obs)
                
               
                available_actions = None
                if eval_available_actions[0] is not None:
                    avail_acts = [eval_available_actions[:, i] for i in range(self.num_agents)]
                    available_actions = np.stack(avail_acts, axis=1) # [Batch, N, Dim]

                
                actions_pt, _ = self.guider.get_actions(
                    current_share_obs, available_actions, deterministic=True
                )
                
                
                eval_actions = _t2n(actions_pt).transpose(1, 0, 2)
                
           
            else:
                eval_actions_collector = []
                for agent_id in range(self.num_agents):
                    eval_actions, temp_rnn_state = self.actor[agent_id].act(
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id]
                        if eval_available_actions[0] is not None
                        else None,
                        deterministic=True,
                    )
                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))

                eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

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
            self.logger.eval_per_step(
                eval_data
            )  

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[
                eval_dones_env == True
            ] = np.zeros( 
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(
                        eval_i
                    )  

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(
                    eval_episode
                )  
                break

    @torch.no_grad()
    def render(self):
        """Render the model."""
        print("start rendering")
        if self.manual_expand_dims:
            
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = (
                    np.expand_dims(np.array(eval_available_actions), axis=0)
                    if eval_available_actions is not None
                    else None
                )
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    for agent_id in range(self.num_agents):
                        eval_actions, temp_rnn_state = self.actor[agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            eval_available_actions[:, agent_id]
                            if eval_available_actions is not None
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = (
                        np.expand_dims(np.array(eval_available_actions), axis=0)
                        if eval_available_actions is not None
                        else None
                    )
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        else:
            
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    for agent_id in range(self.num_agents):
                        eval_actions, temp_rnn_state = self.actor[agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            eval_available_actions[:, agent_id]
                            if eval_available_actions[0] is not None
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions)
                    rewards += eval_rewards[0][0][0]
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0][0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        if "smac" in self.args["env"]: 
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

   
    def prep_rollout(self):
        """Prepare for rollout."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_rollout()
        self.critic.prep_rollout()
       
        if getattr(self, "use_guider", False):
            self.guider.eval()

   
    def prep_training(self):
        """Prepare for training."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
        self.critic.prep_training()
       
        if getattr(self, "use_guider", False):
            self.guider.train()

   
    def save(self, episode=None):
        """Save model parameters.
        Args:
            episode (int): 当前的 episode，用于生成 backup 模型和记录进度
        """
        save_backup_interval = self.algo_args["train"].get("save_backup_interval", 100)
        
       
        checkpoint = {
            "episode": episode,
            "actors": {},
            "critic_state_dict": self.critic.critic.state_dict(),
            "critic_optimizer_state_dict": self.critic.critic_optimizer.state_dict()
        }
       
        if getattr(self, "use_guider", False):
            checkpoint["guider_state_dict"] = self.guider.state_dict()
            checkpoint["guider_optimizer_state_dict"] = self.guider_optimizer.state_dict()

        
        for agent_id in range(self.num_agents):
            checkpoint["actors"][f"agent_{agent_id}"] = self.actor[agent_id].get_checkpoint()

      
        if self.value_normalizer is not None:
            checkpoint["value_normalizer_state_dict"] = self.value_normalizer.state_dict()

       
        latest_path = os.path.join(self.save_dir, "latest.pt")
        torch.save(checkpoint, latest_path)

       
        if episode is not None and episode % save_backup_interval == 0:
            backup_path = os.path.join(self.save_dir, f"backup_ep{episode}.pt")
            torch.save(checkpoint, backup_path)
            
       
        for agent_id in range(self.num_agents):
            torch.save(
                self.actor[agent_id].actor.state_dict(),
                str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt",
            )
        torch.save(
            self.critic.critic.state_dict(), str(self.save_dir) + "/critic_agent" + ".pt"
        )
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir) + "/value_normalizer" + ".pt",
            )

   
    def restore(self):
        """Restore model parameters for perfect resume or rendering."""
        model_dir = str(self.algo_args["train"]["model_dir"])
        
        
        latest_path = os.path.join(model_dir, "latest.pt")
        
        if os.path.exists(latest_path) and not self.algo_args["render"]["use_render"]:
            print(f"==> Resuming perfectly from {latest_path}")
            checkpoint = torch.load(latest_path, map_location=self.device)
            
           
            self.start_episode = checkpoint.get("episode", 0) + 1
            
           
            for agent_id in range(self.num_agents):
                self.actor[agent_id].load_checkpoint(checkpoint["actors"][f"agent_{agent_id}"])
                
           
            self.critic.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

           
            if getattr(self, "use_guider", False) and "guider_state_dict" in checkpoint:
                self.guider.load_state_dict(checkpoint["guider_state_dict"])
                self.guider_optimizer.load_state_dict(checkpoint["guider_optimizer_state_dict"])
            
           
            if self.value_normalizer is not None and "value_normalizer_state_dict" in checkpoint:
                self.value_normalizer.load_state_dict(checkpoint["value_normalizer_state_dict"])
                
        else:
           
            print("==> Resuming from separated .pt files (Legacy Mode / Render Mode)")
            for agent_id in range(self.num_agents):
                policy_actor_state_dict = torch.load(model_dir + "/actor_agent" + str(agent_id) + ".pt", map_location=self.device)
                self.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
                
            if not self.algo_args["render"]["use_render"]:
                policy_critic_state_dict = torch.load(model_dir + "/critic_agent" + ".pt", map_location=self.device)
                self.critic.critic.load_state_dict(policy_critic_state_dict)
                if self.value_normalizer is not None:
                    value_normalizer_state_dict = torch.load(model_dir + "/value_normalizer" + ".pt", map_location=self.device)
                    self.value_normalizer.load_state_dict(value_normalizer_state_dict)

    def close(self):
        """Close environment, writter, and logger."""
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.logger.close()
