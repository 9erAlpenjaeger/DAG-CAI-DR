from typing import Any, Dict, Generator, List, Optional, Union, Type, NamedTuple

import torch as th
import numpy as np
from gym import spaces
from torch.nn import functional as F
import pygmo as pg

from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

#SelfA2C = TypeVar("SelfA2C", bound="A2C")
TensorDict = Dict[Union[str, int], th.Tensor]
OBJ_NUM = 3
''' buffer with gradient info of tensor'''
class DictRolloutBufferSamples_tensor(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    objs: np.array
    dense_rews: np.array
    rglr: th.Tensor
    
class DictRolloutBuffer_tensor(DictRolloutBuffer):
    def reset(self) -> None:
        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros((self.buffer_size, self.n_envs, *obs_input_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.objs = np.zeros((self.buffer_size, self.n_envs, OBJ_NUM), dtype=np.float32) 
        self.dense_rews = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=th.float32, device = self.device) # log_prob is kept in th.tensor in so that grad info won't be lost
        self.rglr = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device = self.device)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()
        
    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        obj: np.ndarray,
        dense_rew: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        rglr: th.Tensor,
    ) -> None:  # pytype: disable=signature-mismatch
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key]).copy()
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_
        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.objs[self.pos] = np.array(obj)
        self.dense_rews[self.pos] = np.array(dense_rew).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob#.clone().detach()
        self.rglr[self.pos] = rglr#.clone().detach()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    def get(
        self,
        batch_size: Optional[int] = None,
    ):# -> Generator[DictRolloutBufferSamples_tensor, None, None]:  # type: ignore[signature-mismatch] #FIXME
        self.generator_ready = True
        yield self._get_samples()
        
    def _get_samples(
        self,
        batch_inds: np.ndarray = None,
        env: Optional[VecNormalize] = None,
    ) -> DictRolloutBufferSamples_tensor:  # type: ignore[signature-mismatch] #FIXME
        return DictRolloutBufferSamples_tensor(
            observations={key: self.to_torch(obs) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions),
            old_values=self.to_torch(self.values),
            old_log_prob=self.log_probs,#.clone().detach(),
            advantages=self.to_torch(self.advantages),
            returns=self.to_torch(self.returns.flatten()),
            objs=self.objs,
            dense_rews = self.dense_rews,
            rglr=self.rglr#.clone().detach()
        )

class ReinListModel(OnPolicyAlgorithm):

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        c_logits: float = 0.001,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        reference_points = None
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.normalize_advantage = normalize_advantage
        self.c_logits = c_logits
        self.reference_points = np.array(reference_points)
        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)
        if _init_setup_model:
            self._setup_model()
            
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer_tensor #if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        
    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        with th.no_grad():
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
        actions, log_prob, rglr = self.policy.evaluate_actions(obs_tensor)
        actions = actions.clone().detach().cpu().numpy()
        ''' for VecEnv'''
        env.step_async(actions)
        new_obs, rewards, dones, infos = env.step_wait()
        cnt = 0
        objs_batch = []
        dense_rews_batch = []
        for reward in rewards:
            self.logger.record("reward/reward%d" %(cnt), reward)
            cnt+=1
        
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
            actions, values, log_probs, rglr = self.policy(obs_tensor)
            actions = actions.clone().detach().cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            env.step_async(clipped_actions)
            new_obs, rewards, dones, infos = env.step_wait()  
            objs = list()
            dense_rews = list()
            for info in infos:
                objs.append(info['obj'])
                dense_rews.append(info['dense_rew'])
            objs_batch.append(objs)
            dense_rews_batch.append(dense_rews)
            #new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer({})
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
            rollout_buffer.add(self._last_obs, actions, rewards, objs, dense_rews, self._last_episode_starts, values, log_probs, rglr)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # This will only loop once (get all data in one go)
        #print(self.reference_points)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            values = rollout_data.old_values.flatten()
            rglr = rollout_data.rglr#.clone().detach()
            objs = rollout_data.objs
            dense_rews = rollout_data.dense_rews
            log_prob = rollout_data.old_log_prob#.clone().detach()
            advantages = rollout_data.advantages

            objs = objs.transpose(1,0,2)
            dense_rews = th.tensor(dense_rews, dtype=th.float32, device=self.device)

            hvs = []
            for reference_point, obj in zip(self.reference_points, objs):
                #hv = hv = pg.hypervolume(obj)
                #hv_value = hv.compute(reference_point)
                #hvs.append(hv_value)
                hvs.append(0)
                #print(hv_value)

            if self.normalize_advantage:
                m = advantages.mean(0, keepdim = True)
                s = advantages.std(0, unbiased=False, keepdim=True)
                advantages -= m
                advantages /= (s+1e-10)


            #log_prob = log_prob.flatten()
            #rglr = rglr.flatten()
            advantages = advantages.flatten()
            #rglr = rglr.mean()
            # Policy gradient loss
            #policy_loss = - (advantages * log_prob).mean()

            #loss = -(advantages * log_prob.sum(dim=2)) + self.c_logits * rglr # sparse rew
            policy_loss = -dense_rews * log_prob 
            regular_item = (self.c_logits * rglr).unsqueeze(-1)
            non_zero_mask = ~th.isclose(policy_loss, torch.tensor(0.0), atol=1e-8)

            loss = policy_loss + regular_item 
            
            loss = loss[non_zero_mask]
            if loss.numel() > 0:
                loss = loss.mean()
            else:
                loss = torch.tensor(0.0).to(device=loss.device)

            #print(loss)

            value_loss = F.mse_loss(rollout_data.returns, values)
            # Optimization step
            self.policy.optimizer.zero_grad()
            #rglr.mean().backward()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/regularized_logits", rglr.mean().item())
        self.logger.record("train/loss", loss.item())
        #self.logger.record("train/entropy_loss", entropy_loss.item())
        #self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.mean().item())
        for hv,i in zip(hvs, range(len(hvs))):
            self.logger.record("train/hv" + str(i), hv)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "A2C",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
