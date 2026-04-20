import os

import gym
import time
import numpy as np
from functools import partial
from datetime import datetime
#import pygmo as pg
import argparse

import torch
import torch.nn.functional as F
#from torch_scatter import scatter
from torchvision import ops
from torch_geometric.nn.models import GCN, GAT
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch


from Graphormer.graphormer.modules import GraphormerGraphEncoder


from gymenvs.cloudgym_oneshot import CloudGym_oneshot, cat_action_machinepri
from gymenvs.cloudgym_oneshot_tpch import CloudGym_oneshot_tpch
from gymenvs.cloudgym_oneshot_jssp import CloudGym_oneshot_jssp

from RL_models.utils import (RectangleAttn,
                             cat_list, 
                             GumbelSort, 
                             GumbelTopoSortInTuples, 
                             GumbelTopoSortAdj, 
                             DemonstrationBuffer
                            )

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, Type, NamedTuple
from stable_baselines3.common.type_aliases import (
    DictRolloutBufferSamples,
    RolloutBufferSamples,
    GymEnv, 
    MaybeCallback, 
    Schedule
)
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy, 
    ActorCriticPolicy, 
    BasePolicy, 
    MultiInputActorCriticPolicy
)
from stable_baselines3.common.utils import explained_variance, obs_as_tensor



global_flops = 0
EMB_DIM = CloudGym_oneshot.raw_node_attr_length
HIDDEN_DIM = 16 # EMB_DIM
LAYERS_NUM = 4
GOLBAL_EMB_DIM = EMB_DIM
max_node_num = CloudGym_oneshot.max_node_num
max_edge_num = CloudGym_oneshot.max_edge_num

machine_num = CloudGym_oneshot.machine_num

NULL_ACTION = CloudGym_oneshot.NULL_ACTION



#GNN = ' '

#SelfA2C = TypeVar("SelfA2C", bound="A2C")
TensorDict = Dict[Union[str, int], torch.Tensor]
OBJ_NUM = 3





''' buffer with gradient info of tensor'''
class DictRolloutBufferSamples_tensor(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    objs: np.array
    dense_rews: np.array
    rglr: torch.Tensor

    machine_pris: torch.Tensor
    machine_available_masks: np.array
    machine_chosen_masks: np.array
    
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
        self.log_probs = torch.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=torch.float32, device = self.device) # log_prob is kept in torch.tensor in so torchat grad info won't be lost
        self.rglr = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device = self.device)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False

        self.machine_pris = torch.zeros((self.buffer_size, self.n_envs, self.action_dim, machine_num), dtype=torch.float32, device=self.device)
        self.machine_available_masks = np.zeros((self.buffer_size, self.n_envs, self.action_dim, machine_num), dtype=bool)
        self.machine_chosen_masks = np.zeros((self.buffer_size, self.n_envs, self.action_dim, machine_num), dtype=bool)
        
        super(RolloutBuffer, self).reset()
        
    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        obj: np.ndarray,
        dense_rew: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        rglr: torch.Tensor,

        machine_pri: torch.Tensor,
        machine_available_mask: np.ndarray,
        machine_chosen_mask: np.ndarray,
    ) -> None:  # pytype: disable=signature-mismatch
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key]).copy()
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], gym.spaces.Discrete):
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
        self.rglr[self.pos] = rglr.flatten()#.clone().detach()

        self.machine_pris[self.pos] = machine_pri
        self.machine_available_masks[self.pos] = np.array(machine_available_mask).copy()
        self.machine_chosen_masks[self.pos] = np.array(machine_chosen_mask).copy()
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
            rglr=self.rglr, #.clone().detach()

            machine_pris = self.machine_pris,
            machine_available_masks = self.to_torch(self.machine_available_masks),
            machine_chosen_masks = self.to_torch(self.machine_chosen_masks)
        )

class ReinforceOneshot(OnPolicyAlgorithm):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        use_env_for_eval: bool = False,
        env_for_eval: Union[GymEnv, str] = None, # the Vectorized GymEnv only for evaluation (in order to evaluate the generality of the model)
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.0,
        c_logits: float = 0.001,
        is_dw: bool = True,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = True,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        reference_points = None,
        use_demonstrations = False,
        demonstrations = None
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
                gym.spaces.Box,
                gym.spaces.Discrete,
                gym.spaces.MultiDiscrete,
                gym.spaces.MultiBinary,
            ),
        )
        self.global_cnt = 0
        self.is_dw = is_dw
        
        self.use_env_for_eval = use_env_for_eval
        self.env_for_eval = env_for_eval

        self.use_demonstrations = use_demonstrations
        self.demonstrations = demonstrations
        
        self.freeze_flag = False
        
        self.normalize_advantage = normalize_advantage
        self.c_logits = c_logits
        self.reference_points = np.array(reference_points)
        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = torch.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=1e-5)
        if _init_setup_model:
            self._setup_model()
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer_tensor #if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer

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
        if self.use_env_for_eval:
            _last_obs_for_eval = self.env_for_eval.reset()
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        
        with torch.no_grad():
            # Convert to pytorch tensor or to TensorDict
            if self.use_env_for_eval:
                obs_tensor_for_eval = obs_as_tensor(_last_obs_for_eval, self.device)
            obs_tensor = obs_as_tensor(self._last_obs, self.device)

        # evaluate buffered envs
        actions, machine_pris, values, log_probs, rglr = self.policy.evaluate_actions(obs_tensor, None, None, False)
        actions = actions.clone().detach().cpu().numpy()
        machine_pris = machine_pris.clone().detach().cpu().numpy()
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
        joint_action = cat_action_machinepri(clipped_actions, machine_pris)
        env.step_async(joint_action)
        new_obs, rewards, dones, infos = env.step_wait()  
       
        for cnt, reward in enumerate(rewards):
            self.logger.record("reward/reward%d" %(cnt), reward)

        # evaluate non-buffered envs
        if self.use_env_for_eval:
            _actions, _machine_pris, _values, _log_probs, _rglr = self.policy.evaluate_actions(obs_tensor_for_eval, None, None, False)
            _actions = _actions.clone().detach().cpu().numpy()
            _machine_pris = _machine_pris.clone().detach().cpu().numpy()
            if isinstance(self.action_space, gym.spaces.Box):
                _clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            _joint_action = cat_action_machinepri(_clipped_actions, _machine_pris)
            self.env_for_eval.step_async(_joint_action)
            _new_obs, _rewards, _dones, _infos = self.env_for_eval.step_wait()
            for cnt, _reward in enumerate(_rewards):
                self.logger.record("reward/un_reward%d" %(cnt), _reward)
        
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                
            if self.use_demonstrations: # 如果使用demonstration进行若干轮（self.demonstrations.max_episode)的更新
                demon_actions = self.demonstrations.draw()
                actions, machine_pris, values, log_probs, rglr = self.policy.evaluate_actions(obs_tensor, demon_actions) 
                if self.demonstrations.episode_counter >= self.demonstrations.max_episode: # 当demonstration到达上限，进行正常训练
                    self.use_demonstrations = False 
            else:
                actions, machine_pris, values, log_probs, rglr = self.policy(obs_tensor)

            actions = actions.clone().detach().cpu().numpy()
            machine_pris_for_gym = machine_pris.clone().detach().cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            joint_action = cat_action_machinepri(clipped_actions, machine_pris_for_gym)
            env.step_async(joint_action)
            new_obs, rewards, dones, infos = env.step_wait()  
            objs = list()
            dense_rews = list()
            machine_available_masks = list()
            machine_chosen_masks = list()
            for info in infos:
                objs.append(info['obj'])
                dense_rews.append(info['dense_rew'])
                machine_available_masks.append(info['machine_available_mask'])
                machine_chosen_masks.append(info['machine_chosen_mask'])
            #objs_batch.append(objs)
            #dense_rews_batch.append(dense_rews)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer({})
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
            rollout_buffer.add(self._last_obs, 
                               actions, rewards, 
                               objs, dense_rews, 
                               self._last_episode_starts, 
                               values, 
                               log_probs, 
                               rglr, 
                               machine_pris, 
                               machine_available_masks, 
                               machine_chosen_masks)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()
        return True


    def train(self) -> None:
        if self.global_cnt < 1:
            self.global_cnt += 1
            return
        self.global_cnt += 1
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        
        # 训练若干步后冻结GNN
        if self.freeze_flag == False and self.num_timesteps >= 800000:
            self.freeze_flag = True
            for param in self.policy.features_extractor.gnn.parameters():
                param.requires_grad = False

            # 重建优化器，只包括仍然参与训练的参数
            self.policy.optimizer = self.policy.optimizer_class(
                filter(lambda p: p.requires_grad, self.policy.parameters()),
                lr=self.lr_schedule(1), #self.policy.lr_schedule(1),
                **self.policy.optimizer_kwargs #**self.policy.optimizer_kwargs,
            )        

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            
            values = rollout_data.old_values.flatten()
            rglr = rollout_data.rglr#.clone().detach()
            objs = rollout_data.objs
            dense_rews = rollout_data.dense_rews
            dense_rews = torch.tensor(dense_rews, dtype=torch.float32, device=self.device)
            log_prob = rollout_data.old_log_prob#.clone().detach()
            advantages = rollout_data.advantages
            actions = rollout_data.actions

            machine_pris = rollout_data.machine_pris
            machine_available_masks = rollout_data.machine_available_masks
            machine_chosen_masks = rollout_data.machine_chosen_masks

            zero_mask = (dense_rews != 0.0) | (log_prob != 0.0) # | (new_log_prob != 0.0)

            entropy_loss = -torch.sum(-log_prob, dim=-1)/zero_mask.sum(dim=-1).clamp(min=1)
            
            hvs = []
            #for reference_point, obj in zip(self.reference_points, objs):
                #hv = hv = pg.hypervolume(obj)
                #hv_value = hv.compute(reference_point)
                #hvs.append(hv_value)
                #hvs.append(0)

            if self.normalize_advantage:
                m = advantages.mean(0, keepdim = True)
                s = advantages.std(0, unbiased=False, keepdim=True)
                advantages -= m
                advantages /= (s+1e-10)
            #log_prob = log_prob.flatten()
            #rglr = rglr.flatten()
            #advantages = advantages.flatten()

            if self.normalize_advantage:
                flt = dense_rews.flatten()
                #flt = flt[log_prob.flatten() != 0.0]
                flt = flt[flt != 0.0]
                m = flt.mean()
                s = flt.std()
                normalized_dense_rews = (dense_rews - m) / (s + 1e-10)
                dense_rews = normalized_dense_rews
                dense_rews = dense_rews.clamp(-2.0, 2.0) # 进行advantage clipping操作（作用可能同样有限）     


            def log_prob_for_tp_dispatch(machine_pris, machine_available_masks, machine_chosen_masks):
                masked_machine_pris = torch.where(machine_available_masks, machine_pris, torch.full_like(machine_pris, -1e9))
                # 问题出在padding上？被pad的项置什么？
                tp_log_probs = F.log_softmax(masked_machine_pris, dim=-1)
                selected_tp_log_probs = tp_log_probs * machine_chosen_masks
                tp_log_probs = selected_tp_log_probs.sum(dim = -1)
                return tp_log_probs
            
            use_ratio_loss = False
            use_tp = False
            if use_ratio_loss:
                with torch.no_grad():
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                new_actions, new_machine_pris, values, new_log_prob, new_rglr = self.policy.evaluate_actions(obs_tensor, actions, machine_pris)
                new_tp_log_probs = log_prob_for_tp_dispatch(new_machine_pris, machine_available_masks, machine_chosen_masks)       
            clip_range = 0.15
            
            if not self.is_dw:
                ''' sparse rew'''
                if use_tp:
                    tp_log_probs = log_prob_for_tp_dispatch(machine_pris, machine_available_masks, machine_chosen_masks)
                    if use_ratio_loss:
                        ratio = torch.exp(new_tp_log_probs.sum(dim=-1) - tp_log_probs.sum(dim=-1))
                        policy_loss1 = advantages * ratio
                        policy_loss2 = advantages * torch.clamp(ratio, 1-clip_range, 1+clip_range)
                        entropy_loss = -torch.mean(new_tp_log_probs)
                        loss = -torch.min(policy_loss1, policy_loss2) + 0.01 * entropy_loss.mean()
                        loss = loss.mean()
                    else:
                        tp_dispatch_loss = -(advantages * tp_log_probs.sum(dim=2))
                        loss = tp_dispatch_loss.mean()
                else:
                    loss = -(advantages * log_prob.sum(dim=2)) + self.c_logits * rglr + 0.01 * entropy_loss.mean()# sparse rew
                    loss = loss.mean()
            else:
                if use_tp:
                    tp_log_probs = log_prob_for_tp_dispatch(machine_pris, machine_available_masks, machine_chosen_masks)
                    if use_ratio_loss:
                        ratio = torch.exp(new_tp_log_probs - tp_log_probs)
                        policy_loss1 = dense_rews * ratio 
                        policy_loss2 = dense_rews * torch.clamp(ratio, 1-clip_range, 1+clip_range)
                        policy_loss = -torch.min(policy_loss1, policy_loss2)
                        zero_mask = torch.any(machine_chosen_masks, dim=-1)
                        policy_loss = policy_loss.masked_fill(~zero_mask, 0.0)
                        entropy_loss = -torch.mean(new_tp_log_probs)
                        mean_policy_loss = policy_loss.sum() / zero_mask.sum().clamp(min=1) 
                        loss = mean_policy_loss + 0.01 * entropy_loss
                        loss = loss.mean()
                    else:
                        tp_dispatch_loss = -(dense_rews * tp_log_probs)
                        loss = tp_dispatch_loss.mean()
                         
                else:
                    loss = -(dense_rews * log_prob) + self.c_logits * rglr + 0.01 * entropy_loss.mean()
                    loss = loss.mean()

            #loss = tp_dispatch_loss.mean()
            value_loss = F.mse_loss(rollout_data.returns, values)
            # Optimization step
            self.policy.optimizer.zero_grad()
            #rglr.mean().backward()
            torch.autograd.set_detect_anomaly(True)
            loss.backward()

            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/regularized_logits", rglr.mean().item())
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/entropy_loss", entropy_loss.mean().item())
        #self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.mean().item())
        self.logger.record("train/clip_range", clip_range)

        dense_rews = dense_rews.masked_fill(~zero_mask, 0)
        positive_mask = dense_rews > 0.0 # count positive adv
        pos_rew_num = positive_mask.sum(dim=-1).float()     
        self.logger.record("adv/average", (dense_rews.sum() / zero_mask.sum() ).item())
        #self.logger.record("adv/std", dense_rews.std().item())
        self.logger.record('adv/num of positivedense_rew', pos_rew_num.mean().item())

        positive_mask = dense_rews > 0.0 # count positive adv
        pos_rew_num = positive_mask.sum(dim=-1).float()     
        self.logger.record("logprob/average", (log_prob.sum(dim=-1)/zero_mask.sum(dim=-1)).mean().item())
        for hv,i in zip(hvs, range(len(hvs))):
            self.logger.record("train/hv" + str(i), hv)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
            
            


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


class BidirGNN(torch.nn.Module):
    def __init__(self, gnn_type, in_channels, hidden_channels, num_layers):
        super(BidirGNN, self).__init__()
        self.gnn_type = gnn_type
        hidden_channels = int(hidden_channels / 2) 
        if self.gnn_type == 'GCN':
            self.gnn0 = GCN(in_channels = in_channels, hidden_channels = hidden_channels, num_layers = num_layers)
            self.gnn1 = GCN(in_channels = in_channels, hidden_channels = hidden_channels, num_layers = num_layers)
        elif self.gnn_type == 'GAT':
            self.gnn0 = GAT(in_channels = in_channels, hidden_channels = hidden_channels, num_layers = num_layers)
            self.gnn1 = GAT(in_channels = in_channels, hidden_channels = hidden_channels, num_layers = num_layers)
        elif self.gnn_type == 'Graphormer':
            g_args = {
                'num_atoms': 1,
                'num_in_degree': 10,
                'num_out_degree': 10, 
                'num_edges': 1000,
                'num_spatial': 10,
                'num_edge_dis': 10,
                'edge_type': None,
                'multi_hop_max_dist':10, 
                'num_encoder_layers':4,
                'embedding_dim': hidden_channels,
                'ffn_embedding_dim': hidden_channels,
                'num_attention_heads': 8,
            }
            self.gnn0 = GraphormerGraphEncoder(**g_args)
            self.gnn1 = GraphormerGraphEncoder(**g_args)
        else:
            self.gnn = None

    def forward(self, x, edge_index, adj_mat_ls):
        if self.gnn_type == 'Graphormer':
            data0 = Data(x=x.unsqueeze(0), edge_index=edge_index.unsqueeze(0))
            setattr(data0, 'adj_mat', adj_mat_ls)
            data1 = Data(x=x.unsqueeze(0), edge_index=edge_index[[1,0], :].unsqueeze(0))
            setattr(data1, 'adj_mat', adj_mat_ls.permute(0,2,1))
            #batch0 = Batch.from_data_list([data0])
            #batch1 = Batch.from_data_list([data1])
            _, _, out_0 = self.gnn0(data0)
            _, _, out_1 = self.gnn1(data1)
            
        else:
            out_0 = self.gnn0(x, edge_index)
            out_1 = self.gnn1(x, edge_index[[1,0], :])
        out = torch.cat([out_0, out_1], dim = -1)
        return out

class FeatureEmbedding(BaseFeaturesExtractor):
    '''
    extract the feature of the scheduling network in the form of pyg 
    batch is required
    '''
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 in_channels: Union[int, Dict[str, int]],
                 out_channels = 128,
                 hidden_channels=128,
                 heads=8,
                 agg = 'attn_x',
                 gnn = None):
        super().__init__(observation_space, features_dim = out_channels)
        
        self.gnn_type = gnn
        self.gnn = BidirGNN(gnn_type = gnn, in_channels = EMB_DIM, hidden_channels = HIDDEN_DIM, num_layers = 2)

    def forward(self, observations):
        node_nums_ls = observations['node_num']
        edge_nums_ls = observations['edge_num']
        #irr_pair_nums_ls = observations['irr_pair_num']
        xs_ls =observations['x']
        edge_attrs_ls = observations['edge_attr']  
        edge_indexs_ls = observations['edge_index'].to(dtype=torch.int32)
        valid_masks_ls = observations['valid_mask'].to(dtype=torch.bool)

        adj_mat_ls = observations['adj_mat']
        
        global_emb_ls, nodes_emb_ls = [], []

        batch_size = len(node_nums_ls)
        data_list = []

        
        for node_num, edge_num, _x, _edge_index in zip(node_nums_ls, edge_nums_ls, xs_ls, edge_indexs_ls):
            node_num = node_num.to(dtype=torch.int32)
            edge_num = edge_num.to(dtype=torch.int32)   

            x = _x[0:node_num].to(dtype=torch.float32)  
            m = x.mean(0, keepdim=True)
            s = x.std(0, unbiased=False, keepdim=True) 
            x = (x - m) / (s + 1e-8)    
            x_ = x.clone().detach()     

            edge_index = _edge_index[:, 0:edge_num].to(dtype=torch.int32) 
            edge_index_ = edge_index.clone().detach()

            data_list.append(Data(x=x_, edge_index=edge_index_))
        batch = Batch.from_data_list(data_list)
        node2graph = batch.batch
        edge2graph = batch.batch[batch.edge_index[0]]

        node_embs = self.gnn(batch.x, batch.edge_index, adj_mat_ls)
        

        
        for i in range(batch_size):
            node_mask = (batch.batch == i)
            x = node_embs[node_mask]
            node_num = x.shape[0]
            nodes_emb = F.pad(x, (0, 0, 0, max_node_num - node_num))
            nodes_emb_ls.append(nodes_emb)

        global_emb_ls = torch.zeros(batch_size).to(node_embs.device)
        nodes_emb_ls = cat_list(nodes_emb_ls)

        return global_emb_ls, nodes_emb_ls, valid_masks_ls, adj_mat_ls, edge_indexs_ls


class CustomActorCritic(MultiInputActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        ptrnet_kwargs = None,
        *args,
        **kwargs,
    ):
        self.attn_type = ptrnet_kwargs['attn_type']
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs,)
        self.ortho_init = False
        self.global_flops = 0

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomPolicyNetwork(self.features_dim)

    def _build(self, lr_schedule) -> None:
        # self._build_mlp_extractor()
        # latent_dim_pi = self.mlp_extractor.latent_dim_pi
        if self.attn_type == 'CAI' or self.attn_type == 'WOCAI':
            self.action_net = ActorNet(feature_dim = HIDDEN_DIM, attn_type = self.attn_type)
        self.value_net = CriticNet(feature_dim = GOLBAL_EMB_DIM, pi_dim = 32)
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                #self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,}
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Preprocess the observation if needed
        global_emb_ls, nodes_emb_ls, valid_masks_ls, adj_mat_ls, edge_idx_ls = self.extract_features(obs)
        # Evaluate the values for the given observations
        value_ls = self.value_net(global_emb_ls)
        action_ls, log_prob_ls, rglr_ls = self.action_net(nodes_emb_ls = nodes_emb_ls, 
                                                          valid_masks_ls = valid_masks_ls, 
                                                          adj_mat_ls = adj_mat_ls, 
                                                          edge_idx_ls = edge_idx_ls)
        machine_pri_ls = self.tp_action_net(nodes_emb_ls = nodes_emb_ls,
                                         valid_masks_ls = valid_masks_ls,
                                         is_evaluate = False,
                                         old_machine_pri_ls = None)
        #log_prob = log_prob_ls.sum()
        return action_ls, machine_pri_ls, value_ls, log_prob_ls, rglr_ls

    def predict_values(self, obs):
        global_emb_ls, nodes_emb_ls, valid_masks_ls, adj_mat_ls, edge_idx_ls = self.extract_features(obs)
        value_ls = self.value_net(global_emb_ls)
        return value_ls

    def evaluate_actions(self, obs, old_actions_ls, old_machine_pri_ls, is_evaluating_actions = True):
        # evaluate a batch
        # Preprocess the observation if needed
        global_emb_ls, nodes_emb_ls, valid_masks_ls, adj_mat_ls, edge_idx_ls = self.extract_features(obs)
        value_ls = self.value_net(global_emb_ls)
        if is_evaluating_actions:
            action_ls, log_prob_ls, rglr_ls = self.action_net(nodes_emb_ls = nodes_emb_ls, 
                                                            valid_masks_ls = valid_masks_ls, 
                                                            adj_mat_ls = adj_mat_ls,
                                                            is_evaluate = True,
                                                            old_actions_ls = old_actions_ls,
                                                            edge_idx_ls = edge_idx_ls)
            machine_pri_ls = self.tp_action_net(nodes_emb_ls = nodes_emb_ls,
                                                valid_masks_ls = valid_masks_ls,
                                                is_evaluate = True,
                                                old_machine_pri_ls = old_machine_pri_ls)
        else:
            action_ls, log_prob_ls, rglr_ls = self.action_net(nodes_emb_ls = nodes_emb_ls, 
                                                            valid_masks_ls = valid_masks_ls, 
                                                            adj_mat_ls = adj_mat_ls,
                                                            edge_idx_ls = edge_idx_ls)  
            machine_pri_ls = self.tp_action_net(nodes_emb_ls = nodes_emb_ls,
                                                valid_masks_ls = valid_masks_ls,
                                                is_evaluate = False,
                                                old_machine_pri_ls = None)          
        return action_ls, machine_pri_ls, value_ls, log_prob_ls, rglr_ls

'''
def parse_args():
    parse = argparse.ArgumentParser(description='initialize DRL parameters')
    parse.add_argument('--dataset_name', type=str)
    parse.add_argument('--n_steps', type=int)
    parse.add_argument('--GNN_model', type=str)
    args = parse.parse_args()
    return args
'''


class CriticNet(torch.nn.Module):
    def __init__(self, feature_dim, pi_dim):
        super().__init__()
        self.value_net = ops.MLP(in_channels = feature_dim,
                                    hidden_channels= [32, 16, 1],
                                    activation_layer =torch.nn.ReLU)

    def forward(self, global_emb_ls):
        value_ls = torch.zeros(len(global_emb_ls)).to(global_emb_ls.device)
        return value_ls

class ActorNet(torch.nn.Module):
    def __init__(self, feature_dim, attn_type):
        super().__init__()
        self.attn_type = attn_type
        
        self.logits0 = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
            ) 

        ''' for gumbel-k '''
        self.gumbel_toposort_in_tuple = GumbelSort()
        self.gumbel_topo_adj = GumbelTopoSortAdj() 

        
    def MLP_losits(self, valid_nodes_emb):
        logits = self.logits0(valid_nodes_emb) 
        logits = torch.reshape(logits, (-1,))
        ''' normalized_logits'''
        norm_logits = 5.0 * (logits - logits.mean()) / (logits.std() + 1e-20)
        return logits, norm_logits 

    def forward(self, 
                nodes_emb_ls, 
                valid_masks_ls, 
                adj_mat_ls,
                is_evaluate = False,
                old_actions_ls = None, 
                edge_idx_ls = None
                ):
        if is_evaluate == False:
            action_ls = []
            log_prob_ls = []
            rglr_ls = []
            for nodes_emb, valid_mask, adj_mat in zip(nodes_emb_ls, valid_masks_ls, adj_mat_ls):
                valid_nodes_emb = nodes_emb[valid_mask]
                
                logits, norm_logits = self.MLP_losits(valid_nodes_emb)
                norm_rglr = torch.mean(torch.pow(logits, 2))
                logits = norm_logits
                if self.attn_type == 'CAI':
                    log_prob, gumbel_logits = self.gumbel_topo_adj(logits, adj_mat, is_evaluate) #待补充
                elif self.attn_type == 'WOCAI':
                    log_prob, gumbel_logits = self.gumbel_toposort_in_tuple(logits, is_evaluate)
                log_prob = F.pad(log_prob,(0, max_node_num - log_prob.shape[0]))
                action_ls.append(F.pad(gumbel_logits, (0, max_node_num - gumbel_logits.shape[0])))
                log_prob_ls.append(log_prob)       
                rglr_ls.append(norm_rglr)    
            action_ls = cat_list(action_ls)
            log_prob_ls = cat_list(log_prob_ls)
            rglr_ls = cat_list(rglr_ls)
            return action_ls, log_prob_ls, rglr_ls 
        else:
            batchsize = old_actions_ls.shape[0]
            given_logits_ls = old_actions_ls.permute(1, 0, 2) 
            action_ls = []
            log_prob_ls = []
            rglr_ls = []
            for nodes_emb, valid_mask, adj_mat, given_logits_s in zip(nodes_emb_ls, valid_masks_ls, adj_mat_ls, given_logits_ls):
                valid_nodes_emb = nodes_emb[valid_mask]
                actions = []
                log_probs = []
                rglrs = []
                logits, norm_logits = self.MLP_losits(valid_nodes_emb)
                norm_rglr = torch.mean(torch.pow(logits, 2))
                logits = norm_logits
                for given_logits in given_logits_s:
                    #log_prob, gumbel_logits = self.gumbel_toposort_in_tuple(logits, is_evaluate, given_logits)
                    log_prob, gumbel_logits = self.gumbel_topo_adj(logits, adj_mat, is_evaluate, given_logits)
                    log_prob = F.pad(log_prob,(0, max_node_num - log_prob.shape[0]))
                    log_probs.append(log_prob)    
                log_probs = cat_list(log_probs)   
                rglrs.append(norm_rglr)         
                log_prob_ls.append(log_probs)
            log_prob_ls = cat_list(log_prob_ls)
            rglr_ls = cat_list(rglrs).unsqueeze(0).expand(batchsize, -1) 
            log_prob_ls = log_prob_ls.permute(1, 0, 2)
            return None, log_prob_ls, rglr_ls

class ActorNet_tpDispatch_only(torch.nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.t_p_dispatch = RectangleAttn(feature_dim, 1, feature_dim)

    def forward(self, 
                nodes_emb_ls, 
                valid_masks_ls, 
                is_evaluate = False, 
                old_machine_pri_ls = None):
        #if is_evaluate:
        #    return old_machine_pri_ls
        machine_pri_ls = []
        for nodes_emb, valid_mask in zip(nodes_emb_ls, valid_masks_ls):
            
            valid_nodes_emb = nodes_emb[valid_mask]
            node_num = valid_nodes_emb.shape[0]
            machine_emb = torch.tensor([1,2,3,4], dtype=torch.float, device=nodes_emb.device)
            machine_emb = machine_emb.unsqueeze(-1)
            
            machine_pri = self.t_p_dispatch(valid_nodes_emb, machine_emb)

            machine_pri = F.pad(machine_pri, (0, 0, 0, max_node_num - node_num))
            machine_pri_ls.append(machine_pri)
        machine_pri_ls = cat_list(machine_pri_ls)
        return machine_pri_ls
        
