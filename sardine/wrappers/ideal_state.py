'''
SARDINE
Copyright (c) 2023-present NAVER Corp. 
MIT license
'''

"""Fully observable MDP with ideal state."""
from typing import Tuple, Dict, Literal
from ..policies import Policy
from collections import OrderedDict
import numpy as np
import gymnasium as gym

_DATASET_FORMATS = Literal["dict", "sb3_rollout", "sb3_replay"]

class IdealState(gym.Wrapper):
    '''
        Observable variant of Sardine, i.e., the user state is known.
    '''
    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env)

        ### Observation = [cur_user_embedd, norm_recent_topics_hist, norm_bored_timeout]
        self.observation_space = gym.spaces.Box(low = 0, high = 1, shape=(env.unwrapped.num_topics * 3,), dtype=np.float32)

        # Note: when boredom_influence = "item" the user embedding will be static

    def reset(self, seed = None, options = None) -> Tuple[np.ndarray, Dict]:
        _, info = super().reset(seed = seed)
        return info["user_state"], info

    def step(self, slate) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        _, reward, terminated, truncated, info = super().step(slate)
        return info["user_state"], reward, terminated, truncated, info
    
    def _append_dict_values(self, old_dict, append_dict):
        for k in old_dict.keys():
            if isinstance(old_dict[k], dict):
                self._append_dict_values(old_dict[k], append_dict[k])
                continue
            old_dict[k].append(append_dict[k])
        return

    def _to_numpy(self, ep_dict):
        for k in ep_dict.keys():
            if isinstance(ep_dict[k], dict):
                self._to_numpy(ep_dict[k])
                continue
            ep_dict[k] = np.array(ep_dict[k])
        return
    
    def generate_dataset(self, n_users, policy: Policy, seed = None, dataset_type: _DATASET_FORMATS = "dict"):
        """
            Generate a dataset of trajectories from the environment.
            If sb3_rollout_buffer is toggled, the format will be a RolloutBuffer
            as in Stable Baselines 3 (requires PyTorch).
            Otherwise it is a dictionary of dictionaries.
        """
        dataset = OrderedDict()

        observation, _ = self.reset(seed = seed)
        self.action_space.seed(seed)
        u = 0
        if dataset_type == "sb3_rollout":
            ## TOCHANGE
            try:
                import torch
            except ModuleNotFoundError:
                raise ModuleNotFoundError("You need to install pytorch to generate a dataset in DB3 format.")
            from ..buffer import RolloutBuffer
            dataset = RolloutBuffer(n_users * self.env.unwrapped.H, 
                self.observation_space, 
                self.action_space, 
                device = "cpu", gamma = 1.0)
            ep_starts = np.array(True)
        elif dataset_type == "sb3_replay":
            try:
                import torch
            except ModuleNotFoundError:
                raise ModuleNotFoundError("You need to install pytorch to generate a dataset in DB3 format.")
            from ..buffer import DictReplayBuffer
            obs_space = gym.spaces.Dict(
                {
                    "state": self.observation_space,
                    "clicks": gym.spaces.MultiBinary(self.env.unwrapped.slate_size),
                }
            )
            dataset = DictReplayBuffer(n_users * self.H, 
                obs_space, 
                self.action_space, 
                device = "cpu",
                handle_timeout_termination = False)
        else:
            episode_dict = {"observation": [], "action": [], "reward": []}
        while u < n_users:
            action = policy.get_action(observation)
            next_obs, reward, terminated, truncated, info = self.step(action)
            done = terminated or truncated

            if dataset_type == "sb3_rollout":
                ## TOCHANGE
                dataset.add(observation, info["slate"], reward, ep_starts, torch.zeros(1), torch.ones(1))
                ep_starts = np.array(done)
            elif dataset_type == "sb3_replay":
                dataset.add({"state": observation, "clicks": info["clicks"]}, {"state": next_obs, "clicks": info["clicks"]}, info["slate"], reward, done, None)
            else:
                self._append_dict_values(episode_dict, {"observation": {"state": observation, "clicks": info["clicks"]}, "action": info["slate"], "reward": reward})

            if done:
                observation, _ = self.reset()
                if dataset_type == "dict":
                    self._to_numpy(episode_dict)
                    dataset[u] = episode_dict
                    episode_dict = {"observation": [], "action": [], "reward": []}
                u += 1
            else:
                observation = next_obs
        if dataset_type == "sb3_rollout":
            dataset.compute_returns_and_advantage(torch.zeros(1), np.array(True))
        return dataset