'''
SARDINE
Copyright (c) 2023-present NAVER Corp. 
MIT license
'''
import numpy as np
import os
from typing import List, Dict, Tuple, Literal
from .policies import Policy
from collections import deque, OrderedDict

from .consts import DATA_REC_SIM_EMBEDDS

import gymnasium as gym
from gymnasium import spaces

# Pre-defined boredom types and click models
_BOREDOM_TYPES = Literal["user_tloi", "user_car"]
_CLICK_MODELS = Literal["tdPBM", "mixPBM"]
_DATASET_FORMATS = Literal["dict", "sb3_rollout", "sb3_replay"]

class Sardine(gym.Env):
    '''
        a Simulator for Automated Recommendation in Dynamic and INteractive Environments
    '''
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_items : int, slate_size : int, num_topics : int, episode_length : int, 
                env_alpha : float, env_propensities : List[float], env_offset : float, env_slope: float, env_omega : float, 
                recent_items_maxlen : int, boredom_threshold : int, boredom_moving_window : int, 
                env_embedds : str, click_model : _CLICK_MODELS, rel_threshold : float, 
                diversity_penalty : float, diversity_threshold : int, click_prop : float,
                boredom_type : _BOREDOM_TYPES, rel_penalty : bool, render_mode=None, **kwargs):
        super().__init__()

        ### General parameters of the environment
        self.num_items = num_items
        self.num_topics = num_topics
        self.slate_size = slate_size
        self.H = episode_length

        ### Observation and action spaces
        self.observation_space = spaces.Dict(
            {
                "slate": spaces.MultiDiscrete([num_items] * slate_size),
                "clicks": spaces.MultiBinary(n = slate_size),
                "hist": spaces.Box(low = 0, high = 1, shape=(num_topics,), dtype=np.float32)
            }
        )
        self.action_space = spaces.MultiDiscrete([num_items] * slate_size)

        ### Click model
        self._set_propensities(click_model, click_prop, env_propensities, env_alpha)

        ### User preference model
        self.offset = env_offset
        self.slope = env_slope
        self.omega = env_omega
        self.rel_threshold = rel_threshold
        self.diversity_penalty = diversity_penalty
        self.diversity_threshold = diversity_threshold

        ### Boredom model
        self.recent_items_maxlen = recent_items_maxlen
        self.boredom_thresh = boredom_threshold
        self.boredom_moving_window = boredom_moving_window

        self.rel_penalty = rel_penalty
        self.boredom_type = boredom_type

        ### Item generation
        self._init_item_embeddings(env_embedds)
        self._set_topic_for_items()

    def _set_propensities(self, click_model: str, click_prop : float = None, env_propensities : List[float] = None, env_alpha : float = 1.0):
        '''
            Setting click propensities of the PBM click model.
        '''
        self.alpha = env_alpha # Discount applied from relevance to attractiveness

        if click_model=="tdPBM":
            self.propensities = np.power(click_prop, np.arange(self.slate_size))
        elif click_model=="mixPBM":
            probs = [0.5, 0.5]
            props = np.power(click_prop, np.arange(self.slate_size))
            #props = gammas[:, np.newaxis].repeat(self.slate_size, axis = 1)
            self.propensities = probs[0] * props + probs[1] * np.flip(props, axis = 0)
        else:
            self.propensities = env_propensities

    def _init_item_embeddings(self, env_embedds : str):
        """
        Initializes item embeddings
        """
        if env_embedds is None: # Generate new item embeddings
            # Item embeddings with only a certain number of topics
            # Values for other topics are completely zeroed out
            num_topics_per_item = 2 # Average number of topics per item
            self.item_embedd = np.random.rand(self.num_items, self.num_topics)

            # Create a boolean tensor of shape [num_items, num_topics] filled with False values
            mask = np.zeros((self.num_items, self.num_topics), dtype=bool)
            # Force items to have between 2 and 3 topics
            # Since all items which have single topics are identical to each other (i.e., same one-hot vector),
            # Utilize items between 2 and 3 topics
            num_true_values = np.random.randint(num_topics_per_item, num_topics_per_item + 2, (self.num_items,))
            for i in range(self.num_items):
                indices = np.random.permutation(self.num_topics)[:num_true_values[i]]
                mask[i, indices] = True
            self.item_embedd *= mask
            embedd_norm = np.linalg.norm(self.item_embedd, axis = -1)

            self.item_embedd /= embedd_norm[:, np.newaxis]
            np.save(os.path.join(DATA_REC_SIM_EMBEDDS, "item_embeddings"), self.item_embedd)
        else: # Load existing item embeddings
            self.item_embedd = np.load(os.path.join(DATA_REC_SIM_EMBEDDS, env_embedds))

    def _set_topic_for_items(self):
        """
        Sets main topic for each item
        """
        # with m > 1:
        self.item_comp = np.argmax(self.item_embedd, axis = 1)
        self.max_score = np.max(np.linalg.norm(self.item_embedd, axis = 1))

    def _compute_clicks(self, rels : np.ndarray, comps : np.ndarray) -> np.ndarray:
        '''
            PBM click model
        '''
        attr = self.alpha * rels
        if np.max(np.unique(comps, return_counts = True)[1]) >= self.diversity_threshold:
            attr /= self.diversity_penalty ### When too many similar are in the slate, the overall attractiveness of the slate decreases.
        click_probs = attr * self.propensities
        clicks = self.np_random.binomial(n = 1, p = click_probs)
        return clicks

    def _reset_user_embedds(self):
        '''
            Resets the user embedding
        '''
        # User embedding where users are only interested in a certain number of topics
        # Values for other topics are completely zeroed out

        num_topics_per_user = 4 # Average number of topics per user
        threshold = 1 - float(num_topics_per_user) / self.num_topics
        self.user_embedd = self.np_random.uniform(size = (self.num_topics,))
        mask = self.np_random.uniform(size = (self.num_topics,)) > threshold
        while sum(mask) <= num_topics_per_user - 2 or sum(mask) >= num_topics_per_user + 2: # Force users to have between 3 and 5 topics
            mask = self.np_random.uniform(size = (self.num_topics,)) > threshold
        self.user_embedd *= mask
        embedd_norm = np.linalg.norm(self.user_embedd)
        self.user_embedd /= embedd_norm

    def _initial_reco(self):
        """
        Initial slate recommendation with random items
        """
        return self.np_random.integers(low = 0, high = self.item_embedd.shape[0], size = (self.slate_size,))

    def reset(self, seed = None, options = None) -> Tuple[Dict, Dict]:
        '''
            The initial ranker returns the most qualitative document in each topic (or the 10 first topics, or multiple top_docs per topic)
        '''
        super().reset(seed=seed)

        self.boredom_counter = 0
        self.t = 0  # Index of the trajectory-wide timestep
        self.clicked_items = deque([], self.recent_items_maxlen)
        self.clicked_item_topics = deque([], self.recent_items_maxlen)
        self.clicked_step = deque([], self.recent_items_maxlen)
        self.all_clicked_items = []
        self.bored = np.zeros(self.num_topics, dtype = bool)
        self.bored_timeout = self.boredom_moving_window * np.ones(self.num_topics, dtype = int)

        ## User embeddings
        self._reset_user_embedds()

        ## Initial recommendation
        slate = self._initial_reco()

        ## Compute relevances
        slate_embedd = self.item_embedd[slate]    # slate_size, num_topics
        score = slate_embedd @ self.user_embedd   # slate_size
        norm_score = score / self.max_score # Normalize score
        if self.rel_threshold is None:
            relevances = 1 / (1 + np.exp(-(norm_score - self.offset) * self.slope))    ## Rescale relevance
        else:
            relevances = np.where(norm_score > self.rel_threshold, 1, 0)

        ## First interaction
        clicks = self._compute_clicks(relevances, self.item_comp[slate])
        clicked_items = np.where(clicks)[0]
        self.clicked_items.extend(slate[clicked_items])
        self.clicked_item_topics.extend(self.item_comp[slate[clicked_items]])
        self.clicked_step.extend(self.t * np.ones_like(clicked_items))
        self.all_clicked_items.extend(slate[clicked_items])

        ## Update the user state for the next step
        user_state = self._update_user_state(slate, clicked_items)

        info = {'user_state' : user_state, 'terminated' : False, 'clicks' : clicks}
        obs = {'slate' : slate, 'clicks' : clicks, 'hist' : self.norm_recent_topics_hist}
        return obs, info

    def get_st_oracle_slate(self, slate, relevances):
        """
        Short term oracle for a given user embedding.
        Completes a slate whose missing elements are replaced with -1
        """
        ind = np.argpartition(relevances, - self.slate_size)[- self.slate_size:]
        topk_relevances = np.argsort(relevances[ind])
        oracle_slate = ind[np.flip(topk_relevances)]
        antioracle_slate = ind[topk_relevances]

        return np.where(slate == -1, oracle_slate, np.where(slate == -2, antioracle_slate, slate))

    def _adjust_user_embedds(self, cur_u_embedd, bored_topics):
        # Boredom factor influences the user embeddings
        if self.boredom_type == "user_tloi": # Temporary loss-of-interest boredom: boring topic components are zeroed out
            ### Set bored component to 0
            for bt in bored_topics:
                cur_u_embedd[bt] = 0.0

        if self.boredom_type == "user_car": # Churn-and-return boredom: all components are zeroed out
            ### Set user embedding to 0 if there is any boredom
            if (self.bored == True).sum() > 0:
                cur_u_embedd[:] = 0.0

        return cur_u_embedd

    def _clicked_item_influence(self, slate, clicked_items):
        """
        Influence the user embedding with the clicked items in the slate
        """
        if len(slate[clicked_items]) > 0:
            # Compute the average of item embeddings for the clicked items in the slate
            slate_item_embedd = np.mean([self.item_embedd[it] for it in slate[clicked_items]])

            self.user_embedd = self.omega * self.user_embedd + (1 - self.omega) * slate_item_embedd
            embedd_norm = np.linalg.norm(self.user_embedd)
            self.user_embedd /= embedd_norm

    def _update_user_state(self, slate, clicked_items):
        ## Increment time step
        self.t += 1

        ## We remove old clicks from boredom "log"
        while len(self.clicked_step) > 0 and self.clicked_step[0] < self.t - self.boredom_moving_window:
            self.clicked_item_topics.popleft()
            self.clicked_step.popleft()
        self.all_clicked_items.extend(slate[clicked_items])

        ## Update bored_timeout in the next step
        self.bored_timeout -= self.bored.astype(int) # Remove one to timeout for bored topics
        self.bored = self.bored & (self.bored_timeout != 0) # "Unbore" timed out components
        self.bored_timeout[self.bored == False] = self.boredom_moving_window # Reset timer for "unbored" components

        ## Bored anytime recently items from one topic have been clicked more than boredom_threshold
        if len(self.clicked_item_topics) > 0:
            recent_topics = np.concatenate([it[np.newaxis] for it in self.clicked_item_topics])
            recent_topics_hist = np.bincount(recent_topics, minlength = self.num_topics)
            bored_topics = np.arange(self.num_topics)[recent_topics_hist >= self.boredom_thresh]
            ## Then, boredom is triggered for the topics on which have been clicked more than boredom_thresh
            self.bored[bored_topics] = True
        else:
            ## No clicked items in recent history
            recent_topics_hist = np.zeros(self.num_topics)
        bored_topics = np.nonzero(self.bored)[0]

        ## Let clicked items influence user behavior
        self._clicked_item_influence(slate, clicked_items)

        ## Apply boredom and short-term interest to the user embedding
        self.cur_user_embedd = self._adjust_user_embedds(self.user_embedd.copy(), bored_topics)

        ## Define user state and normalize vectors between 0 and 1
        self.norm_recent_topics_hist = np.clip(recent_topics_hist / self.boredom_thresh, 0, 1).astype('float32') # Clip between 0 and 1
        norm_bored_timeout = self.bored_timeout / self.boredom_moving_window
        bored = self.bored.astype(np.float32)
        user_state = np.concatenate([self.cur_user_embedd, self.norm_recent_topics_hist, norm_bored_timeout], axis=0, dtype=np.float32)
        #user_state = np.concatenate([self.user_embedd, self.norm_recent_topics_hist, norm_bored_timeout], axis=0, dtype=np.float32)
        #user_state = np.concatenate([self.cur_user_embedd, self.norm_recent_topics_hist, norm_bored_timeout, bored], axis=0, dtype=np.float32)

        return user_state

    def step(self, slate) -> Tuple[Dict, float, bool, bool, Dict]:
        '''
            Simulates user interaction.
        '''

        ## Compute relevances
        scores = self.item_embedd @ self.cur_user_embedd

        norm_scores = scores / self.max_score # Normalize score
        if self.rel_threshold is None:
            relevances = 1 / (1 + np.exp(-(norm_scores - self.offset) * self.slope))    ## Rescale relevance
        else:
            relevances = np.where(norm_scores > self.rel_threshold, 1, 0)

        ## Reduce overall relevance score if user is bored => Reflects interest in platform
        if self.rel_penalty:
            relevances *= (0.5 ** (self.bored == True).sum())

        # Short Time Oracle acts as greedily as possible.
        st_oracle = (-1 in slate) or (-2 in slate)
        if st_oracle:
            slate = self.get_st_oracle_slate(slate, relevances)
        relevances = relevances[slate]

        info = {}
        info["slate"] = slate
        info["slate_components"] = self.item_comp[slate]
        info["scores"] = norm_scores[slate]
        info["bored"] = self.bored
        info["relevances"] = relevances

        ## Interaction
        clicks = self._compute_clicks(relevances, self.item_comp[slate])
        clicked_items = np.where(clicks)[0]
        self.clicked_items.extend(slate[clicked_items])
        self.clicked_item_topics.extend(self.item_comp[slate[clicked_items]])
        self.clicked_step.extend(self.t * np.ones_like(clicked_items))
        info["clicks"] = clicks

        ## Update the user state for the next step
        user_state = self._update_user_state(slate, clicked_items)
        info["user_state"] = user_state

        ## Set terminated and return
        if self.t > self.H:
            terminated = True
            info["terminated"] = True
        else:
            terminated = False
            info["terminated"] = False

        obs = {'slate' : slate, 'clicks' : clicks, 'hist' : self.norm_recent_topics_hist}
        return obs, np.sum(clicks), terminated, False, info

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

    def generate_dataset(self, n_users : int, policy: Policy, seed: int = None, dataset_type: _DATASET_FORMATS = "dict"):
        """
            Generate a dataset of trajectories from the environment.
            If dataset_type in ["sb3_replay", sb3_rollout"], dataset will be a
            Replay/Rollout Buffer as in Stable Baselines 3 (requires PyTorch).
            Otherwise it is a dictionary of dictionaries.
        """
        dataset = OrderedDict()

        observation, _ = self.reset(seed = seed)
        self.action_space.seed(seed)
        u = 0
        if dataset_type == "sb3_rollout":
            try:
                import torch
            except ModuleNotFoundError:
                raise ModuleNotFoundError("You need to install pytorch to generate a dataset in DB3 format.")
            from .buffer import DictRolloutBuffer
            dataset = DictRolloutBuffer(n_users * self.H,
                self.observation_space,
                self.action_space,
                device = "cpu", gamma = 1.0)
            ep_starts = np.array(True)
        elif dataset_type == "sb3_replay":
            try:
                import torch
            except ModuleNotFoundError:
                raise ModuleNotFoundError("You need to install pytorch to generate a dataset in DB3 format.")
            from .buffer import DictReplayBuffer
            dataset = DictReplayBuffer(n_users * self.H,
                self.observation_space,
                self.action_space,
                device = "cpu",
                handle_timeout_termination = False)
        else:
            episode_dict = {"observation": {"slate": [], "clicks": [], "hist": []},
                            "action": [],
                            "reward": []}
        while u < n_users:
            action = policy.get_action(observation)
            next_obs, reward, terminated, truncated, info = self.step(action)
            done = terminated or truncated

            if dataset_type == "sb3_rollout":
                dataset.add(observation, info["slate"], reward, ep_starts, torch.zeros(1), torch.ones(1))
                ep_starts = np.array(done)
            elif dataset_type == "sb3_replay":
                dataset.add(observation, next_obs, info["slate"], reward, done, None)
            else:
                self._append_dict_values(episode_dict, {"observation": observation, "action": info["slate"], "reward": reward})

            if done:
                observation, _ = self.reset()
                if dataset_type == "dict":
                    self._to_numpy(episode_dict)
                    dataset[u] = episode_dict
                    episode_dict = {"observation": {"slate": [], "clicks": [], "hist": []},
                            "action": [],
                            "reward": []}
                u += 1
            else:
                observation = next_obs
        if dataset_type == "sb3_rollout":
            dataset.compute_returns_and_advantage(torch.zeros(1), np.array(True))
        return dataset