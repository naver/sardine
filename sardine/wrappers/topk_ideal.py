'''
SARDINE
Copyright (c) 2023-present NAVER Corp. 
MIT license
'''

"""TopK ranking with ideal item embeddings."""
from typing import Union

import numpy as np

import gymnasium as gym


class TopKIdeal(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        min_action: Union[float, np.ndarray],
        max_action: Union[float, np.ndarray],
    ):
        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)
        super().__init__(env)

        self.env = env
        self.slate_size = env.slate_size
        self.embedd_dim = env.num_topics

        self.action_space = gym.spaces.Box(low = np.zeros(self.embedd_dim, dtype=np.float32) + min_action,
                                        high = np.zeros(self.embedd_dim, dtype=np.float32) + max_action,
                                        shape=(self.embedd_dim,), dtype=np.float32)

    def action(self, action):
        dot_product = self.env.item_embedd @ action
        ind = np.argpartition(dot_product, - self.slate_size)[- self.slate_size:]
        slate = ind[np.flip(np.argsort(dot_product[ind]))]
        return slate
