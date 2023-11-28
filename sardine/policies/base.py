'''
SARDINE
Copyright (c) 2023-present NAVER Corp. 
MIT license
'''

from abc import ABC, abstractmethod
import numpy as np

class Policy(ABC):
    def __init__(self, env = None, slate_size : int = 10, 
                num_items : int = 1000, seed = None):

        if env is None:
            self.slate_size = slate_size
            self.num_items = num_items
        else:
            self.slate_size = env.unwrapped.slate_size
            self.num_items = env.unwrapped.num_items

        self.generator = np.random.default_rng(seed=seed)
    
    @abstractmethod
    def get_action(self, observation):
        pass