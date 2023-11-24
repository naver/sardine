'''
SARDINE
Copyright (c) 2023-present NAVER Corp. 
MIT license
'''

from .base import Policy
import numpy as np

class EpsilonGreedyOracle(Policy):
    def __init__(self, epsilon : float, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def get_action(self, observation):
        bernoulli_vector = self.generator.binomial(1, self.epsilon, size = (self.slate_size,))
        return np.where(bernoulli_vector, 
                        np.random.randint(self.num_items, size = (self.slate_size,)), 
                        - np.ones((self.slate_size,), dtype = int))

class EpsilonGreedyAntiOracle(Policy):
    def __init__(self, epsilon : float, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    
    def get_action(self, observation):
        bernoulli_vector = self.generator.binomial(1, self.epsilon, size = (self.slate_size,))
        return np.where(bernoulli_vector, 
                        np.random.randint(self.num_items, size = (self.slate_size,)), 
                        - 2 * np.ones((self.slate_size,), dtype = int))