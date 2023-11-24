'''
SARDINE
Copyright (c) 2023-present NAVER Corp. 
MIT license
'''
from gymnasium.envs.registration import register
from .version import __version__

register(
    id="sardine/SingleItem-Static-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1000,
        "slate_size": 1,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "click_prop": 0.85,
        "env_offset": 0.65,
        "env_slope": 100,
        "env_omega": 1.0,
        "recent_items_maxlen": 10,
        "boredom_threshold": 11,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_focused_small.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SingleItem-BoredInf-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1000,
        "slate_size": 1,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.65,
        "env_slope": 100,
        "env_omega": 0.95,
        "recent_items_maxlen": 10,
        "boredom_threshold": 5,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_focused_small.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.85,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SingleItem-Uncertain-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1000,
        "slate_size": 1,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "click_prop": 0.85,
        "env_offset": 0.65,
        "env_slope": 10, # 10 or 20, instead of 100
        "env_omega": 1.0,
        "recent_items_maxlen": 10,
        "boredom_threshold": 11,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_focused_small.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SlateTopK-Bored-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1000,
        "slate_size": 10,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.65,
        "env_slope": 100,
        "env_omega": 1.0,
        "recent_items_maxlen": 10,
        "boredom_threshold": 5,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_focused_small.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.85,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SlateTopK-BoredInf-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1000,
        "slate_size": 10,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.65,
        "env_slope": 100,
        "env_omega": 0.95,
        "recent_items_maxlen": 10,
        "boredom_threshold": 5,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_focused_small.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.85,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SlateTopK-Uncertain-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1000,
        "slate_size": 10,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.65,
        "env_slope": 10,
        "env_omega": 0.95,
        "recent_items_maxlen": 10,
        "boredom_threshold": 5,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_focused_small.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.85,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SlateRerank-Static-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 10,
        "slate_size": 10,
        "num_topics": 10,
        "episode_length": 10,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.3,
        "env_slope": 5,
        "env_omega": 1.0,
        "recent_items_maxlen": 10,
        "boredom_threshold": 11,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_rerank.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.7,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SlateRerank-Bored-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 10,
        "slate_size": 10,
        "num_topics": 10,
        "episode_length": 10,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.3,
        "env_slope": 5,
        "env_omega": 1.0,
        "recent_items_maxlen": 10,
        "boredom_threshold": 4,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_rerank.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.7,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)
