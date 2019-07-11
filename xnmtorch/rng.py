import random

import numpy as np
import torch


def set_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_state():
    return torch.random.get_rng_state(), np.random.get_state(), random.getstate()


def set_state(state):
    torch.random.set_rng_state(state[0])
    np.random.set_state(state[1])
    random.setstate(state[2])
