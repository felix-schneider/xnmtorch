from torch import nn

from xnmtorch.persistence import Serializable
from xnmtorch.persistence.serializable import add_alias


class Initializer:
    def initialize(self, tensor):
        raise NotImplementedError


class UniformInitializer(Initializer, Serializable):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def initialize(self, tensor):
        nn.init.uniform_(tensor, self.min, self.max)


class NormalInitializer(Initializer, Serializable):
    def __init__(self, mean=0.0, std=None):
        self.mean = mean
        self.std = std

    def initialize(self, tensor):
        if self.std is None:
            std = tensor.size(-1) ** -0.5
        else:
            std = self.std

        nn.init.normal_(tensor, self.mean, std)


class ConstantInitializer(Initializer, Serializable):
    def __init__(self, val):
        self.val = val

    def initialize(self, tensor):
        nn.init.constant_(tensor, self.val)


class XavierUniform(Initializer, Serializable):
    def __init__(self, gain=1.0):
        self.gain = gain

    def initialize(self, tensor):
        nn.init.xavier_uniform_(tensor, self.gain)


add_alias("!GlorotUniform", XavierUniform)


class XavierNormal(Initializer, Serializable):
    def __init__(self, gain=1.0):
        self.gain = gain

    def initialize(self, tensor):
        nn.init.xavier_normal_(tensor, self.gain)


