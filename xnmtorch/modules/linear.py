import torch
from torch import nn
import torch.nn.functional as F

from xnmtorch.modules.initializers import XavierUniform, ConstantInitializer
from xnmtorch.persistence import Serializable
from xnmtorch.persistence.serializable import bare


def group_linear(linears, input, bias=False):
    weights = [linear.weight for linear in linears]

    weight = torch.cat(weights, dim=0)

    if bias:
        biases = [linear.bias for linear in linears]
        bias_ = torch.cat(biases)
    else:
        bias_ = None

    return F.linear(input, weight, bias_)


class Linear(nn.Linear, Serializable):
    def __init__(self, in_features, out_features, bias=True, weight_norm=False, initializer=bare(XavierUniform),
                 bias_initializer=bare(ConstantInitializer, val=0)):
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        super().__init__(in_features, out_features, bias)
        self.weight_norm = weight_norm
        if weight_norm:
            nn.utils.weight_norm(self, name='weight')

    def reset_parameters(self):
        self.initializer.initialize(self.weight)
        if self.bias is not None:
            self.bias_initializer.initialize(self.bias)


class MaxOut(nn.Module, Serializable):
    """
    Project the input up `pool_size` times, then take the maximum of the outputs.
    """

    def __init__(self, in_features, out_features, pool_size):
        super().__init__()
        self.in_features = in_features
        self.out_fetures = out_features
        self.pool_size = pool_size
        self.linear = nn.Linear(in_features, out_features * pool_size)

    def forward(self, inputs):
        original_size = inputs.size()

        projected = self.linear(inputs).view(*original_size[:-1], self.out_fetures, self.pool_size)
        out, _ = projected.max(-1)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, pool_size={}' \
            .format(self.in_features, self.out_fetures, self.pool_size)

