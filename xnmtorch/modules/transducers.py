from torch import nn


class SequenceTransducer(nn.Module):
    def forward(self, inputs, mask=None):
        raise NotImplementedError


class IncrementalModule(nn.Module):
    def forward(self, inputs):
        raise NotImplementedError

    def forward_step(self, inputs, state: dict):
        raise NotImplementedError
