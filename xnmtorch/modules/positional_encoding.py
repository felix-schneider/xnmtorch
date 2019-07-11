import math

import torch
from torch.nn import Parameter

from xnmtorch.modules.initializers import XavierUniform, Initializer
from xnmtorch.modules.transducers import SequenceTransducer, IncrementalModule
from xnmtorch.persistence import Ref, Serializable
from xnmtorch.persistence.serializable import bare


class PositionalEncoding(SequenceTransducer, IncrementalModule):
    def __init__(self, model_dim, batch_first=True):
        super().__init__()
        self.model_dim = model_dim
        self.batch_first = batch_first

    def mask_outputs(self, outputs, input_mask):
        if self.batch_first:
            outputs.masked_fill_(input_mask.eq(0).unsqueeze(2), 0)
        else:
            outputs.masked_fill_(input_mask.eq(0).transpose(0, 1).unsqueeze(2), 0)


class SinusoidalPositionalEncoding(PositionalEncoding, Serializable):
    """
    Adds positional embeddings to standard word embeddings
    This matches the original TensorFlow implementation at
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py.

    Args:
        model_dim:   dimension of model
        batch_first: Whether the input and output have the batch or the time dimension first

    Inputs Shapes:
        inputs: batch_size x len_seq x model_dim  or  len_seq x batch_size x model_dim
        input_mask: batch_size x len_seq regardless of batch_first

    Outputs Shapes:
        out:   batch_size x len_seq x model_dim  or  len_seq x batch_size x model_dim

    """

    def __init__(self, model_dim=Ref("exp_global.default_layer_dim"), batch_first=True, initial_length=1024):
        super().__init__(model_dim, batch_first)
        self.register_buffer('pos_emb', None)
        self.current_length = -1
        shape = [1, 1]
        shape[0 if batch_first else 1] = initial_length
        self.generate(initial_length, torch.zeros(*shape))
        self._register_load_state_dict_pre_hook(self._fix_embedding_size)

    def generate(self, new_max_len, inputs):
        position = torch.arange(new_max_len).type_as(inputs)
        num_timescales = self.model_dim // 2
        log_timescale_increment = math.log(10000) / (num_timescales - 1)
        inv_timescales = torch.exp(
            torch.arange(0, num_timescales).type_as(inputs) * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        pos_emb = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), 1)
        self.pos_emb = pos_emb.to(inputs.device)
        self.current_length = new_max_len

    def forward(self, inputs, mask=None):
        seq_len = inputs.size(1 if self.batch_first else 0)

        if seq_len > self.current_length:
            self.generate(seq_len, inputs)

        emb = self.pos_emb[:seq_len, :]

        if not self.batch_first:
            emb = emb.unsqueeze(1)

        return emb

    def forward_step(self, inputs, state: dict):
        last_timestep = state.get(id(self), 0)
        current_timestep = last_timestep + 1
        state[id(self)] = current_timestep

        if current_timestep > self.current_length:
            self.generate(current_timestep * 2, inputs.device)

        emb = self.pos_emb[current_timestep-1, :]
        return emb

    def _fix_embedding_size(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
        old_emb = state_dict[prefix + 'pos_emb']
        self.pos_emb.resize_(*list(old_emb.size()))


class LearnedPositionalEncoding(PositionalEncoding, Serializable):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, max_length, model_dim=Ref("exp_global.default_layer_dim"), batch_first=True,
                 initializer: Initializer = bare(XavierUniform)):
        super().__init__(model_dim, batch_first)
        self.max_length = max_length
        self.pos_emb = Parameter(torch.zeros(max_length, model_dim))
        self.initializer = initializer
        self.reset_parameters()

    def reset_parameters(self):
        self.initializer.initialize(self.pos_emb)

    def forward(self, inputs, mask=None):
        seq_len = inputs.size(1 if self.batch_first else 0)
        needed_length = seq_len
        assert needed_length <= self.max_length

        emb = self.pos_emb[:needed_length, :]

        if not self.batch_first:
            emb = emb.unsqueeze(1)

        return emb

    def forward_step(self, inputs, state: dict):
        last_timestep = state.get(id(self), 0)
        current_timestep = last_timestep + 1
        state[id(self)] = current_timestep

        assert current_timestep <= self.max_length + 1

        emb = self.pos_emb[current_timestep - 1, :]
        return emb
