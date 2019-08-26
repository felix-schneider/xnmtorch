import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from xnmtorch.data.vocab import Vocab
from xnmtorch.modules.embeddings import DenseWordEmbedding
from xnmtorch.modules.initializers import Initializer, XavierUniform, ConstantInitializer
from xnmtorch.persistence import Ref, Serializable
from xnmtorch.persistence.serializable import bare


class Generator(nn.Module):
    def forward(self, decoder_outputs, source_indices, decoder_mask=None, source_mask=None):
        raise NotImplementedError


class DefaultGenerator(Generator, Serializable):
    def __init__(self,
                 vocab: Vocab = None,
                 dim: int = Ref("exp_global.default_layer_dim"),
                 initializer: Initializer = bare(XavierUniform),
                 bias=True,
                 bias_initializer=bare(ConstantInitializer, val=0),
                 multiple: int = Ref("exp_global.multiple", 1),
                 embedding: DenseWordEmbedding = None):
        super().__init__()
        self._shared = embedding is not None
        self._vocab = vocab
        self._embedding = embedding
        self._dim = dim
        self.multiple = multiple

        if self._vocab is not None:
            assert self._embedding is None and self._dim is not None
            size = len(self._vocab)
            if size % multiple != 0:
                size += multiple - (size % multiple)
            self.weights = Parameter(torch.empty(len(self._vocab), self._dim))
            self.vocab_size = len(self._vocab)
        else:
            assert self._embedding is not None
            self.weights = self._embedding.weights
            if hasattr(self._embedding, "multiple"):
                self.multiple = self._embedding.multiple
                self.vocab_size = len(self._embedding.vocab)
            else:
                self.vocab_size = None

        if bias:
            self.bias = Parameter(torch.empty(self.weights.size(0)))
        else:
            self.register_parameter('bias', None)

        self.initializer = initializer
        self.bias_initializer = bias_initializer

        self.reset_parameters()

        self._register_load_state_dict_pre_hook(self._load_params)

    def reset_parameters(self):
        if not self._shared:
            self.initializer.initialize(self.weights)
        if self.bias is not None:
            self.bias_initializer.initialize(self.bias)

    def forward(self, decoder_outputs, source_indices, decoder_mask=None, source_mask=None):
        logits = F.linear(decoder_outputs, self.weights, self.bias)

        if self.multiple != 1 and self.vocab_size is not None:
            extra = self.multiple - (self.vocab_size % self.multiple)
            logits[..., -extra:] = float("-inf")

        return F.log_softmax(logits, -1)

    def _load_params(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if not self._shared:
            self.weights.reqires_grad_(False)
            self.weights.resize_(len(self._vocab), self.dim)
            self.weights.reqires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(False)
            self.bias.resize_(self.weights.size(0))
            self.bias.requires_grad_(True)
