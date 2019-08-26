import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

from xnmtorch.data.vocab import Vocab
from xnmtorch.modules.initializers import NormalInitializer
from xnmtorch.persistence import Serializable, Ref
from xnmtorch.persistence.serializable import bare


class WordEmbedding(nn.Module):
    pass


def embedding_dropout(inputs, dropout, weight, padding_idx=None, scale=None):
    mask = weight.new_empty((weight.size(0), 1)) \
        .bernoulli_(1 - dropout) \
        .expand_as(weight).div(1 - dropout)
    masked_embed_weight = mask * weight

    if scale is not None:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    return F.embedding(inputs, masked_embed_weight, padding_idx)


class DenseWordEmbedding(WordEmbedding):
    @property
    def weights(self):
        raise NotImplementedError


class StandardWordEmbedding(DenseWordEmbedding, Serializable):
    def __init__(self,
                 vocab: Vocab,
                 dim: int = Ref("exp_global.default_layer_dim"),
                 initializer=bare(NormalInitializer),
                 dropout: float = Ref("exp_global.dropout", 0.0),
                 multiple: int = Ref("exp_global.multiple", 1)):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.multiple = multiple
        self.initializer = initializer
        self.dropout = dropout

        self.size = len(self.vocab)
        if self.size % self.multiple != 0:
            self.size += self.multiple - (self.size % self.multiple)
        self.embeddings = Parameter(torch.empty(self.size, self.dim))

        self.reset_parameters()

        self._register_load_state_dict_pre_hook(self._load_params)

    def reset_parameters(self):
        self.initializer.initialize(self.embeddings)

    @property
    def weights(self):
        return self.embeddings

    def forward(self, inputs):
        if self.dropout > 0 and self.training:
            return embedding_dropout(inputs, self.dropout, self.embeddings, self.vocab.pad_index)
        else:
            return F.embedding(inputs, self.embeddings, self.vocab.pad_index)

    def _load_params(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.size = len(self.vocab)
        if self.size % self.multiple != 0:
            self.size += self.multiple - (self.size % self.multiple)
        self.embeddings.requires_grad_(False)
        self.embeddings.resize_(self.size, self.dim)
        self.embeddings.requires_grad_(True)


class PreTrainedWordEmbedding(DenseWordEmbedding, Serializable):
    def __init__(self, vocab,
                 path,
                 freeze=False,
                 dropout: float = Ref("exp_global.dropout")):
        super().__init__()
        self.vocab = vocab
        self.path = path
        self.freeze = freeze
        self.dropout = dropout
        requires_grad = not self.freeze
        if self.path is not None:
            self.word_embeddings = nn.Parameter(torch.as_tensor(np.load(self.path).astype(np.float32)), requires_grad)
            self.special_embeddings = nn.Parameter(torch.empty((self.vocab.num_specials, self.word_embeddings.size(-1))))
            NormalInitializer().initialize(self.special_embeddings)
        else:
            self.word_embeddings = nn.Parameter(torch.Tensor(), requires_grad)
            self.special_embeddings = nn.Parameter(torch.Tensor())

        self.save_processed_arg("path", None)

        self._register_load_state_dict_pre_hook(self._fix_embedding_size)

    @property
    def weights(self):
        return torch.cat((self.special_embeddings, self.word_embeddings))

    def forward(self, inputs):
        if self.dropout > 0 and self.training:
            return embedding_dropout(inputs, self.dropout, self.weights, self.vocab.pad_index)
        else:
            return F.embedding(inputs, self.weights, self.vocab.pad_index)

    def _fix_embedding_size(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
        old_emb = state_dict[prefix + "word_embeddings"]
        self.special_embeddings.requires_grad_(False)
        if not self.freeze:
            self.word_embeddings.requires_grad_(False)
        self.word_embeddings.resize_(*list(old_emb.size()))
        self.special_embeddings.resize_((self.vocab.num_specials, old_emb.size(-1)))
        self.special_embeddings.requires_grad_(True)
        if not self.freeze:
            self.word_embeddings.requires_grad_(True)
