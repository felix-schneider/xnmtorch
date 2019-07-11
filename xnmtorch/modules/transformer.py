from typing import Optional

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from xnmtorch.modules.attention import get_self_attention_bias, get_encoder_attention_bias, MultiHeadAttention
from xnmtorch.modules.initializers import XavierUniform
from xnmtorch.modules.linear import Linear
from xnmtorch.modules.masking import MaskedFunction
from xnmtorch.modules.positional_encoding import PositionalEncoding, SinusoidalPositionalEncoding
from xnmtorch.modules.transducers import SequenceTransducer, IncrementalModule
from xnmtorch.persistence import Serializable, Ref
from xnmtorch.persistence.serializable import bare


class PrePostProcessing(nn.Module):
    """
    Applies processing to tensors
    Args:
        model_dim:          dimension of model
        dropout:            dropout probability
        elementwise_affine: Passed to LayerNorm
        gated_residuals:    Use gated residuals with a parameter
    sequence of processing steps:
        n = normalization
        d = dropout
        a = adding previous input to output (residual)
    """

    def __init__(self, model_dim, sequence='nda', dropout=0.0,
                 elementwise_affine=True, gated_residuals=False, masking=False):
        super(PrePostProcessing, self).__init__()
        self.masking = masking
        self.gated_residuals = gated_residuals
        self.steps = sequence

        if self.gated_residuals:
            self.k = nn.Parameter(torch.ones(1))

        if 'n' in self.steps:
            layer_norm = nn.LayerNorm([model_dim], elementwise_affine=elementwise_affine)
            self.layer_norm = MaskedFunction(layer_norm)
        if 'd' in self.steps:
            self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, tensor, input_tensor=None, mask=None):
        output = tensor
        if not self.masking:
            mask = None

        for step in self.steps:
            if step == 'n':
                output = self.layer_norm(output, mask=mask)
            elif step == 'd':
                output = self.dropout(output)
            elif step == 'a':
                if input_tensor is not None:
                    if self.gated_residuals:
                        output = F.relu(self.k) * output + input_tensor
                    else:
                        output = output + input_tensor

        return output


class FeedForward(nn.Module, Serializable):
    """
    Applies position-wise feed forward to inputs

    Args:
        model_dim:      dimension of model
        hidden_dim:     dimension of feed forward
        dropout:        dropout probability
        weight_norm:    use weight normalization on the weights

    Params:
        layer_1: FC layer from model_dim to hidden_dim
        layer_2: FC layer from hidden_dim to model_dim

    Input Shapes:
        input: batch_size x len x model_dim

    Output Shapes:
        out: batch_size x len x model_dim
    """

    def __init__(self,
                 model_dim,
                 hidden_dim,
                 dropout=Ref("exp_global.dropout", 0.1),
                 weight_norm=False):
        super().__init__()
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.layer_1 = Linear(model_dim, hidden_dim, weight_norm=weight_norm,
                              initializer=XavierUniform(nn.init.calculate_gain("relu")))
        self.layer_2 = Linear(hidden_dim, model_dim, weight_norm=weight_norm)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask=None):
        hidden = F.relu(self.layer_1(inputs), inplace=True)
        hidden = self.dropout(hidden)
        out = self.layer_2(hidden)
        return out


class TransformerEncoderLayer(SequenceTransducer):
    """
    Wraps multi-head attentions and position-wise feed forward into one encoder layer.

    Layers:
        (1)
         Layer norm
         Multi-head self-attention
         Dropout
         Residual with (1)
         (2)
         Layer norm
         Feed-forward
         Dropout
         Residual with (2)

    Feed-Forward:
        Configurable between linear -> ReLU -> linear and Maxout

    Args:
        model_dim:            dimension of model
        feed_forward:         feed-forward network
        attention:            self-attention module
        masked_layers:        whether to use masking for layer norm and feed forward. Useful for sparse masks
        gated_residuals:      whether to use gated residuals

    Params:
        attention:    multi-head self-attentions layer
        feed_forward:  feed forward layer

    Input Shapes:
        inputs:         batch_size x len_query x model_dim  or  len_query x batch_size x model_dim
        input_mask:     batch_size x len_query  or  len_query x batch_size (or broadcastable)
        attention_bias: batch_size x len_query x len_query or broadcastable, regardless of batch_first

    Output Shapes:
        out: batch_size x len_query x model_dim  or  len_query x batch_size x model_dim
    """

    def __init__(self, *, model_dim, feed_forward, attention, dropout=0.1, masked_layers=False, gated_residuals=False):
        super().__init__()
        self.feed_forward = feed_forward
        self.model_dim = model_dim
        self.masked_layers = masked_layers
        self.gated_residuals = gated_residuals

        self.preprocess_attn = PrePostProcessing(self.model_dim, 'n', masking=self.masked_layers)
        self.attention = attention
        self.postprocess_attn = PrePostProcessing(self.model_dim, 'da', dropout,
                                                  gated_residuals=self.gated_residuals)

        self.preprocess_ffn = PrePostProcessing(self.model_dim, 'n', masking=self.masked_layers)
        self.feed_forward = feed_forward
        self.postprocess_ffn = PrePostProcessing(self.model_dim, 'da', dropout,
                                                 gated_residuals=self.gated_residuals)

    def forward(self, inputs, mask=None, self_attention_bias=None):
        residual = inputs
        out = self.preprocess_attn(inputs, mask=mask)
        out, _ = self.attention(out, out, out, self_attention_bias, mask)
        out = self.postprocess_attn(out, residual)

        residual = out
        out = self.preprocess_ffn(out, mask=mask)
        out = self.feed_forward(out, mask=mask if self.masked_layers else None)
        out = self.postprocess_ffn(out, residual)
        return out


class TransformerDecoderLayer(SequenceTransducer, IncrementalModule):
    """
    Wraps multi-head self-attention, encoder-decoder attention and position-wise
    feed forward into one layer of decoder

    Layers:
        (1)
         Layer norm
         Multi-head self-attention
         Dropout
         Residual with (1)
         (2)
         Layer norm
         Multi-head query-context attention
         Dropout
         Residual with (2)
         (3)
         Layer norm
         Feed-forward
         Dropout
         Residual with (3)

    Feed-Forward:
        Configurable between linear -> ReLU -> linear and Maxout

    Args:
        model_dim:            dimension of model
        feed_forward:         feed forward network
        self_attention:       self-attention module
        enc_attention:        encoder attention module
        masked_layers:        whether to use masking for layer norm and feed forward. Useful for sparse masks
        gated_residuals:      whether to use gated residuals
        share_encoder:        instance of TransformerEncoderLayer to share parameters with

    Input Shapes:
        inputs:              len_query x batch_size x model_dim  or  batch_size x len_query x model_dim
        context:             len_context x batch_size x model_dim  or  batch_size x len_context x model_dim
        input_mask:          batch_size x len_query  or  len_query x batch_size
        context_mask:        batch_size x len_context  or  len_context x batch_size
        self_attention_mask: batch_size x len_query x len_query or broadcastable, regardless of batch_first

    Output Shapes:
        out:      len_query x batch_size x model_dim  or  len_query x batch_size x model_dim
    """

    _version = 2

    def __init__(self, *, model_dim, dropout=0.1, feed_forward=None, self_attention=None, enc_attention=None,
                 masked_layers=False, gated_residuals=False, share_encoder: TransformerEncoderLayer = None):
        super().__init__()
        self.feed_forward = feed_forward
        self.model_dim = model_dim
        self.masked_layers = masked_layers
        self.gated_residuals = gated_residuals

        if share_encoder is None:
            assert feed_forward is not None and self_attention is not None
            self.preprocess_self_attn = PrePostProcessing(self.model_dim, 'n', masking=self.masked_layers)
            self.self_attention = self_attention
            self.postprocess_self_attn = PrePostProcessing(self.model_dim, 'da', dropout,
                                                           gated_residuals=self.gated_residuals)

            self.preprocess_ffn = PrePostProcessing(self.model_dim, 'n', masking=self.masked_layers)
            self.feed_forward = feed_forward
            self.postprocess_ffn = PrePostProcessing(self.model_dim, 'da', dropout,
                                                     gated_residuals=self.gated_residuals)
        else:
            assert feed_forward is None and self_attention is None
            self.preprocess_self_attn = share_encoder.preprocess_attn
            self.self_attention = share_encoder.attention
            self.postprocess_self_attn = share_encoder.postprocess_attn

            self.preprocess_ffn = share_encoder.preprocess_ffn
            self.feed_forward = share_encoder.feed_forward
            self.postprocess_ffn = share_encoder.postprocess_ffn

        if enc_attention is not None:
            self.preprocess_enc_attn = PrePostProcessing(self.model_dim, 'n', masking=self.masked_layers)
            self.enc_attention = enc_attention
            self.postprocess_enc_attn = PrePostProcessing(self.model_dim, 'da', dropout,
                                                          gated_residuals=self.gated_residuals)
        else:
            self.enc_attention = None

        self._register_load_state_dict_pre_hook(self._update_names)

    def forward(self, inputs, mask=None, encoder_output=None, encoder_mask=None,
                self_attention_bias=None, encoder_attention_bias=None):
        assert (self.enc_attention is None) == (encoder_output is None)
        residual = inputs
        out = self.preprocess_self_attn(inputs, mask=mask)
        out, _ = self.self_attention(out, out, out, self_attention_bias, mask)
        out = self.postprocess_self_attn(out, residual)

        residual = out
        if self.enc_attention is not None:
            out = self.preprocess_enc_attn(out, mask=mask)
            out, _ = self.enc_attention(out, encoder_output, encoder_output, encoder_attention_bias,
                                        mask, encoder_mask)
            out = self.postprocess_enc_attn(out, residual)
            residual = out

        out = self.preprocess_ffn(out, mask=mask)
        out = self.feed_forward(out, mask=mask if self.masked_layers else None)
        out = self.postprocess_ffn(out, residual)

        return out

    def forward_step(self, inputs, state: dict, encoder_attention_bias=None):
        residual = inputs
        out = self.preprocess_self_attn(inputs)
        out, _ = self.self_attention.forward_step(out, out, out, state=state)
        out = self.postprocess_self_attn(out, residual)
        residual = out

        if self.enc_attention is not None:
            out = self.preprocess_enc_attn(out)
            out, _ = self.enc_attention.forward_step(out, state["encoder_output"], state["encoder_output"],
                                                     attention_bias=encoder_attention_bias,
                                                     value_mask=state["encoder_mask"],
                                                     state=state, static_kv=True)
            out = self.postprocess_enc_attn(out, residual)
            residual = out

        out = self.preprocess_ffn(out)
        out = self.feed_forward(out)
        out = self.postprocess_ffn(out, residual)

        return out

    def _update_names(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', 1)
        if version == 1 and prefix + 'version' not in state_dict:
            for key in self.preprocess_self_attn.state_dict().keys():
                state_dict[prefix + 'preprocess_self_attn.' + key] = state_dict.pop(prefix + 'preprocess_attn.' + key)
            for key in self.preprocess_enc_attn.state_dict().keys():
                state_dict[prefix + 'preprocess_enc_attn.' + key] = state_dict.pop(
                    prefix + 'preprocess_src_attn.' + key)
            for key in self.self_attention.state_dict().keys():
                state_dict[prefix + 'self_attention.' + key] = state_dict.pop(prefix + 'attention_trg.' + key)
            for key in self.enc_attention.state_dict().keys():
                state_dict[prefix + 'enc_attention.' + key] = state_dict.pop(prefix + 'attention_src.' + key)
        elif version == 1:
            del state_dict[prefix + 'version']


class TransformerEncoder(SequenceTransducer, Serializable):
    def __init__(self,
                 num_layers,
                 model_dim: int = Ref("exp_global.default_layer_dim"),
                 emb_dropout: float = Ref("exp_global.dropout", 0.1),
                 residual_dropout: float = Ref("exp_global.dropout", 0.1),
                 positional_encoding: PositionalEncoding = bare(SinusoidalPositionalEncoding),
                 attention=bare(MultiHeadAttention, num_heads=8),
                 feed_forward=bare(FeedForward, hidden_dim=2048),
                 batch_first=False,
                 masked_layers=False,
                 gated_residuals=False,
                 checkpointing_every: Optional[int] = None):
        super().__init__()
        self.model_dim: int = model_dim
        self.batch_first = batch_first
        self.masked_layers = masked_layers
        self.gated_residuals = gated_residuals

        self.preprocess = PrePostProcessing(model_dim, 'd', emb_dropout)
        self.postprocess = PrePostProcessing(model_dim, 'n', masking=masked_layers)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                model_dim=model_dim,
                dropout=residual_dropout,
                feed_forward=feed_forward if i == 0 else feed_forward.clone(),
                attention=attention if i == 0 else attention.clone(),
                masked_layers=masked_layers,
                gated_residuals=gated_residuals
            )
            for i in range(num_layers)
        ])

        self.positional_encoding = positional_encoding
        self.checkpointing_every = checkpointing_every

    shared_params = [{".model_dim", ".positional_encoding.model_dim",
                      ".attention.model_dim", ".feed_forward.model_dim"},
                     {".batch_first", ".positional_encoding.batch_first",
                      ".attention.batch_first", ".feed_forward.batch_first"},
                     {".masked_layers", ".positional_encoding.masked_layers",
                      ".attention.masked_layers", ".feed_forward.masked_layers"},
                     {".gated_residuals", ".positional_encoding.gated_residuals",
                      ".attention.gated_residuals", ".feed_forward.gated_residuals"}]

    def forward(self, inputs, mask=None):
        pos_emb = self.positional_encoding(inputs) if self.positional_encoding is not None else None

        inputs *= math.sqrt(self.model_dim)

        if pos_emb is not None:
            inputs += pos_emb

        inputs = self.preprocess(inputs)

        self_attention_bias = get_self_attention_bias(inputs, self.batch_first, False, mask)

        for i, layer in enumerate(self.layers):
            if self.checkpointing_every is not None and (i + 1) % self.checkpointing_every == 0:
                inputs = checkpoint(inputs, mask, self_attention_bias)
            else:
                inputs = layer(inputs, mask, self_attention_bias)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        outputs = self.postprocess(inputs, mask=mask)
        return outputs


class TransformerDecoder(SequenceTransducer, IncrementalModule, Serializable):
    def __init__(self,
                 num_layers: int,
                 model_dim: int = Ref("exp_global.default_layer_dim"),
                 emb_dropout: float = Ref("exp_global.dropout", 0.1),
                 residual_dropout: float = Ref("exp_global.dropout", 0.1),
                 positional_encoding: PositionalEncoding = bare(SinusoidalPositionalEncoding),
                 attention=bare(MultiHeadAttention, num_heads=8),
                 self_attention=None,
                 enc_attention=None,
                 feed_forward=bare(FeedForward, hidden_dim=2048),
                 batch_first=False,
                 masked_layers=False,
                 gated_residuals=False,
                 checkpointing_every: Optional[int] = None,
                 share_encoder=None):
        super().__init__()
        self.model_dim = model_dim
        self.batch_first = batch_first
        self.masked_layers = masked_layers
        self.gated_residuals = gated_residuals

        self.preprocess = PrePostProcessing(model_dim, 'd', emb_dropout)
        self.postprocess = PrePostProcessing(model_dim, 'n', masking=masked_layers)

        if share_encoder is None:
            if self_attention is None and attention is not None:
                self_attention = attention.clone()
            if enc_attention is None and attention is not None:
                enc_attention = attention.clone()

            self.layers = nn.ModuleList([
                TransformerDecoderLayer(
                    model_dim=model_dim,
                    dropout=residual_dropout,
                    feed_forward=feed_forward if i == 0 else feed_forward.clone(),
                    self_attention=self_attention if i == 0 else self_attention.clone(),
                    enc_attention=enc_attention if i == 0 or enc_attention is None else enc_attention.clone(),
                    masked_layers=masked_layers,
                    gated_residuals=gated_residuals
                )
                for i in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                TransformerDecoderLayer(
                    model_dim=model_dim,
                    enc_attention=enc_attention if i == 0 or enc_attention is None else enc_attention.clone(),
                    masked_layers=masked_layers,
                    gated_residuals=gated_residuals,
                    share_encoder=share_encoder.layers[i]
                )
                for i in range(num_layers)
            ])

        self.positional_encoding = positional_encoding
        self.checkpointing_every = checkpointing_every

    shared_params = [{".model_dim", ".positional_encoding.model_dim", ".self_attention.model_dim",
                      ".enc_attention.model_dim", ".feed_forward.model_dim", ".attention.model_dim"},
                     {".batch_first", ".positional_encoding.batch_first", ".self_attention.batch_first",
                      ".enc_attention.batch_first", ".feed_forward.batch_first", ".attention.batch_first"},
                     {".masked_layers", ".positional_encoding.masked_layers", ".self_attention.masked_layers",
                      ".enc_attention.masked_layers", ".feed_forward.masked_layers", ".attention.masked_layers"},
                     {".gated_residuals", ".positional_encoding.gated_residuals", ".self_attention.gated_residuals",
                      ".enc_attention.gated_residuals", ".feed_forward.gated_residuals", ".attention.gated_residuals"}]

    def forward(self, inputs, mask=None, encoder_outputs=None, encoder_mask=None):
        pos_emb = self.positional_encoding(inputs) if self.positional_encoding is not None else None

        inputs *= math.sqrt(self.model_dim)

        if pos_emb is not None:
            inputs += pos_emb

        inputs = self.preprocess(inputs)

        self_attention_bias = get_self_attention_bias(inputs, self.batch_first, True, mask)
        enc_attention_bias = get_encoder_attention_bias(encoder_outputs, self.batch_first, encoder_mask)

        for i, layer in enumerate(self.layers):
            if self.checkpointing_every is not None and (i + 1) % self.checkpointing_every == 0:
                inputs = checkpoint(layer, inputs, mask, encoder_outputs, encoder_mask,
                                    self_attention_bias, enc_attention_bias)
            else:
                inputs = layer(inputs, mask, encoder_outputs, encoder_mask,
                               self_attention_bias=self_attention_bias,
                               encoder_attention_bias=enc_attention_bias)

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        outputs = self.postprocess(inputs, mask=mask)
        return outputs

    def forward_step(self, inputs, state: dict):
        pos_emb = self.positional_encoding.forward_step(inputs, state) if self.positional_encoding is not None else None

        inputs *= math.sqrt(self.model_dim)

        if pos_emb is not None:
            inputs += pos_emb

        inputs = self.preprocess(inputs)

        encoder_mask = state["encoder_mask"]

        encoder_attention_bias = get_encoder_attention_bias(inputs, self.batch_first, encoder_mask)

        for layer in self.layers:
            inputs = layer.forward_step(inputs, state,
                                        encoder_attention_bias=encoder_attention_bias)

        outputs = self.postprocess(inputs)
        return outputs
