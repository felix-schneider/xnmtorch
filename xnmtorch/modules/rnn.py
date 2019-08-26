from torch import nn

from xnmtorch.modules.transducers import SequenceTransducer
from xnmtorch.persistence import Serializable, Ref


class LSTM(nn.LSTM, SequenceTransducer, Serializable):
    def __init__(self,
                 input_size=Ref("exp_global.default_layer_dim"),
                 output_size=None,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 bidirectional=False,
                 dropout=Ref("exp_global.dropout", 0.0)):
        hidden_size = output_size if output_size is not None else input_size
        if bidirectional:
            assert hidden_size % 2 == 0
            hidden_size //= 2
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         bias=bias,
                         batch_first=batch_first,
                         dropout=dropout,
                         bidirectional=bidirectional)

    def forward(self, inputs, mask=None):
        input_lengths = mask.sum(1 if self.batch_first else 0)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, self.batch_first, False)
        outputs, _ = super().forward(inputs)
        return nn.utils.rnn.pad_packed_sequence(outputs, self.batch_first)[0]
