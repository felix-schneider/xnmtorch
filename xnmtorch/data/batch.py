from torch import Tensor
from torchtext.data import Batch as TorchTextBatch


def cast_to(data, dtype):
    if isinstance(data, Tensor) and data.is_floating_point():
        return data.to(dtype)
    elif isinstance(data, tuple):
        return tuple(cast_to(x, dtype) for x in data)
    else:
        return data


class Batch(TorchTextBatch):
    def to(self, dtype):
        for field in self.fields:
            try:
                data = getattr(self, field)
                data = cast_to(data, dtype)
                setattr(self, field, data)
            except AttributeError:
                continue
        return self

    __iter__ = None  # TODO: fixes a problem in amp that recognized the batch
                     # as being iterable and does not call to()
