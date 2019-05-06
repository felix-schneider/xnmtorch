from typing import Optional, Union, Sequence, List


class Sample:
    """
    A template class to represent a single data example of any type, used for both model input and output.

    Args:
        idx: running sample number (0-based; unique among samples loaded from the same file, but not across files)
        score: a score given to this sample by a model
    """

    def __init__(self, idx: Optional[int] = None, score: Optional[float] = None):
        self.idx = idx
        self.score = score

    def __getitem__(self, item):
        """
        Get an item or a slice of the sample

        Args:
            item: index or slice

        Returns:
            Depends on implementation
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Return length of input, included padding.

        Returns:
            length
        """
        raise NotImplementedError

    @property
    def len_unpadded(self) -> int:
        """
        Return length of input prior to applying any padding.

        Returns:
            unpadded length
        """
        raise NotImplementedError

    def pad(self, pad_len: int) -> 'Sample':
        """
        Return a new, padded version of this sample (or self if pad_len is zero).
        Args:
            pad_len: length to pad to?

        Returns:
            padded sample
        """
        raise NotImplementedError

    def truncate(self, trunc_len: int) -> 'Sample':
        """
        Create a new, right-truncated version of the sample

        Args:
            trunc_len: length to truncate to?

        Returns:
            truncated sample
        """
        raise NotImplementedError

    def unpad(self) -> 'Sample':
        """
        Return the unpadded sample
        If self is unpadded, return self, if not return reference to original unpadded sentence if possible, otherwise
        create a new sentence.
        """
        if len(self) == self.len_unpadded:
            return self
        else:
            return self[:self.len_unpadded]


class StringSample(Sample):
    """
    A base class for sentences based on readable strings.

    Args:
      idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
      score: a score given to this sentence by a model
      output_procs: output processors to be applied when calling sent_str()
    """

    def __init__(self, idx: Optional[int], score: Optional[float] = None,
                 output_procs: Union[OutputProcessor, Sequence[OutputProcessor]] = []):
        super().__init__(idx=idx, score=score)
        self.output_procs = output_procs

    def tokens(self, **kwargs) -> List[str]:
        """
        Return list of readable string tokens.

        Args:
          **kwargs: should accept arbitrary keyword args

        Returns: list of tokens.
        """
        raise NotImplementedError

    def sent_str(self, custom_output_procs=None, **kwargs) -> str:
        """
        Return a single string containing the readable version of the sentence.

        Args:
          custom_output_procs: if not None, overwrite the sentence's default output processors
          **kwargs: should accept arbitrary keyword args

        Returns: readable string
        """
        out_str = " ".join(self.str_tokens(**kwargs))
        pps = self.output_procs
        if custom_output_procs is not None:
            pps = custom_output_procs
        if isinstance(pps, OutputProcessor):
            pps = [pps]
        for pp in pps:
            out_str = pp.process(out_str)
        return out_str

    def __repr__(self):
        return f'"{self.sent_str()}"'

    def __str__(self):
        return self.sent_str()


class ScalarSample(Sample):
    """
    A sentence represented by a single integer value, optionally interpreted via a vocab.

    This is useful for classification-style problems.

    Args:
      value: scalar value
      idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
      vocab: optional vocab to give different scalar values a string representation.
      score: a score given to this sentence by a model
    """

    def __init__(self, value: int, idx: Optional[int] = None, vocab: Optional[Vocab] = None,
                 score: Optional[float] = None):
        super().__init__(idx=idx, score=score)
        self.value = value
        self.vocab = vocab

    def __getitem__(self, item):
        if isinstance(item, int):
            if item != 0:
                raise IndexError(item)
            return self.value
        else:
            if not isinstance(item, slice):
                raise IndexError(item)
            if item.start != 0 or item.stop != 1:
                raise IndexError(item)
            return self

    def __len__(self) -> int:
        return 1

    @property
    def len_unpadded(self) -> int:
        return 1

    def pad(self, pad_len: int) -> 'Sample':
        if pad_len != 1:
            raise ValueError("ScalarSentence cannot be padded")
        return self

    def truncate(self, trunc_len: int) -> 'Sample':
        if trunc_len < 1:
            raise ValueError("ScalarSentence cannot be truncated")
        return self

    def unpad(self) -> 'Sample':
        return self


class CompoundSample(Sample):
    """
    A compound sample contains several sample objects that present different 'views' on the same data examples.

    Args:
      samples: a list of samples
    """

    def __init__(self, samples: Sequence[Sample]):
        super().__init__(idx=samples[0].idx)
        if any(s.idx != self.idx for s in samples[1:]):
            raise ValueError("CompoundSample must contain samples of the same id")
        self.samples = samples

    def __getitem__(self, item):
        raise ValueError("Not supported for CompoundSample")
