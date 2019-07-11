import logging
import os

import torch
from torchtext.data import Field, Dataset as TorchTextDataset, Example

from xnmtorch import settings
from xnmtorch.data.iterators import Iterator, BatchShuffledIterator
from xnmtorch.data.vocab import Vocab
from xnmtorch.persistence import Serializable, Ref


logger = logging.getLogger("dataset")


class Dataset:
    @property
    def sample_name(self):
        return "sample"

    def get_iterator(self, shuffle=False, repeat=False) -> Iterator:
        raise NotImplementedError

    def get_batch_stats(self, batch):
        raise NotImplementedError


class BaseTranslationDataset(Dataset):
    def __init__(self, batch_size,
                 level,
                 sort,
                 sort_within_batch,
                 batch_by_words,
                 batch_first,
                 multiple,
                 has_target=True):
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.sort = sort
        self.sort_within_batch = sort_within_batch
        self.multiple = multiple
        self.batch_by_words = batch_by_words
        self.level = level
        self.has_target = has_target

    @staticmethod
    def sort_key(ex):
        return len(ex.trg), len(ex.src)

    def get_sample_size(self, sample_len, current_count, current_size):
        if current_count == 0:
            old_len = 0
        else:
            old_len = int(current_size / current_count)

        real_len = max(sample_len, old_len)
        if real_len % self.multiple != 0:
            real_len += self.multiple - (real_len % self.multiple)

        return real_len * (current_count + 1)

    def get_batch_size(self, example_index, current_count, current_size):
        new_example = self[example_index]
        return max(self.get_sample_size(len(new_example.src), current_count, current_size),
                   self.get_sample_size(len(new_example.trg) + 1, current_count, current_size))

    def get_iterator(self, shuffle=False, repeat=False) -> Iterator:
        device = None
        if settings.CUDA:
            device = "cuda"

        if shuffle and self.batch_by_words:
            return BatchShuffledIterator(self, self.batch_size, device=device, batch_size_fn=self.get_batch_size,
                                         repeat=repeat, sort=self.sort, shuffle=shuffle,
                                         sort_within_batch=self.sort_within_batch)
        else:
            return Iterator(self, self.batch_size, device=device,
                            batch_size_fn=self.get_batch_size if self.batch_by_words else None,
                            repeat=repeat, sort=self.sort, shuffle=shuffle,
                            sort_within_batch=self.sort_within_batch)

    @property
    def sample_name(self):
        return "word" if self.batch_by_words else "sent"

    @torch.no_grad()
    def get_batch_stats(self, batch):
        src_idx, src_len = batch.src
        trg_idx, trg_len = batch.trg
        return {f"source {self.level}s": sum(src_len), f"target {self.level}s": sum(trg_len),
                "num_samples": len(src_len)}

    def postprocess(self, length, samples, vocab):
        # already padded
        if length % self.multiple != 0:
            pad_amount = self.multiple - (length % self.multiple)
            for sample in samples:
                sample.extend([vocab.pad_index] * pad_amount)

    def postprocess_src(self, samples, vocab):
        self.postprocess(len(samples[0]), samples, vocab)
        return samples

    def postprocess_trg(self, samples, vocab):
        self.postprocess(len(samples[0]) - 1, samples, vocab)
        return samples


class TranslationDataset(BaseTranslationDataset, TorchTextDataset, Serializable):
    def __init__(self, path, batch_size,
                 extensions=(".src", ".trg"),
                 src_vocab: Vocab = Ref("model.src_vocab"),
                 trg_vocab: Vocab = Ref("model.trg_vocab"),
                 level=Ref("model.level"),
                 sort=False,
                 sort_within_batch=False,
                 batch_by_words=True,
                 batch_first=Ref("model.batch_first", True),
                 multiple: int = Ref("exp_global.multiple", 1),
                 max_len=1000,
                 subword_model=None,
                 subword_alpha=0.1,
                 subword_nbest=64):
        self.max_len = max_len
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        tokenize = None
        if level != "word":
            tokenize = list

        if subword_model is not None:
            import sentencepiece as spm
            self.subword_model = spm.SentencePieceProcessor()
            self.subword_model.load(subword_model)
            tokenize = self.split_subwords
        else:
            self.subword_model = None
        self.subword_alpha = subword_alpha
        self.subword_nbest = subword_nbest

        logger.info(f"Loading {path}")
        src = Field(batch_first=batch_first, tokenize=tokenize, include_lengths=True,
                    preprocessing=None, postprocessing=self.postprocess_src)
        src.vocab = src_vocab
        if os.path.exists(os.path.expanduser(path + extensions[1])):
            has_target = True
            trg = Field(batch_first=batch_first, tokenize=tokenize, include_lengths=True,
                        init_token=src_vocab.bos_token, eos_token=trg_vocab.eos_token, is_target=True,
                        preprocessing=None, postprocessing=self.postprocess_trg)
            trg.vocab = trg_vocab
            fields = [('src', src), ('trg', trg)]

            TorchTextDataset.__init__(self,
                                      self.load_parallel_data(
                                          os.path.expanduser(path + extensions[0]),
                                          os.path.expanduser(path + extensions[1]),
                                          fields),
                                      fields)
        else:
            has_target = False
            fields = [('src', src)]

            TorchTextDataset.__init__(self,
                                      self.load_source_data(
                                          os.path.expanduser(path + extensions[0]),
                                          fields[0]),
                                      fields)
        BaseTranslationDataset.__init__(self, batch_size, level, sort, sort_within_batch, batch_by_words,
                                        batch_first, multiple, has_target)

    def split_subwords(self, text):
        return self.subword_model.SampleEncodeAsPieces(text, self.subword_nbest, self.subword_alpha)

    def load_parallel_data(self, source_filename, target_filename, fields):
        examples = []
        with open(source_filename) as src_file, open(target_filename) as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                example = Example.fromlist([src_line, trg_line], fields)
                if self.filter_example(example):
                    examples.append(example)
        return examples

    @staticmethod
    def load_source_data(filename, field):
        examples = []
        with open(filename) as src_file:
            for src_line in src_file:
                src_line = src_line.strip()
                examples.append(Example.fromlist([src_line], [field]))
        return examples

    def filter_example(self, example):
        return len(example.src) <= self.max_len and len(example.trg) <= self.max_len


class H5TranslationDataset(BaseTranslationDataset, TorchTextDataset, Serializable):
    class ExampleWrapper:
        CACHE_SIZE = 100

        def __init__(self, ds, fields):
            self.fields = fields
            self.ds = ds
            self.cache_idx = 0
            self.cache = self.ds[0:self.CACHE_SIZE]

        def __getitem__(self, i):
            if not (self.cache_idx <= i < self.cache_idx + self.CACHE_SIZE):
                self.cache_idx = i
                self.cache = self.ds[i:i + self.CACHE_SIZE]
            return Example.fromlist(self.cache[i - self.cache_idx], self.fields)

        def __len__(self):
            return len(self.ds)

    def __init__(self, path, batch_size,
                 src_vocab: Vocab = Ref("model.src_vocab"),
                 trg_vocab: Vocab = Ref("model.trg_vocab"),
                 level=Ref("model.level"),
                 sort_within_batch=False,
                 batch_by_words=True,
                 batch_first=Ref("model.batch_first", True),
                 multiple: int = Ref("exp_global.multiple", 1),):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        import h5py
        src = Field(batch_first=batch_first, include_lengths=True,
                    postprocessing=self.postprocess_src, use_vocab=False, pad_token=src_vocab.pad_index)
        trg = Field(batch_first=batch_first, include_lengths=True,
                    init_token=trg_vocab.bos_index, eos_token=trg_vocab.eos_index, pad_token=trg_vocab.pad_index,
                    is_target=True, postprocessing=self.postprocess_trg, use_vocab=False)
        trg.vocab = trg_vocab
        self.data = h5py.File(path, "r")
        self.src_lengths = self.data["src_len"][:]
        self.trg_lengths = self.data["trg_len"][:]
        fields = [("src", src), ("trg", trg)]
        logger.info(f"Loading {path}")

        TorchTextDataset.__init__(self,
                                  self.ExampleWrapper(self.data["examples"], fields),
                                  fields)
        BaseTranslationDataset.__init__(self, batch_size, level, False, sort_within_batch, batch_by_words,
                                        batch_first, multiple)

    def postprocess_src(self, samples, vocab):
        # Because of use_vocab=False, this function is always called with vocab=None
        self.postprocess(len(samples[0]), samples, self.src_vocab)
        return samples

    def postprocess_trg(self, samples, vocab):
        self.postprocess(len(samples[0]) - 1, samples, self.trg_vocab)
        return samples

    def get_batch_size(self, example_index, current_count, current_size):
        return max(self.get_sample_size(self.src_lengths[example_index], current_count, current_size),
                   self.get_sample_size(self.trg_lengths[example_index] + 1, current_count, current_size))
