from apex import amp
from torch import nn, Tensor
from torchtext.data import Batch

from xnmtorch import settings
from xnmtorch.data.vocab import Vocab
from xnmtorch.eval.search_strategies import SearchStrategy, GreedySearch
from xnmtorch.modules.embeddings import WordEmbedding
from xnmtorch.modules.generators import Generator
from xnmtorch.modules.transducers import SequenceTransducer, IncrementalModule
from xnmtorch.persistence import Serializable


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._initialized = False

    def forward(self, input: Batch):
        raise NotImplementedError

    def inference(self, input: Batch):
        return self(input)

    def initialize(self, optimizer=None):
        if not self._initialized:
            amp.initialize(self, optimizer, enabled=settings.CUDA, opt_level=settings.FP16)
        self._initialized = True


class AutoregressiveModel(Model):
    # Uses its own output
    def get_initial_state(self, input: Batch) -> dict:
        raise NotImplementedError

    def reorder_state(self, state, indices):
        def reorder(module):
            if hasattr(module, "reorder_state") and module is not self:
                module.reorder_state(state, indices)
        self.apply(reorder)

    def inference(self, input: Batch, search_strategy: SearchStrategy = None):
        if search_strategy is None:
            search_strategy = GreedySearch()
        initial_state = self.get_initial_state(input)
        _, src_lengths = input.src
        return search_strategy.generate_output(self, initial_state, src_lengths)

    def inference_step(self, input: Tensor, state: dict):
        raise NotImplementedError

    def get_finish_mask(self, scores, outputs):
        raise NotImplementedError

    def postprocess_output(self, output):
        return output


class TranslationModel(AutoregressiveModel, Serializable):
    def __init__(self,
                 src_vocab: Vocab,
                 trg_vocab: Vocab,
                 src_embedding: WordEmbedding,
                 trg_embedding: WordEmbedding,
                 encoder: SequenceTransducer,
                 decoder: (SequenceTransducer, IncrementalModule),
                 generator: Generator,
                 batch_first: bool = True,
                 level: str = "word", ):
        super().__init__()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.level = level
        assert level in {"word", "char"}
        self.src_embedding = src_embedding
        self.trg_embedding = trg_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.batch_first = batch_first
        self.loss = nn.NLLLoss(ignore_index=trg_vocab.pad_index, reduction="none")

        self._register_state_dict_hook(self.save_vocabs)
        self._register_load_state_dict_pre_hook(self.load_vocabs)

    shared_params = [{".src_vocab", ".src_embedder.vocab"},
                     {".trg_vocab", ".trg_embedder.vocab"},
                     {".batch_first", ".encoder.batch_first", ".decoder.batch_first"}]

    def get_initial_state(self, input: Batch) -> dict:
        src_indices, src_lengths = input.src
        embedded = self.src_embedding(src_indices)
        src_pad_mask = 1 - src_indices.eq(self.src_vocab.pad_index)
        encoder_output = self.encoder(embedded, src_pad_mask)
        state = {"encoder_output": encoder_output, "src_indices": src_indices, "encoder_mask": src_pad_mask}
        return state

    def reorder_state(self, state, indices):
        for k in ("encoder_output", "src_indices", "encoder_mask"):
            state[k] = state[k].index_select(0 if self.batch_first else 1, indices)
        super().reorder_state(state, indices)

    def inference(self, input: Batch, search_strategy: SearchStrategy = None):
        output = super().inference(input, search_strategy)

        for sample in output:
            for search_output in sample:
                search_output["outputs"] = self.trg_vocab.indices_to_str(search_output["outputs"][:-1])

        if hasattr(input, "trg"):
            for sample, ref, ref_len in zip(output, *input.trg):
                ref_str = self.trg_vocab.indices_to_str(ref[1:ref_len-1])
                for search_output in sample:
                    search_output["ref"] = ref_str

        return output

    def postprocess_output(self, output):
        return self.trg_vocab.postprocess(output)

    def inference_step(self, input: Tensor, state: dict):
        if input is None:
            batch_size = state["src_indices"].size(0 if self.batch_first else 1)
            input = state["src_indices"].new_full((batch_size,), self.trg_vocab.bos_index)
        embedded = self.trg_embedding(input)
        decoder_output = self.decoder.forward_step(embedded, state)
        log_probs = self.generator(decoder_output, state["src_indices"], source_mask=state["encoder_mask"])
        return log_probs

    def get_finish_mask(self, scores, outputs):
        return outputs.eq(self.trg_vocab.eos_index)

    def forward(self, input: Batch):
        src_indices, src_lengths = input.src
        embedded = self.src_embedding(src_indices)
        src_pad_mask = src_indices.ne(self.src_vocab.pad_index)
        encoder_output = self.encoder(embedded, src_pad_mask)

        trg_indices, trg_lengths = input.trg
        if self.batch_first:
            decoder_input = trg_indices[:, :-1]
            targets = trg_indices[:, 1:].contiguous()
        else:
            decoder_input = trg_indices[:-1]
            targets = trg_indices[1:]
        # trg_indices has eos, which we don't want to attend to
        trg_pad_mask = decoder_input.ne(self.trg_vocab.pad_index) - decoder_input.eq(self.trg_vocab.eos_index)

        decoder_embedded = self.trg_embedding(decoder_input)

        decoder_outputs = self.decoder(decoder_embedded, trg_pad_mask, encoder_output, src_pad_mask)
        log_probs = self.generator(decoder_outputs, src_indices, trg_pad_mask, src_pad_mask)

        nll = self.loss(log_probs.view(-1, log_probs.size(-1)), targets.view(-1))

        return {"log_probs": log_probs, "nll": nll}

    @staticmethod  # this is a bit counter-intuitive... save_state_dict_hook is called with self as first arg...
    def save_vocabs(self, state_dict, prefix, local_metadata):
        state_dict[prefix + "src_vocab"] = self.src_vocab.state_dict()
        if self.src_vocab is not self.trg_vocab:
            state_dict[prefix + "trg_vocab"] = self.trg_vocab.state_dict()

    def load_vocabs(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + "src_vocab"
        if key not in state_dict:
            missing_keys.append(key)
            return
        self.src_vocab.load_state_dict(state_dict[key])
        del state_dict[key]
        if self.src_vocab is not self.trg_vocab:
            key = prefix + "trg_vocab"
            if key not in state_dict:
                missing_keys.append(key)
                return
            self.trg_vocab.load_state_dict(state_dict[key])
            del state_dict[key]

