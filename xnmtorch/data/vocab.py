import logging
import re
from collections import Counter

from torchtext.vocab import Vocab as TorchVocab

from xnmtorch.persistence import Serializable


bpe_re = re.compile("(@@ )|(@@ ?$)")

logger = logging.getLogger("vocab")


class Vocab(TorchVocab, Serializable):
    unk_token = TorchVocab.UNK
    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    
    def __init__(self, path, bpe=False, sentence_piece=False):
        counter = Counter()
        if path is not None:
            if not sentence_piece:
                with open(path) as file:
                    for line in file:
                        word, count = line.split()
                        counter[word] = int(count)
            else:
                with open(path) as file:
                    for line in file:
                        word, logprob = line.split("\t")
                        counter[word] = 1
            logger.info(f"Loaded vocab {path} with {len(counter) + 4:,d} words")
            self.save_processed_arg("path", None)
        super().__init__(counter, specials=[self.unk_token, self.pad_token, self.bos_token, self.eos_token])
        self.bpe = bpe
        self.sentence_piece = sentence_piece

    @property
    def pad_index(self):
        return self.stoi[self.pad_token]

    @property
    def bos_index(self):
        return self.stoi[self.bos_token]

    @property
    def eos_index(self):
        return self.stoi[self.eos_token]

    def indices_to_str(self, indices):
        return " ".join(self.itos[x] for x in indices)

    def postprocess(self, text):
        if self.bpe:
            return bpe_re.sub("", text)
        elif self.sentence_piece:
            return text.replace(" ", "").replace("\u2581", " ").lstrip()
        else:
            return text

    def state_dict(self):
        return {"words": self.itos}

    def load_state_dict(self, state_dict):
        self.itos = state_dict["words"]
        assert self.unk_index == self.itos.index(self.unk_token)
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

