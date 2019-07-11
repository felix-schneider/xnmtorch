import math

from torch.utils.tensorboard import SummaryWriter

from xnmtorch.persistence import Serializable


class Metric:
    higher_is_better = True

    @property
    def value(self):
        raise NotImplementedError

    @property
    def label(self):
        raise NotImplementedError

    def __str__(self):
        return f"{self.label}: {self.value}"

    def __repr__(self):
        return self.__str__()

    def better_than(self, another_score: 'Metric') -> bool:
        if another_score is None or another_score.value is None:
            return True
        elif self.value is None:
            return False
        assert type(self) is type(another_score)
        if self.higher_is_better:
            return self.value > another_score.value
        else:
            return self.value < another_score.value

    def write_value(self, writer: SummaryWriter, global_step=None):
        writer.add_scalar(self.label, self.value, global_step)


class ScalarMetric(Metric, Serializable):
    def __init__(self, label, value, precision="2f"):
        self._label = label
        self._value = value
        self.precision = precision

    @property
    def value(self):
        return self._value

    @property
    def label(self):
        return self._label

    def __str__(self):
        return f"{self.label}: {self.value:.{self.precision}}"


class Perplexity(Metric, Serializable):
    higher_is_better = False

    def __init__(self, loss, num_samples):
        self.loss = loss
        self.num_samples = num_samples

    @property
    def value(self):
        return math.exp(self.loss / self.num_samples)

    @property
    def label(self):
        return "ppl"

    def __str__(self):
        return f"{self.label}: {self.value:.2f}"


class BLEUScore(Metric, Serializable):
    def __init__(self, bleu, lowercase):
        self.bleu = bleu
        self.lowercase = lowercase

    @property
    def value(self):
        return self.bleu

    @property
    def label(self):
        return ("lowercase " if self.lowercase else "") + "BLEU"

    def __str__(self):
        return f"{self.value:.2f} {self.label}"


class Evaluator:
    def evaluate(self, hypotheses, references):
        raise NotImplementedError


class BLEU(Evaluator, Serializable):
    def __init__(self, lowercase=False):
        self.lowercase = lowercase

    def evaluate(self, hypotheses, references):
        if self.lowercase:
            hypotheses = map(str.lower, hypotheses)
            references = map(str.lower, references)

        import sacrebleu
        bleu = sacrebleu.raw_corpus_bleu(hypotheses, [references])
        return BLEUScore(bleu.score, self.lowercase)
