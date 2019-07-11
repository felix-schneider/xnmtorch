import os
from collections import namedtuple
from typing import Optional, Sequence

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Batch

from xnmtorch import settings
from xnmtorch.data.datasets import Dataset
from xnmtorch.eval.eval_tasks import EvalTask
from xnmtorch.losses import Loss
from xnmtorch.models import Model
from xnmtorch.persistence import Serializable, Ref

CheckpointReport = namedtuple("CheckpointReport", ["has_improved", "eval_scores"])


class TrainingTask:
    def __init__(self,
                 dataset: Dataset,
                 loss: Loss,
                 dev_tasks: Optional[Sequence[EvalTask]] = None,
                 model: Model = Ref("model"),
                 name="train",
                 report_dir=settings.DEFAULT_REPORT_DIR):
        self.model = model
        self.loss = loss
        self.dataset = dataset
        self.dev_tasks = dev_tasks
        self.current_best_score = None
        self.name = name
        self.iterator = dataset.get_iterator(shuffle=True, repeat=True)
        self.writer = SummaryWriter(os.path.join(report_dir, self.name))

    def get_output_and_loss(self, batch: Batch) -> (dict, Tensor):
        model_output = self.model(batch)
        return model_output, self.loss(model_output)

    def evaluate(self, step_num=None) -> CheckpointReport:
        dev_scores = []
        has_improved = False
        if self.dev_tasks is not None and len(self.dev_tasks) > 0:
            for dev_task in self.dev_tasks:
                dev_score = dev_task.eval(step_num=step_num)
                dev_scores.extend(dev_score)
            main_score = dev_scores[0]
            if self.current_best_score is None or main_score.better_than(self.current_best_score):
                self.current_best_score = main_score
                has_improved = True
        else:
            has_improved = True
        return CheckpointReport(has_improved, dev_scores)

    def state_dict(self) -> dict:
        return {"iterator": self.iterator.state_dict(),
                "current_best": None if self.current_best_score is None else self.current_best_score.dump()}

    def load_state_dict(self, state_dict: dict):
        self.iterator.load_state_dict(state_dict["iterator"])
        self.current_best_score = state_dict["current_best"]
        if self.current_best_score is not None:
            self.current_best_score = Serializable.load(self.current_best_score)
