import logging
import os
from pickle import UnpicklingError
from typing import Sequence, Union, Optional

import torch
import yaml
from apex import amp
from yaml.parser import ParserError

from xnmtorch import settings
from xnmtorch.eval.eval_tasks import EvalTask
from xnmtorch.eval.metrics import Metric
from xnmtorch.models import Model
from xnmtorch.persistence.serializable import Serializable, UninitializedYamlObject
from xnmtorch.train.regimens import TrainingRegimen
from xnmtorch.train.train_tasks import CheckpointReport


logger = logging.getLogger("experiment")


class ExpGlobal(Serializable):
    def __init__(self,
                 checkpoint_dir: str = settings.DEFAULT_CHECKPOINT_DIR,
                 report_dir: str = settings.DEFAULT_REPORT_DIR,
                 dropout: float = 0.3,
                 weight_noise: float = 0.0,  # currently unused
                 multiple: int = 1,
                 default_layer_dim: int = 512,
                 save_num_checkpoints: int = 1):
        self.checkpoint_dir = checkpoint_dir
        self.report_dir = report_dir
        self.dropout = dropout
        self.weight_noise = weight_noise
        self.multiple = multiple
        self.default_layer_dim = default_layer_dim
        self.save_num_checkpoints = save_num_checkpoints


class Experiment(Serializable):
    def __init__(self,
                 exp_global: ExpGlobal,
                 model: Model,
                 train: Optional[TrainingRegimen] = None,
                 evaluate: Optional[Union[EvalTask, Sequence[EvalTask]]] = None,
                 keep_checkpoints="best"):
        self.exp_global = exp_global
        self.model = model
        if settings.CUDA:
            self.model.cuda()

        self.train = train
        if evaluate is not None and not isinstance(evaluate, Sequence):
            evaluate = [evaluate]
        self.evaluate = evaluate
        self.keep_checkpoints = keep_checkpoints
        assert keep_checkpoints in ("best", "recent")

    @classmethod
    def load_experiment(cls, filename, spec_filename=None, for_training=False):
        try:
            # try to load as a checkpoint
            checkpoint = torch.load(filename, map_location="cpu")
            spec = checkpoint["spec"]
            state = checkpoint["state"]
        except UnpicklingError:
            # was just a spec
            with open(filename) as spec_file:
                spec = spec_file.read()
            state = None

        if state is None and spec_filename is not None:
            raise ValueError("Was instructed to load a spec, but no state was given")
        elif spec_filename is not None:
            with open(spec_filename) as spec_file:
                spec = spec_file.read()

        experiment = yaml.full_load(spec)

        if not isinstance(experiment, UninitializedYamlObject) or experiment.cls is not Experiment:
            raise ValueError(f"Top level object must be an Experiment")

        if not for_training:
            del experiment.yaml_args["train"]

        experiment = experiment.initialize()

        if state is not None:
            experiment.load_state_dict(state)

        return experiment

    @classmethod
    def load_model(cls, filename):
        checkpoint = torch.load(filename, map_location="cpu")
        spec = checkpoint["spec"]
        state = checkpoint["state"]
        experiment = yaml.full_load(spec)

        if not isinstance(experiment, UninitializedYamlObject) or experiment.cls is not Experiment:
            raise ValueError(f"Top level object must be an Experiment")

        if "train" in experiment.yaml_args:
            del experiment.yaml_args["train"]
        if "evaluate" in experiment.yaml_args:
            del experiment.yaml_args["evaluate"]

        experiment = experiment.initialize()
        experiment.load_state_dict(state)
        return experiment.model

    def save(self, dev_report: CheckpointReport):
        cp_dir = self.exp_global.checkpoint_dir
        os.makedirs(cp_dir, exist_ok=True)

        primary_metric = dev_report.eval_scores[0]
        filename = f"checkpoint_{str(primary_metric).replace(' ', '_')}_{self.train.step_num}.pt"

        existing_checkpoints = os.listdir(cp_dir)
        logger.info(f"Saving checkpoint as {filename}")
        if len(existing_checkpoints) >= self.exp_global.save_num_checkpoints:
            if self.keep_checkpoints == "best":
                worst = sorted(existing_checkpoints)[0 if primary_metric.higher_is_better else -1]
                logger.info(f"Removing {worst}")
                os.remove(os.path.join(cp_dir, worst))
            else:
                oldest = min(existing_checkpoints, key=lambda x: os.path.getctime(os.path.join(cp_dir, x)))
                logger.info(f"Removing {oldest}")
                os.remove(os.path.join(cp_dir, oldest))

        spec = self.dump()
        state = self.state_dict()

        checkpoint = {"spec": spec, "state": state}
        torch.save(checkpoint, os.path.join(cp_dir, filename))

    def state_dict(self) -> dict:
        state_dict = {"model": self.model.state_dict()}
        if self.train is not None:
            state_dict["train"] = self.train.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict["model"], strict=False)
        if self.train is not None:
            self.train.load_state_dict(state_dict["train"])

    def __call__(self) -> Sequence[Metric]:
        if self.train is not None:
            self.train.initialize_model()
            self.train.run_training(save_fct=self.save)
        else:
            self.model.initialize()

        eval_scores = []
        if self.evaluate is not None:
            for task in self.evaluate:
                eval_scores.extend(task.eval())
        return eval_scores


def _load_model(loader: yaml.Loader, node):
    path = loader.construct_python_str(node)
    return Experiment.load_model(path)


yaml.FullLoader.add_constructor("!LoadModel", _load_model)
