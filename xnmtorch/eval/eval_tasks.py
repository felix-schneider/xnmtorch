import math
import os
from datetime import datetime
from typing import Sequence, Union, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from xnmtorch import settings, logging
from xnmtorch.data.datasets import Dataset
from xnmtorch.eval.metrics import Metric, Perplexity, Evaluator
from xnmtorch.eval.search_strategies import SearchStrategy, BeamSearch
from xnmtorch.models import Model, AutoregressiveModel
from xnmtorch.persistence import Serializable, Ref
from xnmtorch.persistence.serializable import bare


class EvalTask:
    def __init__(self, name="dev", report_dir=settings.DEFAULT_REPORT_DIR):
        self.name = name
        self.writer = SummaryWriter(os.path.join(report_dir, self.name))
        self.logger = logging.getLogger(name)

    def eval(self, step_num=None) -> Sequence[Metric]:
        raise NotImplementedError


class PerplexityEvalTask(EvalTask, Serializable):
    def __init__(self,
                 dataset: Dataset,
                 model: Model = Ref("model"),
                 name="dev",
                 report_dir=Ref("exp_global.report_dir")):
        super().__init__(name, report_dir)
        self.dataset = dataset
        self.model = model

    @torch.no_grad()
    def eval(self, step_num=None) -> Sequence[Metric]:
        self.logger.info(f"Starting {self.name}")
        self.model.eval()
        start_time = datetime.now()
        total_loss = 0
        total_losses = 0
        total_samples = 0
        for i, batch in enumerate(self.dataset.get_iterator(shuffle=False, repeat=False)):
            if i == 0:
                self.logger.info("Batching complete, evaluating")
            model_outputs = self.model(batch)
            total_loss += model_outputs["nll"].sum().item()
            total_losses += model_outputs["nll"].ne(0.0).sum().item()
            total_samples += batch.batch_size
        metric = Perplexity(total_loss, total_losses)
        metric.write_value(self.writer, step_num)
        total_time = datetime.now() - start_time
        self.logger.info(f"{self.name} complete in {total_time}. "
                         f"{total_samples / total_time.total_seconds():.1f} {self.dataset.sample_name}s/s")
        self.logger.info(str(metric))
        return [metric]


class DecodingEvalTask(EvalTask, Serializable):
    def __init__(self,
                 dataset: Dataset,
                 metrics: Union[Evaluator, Sequence[Evaluator]],
                 search_strategy: SearchStrategy = bare(BeamSearch, beam_size=5),
                 model: AutoregressiveModel = Ref("model"),
                 name="dev",
                 report_dir=Ref("exp_global.report_dir"),
                 result_path: Optional[str] = None,
                 print_output=False):
        super().__init__(name, report_dir)
        self.dataset = dataset
        if not isinstance(metrics, Sequence):
            metrics = [metrics]
        self.metrics = metrics
        self.model = model
        self.search_strategy = search_strategy
        self.result_path = result_path
        self.print_output = print_output
        self.result_file = None

    @torch.no_grad()
    def eval(self, step_num=None) -> Sequence[Metric]:
        self.logger.info(f"Starting {self.name}")
        self.model.eval()
        start_time = datetime.now()
        total_samples = 0
        results = []
        references = []
        iterator = self.dataset.get_iterator(shuffle=False, repeat=False)
        try:
            for i, batch in enumerate(iterator):
                if i == 0:
                    self.logger.info("Batching complete, evaluating")
                total_samples += batch.batch_size
                res = self.model.inference(batch, self.search_strategy)
                batch_results = [search_outputs[0]["outputs"] for search_outputs in res]
                if self.print_output:
                    print("\n".join(batch_results))
                self.write_results(batch_results)
                if len(self.metrics) != 0:
                    results.extend(batch_results)
                if "ref" in res[0][0]:
                    references.extend(search_outputs[0]["ref"] for search_outputs in res)
                self.logger.debug(f"{i}: {batch.batch_size} {self.dataset.sample_name}s")
        finally:
            if self.result_file is not None:
                self.result_file.close()
                self.result_file = None
        total_time = datetime.now() - start_time
        self.logger.info(f"{self.name} complete in {total_time}. "
                         f"{total_samples / total_time.total_seconds():.1f} {self.dataset.sample_name}s/s")
        metrics = [metric.evaluate(results, references) for metric in self.metrics]
        self.logger.info(" | ".join(str(metric) for metric in metrics))
        for metric in metrics:
            metric.write_value(self.writer, step_num)
        return metrics

    def write_results(self, results):
        if self.result_path is None:
            return

        if self.result_file is None:
            if os.path.dirname(self.result_path) != "":
                os.makedirs(os.path.dirname(self.result_path), exist_ok=True)

            if isinstance(results[0], str):
                self.result_file = open(self.result_path, "w")
            else:
                self.result_file = open(self.result_path, "wb")

        if isinstance(results[0], str):
            self.result_file.writelines(r + "\n" for r in results)
        else:
            raise NotImplementedError
            # torch.save(results, self.result_path)
