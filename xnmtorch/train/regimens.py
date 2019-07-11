import gc
import math
import logging
from datetime import datetime
from typing import Optional, Sequence, Callable

import torch
from apex import amp
from torch import nn

from xnmtorch import settings
from xnmtorch.data.datasets import Dataset
from xnmtorch.eval.eval_tasks import EvalTask
from xnmtorch.eval.metrics import ScalarMetric, Perplexity
from xnmtorch.losses import Loss
from xnmtorch.train.lr_schedulers import LearningRateScheduler
from xnmtorch.models import Model
from xnmtorch.train.optimizers import Optimizer
from xnmtorch.persistence import Serializable, Ref
from xnmtorch.train.train_tasks import TrainingTask


class TrainingRegimen:
    def run_training(self, save_fct: Callable):
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict):
        raise NotImplementedError

    @property
    def step_num(self):
        raise NotImplementedError

    @property
    def optimizer(self):
        raise NotImplementedError


def print_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)) and obj.is_cuda:
                print(type(obj), obj.size())
        except:
            pass


class SimpleTrainingRegimen(TrainingTask, TrainingRegimen, Serializable):
    def __init__(self,
                 dataset: Dataset,
                 loss: Loss,
                 optimizer: Optimizer,
                 scheduler: LearningRateScheduler,
                 dev_tasks: Optional[Sequence[EvalTask]] = None,
                 model: Model = Ref("model"),
                 num_training_steps: Optional[int] = None,
                 normalize_gradient=True,
                 max_grad_norm: Optional[float] = None,
                 dev_every: Optional[int] = None,
                 report_every: Optional[int] = None,
                 dev_zero=False,
                 reset_optim=False,
                 update_every: Optional[int] = None,
                 name="train",
                 report_dir=Ref("exp_global.report_dir", settings.DEFAULT_REPORT_DIR)):
        super().__init__(dataset, loss, dev_tasks, model, name, report_dir)
        self.optimizer_gen = optimizer
        self._optimizer = self.optimizer_gen.get_optimizer(self.model.parameters())
        self.scheduler = scheduler
        self.num_training_steps = num_training_steps
        self.normalize_gradient = normalize_gradient
        self.max_grad_norm = max_grad_norm
        self.dev_every = dev_every
        assert dev_every is None or (dev_tasks is not None and len(dev_tasks) > 0)
        self.dev_zero = dev_zero
        self.reset_optim = reset_optim
        self.update_every = update_every
        self.report_every = report_every

        self.current_training_steps = 0
        self.current_epoch = 0

        self.logger = logging.getLogger(name)

    def run_training(self, save_fct: Callable):
        if settings.CUDA:
            capability = torch.cuda.get_device_capability(0)[0]
            if settings.FP16 != "O0":
                if capability < 7:
                    self.logger.warning("You have selected half-precision training which"
                                        "may not be supported by your device")
                self.logger.info("Initializing fp16")
            elif capability >= 7:
                self.logger.info("Your device may support faster training with half-precision training")

        self.set_learning_rate(self.scheduler.learning_rate)

        if self.dev_zero and self.current_training_steps == 0:
            self.checkpoint(save_fct)

        try:
            self.train_loop(save_fct)
        except KeyboardInterrupt:
            self.logger.info("Stopped training due to user interrupt")

        try:
            self.checkpoint(save_fct)
        except KeyboardInterrupt:
            self.logger.info("Canceled checkpoint")

    # noinspection PyProtectedMember
    def train_loop(self, save_fct):
        amp.initialize(self.model, self.optimizer, enabled=settings.CUDA, opt_level=settings.FP16)

        num_params = sum(p.numel() for p in amp.master_params(self.optimizer))
        self.logger.info(f"{num_params:,d} parameters")
        self.logger.info(f"{len(self.dataset):,d} training examples")
        for task in self.dev_tasks:
            if hasattr(task, "dataset"):
                self.logger.info(f"{len(task.dataset):,d} {task.name} examples")

        self.model.train()
        timer = datetime.today()
        seen_examples = 0
        backward_steps = 0
        total_loss = 0
        ooms = 0

        report_samples = 0
        report_loss = 0
        last_index = self.iterator._iterations_this_epoch
        for i, batch in enumerate(self.iterator):
            if self.num_training_steps is not None and \
                    self.current_training_steps >= self.num_training_steps:
                break
            elif i == 0:
                self.logger.info("Starting training")

            try:
                model_output, loss = self.get_output_and_loss(batch)
            except RuntimeError as e:
                if 'out of memory' in str(e) or 'get_temporary_buffer' in str(e):
                    # TODO: Problem with delay_unscale?
                    stats = [f"CUDA OOM on forward {i}",
                             f"Batch size {batch.batch_size} {self.dataset.sample_name}s"] + \
                            [f"{k}: {v}" for k, v in self.dataset.get_batch_stats(batch).items()]
                    self.logger.error(" | ".join(stats))
                    self.logger.error(str(e))
                    self.optimizer.zero_grad()
                    del batch
                    if settings.CUDA:
                        torch.cuda.empty_cache()
                    seen_examples = 0
                    total_loss = 0
                    report_samples = 0
                    report_loss = 0
                    timer = datetime.today()
                    ooms += 1
                    print_tensors()
                    if ooms == 5:
                        raise e
                    continue
                else:
                    raise e

            with torch.no_grad():
                nll = model_output["nll"].sum().item()
                batch_size = model_output["nll"].ne(0.0).sum().item()
            del model_output

            total_loss += nll
            seen_examples += batch_size
            report_loss += nll
            report_samples += batch_size

            if self.logger.getEffectiveLevel() <= logging.DEBUG:
                stats = [f"Step {i}: {self.dataset.sample_name}s",
                         f"ppl {math.exp(nll / batch_size):.2f}",
                         f"Batch size {batch.batch_size} {self.dataset.sample_name}s"] + \
                        [f"{k}: {v}" for k, v in self.dataset.get_batch_stats(batch).items()]
                self.logger.debug(" | ".join(stats))

            train_this_step = self.update_every is None or backward_steps % self.update_every == 0

            try:
                with amp.scale_loss(loss, self.optimizer, delay_unscale=not train_this_step) as scaled_loss:
                    scaled_loss.backward()
            except RuntimeError as e:
                if 'out of memory' in str(e) or 'get_temporary_buffer' in str(e):
                    stats = [f"CUDA OOM on backward {i}",
                             f"Batch size {batch.batch_size} {self.dataset.sample_name}s"] + \
                            [f"{k}: {v}" for k, v in self.dataset.get_batch_stats(batch).items()]
                    self.logger.error(" | ".join(stats))
                    self.logger.error(str(e))
                    self.optimizer.zero_grad()
                    del loss
                    del batch
                    if settings.CUDA:
                        torch.cuda.empty_cache()
                    seen_examples = 0
                    total_loss = 0
                    report_samples = 0
                    report_loss = 0
                    timer = datetime.today()
                    ooms += 1
                    print_tensors()
                    if ooms == 5:
                        raise e
                    continue
                else:
                    raise e
            del loss

            backward_steps += 1

            if train_this_step:
                self.logger.debug(f"Training step {self.current_training_steps + 1}")
                if self.normalize_gradient:
                    self.multiply_gradients(1 / seen_examples)

                if self.max_grad_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_grad_norm)

                if self.report_every is not None and self.current_training_steps % self.report_every == 0:
                    if self.max_grad_norm is None:
                        grad_norm = math.sqrt(sum(p.grad.norm() ** 2 for p in amp.master_params(self.optimizer)
                                                  if p.grad is not None))

                    now = datetime.today()
                    metrics = [
                        Perplexity(report_loss, report_samples),
                        ScalarMetric("gnorm", grad_norm),
                        ScalarMetric("lr", self.get_learning_rate(), precision="4e"),
                        ScalarMetric(f"{self.dataset.sample_name}s/s",
                                     report_samples / (now - timer).total_seconds())
                    ]
                    timer = now
                    for metric in metrics:
                        metric.write_value(self.writer, self.current_training_steps)
                    print_metrics = [f"Step {self.current_training_steps + 1:05d}"] + [str(x) for x in metrics]
                    self.logger.info(" | ".join(print_metrics))
                    report_loss = 0
                    report_samples = 0

                self.optimizer.step()
                self.current_training_steps += 1

                current_index = self.iterator._iterations_this_epoch

                if current_index <= last_index:
                    self.current_epoch += 1
                    self.set_learning_rate(self.scheduler.update_epoch())

                self.set_learning_rate(self.scheduler.update_step())
                last_index = current_index
                self.optimizer.zero_grad()
                total_loss = 0
                seen_examples = 0

                if self.dev_every is not None and self.current_training_steps % self.dev_every == 0:
                    self.checkpoint(save_fct)
                    self.model.train()

    def multiply_gradients(self, factor):
        for param in amp.master_params(self.optimizer):
            if param.grad is not None:
                param.grad.mul_(factor)

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]  # we expect only one group

    def checkpoint(self, save_fct: Callable):
        dev_report = self.evaluate(step_num=self.step_num)
        self.logger.info(str(dev_report))
        if dev_report.has_improved:
            save_fct(dev_report)

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict["current_training_steps"] = self.current_training_steps
        state_dict["current_epoch"] = self.current_epoch
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["scheduler"] = self.scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self.current_training_steps = state_dict["current_training_steps"]
        self.current_epoch = state_dict["current_epoch"]
        if not self.reset_optim:
            if "scheduler" in state_dict:
                self.scheduler.load_state_dict(state_dict["scheduler"])
            self.optimizer.load_state_dict(state_dict["optimizer"])

    @property
    def step_num(self):
        return self.current_training_steps

    @property
    def optimizer(self):
        return self._optimizer



