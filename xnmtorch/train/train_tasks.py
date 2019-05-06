import subprocess
from collections import Iterator

from torch import Tensor
from typing import Union, Sequence, Optional

from xnmtorch import logger
from xnmtorch.persistence import Serializable


class TrainingTask:
    """
    Base class for a training task. Training tasks can perform training steps
    and keep track of the training state, but may not implement the actual training
    loop.

    Args:
      model: The model to train
    """

    def __init__(self, model: models.TrainableModel):
        self.model = model

    def should_stop_training(self):
        """
        Returns:
          True iff training is finished, i.e. training_step(...) should not be called again
        """
        raise NotImplementedError("must be implemented by subclasses")

    def training_step(self, **kwargs) -> Tensor:
        """
        Perform forward pass for the next training step and handle training logic (switching epoch, reshuffling, ..)

        Args:
          **kwargs: depends on subclass implementations
        Returns:
          Loss
        """
        raise NotImplementedError("must be implemented by subclasses")

    def minibatches(self) -> Iterator:
        """
        Infinitely loop over training minibatches.

        Returns:
          Generator yielding (src_batch,trg_batch) tuples
        """

    def checkpoint_needed(self) -> bool:
        raise NotImplementedError("must be implemented by subclasses")

    def checkpoint(self, control_learning_schedule: bool = False) -> bool:
        """
        Perform a dev checkpoint.

        Args:
          control_learning_schedule: If ``False``, only evaluate dev data.
                                     If ``True``, also perform model saving, LR decay etc. if needed.
        Returns:
          ``True`` iff the model needs saving
        """
        raise NotImplementedError("must be implemented by subclasses")

    def cur_num_minibatches(self) -> int:
        """
        Current number of minibatches (may change between epochs, e.g. for randomizing batchers or if reload_command is given)
        """
        raise NotImplementedError("must be implemented by subclasses")

    def cur_num_sentences(self) -> int:
        """
        Current number of parallel sentences (may change between epochs, e.g. if reload_command is given)
        """
        raise NotImplementedError("must be implemented by subclasses")


class SimpleTrainingTask(TrainingTask, Serializable):
    """
    Args:
      model: a trainable supervised model
      src_file: The file for the source data.
      tgt_file: The file for the target data.
      dev_every: dev checkpoints every n training steps (0 for only after epoch)
      batcher: Type of batcher
      loss_calculator:
      run_for_epochs: number of epochs (None for unlimited epochs)
      lr_decay: decay learning rate by multiplying by this factor
      lr_decay_times:  Early stopping after decaying learning rate a certain number of times
      patience: apply LR decay after dev scores haven't improved over this many checkpoints
      initial_patience: if given, allows adjusting patience for the first LR decay
      dev_tasks: A list of tasks to run on the development set
      dev_combinator: A formula to combine together development scores into a single score to
                      choose whether to perform learning rate decay, etc.
                      e.g. 'x[0]-x[1]' would say that the first dev task score minus the
                      second dev task score is our measure of how good we're doing. If not
                      specified, only the score from the first dev task will be used.
      restart_trainer: Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying LR decay (https://arxiv.org/pdf/1706.09733.pdf)
      reload_command: Command to change the input data after each epoch.
                           --epoch EPOCH_NUM will be appended to the command.
                           To just reload the data after each epoch set the command to 'true'.
      sample_train_sents: If given, load a random subset of training sentences before each epoch. Useful when training data does not fit in memory.
      max_num_train_sents: Train only on the first n sentences
      max_src_len: Discard training sentences with source-side longer than this
      max_tgt_len: Discard training sentences with target-side longer than this
      name: will be prepended to log outputs if given
    """

    # TODO: defaults
    def __init__(self,
                 model: models.ConditionedModel,
                 src_file: Union[str, Sequence[str]],
                 tgt_file: str = None,
                 dev_every: int = 0,
                 batcher: batchers.Batcher = None,
                 loss_calculator: loss.LossCalculator = None,
                 run_for_epochs: Optional[int] = None,
                 lr_scheduler: schedulers.LearningRateScheduler = None,
                 dev_tasks: Sequence[eval_tasks.EvalTask] = None,
                 dev_combinator=None,
                 restart_trainer: bool = False,
                 # reload_command: Optional[str] = None,
                 name: Optional[str] = None,
                 max_num_train_sents: Optional[int] = None,
                 max_src_len: Optional[int] = None,
                 max_tgt_len: Optional[int] = None):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.dev_tasks = dev_tasks
        self.dev_combinator = dev_combinator

        self.restart_trainer = restart_trainer
        self.run_for_epochs = run_for_epochs

        self.early_stopping_reached = False

        self.training_state = TrainingState()

        # self.reload_command = reload_command

        self.model = model
        self.loss_calculator = loss_calculator

        self.max_num_train_sents = max_num_train_sents
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.batcher = batcher
        self.dev_loss_tracker = loss_trackers.DevLossTracker(self, dev_every, name)
        self.name = name

    def should_stop_training(self) -> bool:
        """
        Signal stopping if self.early_stopping_reached is marked or we exhausted the number of requested epochs.
        """
        return self.early_stopping_reached \
               or self.run_for_epochs is not None and \
               (self.training_state.epoch_num > self.run_for_epochs
                or (self.training_state.epoch_num == self.run_for_epochs and
                    self.training_state.steps_into_epoch >= self.cur_num_minibatches()))

    def cur_num_minibatches(self) -> int:
        """
        Current number of minibatches (may change between epochs, e.g. for randomizing batchers or if reload_command is given)
        """
        return len(self.src_batches)

    def cur_num_samples(self) -> int:
        """
        Current number of samples (may change between epochs, e.g. if reload_command is given)
        """
        return len(self.src_data)
