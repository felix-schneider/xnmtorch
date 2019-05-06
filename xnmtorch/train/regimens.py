from torch import Tensor


class TrainingRegimen:
    """
    A training regimen is a class that implements a training loop.
    """

    def run_training(self):
        """
        Run training steps in a loop until stopping criterion is reached.

        Args:
          save_fct: function to be invoked to save a model at dev checkpoints
        """
        raise NotImplementedError

    def update(self, trainer: optimizers.Optimizer):
        raise NotImplementedError
