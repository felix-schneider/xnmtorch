from xnmtorch.eval.metrics import Metric
from xnmtorch.persistence import Serializable, Ref


class LearningRateScheduler:
    def update_step(self) -> float:
        raise NotImplementedError

    def update_epoch(self) -> float:
        raise NotImplementedError

    def update_dev_score(self, metric: Metric) -> float:
        raise NotImplementedError

    @property
    def learning_rate(self):
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict):
        raise NotImplementedError


class ConstantLearningRate(LearningRateScheduler, Serializable):
    def __init__(self, lr):
        self.lr = lr

    def update_step(self):
        return self.lr

    def update_epoch(self):
        return self.lr

    def update_dev_score(self, metric: Metric):
        return self.lr

    @property
    def learning_rate(self):
        return self.lr

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict):
        pass


class NoamLearningRate(LearningRateScheduler, Serializable):
    def __init__(self, warmup_steps, model_size=Ref("exp_global.default_layer_dim"), multiplier=1.0):
        self.init_lr = model_size ** (-0.5) * multiplier
        self.lr = self.init_lr
        self.step = 0
        self.warmup_steps = warmup_steps

    def update_step(self):
        self.step += 1
        if self.step < self.warmup_steps:
            self.lr = self.init_lr * self.step * self.warmup_steps ** (-1.5)
        else:
            self.lr = self.init_lr * self.step**(-0.5)
        return self.lr

    def update_epoch(self):
        return self.lr

    def update_dev_score(self, metric: Metric) -> float:
        return self.lr

    def state_dict(self) -> dict:
        return {"step": self.step}

    def load_state_dict(self, state_dict: dict):
        self.step = state_dict["step"]

    @property
    def learning_rate(self):
        return self.lr




