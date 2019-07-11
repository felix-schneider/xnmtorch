from torch import nn

from xnmtorch.persistence import Serializable


class Loss(nn.Module):
    def forward(self, model_output: dict):
        raise NotImplementedError


class MLELoss(Loss, Serializable):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, model_output: dict):
        nll_loss = model_output["nll"]
        if self.label_smoothing > 0:  # label smoothing
            mask = nll_loss.ne(0.0)
            nll_loss = nll_loss.sum()
            log_probs = model_output["log_probs"]
            num_classes = log_probs.size(-1)
            log_probs = log_probs.view(-1, num_classes)
            smooth_loss = -log_probs.sum(dim=-1, keepdim=True)[mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.label_smoothing / (num_classes - 1)
            loss = (1 - self.label_smoothing) * nll_loss + eps_i * smooth_loss
        else:
            loss = nll_loss.sum()

        return loss
