import torch.nn as nn


class ClassScoresLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, target):
        """
        Calculates the policy gradient loss function.
        """
        return -batch[range(batch.shape[0]), target].mean()


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def forward(self, batch, target):
        return super().forward(batch / self.temperature, target)
