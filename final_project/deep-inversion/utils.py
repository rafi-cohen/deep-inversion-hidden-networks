import torch
import torch.nn as nn


class TotalVariationRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        """
        Calculates the Total Variation Regularization of the batch
        """
        reg = (
            torch.sum(torch.abs(batch[:, :, :, :-1] - batch[:, :, :, 1:]))
            + torch.sum(torch.abs(batch[:, :, :-1, :] - batch[:, :, 1:, :])))
        return reg


class l2NormRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        """
        Calculates the l2-Norm Regularization of the batch
        """
        return batch.norm()


class PriorRegularization(nn.Module):
    def __init__(self, a_tv, a_l2):
        super().__init__()
        self.tv = TotalVariationRegularization()
        self.l2 = l2NormRegularization()
        self.a_tv = a_tv
        self.a_l2 = a_l2

    def forward(self, batch):
        """
        Calculates the Prior Regularization of the batch
        """
        return self.a_tv * self.tv(batch) + self.a_l2 * self.l2(batch)


class FeatureRegularization(nn.Module):
    def __init__(self, model):
        super.__init__()
        self.running_means = [
            bn.running_mean for bn in model.modules() if isinstance(bn, nn.BatchNorm2d)]
        self.running_vars = [
            bn.running_var for bn in model.modules() if isinstance(bn, nn.BatchNorm2d)]

    def forward(self, batch, batch_running_means, batch_running_vars):
        assert len(self.running_means) == len(batch_running_means)
        mean_term = [torch.norm(batch_mean - bn_mean) for batch_mean,
                     bn_mean in zip(batch_running_means, self.running_means)]
        var_term = [torch.norm(batch_var - bn_var) for batch_var,
                    bn_var in zip(batch_running_vars, self.running_vars)]
        return sum(mean_term) + sum(var_term)


class DIRegularization(nn.Module):
    def __init__(self, model, a_tv, a_l2, a_f):
        super.__init__()
        self.a_f = a_f
        self.prior = PriorRegularization(a_tv, a_l2)
        self.feature = FeatureRegularization(model)

    def forward(self, batch, batch_running_means, batch_running_vars):
        return self.prior(batch) + self.a_f * self.feature(batch, batch_running_means, batch_running_vars)
