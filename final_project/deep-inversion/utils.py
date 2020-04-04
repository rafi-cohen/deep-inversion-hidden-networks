import torch
import torch.nn as nn


class TotalVariationRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        """
        Calculates the Total Variation Regularization of the batch
        """
        diagonal_diff_top_left = (batch[:, :, :-1, :-1] - batch[:, :, 1:, 1:]).abs().sum()
        diagonal_diff_bottom_left = (batch[:, :, 1:, :-1] - batch[:, :, :-1, 1:]).abs().sum()
        vertical_diff = (batch[:, :, :-1, :] - batch[:, :, 1:, :]).abs().sum()
        horizontal_diff = (batch[:, :, :, :-1] - batch[:, :, :, 1:]).abs().sum()
        total_diff = diagonal_diff_top_left + diagonal_diff_bottom_left + vertical_diff + horizontal_diff
        return total_diff

        # diagonal_diff_top_left = torch.norm(batch[:, :, :-1, :-1] - batch[:, :, 1:, 1:])
        # diagonal_diff_bottom_left = torch.norm(batch[:, :, 1:, :-1] - batch[:, :, :-1, 1:])
        # vertical_diff = torch.norm(batch[:, :, :-1, :] - batch[:, :, 1:, :])
        # horizontal_diff = torch.norm(batch[:, :, :, :-1] - batch[:, :, :, 1:])
        # total_diff = diagonal_diff_top_left + diagonal_diff_bottom_left + vertical_diff + horizontal_diff
        # return total_diff

class l2NormRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        """
        Calculates the l2-Norm Regularization of the batch
        """
        return batch.norm()


class PriorRegularization(nn.Module):
    def __init__(self, a_tv, a_l2, *args, **kwargs):
        super().__init__()
        self.tv = TotalVariationRegularization()
        self.l2 = l2NormRegularization()
        self.a_tv = a_tv
        self.a_l2 = a_l2

    def forward(self, batch):
        """
        Calculates the Prior Regularization of the batch
        """
        tv_reg = self.a_tv * self.tv(batch)
        l2_reg = self.a_l2 * self.l2(batch)
        return tv_reg, l2_reg


# class FeatureRegularization(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.running_means = [
#             bn.running_mean for bn in model.modules() if isinstance(bn, nn.BatchNorm2d)]
#         self.running_vars = [
#             bn.running_var for bn in model.modules() if isinstance(bn, nn.BatchNorm2d)]
#
#     def forward(self, batch, batch_running_means, batch_running_vars):
#         assert len(self.running_means) == len(batch_running_means)
#         mean_term = [torch.norm(batch_mean - bn_mean) for batch_mean,
#                      bn_mean in zip(batch_running_means, self.running_means)]
#         var_term = [torch.norm(batch_var - bn_var) for batch_var,
#                     bn_var in zip(batch_running_vars, self.running_vars)]
#         return sum(mean_term) + sum(var_term)


class DIRegularization(nn.Module):
    def __init__(self, model, a_tv, a_l2, a_f, *args, **kwargs):
        super().__init__()
        self.a_f = a_f
        self.prior = PriorRegularization(a_tv, a_l2)
        # self.feature = FeatureRegularization(model)

    def forward(self, batch):
        return self.prior(batch)
