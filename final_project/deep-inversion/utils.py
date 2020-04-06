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
        return self.a_tv * self.tv(batch) + self.a_l2 * self.l2(batch)


class FeatureRegularization(nn.Module):
    batch_means = batch_vars = bn_means = bn_vars = []

    def __init__(self, model):
        super().__init__()
        self.handles = []
        self._register_hooks(model)

    @staticmethod
    def _hook(module, input, output):
        if isinstance(module, nn.BatchNorm2d):
            current_feature_map = input[0]
            dims = list(range(current_feature_map.dim()))
            dims.pop(1)
            FeatureRegularization.batch_means.append(torch.mean(current_feature_map, dim=dims))
            FeatureRegularization.batch_vars.append(torch.var(current_feature_map, dim=dims, unbiased=False))
            FeatureRegularization.bn_means.append(module.running_mean)
            FeatureRegularization.bn_vars.append(module.running_var)

    def _register_hooks(self, model):
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                handle = module.register_forward_hook(self._hook)
                self.handles.append(handle)

    def _remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    @staticmethod
    def _reset_running_stats():
        FeatureRegularization.batch_means = FeatureRegularization.batch_vars = []
        FeatureRegularization.bn_means = FeatureRegularization.bn_vars = []

    def forward(self, batch):
        mean_term = [torch.norm(batch_mean - bn_mean) for batch_mean, bn_mean
                     in zip(FeatureRegularization.batch_means, FeatureRegularization.bn_means)]
        var_term = [torch.norm(batch_var - bn_var) for batch_var, bn_var
                    in zip(FeatureRegularization.batch_vars, FeatureRegularization.bn_vars)]
        self._reset_running_stats()
        return sum(mean_term) + sum(var_term)


class DIRegularization(nn.Module):
    def __init__(self, model, a_tv, a_l2, a_f, *args, **kwargs):
        super().__init__()
        self.a_f = a_f
        self.prior = PriorRegularization(a_tv, a_l2)
        self.feature = FeatureRegularization(model)

    def forward(self, batch):
        return self.prior(batch) + self.a_f * self.feature(batch)
