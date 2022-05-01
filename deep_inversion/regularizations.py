import torch.nn as nn


class TotalVariationRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        """
        Calculates the Total Variation Regularization of the batch
        """
        diagonal_diff_top_left = (batch[:, :, :-1, :-1] - batch[:, :, 1:, 1:]).norm()
        diagonal_diff_bottom_left = (batch[:, :, 1:, :-1] - batch[:, :, :-1, 1:]).norm()
        vertical_diff = (batch[:, :, :-1, :] - batch[:, :, 1:, :]).norm()
        horizontal_diff = (batch[:, :, :, :-1] - batch[:, :, :, 1:]).norm()
        batch_size = batch.shape[0]
        total_diff = (diagonal_diff_top_left + diagonal_diff_bottom_left + vertical_diff + horizontal_diff) / batch_size
        return total_diff


class l2NormRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        """
        Calculates the l2-Norm Regularization of the batch
        """
        batch_size = batch.shape[0]
        return batch.view(batch_size, -1).norm(dim=1).mean()


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
    def __init__(self, model):
        super().__init__()
        self.reg_value = 0
        self.handles = []
        self._register_hooks(model)

    def _hook(self, module, input):
        if isinstance(module, nn.BatchNorm2d):
            current_feature_map = input[0]
            dims = list(range(current_feature_map.dim()))
            dims.pop(1)
            mean_term = (current_feature_map.mean(dim=dims) - module.running_mean).norm()
            var_term = (current_feature_map.var(dim=dims, unbiased=False) - module.running_var).norm()
            reg_value = mean_term + var_term
            self.reg_value = self.reg_value + reg_value

    def _register_hooks(self, model):
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                handle = module.register_forward_pre_hook(self._hook)
                self.handles.append(handle)

    def _remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def forward(self, batch):
        reg_value = self.reg_value
        self.reg_value = 0
        return reg_value


class DIRegularization(nn.Module):
    def __init__(self, model, a_tv, a_l2, a_f, *args, **kwargs):
        super().__init__()
        self.a_f = a_f
        self.prior = PriorRegularization(a_tv, a_l2)
        self.feature = FeatureRegularization(model)

    def forward(self, batch):
        return self.prior(batch) + self.a_f * self.feature(batch)
