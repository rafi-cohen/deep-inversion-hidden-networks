import gym
import torch
import torch.nn as nn
import torch.nn.functional

from .rl_pg import PolicyAgent, TrainBatch, VanillaPolicyGradientLoss


class AACPolicyNet(nn.Module):
    def __init__(self, in_features: int, out_actions: int, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()
        # DONE:
        #  Implement a dual-head neural net to approximate both the
        #  policy and value. You can have a common base part, or not.
        # ====== YOUR CODE: ======
        NLS = {'relu': torch.nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,
               'logsoftmax': nn.LogSoftmax}
        layers = []
        hidden_dims = kw.get("hidden_dims", [50, 50, 50])
        nonlin = kw.get("nonlin", "relu")
        dropout = kw.get("dropout", 0)
        all_dims = [in_features, *hidden_dims, out_actions + 1]
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            layers += [
                nn.Linear(in_dim, out_dim, bias=True),
                NLS[nonlin]()
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        if isinstance(layers[-1], nn.Dropout):
            layers.pop()  # remove last dropout layer
        layers.pop()  # remove last non-linearity
        self.fc_layers = nn.Sequential(*layers)
        # ========================

    def forward(self, x):
        """
        :param x: Batch of states, shape (N,O) where N is batch size and O
        is the observation dimension (features).
        :return: A tuple of action values (N,A) and state values (N,1) where
        A is is the number of possible actions.
        """
        # DONE:
        #  Implement the forward pass.
        #  calculate both the action scores (policy) and the value of the
        #  given state.
        # ====== YOUR CODE: ======
        x = x.reshape((x.shape[0], -1))
        out = self.fc_layers(x)
        action_scores = out[:, :-1]
        state_values = out[:, -1]
        # ========================

        return action_scores, state_values

    @staticmethod
    def build_for_env(env: gym.Env, device='cpu', **kw):
        """
        Creates a A2cNet instance suitable for the given environment.
        :param env: The environment.
        :param kw: Extra hyperparameters.
        :return: An A2CPolicyNet instance.
        """
        # DONE: Implement according to docstring.
        # ====== YOUR CODE: ======
        in_features = env.observation_space.shape[0]
        out_actions = env.action_space.n
        net = AACPolicyNet(in_features, out_actions, **kw)
        # ========================
        return net.to(device)


class AACPolicyAgent(PolicyAgent):

    def current_action_distribution(self) -> torch.Tensor:
        # DONE: Generate the distribution as described above.
        # ====== YOUR CODE: ======
        action_scores = self.p_net(self.curr_state.reshape(1, -1))[0].squeeze()
        actions_proba = torch.softmax(action_scores, dim=0)
        # ========================
        return actions_proba


class AACPolicyGradientLoss(VanillaPolicyGradientLoss):
    def __init__(self, delta: float):
        """
        Initializes an AAC loss function.
        :param delta: Scalar factor to apply to state-value loss.
        """
        super().__init__()
        self.delta = delta

    def forward(self, batch: TrainBatch, model_output, **kw):

        # Get both outputs of the AAC model
        action_scores, state_values = model_output

        # TODO: Calculate the policy loss loss_p, state-value loss loss_v and
        #  advantage vector per state.
        #  Use the helper functions in this class and its base.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        loss_v *= self.delta
        loss_t = loss_p + loss_v
        return loss_t, dict(loss_p=loss_p.item(), loss_v=loss_v.item(),
                            adv_m=advantage.mean().item())

    def _policy_weight(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO:
        #  Calculate the weight term of the AAC policy gradient (advantage).
        #  Notice that we don't want to backprop errors from the policy
        #  loss into the state-value network.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================
        return advantage

    def _value_loss(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO: Calculate the state-value loss.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================
        return loss_v

