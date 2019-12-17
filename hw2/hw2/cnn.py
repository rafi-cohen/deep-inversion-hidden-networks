import torch
import itertools as it
import torch.nn as nn


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # DONE: Create the feature extractor part of the model:
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every
        channels = list(self.channels)
        prev_out_channels = in_channels
        for _ in range(N // P):
            for _ in range(P):
                curr_out_channels = channels.pop(0)
                layers.extend([nn.Conv2d(in_channels=prev_out_channels,
                                         out_channels=curr_out_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         dilation=1),
                              nn.ReLU()])
                prev_out_channels = curr_out_channels

            layers.append(nn.MaxPool2d(kernel_size=2))

        for _ in range(N % P):
            curr_out_channels = channels.pop(0)
            layers.extend([nn.Conv2d(in_channels=prev_out_channels,
                                     out_channels=curr_out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     dilation=1),
                           nn.ReLU()])
            prev_out_channels = curr_out_channels

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # DONE: Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every
        prev_out_dim = ((in_h * in_w) // 2**(2*(N // P))) * self.channels[-1]
        for out_dim in self.hidden_dims:
            layers.append(nn.Linear(in_features=prev_out_dim, out_features=out_dim))
            layers.append(nn.ReLU())
            prev_out_dim = out_dim
        layers.append(nn.Linear(in_features=prev_out_dim, out_features=self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # DONE: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=False, dropout=0.):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None

        # DONE: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  the main_path, which should contain the convolution, dropout,
        #  batchnorm, relu sequences, and the shortcut_path which should
        #  represent the skip-connection.
        #  Use convolutions which preserve the spatial extent of the input.
        #  For simplicity of implementation, we'll assume kernel sizes are odd.
        # ====== YOUR CODE: ======
        main_path_layers = []
        prev_out_channels = in_channels
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            main_path_layers.append(nn.Conv2d(in_channels=prev_out_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=1,
                                              padding=kernel_size//2,
                                              dilation=1))
            if dropout > 0.:
                main_path_layers.append(nn.Dropout2d(p=dropout))
            if batchnorm:
                main_path_layers.append(nn.BatchNorm2d(num_features=out_channels))
            main_path_layers.append(nn.ReLU())
            prev_out_channels = out_channels
        main_path_layers.pop()
        if dropout > 0.:
            main_path_layers.pop()
        if batchnorm:
            main_path_layers.pop()
        self.main_path = nn.Sequential(*main_path_layers)

        shortcut_path_layers = []
        if in_channels != channels[-1]:
            shortcut_path_layers.append(nn.Conv2d(in_channels=in_channels,
                                                  out_channels=channels[-1],
                                                  kernel_size=1,
                                                  bias=False))
        self.shortcut_path = nn.Sequential(*shortcut_path_layers)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # DONE: Create the feature extractor part of the model:
        #  [-> (CONV -> ReLU)*P -> MaxPool]*(N/P)
        #   \------- SKIP ------/
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs (with a skip over them) should exist at the end,
        #  without a MaxPool after them.
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every
        channels = list(self.channels)
        prev_out_channels = in_channels
        for _ in range(N // P):
            layers.append(ResidualBlock(prev_out_channels, channels[:P], [3]*P))
            layers.append(nn.MaxPool2d(kernel_size=2))
            prev_out_channels = channels[P-1]
            channels = channels[P:]  # Remove first P entries

        remainder = N % P
        if remainder != 0:
            layers.append(ResidualBlock(prev_out_channels, channels[:remainder], [3] * remainder))
        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    # DONE: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # The feature extractor part of the model:
        #  [(CONV -> ReLU -> BatchNorm)*P -> MaxPool -> Dropout]*(N/P)
        N = len(self.channels)
        P = self.pool_every
        channels = list(self.channels)
        prev_out_channels = in_channels
        for _ in range(N // P):
            for _ in range(P):
                curr_out_channels = channels.pop(0)
                layers.extend([nn.Conv2d(in_channels=prev_out_channels,
                                         out_channels=curr_out_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         dilation=1),
                               nn.ReLU(),
                               nn.BatchNorm2d(num_features=curr_out_channels)])
                prev_out_channels = curr_out_channels

            layers.extend([nn.MaxPool2d(kernel_size=2),
                           nn.Dropout2d()])

        for _ in range(N % P):
            curr_out_channels = channels.pop(0)
            layers.extend([nn.Conv2d(in_channels=prev_out_channels,
                                     out_channels=curr_out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     dilation=1),
                           nn.ReLU(),
                           nn.BatchNorm2d(num_features=curr_out_channels)])
            prev_out_channels = curr_out_channels

        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # The classifier part of the model:
        #  Linear
        N = len(self.channels)
        P = self.pool_every
        prev_out_dim = ((in_h * in_w) // 2**(2*(N // P))) * self.channels[-1]
        layers.append(nn.Linear(in_features=prev_out_dim, out_features=self.out_classes))
        seq = nn.Sequential(*layers)
        return seq
    # ========================
