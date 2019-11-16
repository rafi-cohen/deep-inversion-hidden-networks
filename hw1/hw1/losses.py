import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # DONE: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        N = x.shape[0]  # Number of samples
        # Gather the corresponding score according to the ground-truth label for each sample
        y_scores = x_scores.gather(1, y.view(-1, 1))
        # Create a matrix M where M[i,j] is the margin-loss
        # for sample i and class j (i.e. s_j - s_{y_i} + delta).
        M = x_scores - y_scores + self.delta
        # Nullify negative elements (equivalent to calculating max(0, x))
        M[M < 0] = 0
        indices = (torch.arange(N), y)  # indices of m_i,y_i for all i in [0,N-1]
        M[indices] = 0  # zero out m_i,y_i to 'skip' it in the summation
        loss = (torch.sum(M)) / N
        # ========================

        # DONE: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['M'] = M
        self.grad_ctx['N'] = N
        self.grad_ctx['x'] = x
        self.grad_ctx['indices'] = indices
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # DONE:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        M: torch.Tensor = self.grad_ctx['M']
        N: torch.Tensor = self.grad_ctx['N']
        x: torch.Tensor = self.grad_ctx['x']
        indices: torch.Tensor = self.grad_ctx['indices']
        # initialize G with 1 where m_i,j > 0, and 0 otherwise
        G = M.new_zeros(M.shape)
        G[M > 0] = 1
        # in the case of m_i,j where j=y_i, set G_i,j to be minus the numbers of time m_i,j > 0 for every j!=y_i
        G[indices] = -torch.sum(G, dim=1)

        grad = torch.mm(x.t(), G)
        grad /= N  # divide the whole grad by N, instead of dividing each column separately
        # ========================

        return grad
