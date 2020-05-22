import torch

from Module import Module


class MSE(Module):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input x and target y.

    Equation
    --------
        MSE = mean(sum((ground_truth - prediction)^2))

    Shape
    -----
        - Input: `(N, *)` where `*` means, any number of additional dimensions
        - Target: `(N, *)`, same shape as the input

    Examples
    --------
        >>> loss = MSE()
        >>> input = torch.randn(3, 5)
        >>> target = torch.randn(3, 5)
        >>> output = loss.forward(input, target)
    """
    def __init__(self, class_num=2):
        super().__init__()
        self.pred = None
        self.gt = None
        self.class_num = class_num

    def forward(self, pred, gt):
        """Forward pass of loss"""
        y = torch.eye(self.class_num)
        gt = y[gt.long()]
        self.pred = pred
        self.gt = gt
        return (gt - pred).pow(2).sum() / gt.shape[0]

    def backward(self):
        """Backward pass of loss"""
        return -2 * (self.gt - self.pred) / self.gt.shape[0]