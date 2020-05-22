from Module import Module


class Relu(Module):
    """Applies the rectified linear unit function element-wise:
        `ReLU(x) = \max(0, x)`

    Shape
    -----
        - Input: `(N, *)` where `*` means, any number of additional dimensions
        - Output: `(N, *)`, same shape as the input

    Examples
    --------
        >>> l = Relu()
        >>> input = torch.randn(2)
        >>> output = l.forward(input)
    """
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, input):
        """Forward pass for the ReLU activation function"""
        self.input = input.clone()
        return self.input.mul((self.input > 0).float())

    def backward(self, grad):
        """Backward pass for the ReLU activation function"""
        activation = (self.input > 0).float()
        return grad.t().mul(activation)