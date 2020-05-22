from Module import Module


class Tanh(Module):
    """Applies the element-wise function:
        Tanh(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Shape
    -----
        - Input: `(N, *)` where `*` means, any number of additional dimensions
        - Output: `(N, *)`, same shape as the input

    Examples
    --------
        >>> l = nn.Tanh()
        >>> input = torch.randn(2)
        >>> output = l.forward(input)
    """
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, input):
        """Forward pass for the Tanh activation function"""
        self.input = input.clone()
        return self.input.tanh()

    def backward(self, grad):
        """Backward pass for the Tanh activation function"""
        return grad.t().mul(1 - (self.input.tanh() ** 2))
