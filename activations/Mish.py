from Module import Module
import torch

class Mish(Module):
    """Applies the function element-wise:
        `Mish(x) = x * tanh(ln(1+e^x))`

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
        """Forward pass for the Mish activation function"""
        self.input = input.clone()
        return self.input.mul(((1+self.input.exp()).log()).float())

    def backward(self, grad):
        """Backward pass for the Mish activation function"""
        numerator = torch.exp(self.input) * (4*torch.exp(2*self.input) + torch.exp(3*self.input) + 4 *(1 + self.input) + torch.exp(self.input) *(6 + 4 *self.input))
        denominator = (2 + 2* torch.exp(self.input) + torch.exp(2*self.input))**2
        activation = numerator.div(denominator)
        return grad.t().mul(activation)