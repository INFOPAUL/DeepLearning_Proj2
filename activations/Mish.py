from Module import Module
import torch

class Mish(Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, input):
        self.input = input.clone()
        return self.input.mul(((1+self.input.exp()).log()).float())

    def backward(self, grad):
        numerator = torch.exp(self.input) * (4*torch.exp(2*self.input) + torch.exp(3*self.input) + 4 *(1 + self.input) + torch.exp(self.input) *(6 + 4 *self.input))
        denominator = (2 + 2* torch.exp(self.input) + torch.exp(2*self.input))**2
        activation = numerator.div(denominator)
        return grad.t().mul(activation)