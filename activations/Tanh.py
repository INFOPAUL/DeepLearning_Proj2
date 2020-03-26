from Module import Module


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.input = None

    def forward(self, input):
        self.input = input.clone()
        return self.input.tanh()

    def backward(self, grad):
        return grad.t().mul(1 - (self.input.tanh() ** 2))