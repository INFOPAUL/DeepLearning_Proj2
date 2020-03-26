from Module import Module


class Relu(Module):
    def __init__(self):
        super(Relu, self).__init__()
        self.input = None

    def forward(self , input):
        self.input = input.clone()
        return self.input.mul((self.input > 0).float())

    def backward(self , grad):
        activation = (self.input > 0).float()
        return grad.t().mul(activation)