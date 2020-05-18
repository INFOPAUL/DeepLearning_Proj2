from Module import Module


class Sequential(Module):
    def __init__(self, layers):
        super(Sequential, self).__init__()
        self.layers = layers

    def forward(self, input):
        inp = input
        for layer in self.layers:
            
            out = layer.forward(inp)
            inp = out
        return out

    def backward(self, grad):
        grad_tmp = grad
        for layer in reversed(self.layers):
            grad_tmp = layer.backward(grad_tmp)

    def param(self):
        weights = []
        biases = []
        for l in self.layers:
            if l.param():
                weights.extend(l.param()[0])
                biases.extend(l.param()[0])
        return [weights, biases]
