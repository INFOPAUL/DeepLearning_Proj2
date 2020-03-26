import torch



class SGD():
    def __init__(self, model, lr=0.001):
        self.lr = lr
        self.model = model

    def update_rule(self, weight, bias, bias_grad, weight_grad):
        weight -= self.lr  * weight_grad
        if bias != None:
            bias -= self.lr  * bias_grad

    def step(self):
        for layer in self.model.layers:
            layer.update_params(self.update_rule)

    def zero_grad(self):
        for layer in self.model.layers:
            layer.zero_grad()