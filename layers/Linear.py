import math

import torch

from Module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, weight_init=None):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.Tensor(in_features, out_features)
        self.weight_grad = torch.Tensor(in_features, out_features)

        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias_grad = torch.Tensor(out_features)
            self.init_biases()

        if weight_init == None:
            self.init_weights()
        else:
            self.weight = weight_init(self.weight)

    def init_weights(self):
        stdv = 1. / math.sqrt(self.in_features)
        self.weight = torch.Tensor(self.weight.size(0), self.weight.size(1)).uniform_(-stdv, stdv)

    def init_biases(self):
        stdv = 1. / math.sqrt(self.in_features)
        if self.bias is not None:
            self.bias = torch.Tensor(self.out_features).uniform_(-stdv, stdv)

    def forward(self, input):
        self.input = input.clone()
        res = input.mm(self.weight)
        if self.bias != None:
            res.add(self.bias)
        return res

    def backward(self, grad):
        if self.bias != None:
            self.bias_grad += grad.sum(dim=0)

        self.weight_grad += self.input.t().mm(grad)

        return self.weight.mm(grad.t())

    def param(self):
        return [self.weight, self.bias]

    def update_params(self, func):
        func(self.weight, self.bias, self.bias_grad, self.weight_grad)

    def zero_grad(self):
        if self.bias != None:
            self.bias_grad.zero_()
        self.weight_grad.zero_()