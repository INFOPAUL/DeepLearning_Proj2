import math

import torch

from Module import Module


class Linear(Module):
    """Applies a linear transformation to the incoming data: `y = xA^T + b`

    Parameters
    ----------
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias. Default: True

    Shape
    -----
        - Input: `(N, *, in_features)` where `*` means any number of
          additional dimensions
        - Output: `(N, *, out_features)` where all but the last dimension
          are the same shape as the input.

    Attributes
    ----------
        weight_init: the learnable weights of the module of shape
            (out_features, in_features)`. The values are initialized from 
            Uniform(-\sqrt(k), \sqrt(k)), where k = \frac{1}{in_features}.
        bias: the learnable bias of the module of shape (out_features).
            If bias is True, the values are initialized from Uniform(-\sqrt(k), \sqrt(k))
            where k = \frac{1}{in_features}.

    Examples
    --------
        >>> l = Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
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
        """Initialize weights"""
        stdv = 1. / math.sqrt(self.in_features)
        self.weight = torch.Tensor(self.weight.size(0), self.weight.size(1)).uniform_(-stdv, stdv)

    def init_biases(self):
        """Initialize biases"""
        stdv = 1. / math.sqrt(self.in_features)
        if self.bias is not None:
            self.bias = torch.Tensor(self.out_features).uniform_(-stdv, stdv)

    def forward(self, input):
        """Forward pass of the linear layer"""
        self.input = input.clone()
        res = input.mm(self.weight)
        if self.bias is not None:
            res.add(self.bias)
        return res

    def backward(self, grad):
        """Backward pass of the linear layer"""
        if self.bias is not None:
            self.bias_grad += grad.sum(dim=0)

        self.weight_grad += self.input.t().mm(grad)

        return self.weight.mm(grad.t())

    def param(self):
        """Get weights and biases"""
        return [self.weight, self.bias]

    def update_params(self, func):
        """Update model parameters with function func"""
        func(self.weight, self.bias, self.bias_grad, self.weight_grad)

    def zero_grad(self):
        """Put zero weights and biases"""
        if self.bias is not None:
            self.bias_grad.zero_()
        self.weight_grad.zero_()