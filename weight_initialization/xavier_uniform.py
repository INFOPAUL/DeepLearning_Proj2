import math


def xavier_uniform(tensor):
    """Fills the input tensor with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    Uniform(-a, a) where
    a = \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Parameters
    ----------
        tensor: an n-dimensional tensor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> xavier_uniform(w)
    """
    std = math.sqrt(6.0 / float(tensor.size(0) + tensor.size(1)))
    return tensor.uniform_(-std, std)