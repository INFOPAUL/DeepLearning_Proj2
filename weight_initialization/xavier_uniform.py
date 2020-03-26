import math


def xavier_uniform(tensor):
    std = math.sqrt(6.0 / float(tensor.size(0) + tensor.size(1)))
    return tensor.uniform_(-std, std)