import math

import torch


def generate_data(nb = 1000):
    data = torch.rand(2, nb)
    radius = 1 / math.sqrt(2 * math.pi)
    distances = data.pow(2).sum(dim=0)
    label = distances <= radius
    return data.t(), label.int()

def normalize(train_data, test_data):
    mean, std = train_data.mean(), train_data.std()
    return train_data.sub_(mean).div_(std), test_data.sub_(mean).div_(std)
