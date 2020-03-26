import torch

from Module import Module


class MSE(Module):
    def __init__(self, class_num=2):
        super(MSE, self).__init__()
        self.pred = None
        self.gt = None
        self.class_num = class_num

    def forward(self, pred, gt):
        y = torch.eye(self.class_num)
        gt = y[gt.long()]
        self.pred = pred
        self.gt = gt
        return (gt - pred).pow(2).sum() / gt.shape[0]

    def backward(self):
        return -2 * (self.gt - self.pred) / self.gt.shape[0]