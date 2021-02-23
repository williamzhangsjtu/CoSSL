import torch.nn as nn
import torch
import torch.nn.functional as F

class WeightedMultiNCELoss(nn.Module):
    def __init__(self):
        super(WeightedMultiNCELoss, self).__init__()

    def forward(self, score, mask):
        loss = - torch.log((F.softmax(score, dim=1) * mask).sum(1))
        return loss.mean()
