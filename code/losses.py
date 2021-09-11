import torch.nn as nn
import torch
import torch.nn.functional as F

class WeightedMultiNCELoss(nn.Module):
    def __init__(self, lambd):
        super(WeightedMultiNCELoss, self).__init__()
        self.lambd = lambd

    def forward(self, score, mask):
        if isinstance(mask, list):
            loss = - torch.log((F.softmax(score, dim=1) * mask[0]).sum(1)) * (1 - self.lambd)
            loss += - torch.log((F.softmax(score * mask[1], dim=1) * mask[0]).sum(1)) * self.lambd
        else:
            loss = - torch.log((F.softmax(score, dim=1) * mask).sum(1))
        return loss.mean()

BCELoss = nn.BCEWithLogitsLoss
