import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma 
        self.alpha = alpha  
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        class_probs = probs.gather(1, targets.view(-1, 1))


        if self.alpha is not None:
            alpha = self.alpha[targets]
            weights = (1 - class_probs) ** self.gamma * alpha
        else:
            weights = (1 - class_probs) ** self.gamma


        loss = -weights * class_probs.log()

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
