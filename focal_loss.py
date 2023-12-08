import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # 调节因子
        self.alpha = alpha  # 类别权重，若为None则不进行权重调整
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # 提取目标类别的概率
        class_probs = probs.gather(1, targets.view(-1, 1))

        # 计算权重
        if self.alpha is not None:
            alpha = self.alpha[targets]
            weights = (1 - class_probs) ** self.gamma * alpha
        else:
            weights = (1 - class_probs) ** self.gamma

        # 计算损失
        loss = -weights * class_probs.log()

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
