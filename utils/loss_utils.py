# utils/loss_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """焦点损失 - 解决类别不平衡问题"""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失 - 防止模型过拟合"""

    def __init__(self, classes=10, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, inputs, targets):
        inputs = inputs.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * inputs, dim=self.dim))


class L1Regularization(nn.Module):
    """L1正则化"""

    def __init__(self, model, lambda_l1=1e-4):
        super(L1Regularization, self).__init__()
        self.model = model
        self.lambda_l1 = lambda_l1

    def forward(self):
        l1_loss = 0
        for param in self.model.parameters():
            if param.requires_grad and len(param.shape) > 1:  # 只对权重参数应用
                l1_loss += torch.norm(param, 1)
        return self.lambda_l1 * l1_loss


class L2Regularization(nn.Module):
    """L2正则化(权重衰减)"""

    def __init__(self, model, lambda_l2=5e-4):
        super(L2Regularization, self).__init__()
        self.model = model
        self.lambda_l2 = lambda_l2

    def forward(self):
        l2_loss = 0
        for param in self.model.parameters():
            if param.requires_grad and len(param.shape) > 1:  # 只对权重参数应用
                l2_loss += torch.norm(param, 2)
        return self.lambda_l2 * l2_loss


# 损失函数工厂 - 方便根据配置创建不同损失函数
def get_loss_function(loss_type='ce', alpha=1, gamma=2, smoothing=0.1,
                      lambda_l1=0, lambda_l2=5e-4, model=None):
    """
    获取指定类型的损失函数

    参数:
        loss_type: 损失函数类型，'ce'(交叉熵), 'focal'(焦点损失), 'ls'(标签平滑)
        alpha: 焦点损失的alpha参数
        gamma: 焦点损失的gamma参数
        smoothing: 标签平滑的平滑系数
        lambda_l1: L1正则化系数
        lambda_l2: L2正则化系数
        model: 用于计算正则化的模型
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'ls':
        return LabelSmoothingLoss(classes=10, smoothing=smoothing)
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")

# 组合损失函数 - 支持同时使用多种损失和正则化


class CombinedLoss(nn.Module):
    def __init__(self, base_loss, l1_reg=None, l2_reg=None):
        super(CombinedLoss, self).__init__()
        self.base_loss = base_loss
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def forward(self, outputs, targets):
        loss = self.base_loss(outputs, targets)
        if self.l1_reg:
            loss += self.l1_reg()
        if self.l2_reg:
            loss += self.l2_reg()
        return loss
