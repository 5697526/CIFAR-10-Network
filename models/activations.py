# models/activations.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish激活函数: f(x) = x * sigmoid(x)"""

    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Mish激活函数: f(x) = x * tanh(ln(1 + e^x))"""

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SELU(nn.Module):
    """SELU激活函数: 自归一化激活函数"""

    def forward(self, x):
        return F.selu(x)


class GELU(nn.Module):
    """GELU激活函数: 高斯误差线性单元"""

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / torch.tensor(torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


# 激活函数工厂 - 方便根据配置创建不同激活函数
def get_activation(activation_type='relu'):
    """
    获取指定类型的激活函数

    参数:
        activation_type: 激活函数类型，'relu', 'swish', 'mish', 'selu', 'gelu'
    """
    if activation_type == 'relu':
        return nn.ReLU()
    elif activation_type == 'swish':
        return Swish()
    elif activation_type == 'mish':
        return Mish()
    elif activation_type == 'selu':
        return SELU()
    elif activation_type == 'gelu':
        return GELU()
    else:
        raise ValueError(f"不支持的激活函数类型: {activation_type}")
