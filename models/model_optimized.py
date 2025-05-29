# models/model_optimized.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfigurableCNN(nn.Module):
    def __init__(self, num_classes=10, filter_config=None, fc_units=None):
        super(ConfigurableCNN, self).__init__()

        # 默认滤波器配置
        if filter_config is None:
            filter_config = [32, 64, 128]  # 原始配置

        # 默认全连接层配置
        if fc_units is None:
            fc_units = [512, num_classes]  # 原始配置

        self.filter_config = filter_config
        self.num_conv_layers = len(filter_config)  # 卷积层数量

        # 计算特征图尺寸 (假设输入为32x32，池化后)
        # 对于每个卷积块，尺寸减半
        # 初始尺寸: 32x32
        # 经过一次池化: 16x16
        # 经过两次池化: 8x8
        # 经过三次池化: 4x4
        # 经过四次池化: 2x2 (当使用deep配置时)
        if self.num_conv_layers == 3:
            feature_dim = filter_config[-1] * 4 * 4  # 3次池化后尺寸为4x4
        elif self.num_conv_layers == 4:
            feature_dim = filter_config[-1] * 2 * 2  # 4次池化后尺寸为2x2
        else:
            feature_dim = filter_config[-1] * 4 * 4  # 默认为3次池化

        # 卷积层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_channels = 3
        for i, out_channels in enumerate(filter_config):
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc_units = fc_units
        self.fc1 = nn.Linear(feature_dim, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])

        # 激活函数
        self.relu = nn.ReLU()

        # Dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 卷积块
        for i in range(self.num_conv_layers):
            x = self.pool(self.relu(self.bns[i](self.convs[i](x))))
            x = self.dropout(x)

        # 展平与全连接层
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 定义不同的滤波器配置方案
FILTER_CONFIGS = {
    "small": [16, 32, 64],       # 小容量网络
    "medium": [32, 64, 128],     # 中等容量网络(原始配置)
    "large": [64, 128, 256],     # 大容量网络
    "wide": [32, 128, 256],      # 宽卷积网络
    "deep": [32, 64, 128, 256]   # 深度卷积网络，4层卷积
}

# 定义不同的全连接层配置
FC_UNITS_CONFIGS = {
    "default": [512, 10],        # 原始配置
    "simplified": [256, 10],     # 简化全连接层
    "complex": [1024, 512, 10]   # 复杂全连接层
}
