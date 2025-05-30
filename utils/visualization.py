# utils/visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def visualize_filters(model, layer_name, num_cols=8, figsize=(12, 12)):
    """可视化卷积层的滤波器"""
    # 获取指定层的权重
    layer = dict(model.named_modules())[layer_name]
    filters = layer.weight.data.cpu().numpy()

    # 计算网格尺寸
    num_filters = filters.shape[0]
    num_rows = int(np.ceil(num_filters / num_cols))

    # 创建图像网格
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    # 标准化滤波器以便更好地显示
    for i in range(num_filters):
        filter_img = filters[i]
        # 归一化到 [0, 1]
        filter_img = (filter_img - filter_img.min()) / \
            (filter_img.max() - filter_img.min())
        # 如果是3通道，转换为RGB
        if filter_img.shape[0] == 3:
            filter_img = np.transpose(filter_img, (1, 2, 0))
        else:
            # 单通道，取第一个通道
            filter_img = filter_img[0]

        axes[i].imshow(filter_img)
        axes[i].axis('off')

    # 隐藏空白子图
    for i in range(num_filters, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def visualize_feature_maps(model, layer_name, input_image, num_cols=8, figsize=(12, 12)):
    """可视化输入图像通过指定层后的特征图"""
    # 创建一个hook来获取中间层的输出
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # 注册hook
    hook_handle = dict(model.named_modules())[
        layer_name].register_forward_hook(get_activation(layer_name))

    # 前向传播
    model.eval()
    with torch.no_grad():
        model(input_image.unsqueeze(0).to(next(model.parameters()).device))

    # 释放hook
    hook_handle.remove()

    # 获取特征图
    feature_maps = activation[layer_name].squeeze(0).cpu().numpy()

    # 计算网格尺寸
    num_maps = feature_maps.shape[0]
    num_rows = int(np.ceil(num_maps / num_cols))

    # 创建图像网格
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    # 显示原始图像
    axes[0].imshow(np.transpose(input_image.numpy(), (1, 2, 0)))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 显示特征图
    for i in range(1, min(num_maps + 1, len(axes))):
        feature_map = feature_maps[i-1]
        # 归一化到 [0, 1]
        feature_map = (feature_map - feature_map.min()) / \
            (feature_map.max() - feature_map.min())
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].axis('off')

    # 隐藏空白子图
    for i in range(num_maps + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def plot_loss_landscape(model, dataloader, device, num_points=20, figsize=(10, 8)):
    """可视化模型的损失景观"""
    # 获取当前模型参数作为中心点
    params = [p.data.clone().detach() for p in model.parameters()]

    # 生成随机方向
    direction1 = [torch.randn_like(p) for p in params]
    direction2 = [torch.randn_like(p) for p in params]

    # 归一化方向
    for d in [direction1, direction2]:
        norm = torch.sqrt(sum(torch.sum(di**2) for di in d))
        for di in d:
            di.div_(norm)

    # 创建网格
    x = np.linspace(-1, 1, num_points)
    y = np.linspace(-1, 1, num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # 计算每个点的损失
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    for i in range(num_points):
        for j in range(num_points):
            # 扰动模型参数
            for p, p0, d1, d2 in zip(model.parameters(), params, direction1, direction2):
                p.data = p0 + x[i] * d1 + y[j] * d2

            # 计算平均损失
            total_loss = 0
            num_batches = 0
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1

            Z[i, j] = total_loss / num_batches

    # 恢复原始参数
    for p, p0 in zip(model.parameters(), params):
        p.data = p0

    # 绘制3D损失景观
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)

    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig


def saliency_map(model, image, target_class=None, device=None):
    """生成显著性图，显示图像中哪些区域对模型预测最重要"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    image = image.to(device).unsqueeze(0)
    image.requires_grad_()

    # 前向传播
    output = model(image)

    # 如果没有指定目标类别，使用预测的最高概率类别
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # 计算梯度
    model.zero_grad()
    output[0, target_class].backward()

    # 获取梯度的绝对值
    grads = image.grad.data.abs().squeeze(0)

    # 取通道维度的最大值
    saliency, _ = torch.max(grads, dim=0)

    # 归一化到 [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    return saliency.cpu().numpy()


def visualize_saliency(model, image, target_class=None, figsize=(10, 5)):
    """可视化显著性图"""
    # 计算显著性图
    saliency = saliency_map(model, image, target_class)

    # 显示原始图像和显著性图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 显示原始图像
    ax1.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    ax1.set_title('Original Image')
    ax1.axis('off')

    # 显示显著性图
    ax2.imshow(saliency, cmap='hot')
    ax2.set_title('Saliency Map')
    ax2.axis('off')

    plt.tight_layout()
    return fig
