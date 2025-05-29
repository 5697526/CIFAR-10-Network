# utils/grad_cam.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


class GradCAM:
    """
    使用Grad-CAM方法可视化模型关注的区域
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradient = None

        # 注册钩子
        self.hook_handles = []

        # 保存特征图的正向钩子
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        # 保存梯度的反向钩子
        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0].detach()

        # 获取目标层
        target_found = False
        for name, module in self.model.named_modules():
            if name == target_layer:
                self.hook_handles.append(
                    module.register_forward_hook(forward_hook))
                self.hook_handles.append(
                    module.register_backward_hook(backward_hook))
                target_found = True
                break

        if not target_found:
            raise ValueError(f"未找到目标层: {target_layer}")

    def __call__(self, x, class_idx=None):
        """
        计算Grad-CAM热力图

        参数:
            x: 输入图像，形状为 (1, 3, H, W)
            class_idx: 目标类别索引，如果为None则使用预测的最高类别
        """
        # 确保模型处于评估模式
        self.model.eval()

        # 前向传播
        x.requires_grad_()
        output = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 计算权重 (全局平均池化梯度)
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)

        # 加权组合特征图
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze()

        # ReLU激活，因为我们只关心对类别有正向贡献的区域
        cam = F.relu(cam)

        # 归一化
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)

        # 调整为与输入图像相同的尺寸
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                            size=(x.size(2), x.size(3)),
                            mode='bilinear',
                            align_corners=False).squeeze()

        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        """移除注册的钩子"""
        for handle in self.hook_handles:
            handle.remove()


def visualize_grad_cam(model, image, target_layer, class_names=None, figsize=(10, 5)):
    """
    可视化Grad-CAM结果

    参数:
        model: PyTorch模型
        image: 输入图像，形状为 (3, H, W)，已归一化
        target_layer: 目标层名称
        class_names: 类别名称列表
        figsize: 图像大小
    """
    # 创建Grad-CAM对象
    grad_cam = GradCAM(model, target_layer)

    # 准备输入
    input_tensor = image.unsqueeze(0)  # 添加批次维度

    # 计算预测结果
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)

    # 获取预测类别名称
    if class_names is not None:
        pred_class = class_names[pred.item()]
    else:
        pred_class = f"Class {pred.item()}"

    # 计算Grad-CAM热力图
    cam = grad_cam(input_tensor, class_idx=pred.item())
    grad_cam.remove_hooks()

    # 反归一化图像以便显示
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img_np = np.transpose(image.numpy(), (1, 2, 0))
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)

    # 将热力图转换为RGB
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # BGR to RGB

    # 将热力图叠加到原始图像上
    superimposed_img = heatmap * 0.4 + img_np
    superimposed_img = np.clip(superimposed_img, 0, 1)

    # 可视化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    ax1.imshow(img_np)
    ax1.set_title('原始图像')
    ax1.axis('off')

    ax2.imshow(cam, cmap='jet')
    ax2.set_title('Grad-CAM热力图')
    ax2.axis('off')

    ax3.imshow(superimposed_img)
    ax3.set_title(f'叠加图\n预测类别: {pred_class}')
    ax3.axis('off')

    plt.tight_layout()
    return fig
