# visualize_network.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.model_optimized import ConfigurableCNN, FILTER_CONFIGS
from utils.dataloader import get_dataloaders
from utils.visualization import visualize_filters, visualize_feature_maps, \
    plot_loss_landscape, visualize_saliency
import os


def main():
    # 设置中文字体
    plt.rcParams["font.family"] = "Microsoft YaHei"

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建保存可视化结果的目录
    os.makedirs('./visualizations', exist_ok=True)

    # 加载模型
    model = ConfigurableCNN(
        num_classes=10,
        filter_config=FILTER_CONFIGS["large"]
    ).to(device)

    # 加载预训练权重
    try:
        checkpoint = torch.load(
            './results/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("已加载预训练模型")
    except:
        print("未找到预训练模型，将使用随机初始化的模型")

    model.eval()

    # 获取数据加载器
    trainloader, testloader = get_dataloaders(batch_size=1)
    dataiter = iter(testloader)

    # 类别名称
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 1. 可视化卷积层滤波器
    print("可视化卷积层滤波器...")
    fig = visualize_filters(model, 'convs.0', num_cols=8, figsize=(12, 12))
    plt.savefig('./visualizations/conv1_filters.png')
    plt.close(fig)

    fig = visualize_filters(model, 'convs.1', num_cols=8, figsize=(12, 12))
    plt.savefig('./visualizations/conv2_filters.png')
    plt.close(fig)

    # 2. 可视化特征图
    print("可视化特征图...")
    try:
        images, labels = next(dataiter)
        images = (images - images.min()) / (images.max() - images.min())
        fig = visualize_feature_maps(
            model, 'convs.0', images[0], num_cols=8, figsize=(12, 12))
        plt.savefig('./visualizations/conv1_feature_maps.png')
        plt.close(fig)

        fig = visualize_feature_maps(
            model, 'convs.1', images[0], num_cols=8, figsize=(12, 12))
        plt.savefig('./visualizations/conv2_feature_maps.png')
        plt.close(fig)
    except StopIteration:
        print("数据加载器中没有足够的样本用于特征图可视化")

    # 3. 可视化显著性图 - 显示模型关注的区域
    print("可视化显著性图...")
    for i in range(min(5, len(testloader))):  # 可视化前5张图像
        try:
            images, labels = next(dataiter)
            images = (images - images.min()) / (images.max() - images.min())
            fig = visualize_saliency(
                model, images[0], target_class=labels[0].item())
            plt.savefig(f'./visualizations/saliency_map_{i}.png')
            plt.close(fig)
        except StopIteration:
            print(f"数据加载器中没有足够的样本，只可视化了 {i} 张显著性图")
            break

    # 4. 可视化损失景观 (计算量较大，可选执行)
    print("可视化损失景观...")
    fig = plot_loss_landscape(
        model, testloader, device, num_points=10, figsize=(10, 8))
    plt.savefig('./visualizations/loss_landscape.png')
    plt.close(fig)

    print("所有可视化已完成，结果保存在'./visualizations'目录中")


if __name__ == '__main__':
    main()
