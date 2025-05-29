# visualize_grad_cam.py
import torch
import matplotlib.pyplot as plt
from models.model_optimized import ConfigurableCNN, FILTER_CONFIGS
from utils.dataloader import get_dataloaders
from utils.grad_cam import visualize_grad_cam
import os


def main():
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建保存可视化结果的目录
    os.makedirs('./grad_cam_visualizations', exist_ok=True)

    # 加载模型
    model = ConfigurableCNN(
        num_classes=10,
        filter_config=FILTER_CONFIGS["medium"]
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

    # 可视化前10张图像的Grad-CAM结果
    for i in range(10):
        images, labels = dataiter.next()
        image = images[0].to(device)

        # 可视化Grad-CAM (使用最后一个卷积层)
        fig = visualize_grad_cam(
            model,
            image,
            target_layer='conv3',  # 可修改为其他卷积层
            class_names=classes,
            figsize=(12, 4)
        )

        plt.savefig(f'./grad_cam_visualizations/grad_cam_{i}.png')
        plt.close(fig)

    print("Grad-CAM可视化已完成，结果保存在'./grad_cam_visualizations'目录中")


if __name__ == '__main__':
    main()
