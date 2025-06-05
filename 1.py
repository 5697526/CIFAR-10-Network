# visualize_network.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.model_optimized import ConfigurableCNN, FILTER_CONFIGS
from utils.dataloader import get_dataloaders
from utils.visualization import visualize_filters, visualize_feature_maps, \
    plot_loss_landscape, visualize_saliency
import os


# 打印当前工作目录
print(f"当前工作目录: {os.getcwd()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载预训练权重
model_path = './results/best_model.pth'
if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("已加载预训练模型")
    except Exception as e:
        print(f"加载模型时出现错误: {e}，将使用随机初始化的模型")
else:
    print("未找到预训练模型，将使用随机初始化的模型")

# ... 其他代码 ...
