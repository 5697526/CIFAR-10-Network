# train_component_optimizers.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models.model_optimized import ConfigurableCNN, FILTER_CONFIGS, FC_UNITS_CONFIGS
from utils.train_utils import train, evaluate
from utils.dataloader import get_dataloaders
from utils.loss_utils import get_loss_function, CombinedLoss, L1Regularization, L2Regularization
from models.activations import get_activation
import os


def main():
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建结果保存目录
    os.makedirs('./results', exist_ok=True)

    # 优化策略配置
    config = {
        # 网络结构配置
        "filter_config": "medium",
        "fc_units": "default",

        # 激活函数配置
        "activation": "relu",

        # 损失函数配置
        "loss_type": "ce",
        "focal_alpha": 1,
        "focal_gamma": 2,
        "ls_smoothing": 0.1,

        # 正则化配置
        "use_l1": False,
        "lambda_l1": 1e-4,
        "use_l2": True,
        "lambda_l2": 5e-4,

        # 训练超参数
        "num_epochs": 30,
        "batch_size": 128,

        # 优化器配置 - 卷积层
        "conv_optimizer": "sgd",
        "conv_lr": 0.01,
        "conv_momentum": 0.9,
        "conv_weight_decay": 5e-4,

        # 优化器配置 - 全连接层
        "fc_optimizer": "adam",
        "fc_lr": 0.001,
        "fc_beta1": 0.9,
        "fc_beta2": 0.999,
        "fc_weight_decay": 1e-4,
    }

    # 打印当前配置
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 获取数据加载器
    trainloader, testloader = get_dataloaders(config["batch_size"])

    # 初始化模型
    model = ConfigurableCNN(
        num_classes=10,
        filter_config=FILTER_CONFIGS[config["filter_config"]],
        fc_units=FC_UNITS_CONFIGS[config["fc_units"]]
    ).to(device)

    # 初始化损失函数
    base_loss = get_loss_function(
        loss_type=config["loss_type"],
        alpha=config["focal_alpha"],
        gamma=config["focal_gamma"],
        smoothing=config["ls_smoothing"]
    )

    # 初始化正则化
    l1_reg = L1Regularization(
        model, config["lambda_l1"]) if config["use_l1"] else None
    l2_reg = L2Regularization(
        model, config["lambda_l2"]) if config["use_l2"] else None

    # 组合损失函数
    criterion = CombinedLoss(base_loss, l1_reg, l2_reg)

    # 区分卷积层参数和全连接层参数
    conv_params = []
    fc_params = []

    for name, param in model.named_parameters():
        if 'conv' in name or 'bn' in name:
            conv_params.append(param)
        else:
            fc_params.append(param)

    # 为卷积层和全连接层分别创建优化器
    optimizer_groups = []

    # 卷积层优化器
    if config["conv_optimizer"] == "sgd":
        optimizer_groups.append({
            'params': conv_params,
            'lr': config["conv_lr"],
            'momentum': config["conv_momentum"],
            'weight_decay': config["conv_weight_decay"] if config["use_l2"] else 0
        })
    elif config["conv_optimizer"] == "adam":
        optimizer_groups.append({
            'params': conv_params,
            'lr': config["conv_lr"],
            'betas': (config["fc_beta1"], config["fc_beta2"]),
            'weight_decay': config["conv_weight_decay"] if config["use_l2"] else 0
        })

    # 全连接层优化器
    if config["fc_optimizer"] == "sgd":
        optimizer_groups.append({
            'params': fc_params,
            'lr': config["fc_lr"],
            'momentum': config["conv_momentum"],
            'weight_decay': config["fc_weight_decay"] if config["use_l2"] else 0
        })
    elif config["fc_optimizer"] == "adam":
        optimizer_groups.append({
            'params': fc_params,
            'lr': config["fc_lr"],
            'betas': (config["fc_beta1"], config["fc_beta2"]),
            'weight_decay': config["fc_weight_decay"] if config["use_l2"] else 0
        })

    # 创建优化器
    optimizer = optim.Adam(optimizer_groups)

    # 学习率调度
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"])

    # 训练模型
    best_acc = 0.0
    for epoch in range(config["num_epochs"]):
        train_loss = train(model, device, trainloader,
                           criterion, optimizer, epoch)
        test_acc = evaluate(model, device, testloader)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'config': config,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': test_acc
            }, './results/best_model.pth')

        scheduler.step()
        print(
            f'Epoch {epoch+1}/{config["num_epochs"]}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%')

    print(f'Best test accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
