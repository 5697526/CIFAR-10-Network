# train_optimized.py
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
import logging

# 配置日志记录
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    print(f"Using device: {device}")

    # 创建结果保存目录
    os.makedirs('./results', exist_ok=True)

    # 优化策略配置
    config = {
        "filter_config": "large",
        "fc_units": "default",

        "activation": "mish",

        "loss_type": "ls",
        "focal_alpha": 1,
        "focal_gamma": 2,
        "ls_smoothing": 0.05,

        "use_l1": False,
        "lambda_l1": 1e-4,
        "use_l2": True,
        "lambda_l2": 5e-4,

        "num_epochs": 35,
        "batch_size": 128,

        "optimizer": "adam",
        "learning_rate": 0.001,
        "momentum": 0,
        "weight_decay": 0.00015,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
    }

    # 打印当前配置并记录到日志
    logging.info("训练配置:")
    print("训练配置:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")
        print(f"  {key}: {value}")

    # 获取数据加载器
    trainloader, testloader = get_dataloaders(config["batch_size"])

    # 初始化模型
    model = ConfigurableCNN(
        num_classes=10,
        filter_config=FILTER_CONFIGS[config["filter_config"]],
        fc_units=FC_UNITS_CONFIGS[config["fc_units"]]
    ).to(device)

    # 初始化损失函数 - 传递类别数
    base_loss = get_loss_function(
        loss_type=config["loss_type"],
        alpha=config["focal_alpha"],
        gamma=config["focal_gamma"],
        smoothing=config["ls_smoothing"]
    )

    # 初始化正则化 - 仅保留L1正则化（L2通过优化器实现）
    l1_reg = L1Regularization(
        model, config["lambda_l1"]) if config["use_l1"] else None

    # 组合损失函数 - 不再传递L2正则化
    criterion = CombinedLoss(base_loss, l1_reg, None)

    # 初始化优化器 - 仅通过优化器使用L2正则化
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            betas=(config["beta1"], config["beta2"]),
            eps=config["eps"],
            weight_decay=config["weight_decay"]  # 确保L2正则化只应用一次
        )
    else:
        raise ValueError(f"不支持的优化器类型: {config['optimizer']}")

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
        log_message = f'Epoch {epoch+1}/{config["num_epochs"]}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%'
        logging.info(log_message)
        print(log_message)

    logging.info(f'Best test accuracy: {best_acc:.2f}%')
    print(f'Best test accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
