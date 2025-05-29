import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models.model import BasicCNN
from utils.train_utils import train, evaluate
from utils.dataloader import get_dataloaders


def main():
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 超参数设置
    num_epochs = 30
    learning_rate = 0.01
    batch_size = 128
    weight_decay = 5e-4

    # 获取数据加载器
    trainloader, testloader = get_dataloaders(batch_size)

    # 初始化模型、损失函数与优化器
    model = BasicCNN().to(device)
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay
    )  # 优化器
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)  # 学习率调度

    # 训练模型
    best_acc = 0.0
    for epoch in range(num_epochs):
        train_loss = train(model, device, trainloader,
                           criterion, optimizer, epoch)
        test_acc = evaluate(model, device, testloader)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), './results/best_model.pth')

        scheduler.step()
        print(
            f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%')

    print(f'Best test accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
