# hyperparameter_search_main.py
import torch
from models.model_optimized import ConfigurableCNN
from utils.dataloader import get_dataloaders
from utils.hyperparameter_search import RandomSearch, BayesianOptimization
import argparse
import numpy as np
import random


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 超参数搜索')
    parser.add_argument('--method', type=str, default='random', choices=['random', 'bayesian'],
                        help='超参数搜索方法 (random 或 bayesian)')
    parser.add_argument('--trials', type=int, default=20,
                        help='试验次数')
    parser.add_argument('--initial_random', type=int, default=5,
                        help='贝叶斯优化的初始随机试验次数')
    parser.add_argument('--save_dir', type=str, default='./hyperparameter_search',
                        help='保存搜索结果的目录')
    args = parser.parse_args()

    # 确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 获取数据加载器
    trainloader, testloader = get_dataloaders(
        batch_size=32)  # 默认批次大小，每个试验会使用自己的配置

    # 初始化搜索方法
    if args.method == 'random':
        search = RandomSearch(
            model_builder=ConfigurableCNN,
            dataloaders=(trainloader, testloader),
            num_trials=args.trials,
            save_dir=args.save_dir
        )
    else:  # bayesian
        search = BayesianOptimization(
            model_builder=ConfigurableCNN,
            dataloaders=(trainloader, testloader),
            num_trials=args.trials,
            num_initial_random=args.initial_random,
            save_dir=args.save_dir
        )

    # 运行搜索
    search.run()


if __name__ == '__main__':
    main()
