# utils/hyperparameter_search.py
import numpy as np
import torch
import random
import os
from datetime import datetime
from models.model_optimized import ConfigurableCNN, FILTER_CONFIGS, FC_UNITS_CONFIGS
from utils.dataloader import get_dataloaders
from utils.loss_utils import get_loss_function, CombinedLoss, L1Regularization, L2Regularization
from utils.train_utils import train, evaluate
from torch.optim import lr_scheduler
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class HyperparameterSearch:
    """超参数搜索基类"""

    def __init__(self, model_builder, dataloaders, num_trials=10, save_dir='./hyperparameter_search'):
        self.model_builder = model_builder
        self.trainloader, self.testloader = dataloaders
        self.num_trials = num_trials
        self.save_dir = save_dir
        self.results = []

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 定义超参数搜索空间
        self.search_space = {
            # 网络结构
            'filter_config': list(FILTER_CONFIGS.keys()),
            'fc_units': list(FC_UNITS_CONFIGS.keys()),
            'activation': ['mish'],

            # 训练参数
            'batch_size': [64, 128, 192],
            'num_epochs': [25, 30, 35],

            # 优化器参数
            'optimizer': ['adam'],
            'learning_rate': [0.001, 0.0015, 0.005],
            'momentum': [0.85, 0.9, 0.95],
            'weight_decay': [0.0001, 0.00015, 0.00005],

            # 损失函数参数
            'loss_type': ['ls'],
            'focal_alpha': [0.5, 1, 2, 5],
            'focal_gamma': [0, 1, 2, 3],
            'ls_smoothing': [0, 0.05, 0.01]
        }

    def get_random_config(self):
        """生成随机超参数配置"""
        config = {}
        for param, values in self.search_space.items():
            if isinstance(values, list):
                config[param] = random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], (int, float)):
                # 连续范围
                if isinstance(values[0], int):
                    config[param] = random.randint(values[0], values[1])
                else:
                    config[param] = random.uniform(values[0], values[1])
            else:
                raise ValueError(f"不支持的参数类型: {param}")

        # 特定参数依赖关系
        if config['optimizer'] != 'sgd':
            config['momentum'] = 0  # 非SGD优化器不需要动量

        if config['loss_type'] != 'focal':
            config['focal_alpha'] = 1
            config['focal_gamma'] = 2

        if config['loss_type'] != 'ls':
            config['ls_smoothing'] = 0.1

        return config

    def evaluate_config(self, config):
        """评估给定的超参数配置"""
        # 设备配置
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 创建模型
        model = self.model_builder(
            num_classes=10,
            filter_config=FILTER_CONFIGS[config['filter_config']],
            fc_units=FC_UNITS_CONFIGS[config['fc_units']]
        ).to(device)

        # 初始化损失函数
        base_loss = get_loss_function(
            loss_type=config['loss_type'],
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma'],
            smoothing=config['ls_smoothing']
        )

        # 初始化正则化
        l1_reg = L1Regularization(model, 0) if config.get(
            'use_l1', False) else None
        l2_reg = L2Regularization(model, config['weight_decay']) if config.get(
            'use_l2', True) else None

        # 组合损失函数
        criterion = CombinedLoss(base_loss, l1_reg, l2_reg)

        # 初始化优化器
        if config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config['learning_rate'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )

        # 学习率调度
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['num_epochs'])

        # 训练模型
        best_acc = 0.0
        for epoch in range(config['num_epochs']):
            train_loss = train(model, device, self.trainloader,
                               criterion, optimizer, epoch)
            test_acc = evaluate(model, device, self.testloader)

            if test_acc > best_acc:
                best_acc = test_acc

            scheduler.step()

        # 保存结果
        result = {
            'config': config,
            'best_accuracy': best_acc,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.results.append(result)
        self.save_results()

        return best_acc

    def save_results(self):
        """保存搜索结果"""
        # 转换为DataFrame
        df = pd.DataFrame([
            {**r['config'], 'best_accuracy': r['best_accuracy'],
                'timestamp': r['timestamp']}
            for r in self.results
        ])

        # 保存为CSV
        df.to_csv(os.path.join(self.save_dir, 'search_results.csv'), index=False)

        # 保存最佳模型
        if len(self.results) > 0:
            best_result = max(self.results, key=lambda x: x['best_accuracy'])
            with open(os.path.join(self.save_dir, 'best_config.txt'), 'w') as f:
                for key, value in best_result['config'].items():
                    f.write(f"{key}: {value}\n")
                f.write(
                    f"Best accuracy: {best_result['best_accuracy']:.2f}%\n")

    def run(self):
        """运行超参数搜索"""
        raise NotImplementedError("子类必须实现此方法")


class RandomSearch(HyperparameterSearch):
    """随机超参数搜索"""

    def run(self):
        """运行随机搜索"""
        print(f"开始随机超参数搜索，总试验次数: {self.num_trials}")

        for i in range(self.num_trials):
            print(f"\n=== 试验 {i+1}/{self.num_trials} ===")
            config = self.get_random_config()

            print("当前配置:")
            for key, value in config.items():
                print(f"  {key}: {value}")

            accuracy = self.evaluate_config(config)
            print(f"最终准确率: {accuracy:.2f}%")

        # 输出最佳结果
        if len(self.results) > 0:
            best_result = max(self.results, key=lambda x: x['best_accuracy'])
            print(f"\n最佳准确率: {best_result['best_accuracy']:.2f}%")
            print("最佳配置:")
            for key, value in best_result['config'].items():
                print(f"  {key}: {value}")


class BayesianOptimization(HyperparameterSearch):
    """贝叶斯超参数优化"""

    def __init__(self, model_builder, dataloaders, num_trials=10, num_initial_random=5,
                 save_dir='./bayesian_optimization'):
        super().__init__(model_builder, dataloaders, num_trials, save_dir)
        self.num_initial_random = num_initial_random

        # 连续参数的映射
        self.continuous_params = {
            'learning_rate': (0.001, 0.1),
            'momentum': (0.85, 0.95),
            'weight_decay': (0, 0.001),
            'focal_alpha': (0.5, 5),
            'focal_gamma': (0, 3),
            'ls_smoothing': (0, 0.2)
        }

        # 离散参数的映射
        self.discrete_params = {
            'filter_config': list(FILTER_CONFIGS.keys()),
            'fc_units': list(FC_UNITS_CONFIGS.keys()),
            'activation': ['relu', 'swish', 'mish', 'selu', 'gelu'],
            'batch_size': [32, 64, 128, 256],
            'optimizer': ['sgd', 'adam', 'rmsprop'],
            'loss_type': ['ce', 'focal', 'ls']
        }

        # 高斯过程回归模型
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )

    def config_to_features(self, config):
        """将配置转换为特征向量"""
        features = []

        # 添加连续参数
        for param, (min_val, max_val) in self.continuous_params.items():
            if param in config:
                # 归一化到 [0, 1]
                value = (config[param] - min_val) / (max_val - min_val)
                features.append(value)

        # 添加离散参数
        for param, values in self.discrete_params.items():
            if param in config:
                # one-hot编码
                one_hot = [0] * len(values)
                one_hot[values.index(config[param])] = 1
                features.extend(one_hot)

        return np.array(features)

    def expected_improvement(self, X, X_sample, Y_sample, xi=0.01):
        """计算预期改进"""
        mu, sigma = self.gp.predict(X, return_std=True)
        mu_sample = self.gp.predict(X_sample)

        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def run(self):
        """运行贝叶斯优化"""
        print(f"开始贝叶斯超参数优化，总试验次数: {self.num_trials}")
        print(f"初始随机试验次数: {self.num_initial_random}")

        # 初始随机试验
        for i in range(self.num_initial_random):
            print(f"\n=== 随机试验 {i+1}/{self.num_initial_random} ===")
            config = self.get_random_config()

            print("当前配置:")
            for key, value in config.items():
                print(f"  {key}: {value}")

            accuracy = self.evaluate_config(config)
            print(f"最终准确率: {accuracy:.2f}%")

        # 贝叶斯优化迭代
        for i in range(self.num_initial_random, self.num_trials):
            print(f"\n=== 贝叶斯优化试验 {i+1}/{self.num_trials} ===")

            # 准备训练数据
            X = np.array([self.config_to_features(r['config'])
                         for r in self.results])
            Y = np.array([r['best_accuracy']
                         for r in self.results]).reshape(-1, 1)

            # 拟合高斯过程模型
            self.gp.fit(X, Y)

            # 生成候选配置
            num_candidates = 1000
            candidates = [self.get_random_config()
                          for _ in range(num_candidates)]
            candidate_features = np.array(
                [self.config_to_features(c) for c in candidates])

            # 计算预期改进
            ei = self.expected_improvement(candidate_features, X, Y)

            # 选择预期改进最大的配置
            best_idx = np.argmax(ei)
            next_config = candidates[best_idx]

            print("推荐配置:")
            for key, value in next_config.items():
                print(f"  {key}: {value}")

            # 评估新配置
            accuracy = self.evaluate_config(next_config)
            print(f"最终准确率: {accuracy:.2f}%")

        # 输出最佳结果
        if len(self.results) > 0:
            best_result = max(self.results, key=lambda x: x['best_accuracy'])
            print(f"\n最佳准确率: {best_result['best_accuracy']:.2f}%")
            print("最佳配置:")
            for key, value in best_result['config'].items():
                print(f"  {key}: {value}")
