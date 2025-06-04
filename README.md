# CIFAR-10-Network


### 一、项目概述：

本项目旨在 CIFAR - 10 数据集上训练神经网络模型，以实现图像分类任务的性能优化。通过构建不同的神经网络模型，运用多种优化策略和超参数搜索方法，以提高模型在 CIFAR - 10 数据集上的分类准确率。同时，项目还包含了对网络的可视化分析，帮助理解模型的学习过程和决策机制。


### 二、数据集介绍

本实验使用的 [CIFAR - 10 ](https://www.cs.toronto.edu/~kriz/cifar.html) 数据集是一个广泛用于图像分类研究的公开数据集。
- **数据内容**：60,000张32×32像素的彩色图像
- **图像类别**：10个类别（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）
- **数据划分**：
  - 训练集：50,000张（5,000张/类）
  - 测试集：10,000张（1,000张/类）
- **图像通道**：RGB三通道

### 三、文件结构

```
CIFAR-10-Network/
├── analyze_search_results.py         # 分析超参数搜索结果
├── hyperparameter_search_main.py     # 超参数搜索主脚本
├── hyperparameter_search/
│   └── best_config.txt               # 超参数搜索得到的最佳配置
├── models/
│   ├── model.py                      # 基本卷积神经网络和残差网络定义
│   ├── model_optimized.py            # 可配置的卷积神经网络定义
│   └── activations.py                # 自定义激活函数定义
├── results/
│   └── best_config.txt               # 训练得到的最佳配置
├── train.py                          # 使用基本模型进行训练
├── train_optimized.py                # 使用优化模型进行训练
├── train_component_optimizers.py     # 为不同层使用不同优化器进行训练
├── utils/
│   ├── dataloader.py                 # 数据加载器
│   ├── train_utils.py                # 训练和评估工具函数
│   ├── loss_utils.py                 # 损失函数和正则化工具函数
│   ├── custom_optimizer.py           # 自定义优化器
│   ├── hyperparameter_search.py      # 超参数搜索工具函数
│   ├── grad_cam.py                   # Grad - CAM可视化工具函数
│   └── visualization.py              # 网络可视化工具函数
├── visualize_network.py              # 网络可视化脚本
└── visualize_grad_cam.py             # Grad - CAM可视化脚本
```

### 四、实验步骤


#### 1. 环境准备：

使用以下命令安装必要的 Python 库:
```
pip install torch torchvision numpy matplotlib seaborn tqdm
```

#### 2. 数据加载：

使用`utils/dataloader.py`用于加载 CIFAR - 10 数据集并进行预处理。

```
python utils/dataloader.py
```
#### 3. 模型定义：

在`models`目录下定义了不同的模型，包括`BasicCNN`、`ResNet`和`ConfigurableCNN`。可以根据需要选择合适的模型，并配置相应的参数。

#### 4. 训练模型：

可以使用不同的训练脚本来训练模型，如`train.py`、`train_optimized.py`、`train_component_optimizers.py`和`train_custom_optimizer.py`。这些脚本中包含了不同的优化策略和超参数配置。

```
python train_optimized.py
```


#### 5. 超参数搜索：

使用`hyperparameter_search_main.py`脚本进行超参数搜索，可以选择随机搜索或贝叶斯优化方法。

```
python hyperparameter_search_main.py --method random --trials 20
```

#### 6. 结果分析：

使用`analyze_search_results.py`脚本分析超参数搜索结果，包括最佳配置的选择、准确率分布的可视化和参数相关性分析。

```
python analyze_search_results.py --results_file ./hyperparameter_search/search_results.csv --save_dir ./search_analysis
```

#### 7. 网络可视化：

使用`visualize_network.py`和`visualize_grad_cam.py`脚本进行网络可视化，包括滤波器可视化、特征图可视化、显著性图可视化、损失景观可视化和 Grad - CAM 可视化。

```
python visualize_network.py
```

### 五、模型权重下载
我的训练好的最佳模型保存在我的网盘：