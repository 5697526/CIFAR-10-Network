# analyze_search_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def analyze_results(results_file, save_dir='./search_analysis'):
    """分析超参数搜索结果"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载结果
    df = pd.read_csv(results_file)

    # 显示最佳配置
    best_config = df.loc[df['best_accuracy'].idxmax()]
    print("最佳配置:")
    print(best_config)

    # 保存最佳配置
    best_config.to_csv(os.path.join(save_dir, 'best_config.csv'))

    # 可视化准确率分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df['best_accuracy'], kde=True)
    plt.title('准确率分布')
    plt.xlabel('准确率 (%)')
    plt.ylabel('试验次数')
    plt.savefig(os.path.join(save_dir, 'accuracy_distribution.png'))
    plt.close()

    # 分析不同参数对准确率的影响
    categorical_params = ['filter_config', 'fc_units',
                          'activation', 'optimizer', 'loss_type']

    for param in categorical_params:
        if param in df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=param, y='best_accuracy', data=df)
            plt.title(f'{param} 对准确率的影响')
            plt.savefig(os.path.join(save_dir, f'{param}_vs_accuracy.png'))
            plt.close()

    continuous_params = ['learning_rate', 'momentum', 'weight_decay', 'batch_size',
                         'focal_alpha', 'focal_gamma', 'ls_smoothing']

    for param in continuous_params:
        if param in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=param, y='best_accuracy', data=df)
            plt.title(f'{param} 对准确率的影响')
            plt.savefig(os.path.join(save_dir, f'{param}_vs_accuracy.png'))
            plt.close()

    # 相关性分析
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('参数相关性分析')
    plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'))
    plt.close()

    print(f"分析完成，结果保存在 {save_dir} 目录中")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='分析超参数搜索结果')
    parser.add_argument('--results_file', type=str, required=True,
                        help='搜索结果CSV文件路径')
    parser.add_argument('--save_dir', type=str, default='./search_analysis',
                        help='保存分析结果的目录')
    args = parser.parse_args()

    analyze_results(args.results_file, args.save_dir)
