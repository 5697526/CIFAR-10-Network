# analyze_search_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def analyze_results(results_file, save_dir='./search_analysis'):
    """Analyze hyperparameter search results"""
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Load results
    df = pd.read_csv(results_file)

    # Display the best configuration
    best_config = df.loc[df['best_accuracy'].idxmax()]
    print("Best configuration:")
    print(best_config)

    # Save the best configuration
    best_config.to_csv(os.path.join(save_dir, 'best_config.csv'))

    # Visualize accuracy distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['best_accuracy'], kde=True)
    plt.title('Accuracy Distribution')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Trial Count')
    plt.savefig(os.path.join(save_dir, 'accuracy_distribution.png'))
    plt.close()

    # Analyze the impact of categorical parameters on accuracy
    categorical_params = ['filter_config', 'fc_units',
                          'activation', 'optimizer', 'loss_type']

    for param in categorical_params:
        if param in df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=param, y='best_accuracy', data=df)
            plt.title(f'Impact of {param} on Accuracy')
            plt.savefig(os.path.join(save_dir, f'{param}_vs_accuracy.png'))
            plt.close()

    # Analyze the impact of continuous parameters on accuracy
    continuous_params = ['learning_rate', 'momentum', 'weight_decay', 'batch_size',
                         'focal_alpha', 'focal_gamma', 'ls_smoothing']

    for param in continuous_params:
        if param in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=param, y='best_accuracy', data=df)
            plt.title(f'Impact of {param} on Accuracy')
            plt.savefig(os.path.join(save_dir, f'{param}_vs_accuracy.png'))
            plt.close()

    # Correlation analysis
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Analysis')
    plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'))
    plt.close()

    print(f"Analysis completed. Results saved in: {save_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze hyperparameter search results')
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to the CSV file of search results')
    parser.add_argument('--save_dir', type=str, default='./search_analysis',
                        help='Directory to save the analysis results')
    args = parser.parse_args()

    analyze_results(args.results_file, args.save_dir)
