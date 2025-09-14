import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def plot_model_performance(model_name):
    """
    Reads a model's result.json file and plots its performance across datasets.
    The generated chart will be saved in the same directory as the result.json file.

    Args:
        model_name (str): The name of the model, corresponding to the folder in 'pred/'.
    """
    # Construct the path to the model's result directory and the result file itself
    result_dir = os.path.join('pred', model_name)
    result_path = os.path.join(result_dir, 'result.json')

    # Check if the result file exists
    if not os.path.exists(result_path):
        print(f"错误: 在 '{result_path}' 未找到结果文件")
        print("请确保您已经运行了评测脚本，并且该文件已存在。")
        return

    # Load the data from the result.json file
    with open(result_path, 'r', encoding='utf-8') as f:
        try:
            scores = json.load(f)
        except json.JSONDecodeError:
            print(f"错误: 无法从 '{result_path}' 解析JSON。文件可能为空或已损坏。")
            return

    # Prepare data for plotting
    df = pd.DataFrame(list(scores.items()), columns=['Dataset', 'Score']).sort_values('Dataset')

    # --- Plotting ---
    # 定义西湖大学求知橙色
    knowledge_orange = '#E57224'

    # Create the plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Plot the data
    ax.plot(df['Dataset'], df['Score'], marker='o', linestyle='-', color='dodgerblue',
            markerfacecolor=knowledge_orange, markeredgecolor=knowledge_orange, markersize=8, label=model_name)

    # Customize the plot
    ax.set_title(f'{model_name}', fontsize=18, pad=20)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.tick_params(axis='x', rotation=90, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, color='grey', alpha=0.5)
    ax.legend(fontsize=12, frameon=True, facecolor='white', framealpha=0.8)
    ax.set_ylim(0, 105)

    # Adjust layout and save the figure to the same directory as result.json
    plt.tight_layout()
    output_filename = os.path.join(result_dir, f'{model_name}_performance_chart.png')
    plt.savefig(output_filename)
    print(f"图表已保存至: '{output_filename}'")
    plt.show()


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="根据LongBench评测结果绘制模型性能图表。")
    parser.add_argument('--model', type=str, required=True, help='需要绘制结果的模型的名称。')
    
    args = parser.parse_args()
    
    plot_model_performance(args.model)