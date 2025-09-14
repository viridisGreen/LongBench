import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def plot_multiple_model_performance(model_names):
    """
    读取多个模型的 result.json 文件，并将它们的性能数据绘制在一张
    经过美化的、具有西湖大学风格的对比图表中。

    Args:
        model_names (list): 一个或多个模型名称的列表，每个名称对应 'pred/' 下的一个文件夹。
    """
    all_scores = {}
    
    # --- 数据加载 ---
    for model_name in model_names:
        result_path = os.path.join('pred', model_name, 'result.json')

        if not os.path.exists(result_path):
            print(f"警告: 在 '{result_path}' 未找到模型 '{model_name}' 的结果文件，将跳过此模型。")
            continue

        with open(result_path, 'r', encoding='utf-8') as f:
            try:
                scores = json.load(f)
                all_scores[model_name] = scores
            except json.JSONDecodeError:
                print(f"警告: 无法解析 '{result_path}' 的JSON文件，将跳过此模型。")
                continue

    if not all_scores:
        print("错误: 未加载任何有效的模型数据，无法生成图表。")
        return

    # --- 数据准备 ---
    df = pd.DataFrame(all_scores).sort_index()

    # --- 图表绘制 ---
    # 定义西湖大学风格色板和标记形状
    # 主色：求知橙, 辅助色：西湖蓝, 远山黛, 创新紫, 活力绿
    westlake_palette = {
        'colors': ['#E57224', '#004A7F', '#5C5C5C', '#6F42C1', '#28A745', '#DC3545', '#17A2B8'],
        'markers': ['o', 's', '^', 'D', 'v', 'P', '*']
    }

    # 创建图表
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # 为每个模型绘制性能曲线
    for i, model_name in enumerate(df.columns):
        color = westlake_palette['colors'][i % len(westlake_palette['colors'])]
        marker = westlake_palette['markers'][i % len(westlake_palette['markers'])]
        
        ax.plot(df.index, df[model_name], 
                marker=marker, 
                linestyle='-', 
                color=color,
                label=model_name,
                markersize=8,
                linewidth=2.5,
                alpha=0.9)

    # --- 图表美化 ---
    # 设置标题和坐标轴标签
    ax.set_title('LongBench v1 Performance Comparison', fontsize=22, pad=25, fontfamily='Heiti TC', weight='bold')
    ax.set_xlabel('Dataset', fontsize=16, fontfamily='Heiti TC')
    ax.set_ylabel('Score (%)', fontsize=16, fontfamily='Heiti TC')

    # 调整刻度样式
    ax.tick_params(axis='x', rotation=90, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # 设置网格线
    ax.grid(True, which='major', linestyle='--', linewidth=0.7, color='grey', alpha=0.5)

    # 设置图例
    ax.legend(fontsize=14, frameon=True, facecolor='#F8F9FA', framealpha=1, edgecolor='none', loc='upper left', bbox_to_anchor=(1.01, 1))

    # 设置Y轴范围
    ax.set_ylim(0, 105)
    
    # 移除图表上和右侧的边框线，使其更简洁
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')

    # 调整布局并保存图表
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # 留出右侧空间给图例
    output_filename = 'LongBench_performance_comparison_styled.png'
    plt.savefig(output_filename, dpi=300) # 使用更高的DPI以获得更清晰的图片
    
    print(f"对比图表已保存至: '{output_filename}'")
    plt.show()


if __name__ == '__main__':
    # 设置参数解析器
    parser = argparse.ArgumentParser(description="根据LongBench评测结果绘制多个模型的性能对比图。")
    parser.add_argument('--model', type=str, nargs='+', required=True, 
                        help='需要绘制结果的一个或多个模型的名称 (例如: --model model1 model2)。')
    
    args = parser.parse_args()
    
    # 为了在不同系统上获得更好的中文显示效果，可以尝试设置字体
    try:
        plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Microsoft YaHei', 'SimHei'] # 优先使用黑体
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    except Exception as e:
        print(f"设置中文字体失败: {e}")

    plot_multiple_model_performance(args.model)