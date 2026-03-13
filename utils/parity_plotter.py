import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ================= 配置区 =================
# 指向您某次实验生成的预测结果文件
# 例如: '../experiments/sweep_simulation_dl_20260309_1627/blind_test_predictions.csv'
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'sweep_simulation_dl_20260309_1627', 'blind_test_predictions.csv')

# 替换为您 CSV 文件中实际的列名
COL_TRUE = 'True_LVF'       # 真实值的列名 (请根据实际 csv 检查并修改, 有可能是 'y_true' 等)
COL_PRED = 'Predicted_LVF'  # 预测值的列名 (请根据实际 csv 检查并修改, 有可能是 'y_pred' 等)

OUTPUT_DIR = os.path.join(os.path.dirname(CSV_PATH), 'physics_analysis')
# ==========================================

def plot_parity_and_error(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    # 1. 读取数据
    df = pd.read_csv(csv_path)
    
    # 兼容性检查：如果列名不对，尝试猜测常见的列名
    if COL_TRUE not in df.columns or COL_PRED not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        # 简单回退机制
        y_true = df.iloc[:, 0].values
        y_pred = df.iloc[:, 1].values
    else:
        y_true = df[COL_TRUE].values
        y_pred = df[COL_PRED].values

    # 2. 计算误差与评估指标
    errors = y_pred - y_true
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"Metrics -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # 3. 开始绘图 (1行2列的组合图)
    # 采用学术风格配置
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Global Prediction Accuracy and Error Distribution', fontsize=16, fontweight='bold', y=1.02)

    # ---------- 子图 (a): Parity Plot (回归散点图) ----------
    ax1 = axes[0]
    
    # 绘制散点
    # 为了表现密度，可以调整透明度 alpha，或者使用 sns.scatterplot
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor='k', s=40, color='royalblue', ax=ax1)
    
    # 绘制 y=x 理想对角线
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    buffer = (max_val - min_val) * 0.05
    ax1.plot([min_val - buffer, max_val + buffer], 
             [min_val - buffer, max_val + buffer], 
             'r--', lw=2, label='Ideal Prediction (y = x)')
    
    # 标注指标文本框
    textstr = '\n'.join((
        f'$R^2$ = {r2:.4f}',
        f'RMSE = {rmse:.4f}',
        f'MAE = {mae:.4f}'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    ax1.set_title('(a) Predicted vs. True LVF', fontsize=14)
    ax1.set_xlabel('True LVF', fontsize=13)
    ax1.set_ylabel('Predicted LVF', fontsize=13)
    ax1.set_xlim([min_val - buffer, max_val + buffer])
    ax1.set_ylim([min_val - buffer, max_val + buffer])
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='lower right')

    # ---------- 子图 (b): Error Histogram (误差分布直方图) ----------
    ax2 = axes[1]
    
    # 使用 seaborn 绘制带有核密度估计(KDE)的直方图
    sns.histplot(errors, bins=40, kde=True, color='teal', edgecolor='black', alpha=0.6, ax=ax2)
    
    # 添加误差均值线 (理论上应该接近0)
    mean_err = np.mean(errors)
    ax2.axvline(mean_err, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error ($\mu$={mean_err:.4f})')
    # 添加 ±1个标准差的区间
    std_err = np.std(errors)
    ax2.axvline(mean_err + std_err, color='orange', linestyle='dotted', linewidth=2, label=f'+1 Std Dev ($\sigma$={std_err:.4f})')
    ax2.axvline(mean_err - std_err, color='orange', linestyle='dotted', linewidth=2, label=f'-1 Std Dev')

    ax2.set_title('(b) Prediction Error Distribution', fontsize=14)
    ax2.set_xlabel('Absolute Error (Predicted - True LVF)', fontsize=13)
    ax2.set_ylabel('Frequency (Number of Samples)', fontsize=13)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='upper right')

    # 4. 优化布局并保存
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'Parity_and_Error_Distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot successfully saved to: {save_path}")

if __name__ == "__main__":
    plot_parity_and_error(CSV_PATH)