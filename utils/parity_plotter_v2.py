import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ================= 配置区 =================
# 1. 预测结果 CSV 路径
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'sweep_simulation_rf_20260309_1116', 'blind_test_predictions.csv')

# 2. 原始数据 MAT 路径 (用于提取真实的 thickness)
MAT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Z_array.mat')

# 3. 图像输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(CSV_PATH), 'physics_analysis')

# CSV 中预测厚度的列名
COL_PRED_THICKNESS = 'Predicted_Thickness' 
# ==========================================

def calc_lvf(thickness):
    """
    根据厚度计算局部空隙率 (LVF)
    公式: LVF = (50 * thickness - thickness**2) / 625
    """
    return (50 * thickness - thickness**2) / 625

def plot_parity_and_error(csv_path, mat_path):
    # 1. 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    if not os.path.exists(mat_path):
        print(f"Error: MAT file not found at {mat_path}")
        return

    # 2. 读取预测厚度数据
    df_pred = pd.read_csv(csv_path)
    if COL_PRED_THICKNESS in df_pred.columns:
        pred_thickness = df_pred[COL_PRED_THICKNESS].values
    else:
        # 如果列名不对，默认取第一列
        print(f"Warning: Column '{COL_PRED_THICKNESS}' not found. Using the first column.")
        pred_thickness = df_pred.iloc[:, 0].values

    # 3. 读取真实原始数据提取真实厚度
    print(f"Loading raw MAT data from {mat_path}...")
    mat_data = sio.loadmat(mat_path)['Z']
    # 第 2 列 (索引为 1) 是 thickness
    true_thickness = mat_data[:, 1] 

    # 4. 长度一致性校验
    if len(pred_thickness) != len(true_thickness):
        print(f"Warning: The number of predictions ({len(pred_thickness)}) does not match the true data ({len(true_thickness)}).")
        print("Truncating to the shorter length to align data (Ensure your blind test set is the same as Z_array!).")
        min_len = min(len(pred_thickness), len(true_thickness))
        pred_thickness = pred_thickness[:min_len]
        true_thickness = true_thickness[:min_len]

    # 5. 转换为 LVF (核心物理转换)
    y_pred = calc_lvf(pred_thickness)
    y_true = calc_lvf(true_thickness)

    # 6. 计算误差与评估指标
    errors = y_pred - y_true
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"Metrics (LVF scale) -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # ================= 开始绘图 =================
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Global Prediction Accuracy and Error Distribution (LVF)', fontsize=16, fontweight='bold', y=1.02)

    # ---------- 子图 (a): Parity Plot (回归散点图) ----------
    ax1 = axes[0]
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
    sns.histplot(errors, bins=40, kde=True, color='teal', edgecolor='black', alpha=0.6, ax=ax2)
    
    # 添加误差均值线
    mean_err = np.mean(errors)
    ax2.axvline(mean_err, color='red', linestyle='dashed', linewidth=2, label=f'Mean Error ($\mu$={mean_err:.4f})')
    # 添加 ±1个标准差的区间
    std_err = np.std(errors)
    ax2.axvline(mean_err + std_err, color='orange', linestyle='dotted', linewidth=2, label=f'+1 Std Dev ($\sigma$={std_err:.4f})')
    ax2.axvline(mean_err - std_err, color='orange', linestyle='dotted', linewidth=2, label='-1 Std Dev')

    ax2.set_title('(b) Prediction Error Distribution (LVF)', fontsize=14)
    ax2.set_xlabel('Absolute Error (Predicted - True LVF)', fontsize=13)
    ax2.set_ylabel('Frequency (Number of Samples)', fontsize=13)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='upper right')

    # ================= 保存图表 =================
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'Parity_and_Error_Distribution_LVF.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot successfully saved to: {save_path}")

if __name__ == "__main__":
    plot_parity_and_error(CSV_PATH, MAT_PATH)