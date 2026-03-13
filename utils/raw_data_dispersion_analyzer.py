import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

# ================= 配置区 =================
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Z_array.mat')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'raw_data_analysis')
TARGET_FREQS = [0.2, 0.8, 2, 8, 15, 20] # 关注的典型频率 (MHz)
# ==========================================

def load_and_process_data(mat_path):
    print(f"Loading raw data from: {mat_path}")
    mat_data = sio.loadmat(mat_path)['Z']
    
    # 提取基础列
    sigma = mat_data[:, 0]
    thickness = mat_data[:, 1]
    freq = mat_data[:, 2]
    
    # 计算 LVF: (50 * thickness - thickness**2) / 625
    lvf = (50 * thickness - thickness**2) / 625
    
    # 构建基础 DataFrame
    df = pd.DataFrame({
        'sigma': sigma,
        'thickness': thickness,
        'LVF': lvf,
        'freq_mhz': freq,
        # 对频率四舍五入，方便后续提取特定的离散频率
        'freq_round': np.round(freq, decimals=2) 
    })
    
    # 提取 ch1, ch2, ch3, ch4 的测量数据 (对应第1到第4个测量通道)
    # 索引偏移：imag/w 从第3列开始，real从31开始，phase从59开始
    for i, ch in enumerate([1, 2, 3, 4]):
        col_imag = 3 + i
        col_real = 31 + i
        col_phase = 59 + i
        
        df[f'imag_w_ch{ch}'] = mat_data[:, col_imag]
        df[f'real_ch{ch}'] = mat_data[:, col_real]
        df[f'phase_ch{ch}'] = mat_data[:, col_phase]
        
        # 计算角频率 w = 2 * pi * f (注意频率单位是 MHz，转换为 Hz)
        w = 2 * np.pi * df['freq_mhz'] * 1e6
        
        # 计算 amp(Y) = sqrt(real(Y)^2 + imag(Y)^2)
        # 注意: data 中存储的是 imag(Y)/w，所以 imag(Y) = (imag(Y)/w) * w
        imag_Y = df[f'imag_w_ch{ch}'] * w
        df[f'amp_ch{ch}'] = np.sqrt(df[f'real_ch{ch}']**2 + imag_Y**2)

    return df

def plot_dispersion(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sigmas = df['sigma'].unique()
    
    # 我们要绘制的4个物理量
    metrics = {
        'imag_w': 'Capacitance (imag(Y)/w) [F]',
        'real': 'Conductance (real(Y)) [S]',
        'phase': 'Phase Angle [Deg]',
        'amp': 'Admittance Amplitude (amp(Y)) [S]'
    }
    
    for sigma_val in sigmas:
        df_sigma = df[df['sigma'] == sigma_val]
        
        for metric_prefix, y_label in metrics.items():
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Dispersion Analysis: {y_label}\n(Conductivity $\sigma$ = {sigma_val})', fontsize=16)
            
            # 绘制 ch1 到 ch4
            for ch_idx, ax in enumerate(axes.flatten(), 1):
                metric_name = f'{metric_prefix}_ch{ch_idx}'
                
                # 在同一张子图里画出不同频率的曲线
                for f in TARGET_FREQS:
                    df_freq = df_sigma[df_sigma['freq_round'] == f].copy()
                    
                    if not df_freq.empty:
                        # 确保按照 LVF 从小到大排序，以便画出平滑的折线图
                        df_freq = df_freq.sort_values(by='LVF')
                        ax.plot(df_freq['LVF'], df_freq[metric_name], marker='o', markersize=4, label=f'{f} MHz')
                
                ax.set_title(f'Channel {ch_idx}')
                ax.set_xlabel('Local Void Fraction (LVF)')
                ax.set_ylabel(y_label)
                ax.grid(True, linestyle='--', alpha=0.7)
                if ch_idx == 1: # 只在第一个子图显示图例，避免遮挡
                    ax.legend(title='Frequency', loc='best')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # 保存图表
            save_name = f'Dispersion_Sigma_{sigma_val}_{metric_prefix}.png'
            plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=300)
            plt.close()
            print(f"Saved: {save_name}")

if __name__ == '__main__':
    # 检查原始数据是否存在
    if not os.path.exists(DATA_PATH):
        print(f"Error: Raw data file not found at {DATA_PATH}")
        print("Please ensure 'Z_array.mat' is placed in the 'data/raw/' directory.")
    else:
        df_processed = load_and_process_data(DATA_PATH)
        plot_dispersion(df_processed)
        print(f"All plots have been saved to {OUTPUT_DIR}")