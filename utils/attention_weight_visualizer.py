import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def generate_and_save_visualizations(model, X_test_t, y_test_t, scaler_X, scaler_y, sample_index=0, save_base_dir="./experiments/visualizations"):
    model.eval()
    
    # 0. 准备保存目录
    save_dir_heat = os.path.join(save_base_dir, "heatmaps")
    save_dir_bar = os.path.join(save_base_dir, "importance_bars")
    os.makedirs(save_dir_heat, exist_ok=True)
    os.makedirs(save_dir_bar, exist_ok=True)

    # 1. 提取当前样本的数据与注意力权重
    device = next(model.parameters()).device
    sample_x = X_test_t[sample_index:sample_index+1].to(device) 
    
    sample_y = y_test_t[sample_index].item()

    with torch.no_grad():
        pred_y_scaled = model(sample_x)
        attn_matrix_all_heads = model.attention_weights[0].cpu().numpy()
        num_heads = attn_matrix_all_heads.shape[0]
        
        # 🌟 核心升级：动态获取当前模型的序列长度 (例如 28 或 86)
        seq_len = attn_matrix_all_heads.shape[-1]

    # 2. 反归一化获取物理量
    pred_real = scaler_y.inverse_transform(pred_y_scaled.cpu().numpy())[0][0]
    true_real = scaler_y.inverse_transform([[sample_y]])[0][0]
    
    sample_x_real = scaler_X.inverse_transform(sample_x.cpu().numpy())[0]
    sigma_real = sample_x_real[0]
    freq_real = sample_x_real[1]
    
    info_str = (f"基本工况: 电导率 = {sigma_real:.4g} | 激励频率 = {freq_real:.1f} Hz\n"
                f"验证结果: 实际厚度 = {true_real:.2f} | 预测厚度 = {pred_real:.2f}")
    
    print(f"[*] 正在生成可视化图表 | 样本 {sample_index} -> 实际: {true_real:.2f}, 预测: {pred_real:.2f} (序列长度: {seq_len})")

    # ==========================================
    # 🌟 图 1：自适应多头注意力热力图
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 14

    fig1 = plt.figure(figsize=(14, 11)) 
    fig1.suptitle(f"多头自注意力权重矩阵 (测试集样本 {sample_index})\n{info_str}", 
                  fontsize=18, fontweight='bold', y=0.96)

    gs1 = GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.15, hspace=0.3)
    fig1.subplots_adjust(top=0.85) 

    # 动态生成大约 5 个均匀的刻度，防止坐标轴密密麻麻
    ticks = np.linspace(0, seq_len - 1, 5, dtype=int)

    for i in range(min(4, num_heads)):
        row = i // 2
        col = i % 2
        
        ax1 = fig1.add_subplot(gs1[row, col])
        im = ax1.imshow(attn_matrix_all_heads[i], cmap='viridis', aspect='auto')
        ax1.set_title(f"头 no.{i+1}", fontsize=24, fontfamily='serif')
        
        if col == 0: ax1.set_ylabel("序列位置 (Query)", fontsize=18)
        if row == 1: ax1.set_xlabel("序列位置 (Key)", fontsize=18)
            
        # 🌟 使用动态刻度
        ax1.set_xticks(ticks)
        ax1.set_yticks(ticks)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        if col == 1:
            cax = fig1.add_subplot(gs1[row, 2])
            cbar = fig1.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=12)
            
    save_filename_heat = os.path.join(save_dir_heat, f"heatmap_sample_{sample_index}.png")
    fig1.savefig(save_filename_heat, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1) 

    # ==========================================
    # 🌟 图 2：自适应全局重要性柱状图
    # ==========================================
    channel_importance = np.mean(attn_matrix_all_heads, axis=(0, 1))
    top3_indices = np.argsort(channel_importance)[-3:] 

    fig2 = plt.figure(figsize=(12, 7))
    fig2.suptitle(f"{seq_len} 维度全局重要性得分 (测试集样本 {sample_index})\n{info_str}", 
                  fontsize=16, fontweight='bold', y=0.96)
    fig2.subplots_adjust(top=0.85)

    # 🌟 智能前缀：如果是 28 维，大概率是物理通道 (Ch)；否则统称特征 (F)
    prefix = "Ch" if seq_len == 28 else "F"
    channels = [f"{prefix}{i+1}" for i in range(seq_len)]
    x_pos = np.arange(seq_len)

    bars = plt.bar(x_pos, channel_importance, color='steelblue', edgecolor='black', alpha=0.8)

    for idx in top3_indices:
        bars[idx].set_color('crimson')
        bars[idx].set_edgecolor('black')

    # 🌟 动态平均基准线
    baseline = 1.0 / seq_len
    plt.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, label=f"平均基准线 ({baseline:.4f})")

    plt.xlabel("序列输入节点", fontsize=14)
    plt.ylabel("注意力平均贡献得分", fontsize=14)
    
    # 🌟 智能步长：如果特征超过 30 个，每隔几个显示一个标签，防止文字重叠
    step = 1 if seq_len <= 30 else max(1, seq_len // 15)
    plt.xticks(x_pos[::step], channels[::step], rotation=45)
    
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    save_filename_bar = os.path.join(save_dir_bar, f"importance_bar_sample_{sample_index}.png")
    fig2.savefig(save_filename_bar, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig2)