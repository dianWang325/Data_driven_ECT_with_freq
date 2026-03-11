# 硬编码绘图维度，特征维度更新时会图片显示异常，后续可以改为动态适配

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def generate_and_save_visualizations(model, X_test_t, y_test_t, scaler_X, scaler_y, sample_index=0, save_base_dir="./experiments/visualizations"):
    """
    提取多头注意力权重，并生成热力图与通道重要性柱状图。
    
    参数:
        model: 训练好的深度学习模型 (需要有 attention_weights 属性)
        X_test_t: 测试集特征 Tensor
        y_test_t: 测试集标签 Tensor
        scaler_X: 特征的标准化器 (用于反归一化获取真实电导率/频率)
        scaler_y: 标签的标准化器 (用于反归一化获取真实厚度)
        sample_index: 要可视化的测试集样本索引
        save_base_dir: 图片保存的基础目录
    """
    model.eval()
    
    # ==========================================
    # 0. 准备保存目录
    # ==========================================
    save_dir_heat = os.path.join(save_base_dir, "heatmaps")
    save_dir_bar = os.path.join(save_base_dir, "importance_bars")
    os.makedirs(save_dir_heat, exist_ok=True)
    os.makedirs(save_dir_bar, exist_ok=True)

    # ==========================================
    # 1. 提取当前样本的张量数据与注意力权重
    # ==========================================
    sample_x = X_test_t[sample_index:sample_index+1] 
    sample_y = y_test_t[sample_index].item()

    with torch.no_grad():
        # 前向传播与注意力提取
        pred_y_scaled = model(sample_x)
        # 提取第0个样本的注意力矩阵，转为 numpy
        attn_matrix_all_heads = model.attention_weights[0].cpu().numpy()
        num_heads = attn_matrix_all_heads.shape[0]

    # ==========================================
    # 2. 反归一化：获取真实的物理量
    # ==========================================
    # 预测厚度与真实厚度
    pred_real = scaler_y.inverse_transform(pred_y_scaled.cpu().numpy())[0][0]
    true_real = scaler_y.inverse_transform([[sample_y]])[0][0]
    
    # 提取并反归一化 X，以获取电导率 (第0列) 和 频率 (第1列)
    sample_x_real = scaler_X.inverse_transform(sample_x.cpu().numpy())[0]
    sigma_real = sample_x_real[0]  # 电导率
    freq_real = sample_x_real[1]   # 激励频率
    
    # 构造要在图片上显示的样本基本情况文本
    info_str = (f"基本工况: 电导率 = {sigma_real:.4g} | 激励频率 = {freq_real:.1f} Hz\n"
                f"验证结果: 实际厚度 = {true_real:.2f} | 预测厚度 = {pred_real:.2f}")
    
    print(f"[*] 正在生成可视化图表 | 样本 {sample_index} -> 实际厚度: {true_real:.2f}, 预测厚度: {pred_real:.2f}")

    # ==========================================
    # 🌟 图 1：绘制多头注意力热力图
    # ==========================================
    # 设置中文字体，防止乱码 (若在 Linux/Mac 运行，可能需要更改为对应的中文字体名，如 'Arial Unicode MS')
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 14

    fig1 = plt.figure(figsize=(14, 11)) 
    fig1.suptitle(f"多头自注意力权重矩阵 (测试集样本 {sample_index})\n{info_str}", 
                  fontsize=18, fontweight='bold', y=0.96)

    gs1 = GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.15, hspace=0.3)
    fig1.subplots_adjust(top=0.85) 

    # 绘制最多 4 个头的热力图
    for i in range(min(4, num_heads)):
        row = i // 2
        col = i % 2
        
        ax1 = fig1.add_subplot(gs1[row, col])
        im = ax1.imshow(attn_matrix_all_heads[i], cmap='viridis', aspect='auto')
        ax1.set_title(f"头 no.{i+1}", fontsize=24, fontfamily='serif')
        
        if col == 0: ax1.set_ylabel("序列位置 (Query)", fontsize=18)
        if row == 1: ax1.set_xlabel("序列位置 (Key)", fontsize=18)
            
        ax1.set_xticks([0, 10, 20, 27])
        ax1.set_yticks([0, 10, 20, 27])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # 添加 Colorbar (只在第二列添加以保持画面整洁)
        if col == 1:
            cax = fig1.add_subplot(gs1[row, 2])
            cbar = fig1.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=12)
            
    # 保存热力图
    save_filename_heat = os.path.join(save_dir_heat, f"heatmap_sample_{sample_index}.png")
    fig1.savefig(save_filename_heat, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1) 

    # ==========================================
    # 🌟 图 2：绘制全局通道重要性柱状图
    # ==========================================
    # 对所有头、所有 Query 行求平均，得到每个源通道 (Key) 的综合得分
    channel_importance = np.mean(attn_matrix_all_heads, axis=(0, 1))
    top3_indices = np.argsort(channel_importance)[-3:] # 取前三名

    fig2 = plt.figure(figsize=(12, 7))
    fig2.suptitle(f"28 通道物理节点全局重要性得分 (测试集样本 {sample_index})\n{info_str}", 
                  fontsize=16, fontweight='bold', y=0.96)
    fig2.subplots_adjust(top=0.85)

    channels =[f"Ch{i+1}" for i in range(28)]
    x_pos = np.arange(len(channels))

    bars = plt.bar(x_pos, channel_importance, color='steelblue', edgecolor='black', alpha=0.8)

    # 高亮前三名核心通道
    for idx in top3_indices:
        bars[idx].set_color('crimson')
        bars[idx].set_edgecolor('black')

    # 添加平均基准线
    baseline = 1.0 / 28.0
    plt.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, label=f"平均基准线 ({baseline:.4f})")

    plt.xlabel("物理传感器通道 (基于复阻抗信息融合)", fontsize=14)
    plt.ylabel("注意力平均贡献得分", fontsize=14)
    plt.xticks(x_pos, channels, rotation=45)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 保存柱状图
    save_filename_bar = os.path.join(save_dir_bar, f"importance_bar_sample_{sample_index}.png")
    fig2.savefig(save_filename_bar, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig2)