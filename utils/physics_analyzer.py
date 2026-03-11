# 文件路径: utils/physics_analyzer.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_visualize_physics_groups(model, X_test_t, y_test_t, scaler_X, scaler_y, save_base_dir):
    print("\n" + "="*40)
    print("🔬 开始进行基于物理特性的多维分组分析")
    print("="*40)
    
    save_dir = os.path.join(save_base_dir, "physics_analysis")
    os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()

    data_list = []
    seq_len = None  # 用于记录动态序列长度

    with torch.no_grad():
        for i in range(len(X_test_t)):
            sample_x = X_test_t[i:i+1].to(device)
            sample_y = y_test_t[i].item()
            
            pred_y_scaled = model(sample_x)
            attn_matrix = model.attention_weights[0].cpu().numpy() 
            
            # 🌟 动态获取序列长度
            if seq_len is None:
                seq_len = attn_matrix.shape[-1]
                
            channel_importance = np.mean(attn_matrix, axis=(0, 1))
            
            pred_real = scaler_y.inverse_transform(pred_y_scaled.cpu().numpy())[0][0]
            true_real = scaler_y.inverse_transform([[sample_y]])[0][0]
            
            sample_x_real = scaler_X.inverse_transform(sample_x.cpu().numpy())[0]
            sigma_real = sample_x_real[0]
            freq_real = sample_x_real[1]
            
            data_list.append({
                'Index': i,
                'Sigma': sigma_real,
                'Freq': freq_real,
                'Ratio_SF': sigma_real / (freq_real + 1e-9), 
                'True_Thickness': true_real,
                'Pred_Thickness': pred_real,
                'Error_Abs': abs(true_real - pred_real),
                'Importance_Array': channel_importance
            })

    df = pd.DataFrame(data_list)
    df['Ratio_Category'] = pd.qcut(df['Ratio_SF'], q=3, labels=['低比值 (容性主导)', '中比值 (过渡区)', '高比值 (阻性主导)'], duplicates='drop')
    df['Thick_Category'] = pd.qcut(df['True_Thickness'], q=3, labels=['薄环流', '中等环流', '厚环流'], duplicates='drop')

    # ==========================================
    # 可视化 1：误差分析 (保持不变)
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(x='Ratio_Category', y='Error_Abs', data=df, ax=axes1[0], capsize=.1, palette='Blues_d')
    axes1[0].set_title('不同 [电导率/频率] 比值下的预测绝对误差', fontsize=15)
    axes1[0].set_ylabel('平均绝对误差 (MAE)', fontsize=13)
    
    sns.barplot(x='Thick_Category', y='Error_Abs', data=df, ax=axes1[1], capsize=.1, palette='Greens_d')
    axes1[1].set_title('不同 [环流厚度] 下的预测绝对误差', fontsize=15)
    axes1[1].set_ylabel('平均绝对误差 (MAE)', fontsize=13)

    plt.tight_layout()
    fig1.savefig(os.path.join(save_dir, 'Accuracy_Analysis.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ==========================================
    # 🌟 可视化 2：自适应特征重要性分布差异
    # ==========================================
    fig2, axes2 = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # 动态前缀与 X 轴
    prefix = "Ch" if seq_len == 28 else "F"
    channels = [f"{prefix}{i+1}" for i in range(seq_len)]
    x_pos = np.arange(seq_len)

    # --- 2.1: 方差影响 ---
    ratio_groups = df.groupby('Ratio_Category', observed=False)['Importance_Array'].apply(lambda x: np.mean(np.vstack(x), axis=0))
    colors_ratio = ['#1f77b4', '#ff7f0e', '#d62728']
    
    for i, (cat_name, imp_array) in enumerate(ratio_groups.items()):
        variance = np.var(imp_array)
        axes2[0].plot(x_pos, imp_array, marker='o', markersize=4, linewidth=2, color=colors_ratio[i],
                      label=f"{cat_name} (注意力方差: {variance:.6f})")

    axes2[0].set_title('猜想验证 1: [电导率/频率] 比值对特征重要性分布方差的影响', fontsize=16, fontweight='bold')
    axes2[0].set_ylabel('平均注意力得分', fontsize=14)
    axes2[0].legend(fontsize=12)
    axes2[0].grid(True, linestyle='--', alpha=0.6)

    # --- 2.2: 高亮偏移影响 ---
    thick_groups = df.groupby('Thick_Category', observed=False)['Importance_Array'].apply(lambda x: np.mean(np.vstack(x), axis=0))
    colors_thick = ['#2ca02c', '#9467bd', '#8c564b']
    
    for i, (cat_name, imp_array) in enumerate(thick_groups.items()):
        peak_channel = np.argmax(imp_array) + 1
        axes2[1].plot(x_pos, imp_array, marker='s', markersize=4, linewidth=2, color=colors_thick[i],
                      label=f"{cat_name} (最高关注点: {prefix}{peak_channel})")

    axes2[1].set_title('猜想验证 2: [环流厚度] 对核心特征节点(高亮位置)的偏移影响', fontsize=16, fontweight='bold')
    axes2[1].set_ylabel('平均注意力得分', fontsize=14)
    axes2[1].set_xlabel('输入特征序列', fontsize=14)
    
    # 智能步长控制 X 轴显示
    step = 1 if seq_len <= 30 else max(1, seq_len // 15)
    axes2[1].set_xticks(x_pos[::step])
    axes2[1].set_xticklabels(channels[::step], rotation=45)
    
    axes2[1].legend(fontsize=12)
    axes2[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, 'Feature_Importance_Shift.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    csv_path = os.path.join(save_dir, 'physics_analysis_data.csv')
    df.drop(columns=['Importance_Array']).to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"[*] 多维分组分析完成！报告及数据已保存至: {save_dir}")