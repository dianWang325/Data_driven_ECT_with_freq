# 硬编码绘图维度，特征维度更新时会图片显示异常，后续可以改为动态适配

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_visualize_physics_groups(model, X_test_t, y_test_t, scaler_X, scaler_y, save_base_dir):
    """
    基于物理特性的多维分组分析：
    探究电导率/频率比值、几何厚度对模型预测误差及注意力分布的影响。
    """
    print("\n" + "="*40)
    print("🔬 开始进行基于物理特性的多维分组分析")
    print("="*40)
    
    # 创建专门存放物理分析报告的子文件夹
    save_dir = os.path.join(save_base_dir, "physics_analysis")
    os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()

    # ==========================================
    # 1. 收集所有测试集样本的预测结果与注意力权重
    # ==========================================
    data_list = []

    with torch.no_grad():
        for i in range(len(X_test_t)):
            # 确保输入数据在正确的设备上
            sample_x = X_test_t[i:i+1].to(device)
            sample_y = y_test_t[i].item()
            
            # 前向传播
            pred_y_scaled = model(sample_x)
            attn_matrix = model.attention_weights[0].cpu().numpy() # (4, 28, 28)
            
            # 计算该样本的 28通道全局重要性 (对头和Query求平均)
            channel_importance = np.mean(attn_matrix, axis=(0, 1))
            
            # 反归一化获取物理真实值
            pred_real = scaler_y.inverse_transform(pred_y_scaled.cpu().numpy())[0][0]
            true_real = scaler_y.inverse_transform([[sample_y]])[0][0]
            
            # 反归一化 X (注意要先切回 CPU)
            sample_x_real = scaler_X.inverse_transform(sample_x.cpu().numpy())[0]
            sigma_real = sample_x_real[0]  # 第0列: 电导率
            freq_real = sample_x_real[1]   # 第1列: 频率
            
            # 记录到字典
            data_list.append({
                'Index': i,
                'Sigma': sigma_real,
                'Freq': freq_real,
                'Ratio_SF': sigma_real / (freq_real + 1e-9), # 防止除零
                'True_Thickness': true_real,
                'Pred_Thickness': pred_real,
                'Error_Abs': abs(true_real - pred_real),
                'Importance_Array': channel_importance
            })

    # 转换为 Pandas DataFrame
    df = pd.DataFrame(data_list)

    # 2. 类别划分 (使用 qcut 按数据分布的 33%, 66% 分位数划分为 3 类)
    # 添加 duplicates='drop' 以防某些分位数边缘存在大量重复值导致报错
    df['Ratio_Category'] = pd.qcut(df['Ratio_SF'], q=3, labels=['低比值 (容性主导)', '中比值 (过渡区)', '高比值 (阻性主导)'], duplicates='drop')
    df['Thick_Category'] = pd.qcut(df['True_Thickness'], q=3, labels=['薄环流', '中等环流', '厚环流'], duplicates='drop')

    print("[*] 数据提取与分类完毕。数据概览：")
    print(df[['Ratio_SF', 'Ratio_Category', 'True_Thickness', 'Thick_Category']].head())

    # ==========================================
    # 🌟 可视化 1：不同类别的预测准确性差异 (误差分析)
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))

    # 图1.1：不同比值下的绝对误差 MAE
    sns.barplot(x='Ratio_Category', y='Error_Abs', data=df, ax=axes1[0], capsize=.1, palette='Blues_d')
    axes1[0].set_title('不同 [电导率/频率] 比值下的预测绝对误差', fontsize=15)
    axes1[0].set_ylabel('平均绝对误差 (MAE)', fontsize=13)
    axes1[0].set_xlabel('介质电磁特性比值', fontsize=13)

    # 图1.2：不同厚度下的绝对误差 MAE
    sns.barplot(x='Thick_Category', y='Error_Abs', data=df, ax=axes1[1], capsize=.1, palette='Greens_d')
    axes1[1].set_title('不同 [环流厚度] 下的预测绝对误差', fontsize=15)
    axes1[1].set_ylabel('平均绝对误差 (MAE)', fontsize=13)
    axes1[1].set_xlabel('几何厚度分类', fontsize=13)

    plt.tight_layout()
    fig1.savefig(os.path.join(save_dir, 'Accuracy_Analysis.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ==========================================
    # 🌟 可视化 2：特征重要性分布差异 (物理验证核心)
    # ==========================================
    fig2, axes2 = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    channels = [f"Ch{i+1}" for i in range(28)]
    x_pos = np.arange(28)

    # --- 图 2.1: 验证比值对“方差（聚焦度）”的影响 ---
    ratio_groups = df.groupby('Ratio_Category', observed=False)['Importance_Array'].apply(lambda x: np.mean(np.vstack(x), axis=0))

    colors_ratio = ['#1f77b4', '#ff7f0e', '#d62728']
    for i, (cat_name, imp_array) in enumerate(ratio_groups.items()):
        variance = np.var(imp_array)
        axes2[0].plot(x_pos, imp_array, marker='o', linewidth=2, color=colors_ratio[i],
                      label=f"{cat_name} (注意力方差: {variance:.6f})")

    axes2[0].set_title('猜想验证 1: [电导率/频率] 比值对特征重要性分布方差的影响', fontsize=16, fontweight='bold')
    axes2[0].set_ylabel('平均注意力得分', fontsize=14)
    axes2[0].legend(fontsize=12)
    axes2[0].grid(True, linestyle='--', alpha=0.6)

    # --- 图 2.2: 验证厚度对“高亮通道位置”的影响 ---
    thick_groups = df.groupby('Thick_Category', observed=False)['Importance_Array'].apply(lambda x: np.mean(np.vstack(x), axis=0))

    colors_thick = ['#2ca02c', '#9467bd', '#8c564b']
    for i, (cat_name, imp_array) in enumerate(thick_groups.items()):
        peak_channel = np.argmax(imp_array) + 1
        axes2[1].plot(x_pos, imp_array, marker='s', linewidth=2, color=colors_thick[i],
                      label=f"{cat_name} (最高关注点: Ch{peak_channel})")

    axes2[1].set_title('猜想验证 2: [环流厚度] 对核心传感器通道(高亮位置)的偏移影响', fontsize=16, fontweight='bold')
    axes2[1].set_ylabel('平均注意力得分', fontsize=14)
    axes2[1].set_xlabel('物理传感器通道', fontsize=14)
    axes2[1].set_xticks(x_pos)
    axes2[1].set_xticklabels(channels, rotation=45)
    axes2[1].legend(fontsize=12)
    axes2[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, 'Feature_Importance_Shift.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 额外：把分组统计的 DataFrame 保存为 CSV，方便后续写论文时做表格
    csv_path = os.path.join(save_dir, 'physics_analysis_data.csv')
    df.drop(columns=['Importance_Array']).to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"[*] 多维分组分析完成！报告及数据已保存至: {save_dir}")