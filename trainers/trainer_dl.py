# 文件路径: trainers/trainer_dl.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib 
import json
from sklearn.metrics import r2_score, mean_squared_error
from utils.attention_weight_visualizer import generate_and_save_visualizations
from utils.physics_analyzer import analyze_and_visualize_physics_groups

# 引入你在 model/ 目录下写的模型图纸
from models.networks import MLP_v1, MLP_v2, Attention_Net_v1, Attention_Net_v2

def build_model(dl_config, feature_dim):
    """
    模型工厂函数：根据配置和特征维度，动态组装并返回对应的模型实例。
    """

    arch_type = dl_config.get('architecture', 'mlp_v1').lower()
    
    # 注册 MLP_v1
    if arch_type == 'mlp_v1':
        model = MLP_v1(
            input_dim=feature_dim,
            dropout_rate=dl_config.get('dropout_rate', 0.1)
        )
    elif arch_type == 'mlp_v2':
        model = MLP_v2(
            input_dim=feature_dim,
            dropout_rate=dl_config.get('dropout_rate', 0.1)
        )
    elif arch_type == 'attention_v1': 
        # 🌟 注册纯数据驱动版本
        model = Attention_Net_v1(
            input_dim=feature_dim, # 这里的 feature_dim 极其关键，将决定网络大小
            hidden_dim=dl_config.get('hidden_dim', 64),
            num_heads=dl_config.get('num_heads', 4),
            dropout_rate=dl_config.get('dropout_rate', 0.2)
        )
    elif arch_type == 'attention_v2':
        model = Attention_Net_v2(
            input_dim=feature_dim,
            hidden_dim=dl_config.get('hidden_dim', 64),
            num_heads=dl_config.get('num_heads', 4),
            dropout_rate=dl_config.get('dropout_rate', 0.2)
        )
    else:
        raise ValueError(f"❌ 不支持的深度学习架构类型: {arch_type}")
        
    return model

def train_deep_learning(data_dict, dl_config, save_dir="./experiments/dl_model"):
    print("\n" + "="*40)
    print("🚀 启动深度学习引擎")
    print("="*40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = data_dict['train_loader']
    test_loader = data_dict['test_loader']
    scaler_y = data_dict['scaler_y']
    feature_dim = data_dict['feature_dim']  

    # 提取 YAML 配置中的高级训练参数
    epochs = dl_config.get('epochs', 200)
    lr = dl_config.get('learning_rate', 0.005)
    weight_decay = dl_config.get('weight_decay', 1e-5)
    scheduler_patience = dl_config.get('scheduler_patience', 10)
    scheduler_factor = dl_config.get('scheduler_factor', 0.5)

    print(f"[*] 动态构建模型: {dl_config.get('architecture', 'mlp_v1').upper()}")
    model = build_model(dl_config, feature_dim).to(device)
    
    criterion = nn.MSELoss()
    
    # 🌟 接入 L2 正则化 (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 🌟 接入学习率调度器 (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=scheduler_factor, 
        patience=scheduler_patience, 
        verbose=True  # 触发衰减时自动在终端打印信息
    )
    # 创建保存目录和最优权重路径
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'dl_model_best_weights.pth')
    
    # 初始化一个无穷大的 best_val_loss 用于记录历史最优
    best_val_loss = float('inf')

    print(f"[*] 开始训练，总 Epoch 数: {epochs} ...")
    
    for epoch in range(epochs):
# ---------------- 阶段 1: 训练 ----------------
        model.train()  
        running_train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad() 
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward() 
            optimizer.step() 
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        
        # ---------------- 阶段 2: 验证 ----------------
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(test_loader)
        
        # 🌟 核心修改 1: 用验证集的 Loss 更新学习率调度器
        scheduler.step(avg_val_loss)
        
        # 🌟 核心修改 2: 保存验证集上表现最好的模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            is_best = "⭐(Best)"
        else:
            is_best = ""
        
        # 打印日志 (可根据需要调整打印频率)
        if (epoch + 1) % 10 == 0 or epoch == 0 or is_best:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} {is_best} | LR: {current_lr:.6f}")

    print("[*] 训练大循环结束！")

    # ==========================================
    # 最终评估 (加载之前保存的最优权重)
    # ==========================================
    print(f"[*] 正在加载最优权重 ({best_model_path}) 进行最终物理指标评估...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())
            
    y_pred_scaled = np.vstack(all_preds)
    y_test_scaled = np.vstack(all_targets)

    # 归一化空间指标
    mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
    r2 = r2_score(y_test_scaled, y_pred_scaled)
    print(f"    -> [最优模型-归一化空间] MSE: {mse_scaled:.6f} | R2 Score: {r2:.4f}")

    # 真实物理空间指标 (反归一化)
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled).ravel()
    y_test_real = scaler_y.inverse_transform(y_test_scaled).ravel()
    mse_real = mean_squared_error(y_test_real, y_pred_real)
    print(f"    -> [最优模型-真实物理空间] MSE: {mse_real:.6f}")
    metrics = {
        'MSE_Scaled': float(mse_scaled),
        'R2_Score': float(r2),
        'MSE_Real': float(mse_real)
    }
    # ==========================================
    # 🌟 新增：自动挖掘典型样本并生成可视化报告
    # ==========================================
    # 动态防御：只有当模型包含 attention_weights 属性时，才执行可视化
    if hasattr(model, 'attention_weights'):
        print("\n[*] 检测到注意力机制模型，正在自动生成特征重要性图表...")
        
        # 1. 计算所有测试集样本的绝对误差
        errors = np.abs(y_pred_real - y_test_real)
        
        # 2. 对误差进行从小到大排序，获取对应的样本索引
        sorted_indices = np.argsort(errors)
        
        # 提取表现最好（误差最小）的 2 个样本，和表现最差（误差最大）的 2 个样本
        best_2_idx = sorted_indices[:2]
        worst_2_idx = sorted_indices[-2:]
        target_indices = np.concatenate([best_2_idx, worst_2_idx])
        
        # 3. 从字典中提取完整的测试集 Tensor (以便传入画图函数)
        X_test_t = data_dict['X_test_t']
        y_test_t = data_dict['y_test_t']
        scaler_X = data_dict['scaler_X']
        
        # 4. 循环生成并保存图表
        for idx in target_indices:
            # 给最好和最差的样本加上不同的前缀，方便在文件夹里一眼区分
            prefix = "BEST_PREDICT" if idx in best_2_idx else "WORST_PREDICT"
            print(f"    -> 正在绘制 {prefix} 样本 (索引: {idx}, 绝对误差: {errors[idx]:.4f})")
            
            # 为了防止混淆，我们将图表分别存放在实验目录下的 BEST 和 WORST 子文件夹中
            target_save_dir = os.path.join(save_dir, "visualizations", prefix)
            
            generate_and_save_visualizations(
                model=model,
                X_test_t=X_test_t,
                y_test_t=y_test_t,
                scaler_X=scaler_X,
                scaler_y=scaler_y,
                sample_index=idx,
                save_base_dir=target_save_dir 
            )
        print(f"[*] 可视化图表已全部生成，保存在: {os.path.join(save_dir, 'visualizations')} 目录下！")
# 🌟 新增：执行全局物理特性分组分析
        if hasattr(model, 'attention_weights'):
            analyze_and_visualize_physics_groups(
                model=model,
                X_test_t=X_test_t,
                y_test_t=y_test_t,
                scaler_X=scaler_X,
                scaler_y=scaler_y,
                save_base_dir=save_dir
            )
    return model, metrics