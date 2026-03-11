import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# model直接来自sklearn，无需导入自定义模型


def train_random_forest(data_dict, save_dir="./experiments/rf_model", n_estimators=100, max_depth=15, n_jobs=-1, random_state=42, **kwargs):
    """
    接收 get_dataloaders 返回的字典，训练并评估随机森林模型。
    参数由外部的 YAML 配置文件动态传入。
    """
    print("\n" + "="*40)
    print("🚀 启动传统机器学习引擎：随机森林")
    print("="*40)

    # ---------------------------------------------------------
    # 1. 从字典中解包数据 (获取 NumPy 格式的数据)
    # ---------------------------------------------------------
    X_train = data_dict['X_train_np']
    X_test = data_dict['X_test_np']
    y_train_2d = data_dict['y_train_np']
    y_test_2d = data_dict['y_test_np']
    scaler_y = data_dict['scaler_y']  # 提取 y 的缩放器，用于后续还原真实物理量
    
    # 🌟 核心技巧：将 (N, 1) 的二维列向量展平为 (N,) 的一维数组
    y_train = y_train_2d.ravel()
    y_test = y_test_2d.ravel()
    
    print(f"[*] 数据准备完毕 | 训练集形状: X={X_train.shape}, y={y_train.shape}")

    # ---------------------------------------------------------
    # 2. 构建并训练模型 (使用动态传入的参数)
    # ---------------------------------------------------------
    # 打印时也使用动态变量，确保输出与实际训练一致
    print(f"[*] 正在训练随机森林模型 (n_estimators={n_estimators}, max_depth={max_depth})...")
    
    # 将传入的参数喂给模型。**kwargs 允许接收配置文件中可能新增的其他任何 sklearn 支持的参数
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        n_jobs=n_jobs,
        random_state=random_state,
        **kwargs 
    )
    
    rf_model.fit(X_train, y_train)
    print("[*] 模型训练完成！")

    # ---------------------------------------------------------
    # 3. 预测与双重评估 (归一化空间 vs 真实物理空间)
    # ---------------------------------------------------------
    print("[*] 正在测试集上进行预测与评估...")
    y_pred_scaled = rf_model.predict(X_test)
    
    # 指标 1：归一化空间下的评估 (直接用模型输出计算)
    mse_scaled = mean_squared_error(y_test, y_pred_scaled)
    r2 = r2_score(y_test, y_pred_scaled)
    print(f"    -> [归一化空间] MSE: {mse_scaled:.6f} | R2 Score: {r2:.4f}")
    
    # 🌟 指标 2：真实物理空间下的评估 (反归一化)
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_real = scaler_y.inverse_transform(y_test_2d).ravel()
    
    mse_real = mean_squared_error(y_test_real, y_pred_real)
    print(f"    -> [真实物理空间] MSE: {mse_real:.6f}")

    metrics = {
        'MSE_Scaled': float(mse_scaled),
        'R2_Score': float(r2),
        'MSE_Real': float(mse_real)
    }

    # ---------------------------------------------------------
    # 4. 单独保存模型本身
    # ---------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'rf_model.joblib')
    joblib.dump(rf_model, model_path)
    print(f"[*] 模型文件已保存至: {model_path}")
    
    return rf_model, metrics