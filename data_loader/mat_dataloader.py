import os
import joblib
from datetime import datetime
import numpy as np
import scipy.io as sio
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def get_dataloaders(file_path, 
                    mat_key, 
                    target_col=1, 
                    feature_cols=None, 
                    cap_cols=None, 
                    test_size=0.2, 
                    batch_size=128, 
                    random_state=42, 
                    save_scaler_dir='./scaler_tmp'):
    """
    统一的数据加载、特征组合与预处理工厂函数
    
    参数:
        file_path: mat 数据文件路径
        mat_key: mat 字典中的主键名 (如 'C_data', 'Z')
        target_col: 目标值(厚度 Y) 所在的列索引 (默认 1)
        feature_cols: 需要作为特征(X)提取的列索引列表。若为 None，则提取除 target_col 外的所有列。
        cap_cols: 极其微小的电容列索引列表，需要预先乘以 1e12 保护精度。
        batch_size: DataLoader 的批大小
        save_scaler_dir: Scaler 预处理器的保存路径
        
    返回:
        一个包含所有需要的组件的字典 (train_loader, test_loader, scalers, tensors...)
    """
    
    # 1. 加载 MATLAB 数据
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到数据文件: {file_path}")
        
    mat_data = sio.loadmat(file_path)
    raw_data = mat_data[mat_key]
    
    # 2. 自动或手动确认特征列
    if feature_cols is None:
        feature_cols = list(range(raw_data.shape[1]))
        feature_cols.remove(target_col)
        
    # 3. 提取特征矩阵 (X) 和目标向量 (y)
    X = raw_data[:, feature_cols].copy() # 拷贝防止修改原矩阵
    y = raw_data[:, target_col]
    
    # 4. 处理电容的极小值精度保护 (乘以 1e12)
    if cap_cols is not None:
        for col_idx in cap_cols:
            if col_idx in feature_cols:
                # 找到该列在切片后的 X 矩阵中的相对索引位置
                rel_idx = feature_cols.index(col_idx)
                X[:, rel_idx] *= 1e12
                
    feature_dim = X.shape[1]
    print(f"成功加载 {file_path} | 特征维度: {feature_dim} | 样本数: {X.shape[0]}")
    
    # 5. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # 6. 数据标准化 (StandardScaler)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    # 7. 自动保存预处理器 (为后续测试与部署做准备)
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(save_scaler_dir, exist_ok=True)
    joblib.dump(scaler_X, os.path.join(save_scaler_dir, f'scaler_X_{current_time}.pkl'))
    joblib.dump(scaler_y, os.path.join(save_scaler_dir, f'scaler_y_{current_time}.pkl'))
    print(f"Scaler 已保存至 {save_scaler_dir} 目录")
    
    # 8. 转换为 PyTorch Tensor
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    # 9. 构建 DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 10. 将所有有用的对象打包成字典返回
    return {
        # === 深度学习 (PyTorch) 专用组件 ===
        'train_loader': train_loader,
        'test_loader': test_loader,
        'X_test_t': X_test_t,
        'y_test_t': y_test_t,

        # === 传统机器学习 (如随机森林) 专用组件 ===
        'X_train_np': X_train_scaled,   # ⭐ RF 训练必须用到
        'y_train_np': y_train_scaled,   # ⭐ RF 训练必须用到
        'X_test_np': X_test_scaled,     # ⭐ RF 测试必须用到
        'y_test_np': y_test_scaled,     # ⭐ RF 测试必须用到

        # === 通用组件 ===
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_dim': feature_dim  # 返回计算出的维度，方便初始化模型
    }