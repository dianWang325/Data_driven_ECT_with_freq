## 未兼容RF版本的预测脚本，专门针对深度学习模型的盲测预测与物理分析设计。
import os
import glob
import torch
import joblib
import argparse
import yaml
import numpy as np
import scipy.io as sio

# 导入模型工厂与物理分析组件
from trainers.trainer_dl import build_model
from utils.physics_analyzer import analyze_and_visualize_physics_groups

def load_experiment_assets(exp_dir):
    """
    加载配置文件和 Scaler。
    """
    print(f"[*] 正在读取实验档案: {exp_dir}")
    config_path = os.path.join(exp_dir, 'run_config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    scaler_X_files = glob.glob(os.path.join(exp_dir, 'scaler_X_*.pkl'))
    scaler_y_files = glob.glob(os.path.join(exp_dir, 'scaler_y_*.pkl'))
    
    if not scaler_X_files or not scaler_y_files:
        raise FileNotFoundError("在实验目录中找不到对应的 Scaler 文件！")
        
    scaler_X = joblib.load(scaler_X_files[0])
    scaler_y = joblib.load(scaler_y_files[0])
    
    print("    -> 成功加载 run_config.yaml, scaler_X, scaler_y")
    return config, scaler_X, scaler_y

def process_blind_data(file_path, mat_key, scaler_X, target_col=1, cap_cols=None):
    """
    读取全新的 .mat 数据，提取特征 X 与真实标签 y（如果存在）。
    """
    print(f"[*] 正在加载盲测数据: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到盲测数据文件: {file_path}")
        
    mat_data = sio.loadmat(file_path)
    raw_data = mat_data[mat_key]
    
    # 提取真实标签 y (用于误差分析)
    y = raw_data[:, target_col]
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # 提取特征矩阵 X
    feature_cols = list(range(raw_data.shape[1]))
    feature_cols.remove(target_col)
    X = raw_data[:, feature_cols].copy()
    
    if cap_cols is not None:
        for col_idx in cap_cols:
            if col_idx in feature_cols:
                rel_idx = feature_cols.index(col_idx)
                X[:, rel_idx] *= 1e12
                
    # 使用训练时的 scaler_X 进行标准化
    X_scaled = scaler_X.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    print(f"    -> 成功提取特征与标签，样本数: {X.shape[0]}")
    return X_tensor, y_tensor, raw_data

def main():
    # 🌟 1. 改为只接收 yaml 配置文件路径
    parser = argparse.ArgumentParser(description="ECT 模型盲测预测与分析脚本")
    parser.add_argument('--config', type=str, default='configs/predict_config.yaml', 
                        help="预测配置文件的路径")
    args = parser.parse_args()

    # 读取 yaml 配置
    with open(args.config, 'r', encoding='utf-8') as f:
        predict_cfg = yaml.safe_load(f)['prediction']

    exp_dir = predict_cfg['exp_dir']
    new_data = predict_cfg['new_data']
    mat_key = predict_cfg.get('mat_key', 'C')
    output_prefix = predict_cfg.get('output_prefix', 'blind_test')

    # 🌟 2. 自动构建输出路径 (强制保存在实验权重目录下)
    out_csv = os.path.join(exp_dir, f"{output_prefix}_predictions.csv")
    analysis_save_dir = os.path.join(exp_dir, f"{output_prefix}_physics_report")

    # 3. 加载实验资产
    config, scaler_X, scaler_y = load_experiment_assets(exp_dir)
    dl_config = config['dl_params']

    # 4. 预处理盲测数据
    X_tensor, y_tensor, raw_data = process_blind_data(new_data, mat_key, scaler_X)

    # 5. 动态重建模型并加载最优权重
    feature_dim = scaler_X.n_features_in_
    print(f"[*] 正在重建网络架构: {dl_config['architecture'].upper()} (输入维度: {feature_dim})")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(dl_config, feature_dim).to(device)
    
    weights_path = os.path.join(exp_dir, 'dl_model_best_weights.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # 6. 执行预测
    print("[*] 正在执行高速推理...")
    model.eval()
    with torch.no_grad():
        X_tensor_dev = X_tensor.to(device)
        preds_scaled = model(X_tensor_dev)
        
    # 7. 反归一化并保存基础 CSV 结果
    preds_real = scaler_y.inverse_transform(preds_scaled.cpu().numpy()).ravel()
    np.savetxt(out_csv, preds_real, delimiter=",", header="Predicted_Thickness", comments="")
    print(f"✅ 预测结果已保存至: {os.path.abspath(out_csv)}")

    # ==========================================
    # 🌟 8. 执行盲测数据的物理特性分组分析
    # ==========================================
    if hasattr(model, 'attention_weights'):
        # 直接使用我们在上面算好的 analysis_save_dir
        analyze_and_visualize_physics_groups(
            model=model,
            X_test_t=X_tensor,
            y_test_t=y_tensor,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            save_base_dir=analysis_save_dir
        )
        print(f"✅ 盲测深度物理分析报告已生成: {os.path.abspath(analysis_save_dir)}")

if __name__ == '__main__':
    main()