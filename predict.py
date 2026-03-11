## 兼容RF模型的预测脚本，在原有深度学习版本的基础上新增了对随机森林模型的支持，同时保留了针对深度学习模型的物理分析功能。
import os
import glob
import torch
import joblib
import argparse
import yaml
import numpy as np
import scipy.io as sio

# 🌟 新增：导入 sklearn 的误差评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 导入深度学习模型工厂与物理分析组件
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
    读取全新的 .mat 数据，提取特征 X 与真实标签 y。
    """
    print(f"[*] 正在加载盲测数据: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到盲测数据文件: {file_path}")
        
    mat_data = sio.loadmat(file_path)
    raw_data = mat_data[mat_key]
    
    # 提取真实标签 y (这本身就是物理空间下的原值)
    y_numpy = raw_data[:, target_col]
    
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
    
    print(f"    -> 成功提取特征与标签，样本数: {X.shape[0]}")
    return X_scaled, y_numpy, raw_data

def main():
    parser = argparse.ArgumentParser(description="ECT 模型盲测预测与分析脚本")
    parser.add_argument('--config', type=str, default='configs/predict_config.yaml', 
                        help="预测配置文件的路径")
    args = parser.parse_args()

    # 1. 读取预测 yaml 配置
    with open(args.config, 'r', encoding='utf-8') as f:
        predict_cfg = yaml.safe_load(f)['prediction']

    exp_dir = predict_cfg['exp_dir']
    new_data = predict_cfg['new_data']
    mat_key = predict_cfg.get('mat_key', 'C')
    output_prefix = predict_cfg.get('output_prefix', 'blind_test')

    out_csv = os.path.join(exp_dir, f"{output_prefix}_predictions.csv")
    analysis_save_dir = os.path.join(exp_dir, f"{output_prefix}_physics_report")

    # 2. 加载实验资产与配置
    config, scaler_X, scaler_y = load_experiment_assets(exp_dir)
    model_type = config.get('model_type', 'dl').lower()

    # 3. 预处理盲测数据
    X_scaled, y_numpy, raw_data = process_blind_data(new_data, mat_key, scaler_X)

    # ==========================================
    # 4. 双分支路由：模型加载与推理
    # ==========================================
    model = None
    
    if model_type == 'rf':
        print("\n>>> 检测到传统机器学习 (RF) 档案 <<<")
        model_path = os.path.join(exp_dir, 'rf_model.joblib')
        model = joblib.load(model_path)
        print("    -> 成功加载随机森林模型权重！")
        
        print("[*] 正在执行 CPU 高速推理...")
        preds_scaled = model.predict(X_scaled)
        
    elif model_type == 'dl':
        print("\n>>> 检测到深度学习 (DL) 档案 <<<")
        dl_config = config['dl_params']
        feature_dim = scaler_X.n_features_in_
        print(f"[*] 正在重建网络架构: {dl_config['architecture'].upper()} (输入维度: {feature_dim})")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(dl_config, feature_dim).to(device)
        
        weights_path = os.path.join(exp_dir, 'dl_model_best_weights.pth')
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("    -> 成功注入最优权重！")
        
        print("[*] 正在执行 GPU/CPU 高速推理...")
        model.eval()
        with torch.no_grad():
            X_tensor_dev = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            preds_scaled_tensor = model(X_tensor_dev)
            preds_scaled = preds_scaled_tensor.cpu().numpy()
            
    else:
        raise ValueError(f"❌ 不支持的模型类型: {model_type}")

    # ==========================================
    # 🌟 5. 反归一化并计算真实物理世界误差
    # ==========================================
    # 获取模型预测的真实物理厚度
    preds_real = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

    # 计算盲测指标 (y_numpy 是未经过归一化的 .mat 真实原始数据)
    mse_real = mean_squared_error(y_numpy, preds_real)
    mae_real = mean_absolute_error(y_numpy, preds_real)
    r2_real = r2_score(y_numpy, preds_real)

    # 打印酷炫的盲测评估面板
    print("\n" + "="*50)
    print("📊 盲测数据集全局评估指标 (真实物理空间)")
    print("="*50)
    print(f"-> R2 Score (决定系数) : {r2_real:.4f}")
    print(f"-> MSE (均方误差)      : {mse_real:.6f}")
    print(f"-> MAE (平均绝对误差)  : {mae_real:.6f}")
    print("="*50)

    # 保存预测结果为 CSV
    np.savetxt(out_csv, preds_real, delimiter=",", header="Predicted_Thickness", comments="")
    print(f"\n✅ 预测结果与标签对比已保存至: {os.path.abspath(out_csv)}")

    # ==========================================
    # 6. 执行盲测数据的物理特性分组分析 (仅限带注意力的 DL 模型)
    # ==========================================
    if model_type == 'dl' and hasattr(model, 'attention_weights'):
        X_tensor_cpu = torch.tensor(X_scaled, dtype=torch.float32)
        y_scaled = scaler_y.transform(y_numpy.reshape(-1, 1)).ravel()
        y_tensor_cpu = torch.tensor(y_scaled, dtype=torch.float32)
        
        analyze_and_visualize_physics_groups(
            model=model,
            X_test_t=X_tensor_cpu,
            y_test_t=y_tensor_cpu,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            save_base_dir=analysis_save_dir
        )
        print(f"✅ 盲测深度物理分析报告已生成: {os.path.abspath(analysis_save_dir)}")
    elif model_type == 'rf':
        print("💡 注: 当前模型为随机森林，无注意力机制，跳过特征热力图生成。")

if __name__ == '__main__':
    main()