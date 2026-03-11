import os
import argparse
import yaml
import shutil
from datetime import datetime

from data_loader.mat_dataloader import get_dataloaders
from trainers.trainer_rf import train_random_forest
from trainers.trainer_dl import train_deep_learning
from utils.logger import append_to_global_log

def load_yaml_config(config_path):
    """安全地读取 YAML 配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # ==========================================
    # 1. 解析唯一的命令行参数：配置文件路径
    # ==========================================
    parser = argparse.ArgumentParser(description="仿真实验统一训练入口")
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml', 
                        help="YAML 实验配置文件的路径")
    args = parser.parse_args()

    # 读取配置字典
    config = load_yaml_config(args.config)
    model_type = config.get('model_type')

    # ==========================================
    # 2. 创建本次实验的专属保存目录，并备份配置
    # ==========================================
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = config.get('experiment_name', 'default_exp')
    save_dir = f"./experiments/{exp_name}_{model_type}_{current_time}"
    os.makedirs(save_dir, exist_ok=True)
    
    # ⭐ 核心动作：把当前用的配置文件直接复制进实验目录，留作完美复现的档案
    shutil.copy(args.config, os.path.join(save_dir, 'run_config.yaml'))
    print(f"[*] 实验初始化完成！产物及配置档案将保存在: {save_dir}")

    # ==========================================
    # 3. 加载通用的数据 (从配置字典 data 层级获取参数)
    # ==========================================
    data_cfg = config['data']
    print(f"[*] 正在从 {data_cfg['file_path']} 加载数据...")
    
    data_dict = get_dataloaders(
        file_path=data_cfg['file_path'],
        mat_key=data_cfg['mat_key'],
        batch_size=data_cfg['batch_size'],
        test_size=data_cfg['test_size'],
        random_state=data_cfg['random_state'],
        save_scaler_dir=save_dir       # Scaler 依然保存在当前实验目录下
    )

    # ==========================================
    # 4. 根据配置文件中的 model_type，指派不同的“施工队”
    # ==========================================
    if model_type == 'rf':
        print("\n>>> 启动传统机器学习引擎 (RF) <<<")
        rf_cfg = config['rf_params']
        
        # 🌟 接收返回的 model 和 metrics，并使用 **rf_cfg 优雅解包传参
        model, metrics = train_random_forest(
            data_dict=data_dict, 
            save_dir=save_dir,
            **rf_cfg
        )
        
        # 为写日志准备参数字符串
        key_params = f"Trees:{rf_cfg.get('n_estimators')}|Depth:{rf_cfg.get('max_depth')}"
        architecture = "RandomForest"
        
    elif model_type == 'dl':
        print("\n>>> 启动深度学习引擎 (DL) <<<")
        dl_cfg = config['dl_params']
        
        # 🌟 接收返回的 model 和 metrics，直接将整个 dl_cfg 传给训练器
        model, metrics = train_deep_learning(
            data_dict=data_dict, 
            dl_config=dl_cfg,
            save_dir=save_dir
        )
        
        # 为写日志准备参数字符串
        key_params = f"LR:{dl_cfg.get('learning_rate')}|Epochs:{dl_cfg.get('epochs')}|Heads:{dl_cfg.get('num_heads', 'N/A')}"
        architecture = dl_cfg.get('architecture', 'N/A').upper()
        
    else:
        print(f"❌ 配置文件错误：不支持的 model_type '{model_type}'")
        return  # 如果模型类型错误，直接退出程序，不写日志

    # ==========================================
    # 🌟 5. 将本次实验结果写入全局宏观日志
    # ==========================================
    # 从顶层配置中提取实验名称和时间戳（假设你在文件上面已经定义了这些变量）
    global_log_path = "./experiments/all_results_log.csv"
    
    # 提取数据源路径，如果没有写则用 'N/A' 代替
    data_source = config.get('data', {}).get('file_path', 'N/A')
    
    # 构建一条扁平化的记录字典
    log_record = {
        '时间戳': current_time,
        '实验名称': exp_name,
        '流派': model_type.upper(),
        '网络架构': architecture,
        '核心超参数': key_params,
        '测试集_R2': round(metrics['R2_Score'], 4),
        '测试集_MSE(归一化)': round(metrics['MSE_Scaled'], 6),
        '测试集_MSE(真实物理)': round(metrics['MSE_Real'], 6),
        '数据源': data_source,
        '产物目录': save_dir
    }
    
    # 将字典喂给 logger (前提是你在文件顶部 import 了这个函数)
    append_to_global_log(global_log_path, log_record)
    print(f"\n[*] 宏观对比数据已成功追加至: {global_log_path}")

# ==========================================
# 程序执行入口
# ==========================================
"""
运行命令示例:
python main.py --config configs/experiment_config.yaml
"""
if __name__ == '__main__':
    main()