# 文件路径: utils/logger.py

import os
import csv

def append_to_global_log(log_path, record_dict):
    """
    将单次实验的记录追加到全局 CSV 文件中。
    如果文件不存在，会自动创建并写入表头。
    
    参数:
        log_path (str): 全局日志文件的路径 (例如: ./experiments/all_results_log.csv)
        record_dict (dict): 包含本次实验各项指标与参数的字典
    """
    # 检查文件是否已经存在
    file_exists = os.path.isfile(log_path)
    
    # 🌟 修复核心：安全地提取文件夹路径，如果不为空才进行创建
    log_dir = os.path.dirname(log_path)
    if log_dir:  
        os.makedirs(log_dir, exist_ok=True)
    
    # 使用 utf-8-sig 编码，确保用 Windows Excel 打开时中文绝不乱码
    with open(log_path, mode='a', newline='', encoding='utf-8-sig') as f:
        # fieldnames 自动提取字典的键作为 CSV 的表头
        writer = csv.DictWriter(f, fieldnames=record_dict.keys())
        
        # 如果是第一次生成这个文件，先写一行表头
        if not file_exists:
            writer.writeheader()
            
        # 写入本次实验的数据
        writer.writerow(record_dict)