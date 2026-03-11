# 文件路径: models/networks.py

import torch
import torch.nn as nn

class MLP_v1(nn.Module):
    """
    V1 版本深度 MLP 网络
    """
    def __init__(self, input_dim, dropout_rate=0.1):
        super(MLP_v1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),       
            nn.BatchNorm1d(256),      
            nn.ReLU(),
            nn.Dropout(dropout_rate), 
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 1)          
        )

    def forward(self, x):
        return self.net(x)

class MLP_v2(nn.Module):
    """
    V1 版本深度 MLP 网络
    """
    def __init__(self, input_dim, dropout_rate=0.1):
        super(MLP_v2, self).__init__()
        self.net = nn.Sequential(
            # 【关键修改】：输入层维度改为 86，神经元增加到 512
            nn.Linear(input_dim, 512),       
            nn.BatchNorm1d(512),      
            nn.ReLU(),
            nn.Dropout(0.1),          
            
            # 第二层加宽到 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第三层
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # 输出层保持不变，依然预测 1 个厚度值
            nn.Linear(64, 1)          
        )

    def forward(self, x):
        return self.net(x)

class Attention_Net_v1(nn.Module):
    """
    纯数据驱动的全局自注意力网络 (不包含物理空间组合的先验知识)
    将所有的特征 (包含全局变量和所有传感器测量值) 均视为独立的序列节点。
    """
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, dropout_rate=0.2, **kwargs):
        super(Attention_Net_v1, self).__init__()
        
        d_model = hidden_dim
        
        # 1. 独立特征嵌入 (Feature Embedding)
        # 将每一个标量特征 (维度为1) 映射为 d_model 维度的向量
        self.feature_proj = nn.Linear(1, d_model)
        
        # 2. 多头自注意力层
        # 这里序列长度将会是 input_dim (例如 86)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
        # 3. 回归层
        # 展平后的维度 = 序列长度 (input_dim) * 嵌入维度 (d_model)
        in_features = input_dim * d_model
        
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 🌟 同样保留注意力权重属性，方便后续提取
        self.attention_weights = None

    def forward(self, x):
        # x 的原始形状: (Batch, input_dim) 
        # 例如 (Batch, 86)
        
        # 将二维张量扩展为三维，把每一个特征变成一个大小为 1 的特征向量
        # 形状变为: (Batch, input_dim, 1)
        x_seq = x.unsqueeze(-1)
        
        # 独立投射到 d_model 维度: (Batch, input_dim, d_model)
        seq = self.feature_proj(x_seq)
        
        # 注意力计算: 让所有 86 个特征彼此之间自由计算相关性
        # attn_weights 的形状将会是 (Batch, num_heads, input_dim, input_dim)
        attn_output, attn_weights = self.self_attn(
            seq, seq, seq, 
            need_weights=True, 
            average_attn_weights=False 
        )
        self.attention_weights = attn_weights
        
        # 残差连接 + 归一化
        seq = self.norm(seq + attn_output)
        
        # 展平: (Batch, input_dim * d_model)
        att_flat = seq.flatten(start_dim=1)
        
        # 直接输出回归预测
        out = self.regressor(att_flat)
        return out
    

class Attention_Net_v2(nn.Module):
    # 🌟 修改1：统一参数名，加入 dropout_rate，并用 **kwargs 吸收传入的 input_dim 等多余参数
    def __init__(self, hidden_dim=64, num_heads=4, dropout_rate=0.2, **kwargs):
        super(Attention_Net_v2, self).__init__()
        
        # 为了与你原代码一致，将 hidden_dim 赋给 d_model 变量
        d_model = hidden_dim 
        
        # 1. 传感器通道特征嵌入
        self.sensor_proj = nn.Linear(3, d_model)
        
        # 2. 显式使用 MultiheadAttention (方便提取权重)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        
        self.norm = nn.LayerNorm(d_model) # 归一化层 (残差连接使用)
        
        # 3. 全局特征网络
        self.global_proj = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        
        # 4. 回归层
        in_features = 28 * d_model + 16
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # 🌟 修改2：使用传入的 dropout_rate 替代硬编码的 0.2
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 🌟 用于在前向传播时存储注意力权重 🌟
        self.attention_weights = None

    def forward(self, x):
        # ⚠️ 注意：这里硬编码了切片，要求传入的 x 必须恰好是 86 维！
        x_global = x[:, 0:2] 
        
        caps = x[:, 2:30].unsqueeze(-1)    
        conds = x[:, 30:58].unsqueeze(-1)  
        angles = x[:, 58:86].unsqueeze(-1) 
        x_sensor = torch.cat([caps, conds, angles], dim=-1) 
        
        # (Batch, 28, d_model)
        seq = self.sensor_proj(x_sensor) 
        
        # 注意力机制: Query=seq, Key=seq, Value=seq
        attn_output, attn_weights = self.self_attn(
            seq, seq, seq, 
            need_weights=True, 
            average_attn_weights=False 
        )
        # 保存注意力权重用于可视化
        self.attention_weights = attn_weights
        
        # 残差连接 + 归一化
        seq = self.norm(seq + attn_output) 
        
        att_flat = seq.flatten(start_dim=1) 
        glob_feat = self.global_proj(x_global)
        combined = torch.cat([att_flat, glob_feat], dim=1) 
        
        out = self.regressor(combined)
        return out