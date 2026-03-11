from sklearn.ensemble import RandomForestRegressor


# 这个文件似乎用不到
def build_rf_model(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1):
    """
    构建并返回一个随机森林回归模型实例。
    将超参数作为函数参数暴露出来，方便在外层的 train.py 中统一管理和调参。
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    return model