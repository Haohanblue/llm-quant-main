import json
import numpy as np
def get_factor_score(current_factors, factors):
    df = current_factors.copy()  # 避免修改原始数据

    # 1. 获取公式和因子列
    formula_dict = factors
    formula = formula_dict["pandas_formula"]
    factor_list = formula_dict["factor_list"]  # 获取要标准化的列名列表
    
    # 2. 直接用 Pandas 进行 Z-Score 标准化
    for col in factor_list:
        # 填充缺失值为均值
        df[col] = df[col].fillna(df[col].mean())
        std = df[col].std()
        if std != 0:
            df[col] = (df[col] - df[col].mean()) / std
        else:
            print(f"列 {col} 的标准差为 0，跳过标准化。")
    # 3. 计算得分
    df['score'] = eval(formula)
    
    return df

def get_factor_list(factors):
    formula_dict = factors
    factor_list = formula_dict["factor_list"]
    return factor_list

