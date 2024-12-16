import json
import numpy as np
# 读取factors.json
def get_factors():
    with open('factors.json', 'r',encoding='utf-8') as f:
        factors = json.load(f)
    return factors

def get_factor_score(current_factors):
     # 计算 RSI 平均值
    # current_factors['rsi_avg'] = current_factors[['rsi_bfq_6', 'rsi_bfq_12', 'rsi_bfq_24']].mean(axis=1)

    # 计算评分，假设评分公式如下：
    # score = -pe - pb - ps + rsi_avg
    # (mtm_hfq * 0.1) + (macd_hfq * 0.2) + (rsi_hfq_12 * 0.0667) + (brar_ar_hfq * 0.0833) + (psy_hfq * 0.0333) + (trix_hfq * 0.0833) + (rsi_hfq_24 * 0.0833) + (brar_br_hfq * 0.0667) + (psyma_hfq * 0.05) + (obv_hfq * 0.05) + (rsi_hfq_6 * 0.0833) + (vr_hfq * 0.0667) + (dmi_adx_hfq * 0.0333)
    # 您可以根据实际的评分公式进行调整
    # current_factors['score'] = (
    #     -0.1*current_factors['pe'].apply(lambda x: x if x > 0 else 0) -
    #     0.2*current_factors['pb'].apply(lambda x: x if x > 0 else 0) -
    #     0.3*current_factors['ps'].apply(lambda x: x if x > 0 else 0) 
    # )
    df = current_factors
    formula_dict = get_factors()
    formula = formula_dict["pandas_formula"]
    current_factors['score'] = eval(formula)
    return current_factors
def get_factor_list():
    formula_dict = get_factors()
    factor_list = formula_dict["factor_list"]
    return factor_list
