import json
import numpy as np
def get_factor_score(current_factors, factors):
    df = current_factors
    formula_dict = factors
    formula = formula_dict["pandas_formula"]
    current_factors['score'] = eval(formula)
    return current_factors
def get_factor_list(factors):
    formula_dict = factors
    factor_list = formula_dict["factor_list"]
    return factor_list
