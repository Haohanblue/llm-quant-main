# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any
import logging
from template import CoT_consistent_refine_factors_prompt_chinese, materials
from chatGPT import get_GPT_response
import json
import re
import os
import shutil
import time
# 导入回测逻辑模块
from run import BacktestManager, DataHandler

# 配置 FastAPI
app = FastAPI()

# 设置日志
logging.basicConfig(level=logging.INFO)

# 定义公式相关的模型
class FormulaDict(BaseModel):
    factor_list: List[str] = Field(
        default_factory=lambda: [
            "pe",
            "pb",
            "ps",
            "rsi_hfq_6",
            "vol",
            "float_share",
            "kdj_k_hfq",
            "kdj_d_hfq",
            "ema_hfq_5",
            "ema_hfq_20",
            "close_hfq",
            "boll_mid_hfq",
            "boll_upper_hfq",
            "boll_lower_hfq",
            "macd_hfq",
            "macd_dif_hfq"
        ],
        description="因子列表"
    )
    formula: str = Field(
        "(-0.2*pe/(pe+pb+ps) + 0.3*rsi_hfq_6 - 0.5*vol/float_share + "
        "0.4*(kdj_k_hfq-kdj_d_hfq)/(kdj_k_hfq+kdj_d_hfq) + "
        "0.6*(ema_hfq_5-ema_hfq_20)/(ema_hfq_5+ema_hfq_20) + "
        "(close_hfq-boll_mid_hfq)/(boll_upper_hfq-boll_lower_hfq) + "
        "0.5*macd_hfq/(macd_dif_hfq+1))/3",
        description="评分公式"
    )
    pandas_formula: str = Field(
        "((-0.2*df['pe']/(df['pe'] + df['pb'] + df['ps']) + "
        "0.3*df['rsi_hfq_6'] - 0.5*df['vol']/df['float_share'] + "
        "0.4*(df['kdj_k_hfq'] - df['kdj_d_hfq'])/(df['kdj_k_hfq'] + df['kdj_d_hfq']) + "
        "0.6*(df['ema_hfq_5'] - df['ema_hfq_20'])/(df['ema_hfq_5'] + df['ema_hfq_20']) + "
        "(df['close_hfq'] - df['boll_mid_hfq'])/(df['boll_upper_hfq'] - df['boll_lower_hfq']) + "
        "0.5*df['macd_hfq']/(df['macd_dif_hfq'] + 1))/3)",
        description="pandas 评分公式"
    )
    explanation: str = Field(
        "该公式结合多种属性的因子以提供一个多视角的选股工具，"
        "从估值、技术指标和市场情绪等方面评估股票的潜在收益能力。"
        "量化分析中处理了量纲一致性问题，使各因子可以互相比较，"
        "为超额Alpha策略提供支撑。",
        description="公式解释"
    )

# 定义回测参数的输入模型
class BacktestParams(BaseModel):
    start_date: str = Field('2020-01-01', description="回测开始日期")
    end_date: str = Field('2024-11-01', description="回测结束日期")
    rotation_days: int = Field(20, description="轮动天数")
    num_stocks: int = Field(3, description="每次轮动选股数量")
    initial_money: float = Field(1_000_000, description="初始资金")
    commission: float = Field(0.00025, description="交易佣金比例")
    tax: float = Field(0.0005, description="交易税比例")
    slippage: float = Field(0.001, description="滑点比例")
    stop_profit: float = Field(0.4, description="止盈百分比")
    stop_loss: float = Field(0.2, description="止损百分比")
    stock_pool_type: str = Field("中证500", description="股票池类型")
    formula_dict: FormulaDict = Field(default_factory=FormulaDict, description="公式相关参数")
    
class QuestionRequest(BaseModel):
    question: str

class ResponseData(BaseModel):
    answer: str
    extracted_json: dict 

# 启动策略回测
@app.post("/run")
async def run_strategy(params: BacktestParams):
    try:
        # 数据库连接URL
        DB_URL = 'mysql+pymysql://quantuser:Quantuser233.@bj-cynosdbmysql-grp-3upmvv08.sql.tencentcdb.com:27017/stock'
        
        # 股票池映射
        stock_pool_map = {
            "上证指数": '000001.SH',
            "深证成指": '399001.SZ',
            "上证50": '000016.SH',
            "中证500": '000905.SH',
            "中小100": '399005.SZ',
        }

        # 获取股票池
        data = DataHandler(db_url=DB_URL)
        stock_pool = data.get_index_code_list(stock_pool_map[params.stock_pool_type])
        
        logging.info(f"选定的股票池有以下代码：{stock_pool}")
        
        # 创建并运行回测管理器
        manager = BacktestManager(
            db_url=DB_URL,
            stock_pool=stock_pool,
            start_date=params.start_date,
            end_date=params.end_date,
            rotation_days=params.rotation_days,
            num_stocks=params.num_stocks,
            init_money=params.initial_money,
            commission=params.commission,
            tax=params.tax,
            slippage=params.slippage,
            stop_profit=params.stop_profit,
            stop_loss=params.stop_loss,
            formula_data=params.formula_dict.model_dump()
        )
        # 运行回测
        result = manager.run_backtest()
        
        # 计算并输出策略评估指标
        evaluation_metrics = {
            "code": 200,
            "message": "回测完成",
            "data": {
                "strategy_return": result.get("strategy_return"),
                "strategy_return_rate": result.get("strategy_return_rate"),
                "benchmark_return": result.get("benchmark_return"),
                "alpha": result.get("alpha"),
                "beta": result.get("beta"),
                "sharpe": result.get("sharpe"),
                "max_drawdown": result.get("max_drawdown"),
                "win_rate": result.get("win_rate"),
                "max_consecutive_gains": int(result.get("max_consecutive_gains", 0)),
                "max_consecutive_losses": int(result.get("max_consecutive_losses", 0)),
                "trade_log": result.get("trade_log"),
                "strategy_returns_list_rounded": result.get("strategy_returns_list_rounded"),
                "benchmark_returns_list_rounded": result.get("benchmark_returns_list_rounded")
            }
        }
        # 返回JSON格式的结果
        return evaluation_metrics
    
    except Exception as e:
        logging.error(f"Error running strategy: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/ask", response_model=ResponseData)
def ask_question(request: QuestionRequest):
    question = request.question
    prompt = CoT_consistent_refine_factors_prompt_chinese.format(question=question, materials=materials)
    print("=====输入的问题=====")
    print(prompt)
    try:
        # 获取模型的回答
        result = get_GPT_response(model="gpt-4o", message=prompt)
        print("=====输出的回答=====")
        print(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型请求失败: {e}")

    # 提取回答中的 JSON 字符串
    try:
        json_match = re.search(r'```json(.*?)```', result, re.S)
        if not json_match:
            raise ValueError("未找到 JSON 格式的内容")
        json_str = json_match.group(1).strip()
        json_dict = json.loads(json_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"解析 JSON 失败: {e}")
    return ResponseData(answer=result, extracted_json=json_dict)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
