CoT_consistent_refine_factors_prompt = """
You are an stock market analyst, and you are also an expert in 股票因子公式挖掘.
    - Question:{question}
We have retrieved the following Materials for this question，你必须基于以下信息进行因子公式的挖掘，不要用其他的指标。 (Special Note: the materials are made of introduction):
    - Materials: {materials}

Solve the problem step by step as required by the intructions.

Instructions:
1. Given the above question, rephrase and  expand it to help you do better answering. Maintain all information in the original question.
    {{ Insert rephrased and expanded question.}}
2. Analyze the impact that each material may have on the quetion.if you don't understand some factors, you can seek the introduction part for help.
    {{ Insert your analysis and factors step by step. }}
3. Three to five preliminary formulas are made independently, 基于用户提示中挖掘相关的因子，进行公式组合，如果用户没有详细的信息，你可以自行挖掘。切记，你所有用到的指标，必须都来源于我给你的meterials里的指标，如果需要用到其他的数据，你必须基于这些指标来计算，不允许使用不存在的指标，包括后面的天数，并且要和最终公式写在一起，作为同一个公式里的内容。你可以对所给指标进行自由的组合，基础的运算（根号，平方，加减乘除）,聚合运算（最大、最小、平均等，中位），同时你可以根据用户信息或者自行分配各自的权重。
    {{ Insert your Three preliminary formulas and corresponding explanations.}}
4. Evaluate the quality of preliminary formulas in the above steps from the following aspect: the adequacy and logic of explanations. Rate the evaluate results from 1 (lowest) to 5 (highest).
    {{ Insert your evaluation thoughs. }}
5. Based on your evaluation score, if the preliminary formulas are not satisfactory, please update the preliminary formulas.
    {{ Insert your updated formulas. }}
6. Take the average of the newest formula to give the final formula. Output your final probability formula.
    {{ Insert your answer, your final formula must be in JSON format with three keys: "factor_list","formula","pandas_formula","explaination"。factor_list are the original factors you used in the metaries , formula is your final formula genrated from the orinal factors, pandas_formula is change the formula to pandas sytyle such as (
        -0.1*df['pe'].apply(lambda x: x if x > 0 else 0) -
        0.2*df['pb'].apply(lambda x: x if x > 0 else 0) -
        0.3*df['ps'].apply(lambda x: x if x > 0 else 0) 
    ) just ignore this content but pay attention to its style and make sure it is df .and explaination is the reason you choosed the final formula as the answer }}

"""

CoT_consistent_refine_factors_prompt_chinese = """
你是一个股市分析师，同时你也是股票因子公式挖掘的专家。
    - 问题：{question}
我们已经为这个问题收集了以下材料，你必须基于以下信息进行因子公式的挖掘，不要使用其他的指标。 
(特别说明：这些材料由介绍部分组成)：
    - 材料：{materials}
请按要求逐步解决问题。
指示：
1. 根据上述问题，重新表述并扩展它，以帮助你更好地回答。保持原始问题中的所有信息。
    {{ 插入重新表述和扩展后的问题。}}
2. 分析每个材料可能对问题的影响。如果你不理解某些因子，可以查阅介绍部分寻求帮助。
    {{ 插入你的分析和逐步拆解的因子。}}
3. 独立构建三到五个初步公式，基于用户提示中挖掘相关的因子，进行公式组合。
如果用户没有提供详细信息，你可以自行挖掘。切记，所有用到的指标，必须都来源于我给你的材料里的指标，
如果需要用到其他的数据，你必须基于这些指标来计算，不允许使用不存在的指标，包括未来的天数，
并且要与最终公式一起写入，作为同一个公式里的内容。你可以对所给指标进行自由组合，
你可以天马行空的想象，基于选择的指标，进行组合、基本运算（根号、平方、加减乘除），聚合运算（最大、最小、平均等，中位数），
最终是要转化为基于df的公式的，同时你可以根据用户信息或者自行分配各自的权重。
最重要的是，你要自己处理解决量纲的问题，请直接在公式中处理。
    {{ 插入你的三组初步公式和相应的解释。}}
4. 从以下几个方面评估上述初步公式的质量：解释的充分性和逻辑性。请给出你的评估结果，分数从1（最低）到5（最高）。
    {{ 插入你的评估思路。}}
5. 根据你的评估分数，如果初步公式不满意，请更新初步公式。
    {{ 插入你的更新后的公式。}}
6. 取最新公式的平均值，给出最终公式。输出你的最终概率公式，以JSON格式输出，记得带上```json```。
    {{ 插入你的答案，最终公式必须是JSON格式，包含三个键： "factor_list"、"formula"、"pandas_formula"、
    "explanation"。factor_list是你在材料中使用的原始因子，formula是你从原始因子生成的最终公式，
    pandas_formula是将公式转换为pandas样式，该公式必须是一个表达式，并且内嵌量纲的处理过程，以及所有过程，写到一个公式里。如：
        -0.1*df['pe'].apply(lambda x: x if x > 0 else 0) -
        0.2*df['pb'].apply(lambda x: x if x > 0 else 0) -
        0.3*df['ps'].apply(lambda x: x if x > 0 else 0)
    )。忽略这个内容，但注意它的样式，并确保它是df。explanation是你选择最终公式的理由。}}
"""




materials=[
    {
        "id": 1,
        "name": "ts_code",
        "type": "str",
        "description": "股票代码",
        "attribute": "基础",
        "introduction": "唯一标识一只股票的代码，便于市场数据的查找和分析。"
    },
    {
        "id": 2,
        "name": "trade_date",
        "type": "str",
        "description": "交易日期",
        "attribute": "基础",
        "introduction": "指明特定交易的日期，有助于时间序列分析和历史数据对比。"
    },
    {
        "id": 3,
        "name": "open_hfq",
        "type": "float",
        "description": "开盘价",
        "attribute": "基础",
        "introduction": "股票在交易日开始时的第一笔成交价格，是市场情绪的初步反映。"
    },
    {
        "id": 4,
        "name": "high_hfq",
        "type": "float",
        "description": "最高价",
        "attribute": "基础",
        "introduction": "股票在交易日内达到的最高成交价格，显示市场的强势程度。"
    },
    {
        "id": 5,
        "name": "low_hfq",
        "type": "float",
        "description": "最低价",
        "attribute": "基础",
        "introduction": "股票在交易日内达到的最低成交价格，反映市场的弱势状态。"
    },
    {
        "id": 6,
        "name": "close_hfq",
        "type": "float",
        "description": "收盘价",
        "attribute": "基础",
        "introduction": "股票在交易日结束时的最后一笔成交价格，是市场情绪的最终反映。"
    },
    {
        "id": 7,
        "name": "change",
        "type": "float",
        "description": "涨跌额",
        "attribute": "基础",
        "introduction": "表示股票价格的变化额，有助于快速判断股票表现的好坏。"
    },
    {
        "id": 8,
        "name": "vol",
        "type": "float",
        "description": "成交量 （手）",
        "attribute": "基础",
        "introduction": "股票在特定时间内的交易数量，反映市场活跃度和流动性。"
    },
    {
        "id": 9,
        "name": "amount",
        "type": "float",
        "description": "成交额 （千元）",
        "attribute": "基础",
        "introduction": "股票在特定时间内的总成交金额，帮助分析资金流动情况。"
    },
    {
        "id": 10,
        "name": "turnover_rate",
        "type": "float",
        "description": "换手率（%）",
        "attribute": "基础",
        "introduction": "显示股票在特定时间内的流动性，帮助判断市场参与度。"
    },
    {
        "id": 11,
        "name": "turnover_rate_f",
        "type": "float",
        "description": "换手率（自由流通股）",
        "attribute": "基础",
        "introduction": "针对自由流通股的换手率，反映真实流通股的交易情况。"
    },
    {
        "id": 12,
        "name": "volume_ratio",
        "type": "float",
        "description": "量比",
        "attribute": "基础",
        "introduction": "当前成交量与过去某一时期成交量的比值，帮助判断市场活跃度。"
    },
    {
        "id": 13,
        "name": "adj_factor",
        "type": "float",
        "description": "复权因子",
        "attribute": "基础",
        "introduction": "用于调整历史价格，以便反映股票分红、配股等事件的影响。"
    },
    {
        "id": 14,
        "name": "downdays",
        "type": "float",
        "description": "连跌天数",
        "attribute": "基础",
        "introduction": "表示股票连续下跌的天数，有助于判断下跌趋势的持续性。"
    },
    {
        "id": 15,
        "name": "updays",
        "type": "float",
        "description": "连涨天数",
        "attribute": "基础",
        "introduction": "表示股票连续上涨的天数，有助于判断上涨趋势的持续性。"
    },
    {
        "id": 16,
        "name": "pe",
        "type": "float",
        "description": "市盈率（总市值/净利润， 亏损的PE为空）",
        "attribute": "财务型",
        "introduction": "反映公司盈利能力的指标，较低的PE通常表示股票被低估。"
    },
    {
        "id": 17,
        "name": "pe_ttm",
        "type": "float",
        "description": "市盈率（TTM，亏损的PE为空）",
        "attribute": "财务型",
        "introduction": "基于过去12个月（TTM）的净利润计算的市盈率，提供更实时的盈利数据。"
    },
    {
        "id": 18,
        "name": "pb",
        "type": "float",
        "description": "市净率（总市值/净资产）",
        "attribute": "财务型",
        "introduction": "衡量公司市值与净资产的关系，PB小于1通常表示公司被低估。"
    },
    {
        "id": 19,
        "name": "ps",
        "type": "float",
        "description": "市销率",
        "attribute": "财务型",
        "introduction": "反映公司销售收入与市值的比例，适用于无利润的公司评估。"
    },
    {
        "id": 20,
        "name": "ps_ttm",
        "type": "float",
        "description": "市销率（TTM）",
        "attribute": "财务型",
        "introduction": "基于过去12个月销售收入计算的市销率，提供更动态的销售表现数据。"
    },
    {
        "id": 21,
        "name": "dv_ratio",
        "type": "float",
        "description": "股息率 （%）",
        "attribute": "财务型",
        "introduction": "表示公司每股派发的股息与当前股价的比例，衡量投资回报。"
    },
    {
        "id": 22,
        "name": "dv_ttm",
        "type": "float",
        "description": "股息率（TTM）（%）",
        "attribute": "财务型",
        "introduction": "基于过去12个月分红的股息率，提供更稳定的股息收益预期。"
    },
    {
        "id": 23,
        "name": "total_share",
        "type": "float",
        "description": "总股本 （万股）",
        "attribute": "财务型",
        "introduction": "公司发行的所有股票总数，包括流通股和限售股。"
    },
    {
        "id": 24,
        "name": "float_share",
        "type": "float",
        "description": "流通股本 （万股）",
        "attribute": "财务型",
        "introduction": "在市场上可自由交易的股票数量，影响股票流动性。"
    },
    {
        "id": 25,
        "name": "free_share",
        "type": "float",
        "description": "自由流通股本 （万）",
        "attribute": "财务型",
        "introduction": "不受限售条件影响的股票数量，更真实地反映市场可交易的股份。"
    },
    {
        "id": 26,
        "name": "total_mv",
        "type": "float",
        "description": "总市值 （万元）",
        "attribute": "财务型",
        "introduction": "公司当前股价乘以总股本，反映公司的整体市场价值。"
    },
    {
        "id": 27,
        "name": "circ_mv",
        "type": "float",
        "description": "流通市值（万元）",
        "attribute": "财务型",
        "introduction": "当前股价乘以流通股本，显示市场上可交易股票的总价值。"
    },
    {
        "id": 28,
        "name": "atr_hfq",
        "type": "float",
        "description": "真实波动N日平均值-CLOSE, HIGH, LOW, N=20",
        "attribute": "超买超卖型",
        "introduction": "真实波动率的N日平均值，考虑了收盘价、最高价和最低价，主要用于衡量市场的波动性和风险水平。"
    },
    {
        "id": 29,
        "name": "bias1_hfq",
        "type": "float",
        "description": "BIAS乖离率-CLOSE, L1=6, L2=12, L3=24",
        "attribute": "超买超卖型",
        "introduction": "BIAS乖离率，反映当前收盘价与其6日均线的偏离程度，用于识别超买或超卖状态。"
    },
    {
        "id": 30,
        "name": "bias2_hfq",
        "type": "float",
        "description": "BIAS乖离率-CLOSE, L1=6, L2=12, L3=24",
        "attribute": "超买超卖型",
        "introduction": "同样为BIAS乖离率，反映当前收盘价与其12日均线的偏离程度，通过不同的均线组合提供市场趋势的偏离信息，帮助分析买入或卖出时机。"
    },
    {
        "id": 31,
        "name": "bias3_hfq",
        "type": "float",
        "description": "BIAS乖离率-CLOSE, L1=6, L2=12, L3=24",
        "attribute": "超买超卖型",
        "introduction": "继续延伸BIAS乖离率，反映当前收盘价与其24日均线的偏离程度,通过更长的均线参数，提供更全面的市场情绪分析。"
    },
    {
        "id": 32,
        "name": "cci_hfq",
        "type": "float",
        "description": "顺势指标又叫CCI指标-CLOSE, HIGH, LOW, N=14",
        "attribute": "超买超卖型",
        "introduction": "顺势指标（CCI），用于判断市场的超买和超卖情况，基于价格的变化和趋势，通常适用于14日周期。"
    },
    {
        "id": 33,
        "name": "kdj_hfq",
        "type": "float",
        "description": "KDJ指标-CLOSE, HIGH, LOW, N=9, M1=3, M2=3",
        "attribute": "超买超卖型",
        "introduction": "KDJ指标，结合随机指标与动量，帮助投资者识别潜在的反转点和市场趋势，参数包括9日周期及平滑系数。"
    },
    {
        "id": 34,
        "name": "kdj_d_hfq",
        "type": "float",
        "description": "KDJ指标-CLOSE, HIGH, LOW, N=9, M1=3, M2=3",
        "attribute": "超买超卖型",
        "introduction": "KDJ指标中的D值，反映了市场的中期趋势，用于辅助判断买入或卖出的信号。"
    },
    {
        "id": 35,
        "name": "kdj_k_hfq",
        "type": "float",
        "description": "KDJ指标-CLOSE, HIGH, LOW, N=9, M1=3, M2=3",
        "attribute": "超买超卖型",
        "introduction": "KDJ指标中的K值，代表短期趋势变化，常与D值结合使用，以确认交易信号。"
    },
    {
        "id": 36,
        "name": "mfi_hfq",
        "type": "float",
        "description": "MFI指标是成交量的RSI指标-CLOSE, HIGH, LOW, VOL, N=14",
        "attribute": "超买超卖型",
        "introduction": "货币流量指数（MFI），结合价格和成交量的RSI指标，帮助分析资金流入流出情况，以判断市场强弱。"
    },
    {
        "id": 37,
        "name": "mtm_hfq",
        "type": "float",
        "description": "动量指标-CLOSE, N=12, M=6",
        "attribute": "超买超卖型",
        "introduction": "动量指标，基于收盘价的变化，衡量价格变动的速度，用于识别市场趋势的强度。"
    },
    {
        "id": 38,
        "name": "mtmma_hfq",
        "type": "float",
        "description": "动量指标-CLOSE, N=12, M=6",
        "attribute": "超买超卖型",
        "introduction": "动量指标的移动平均版本，通过平滑处理来提供更稳定的趋势分析。"
    },
    {
        "id": 39,
        "name": "roc_hfq",
        "type": "float",
        "description": "变动率指标-CLOSE, N=12, M=6",
        "attribute": "超买超卖型",
        "introduction": "变动率指标，衡量一定周期内价格变化的幅度，通常用于识别趋势的反转。"
    },
    {
        "id": 40,
        "name": "maroc_hfq",
        "type": "float",
        "description": "变动率指标-CLOSE, N=12, M=6",
        "attribute": "超买超卖型",
        "introduction": "变动率的移动平均版本，结合了变动率的特点，为交易决策提供额外的平滑信号。"
    },
    {
        "id": 41,
        "name": "rsi_hfq_12",
        "type": "float",
        "description": "RSI指标-CLOSE, N=12",
        "attribute": "超买超卖型",
        "introduction": "相对强弱指数（RSI），通过12日价格变化衡量超买超卖情况，帮助分析市场的动能。"
    },
    {
        "id": 42,
        "name": "rsi_hfq_24",
        "type": "float",
        "description": "RSI指标-CLOSE, N=24",
        "attribute": "超买超卖型",
        "introduction": "RSI的24日版本，提供更长周期的市场强弱分析，以补充短期信号。"
    },
    {
        "id": 43,
        "name": "rsi_hfq_6",
        "type": "float",
        "description": "RSI指标-CLOSE, N=6",
        "attribute": "超买超卖型",
        "introduction": "RSI的6日版本，适合快速反应的交易策略，以捕捉短期市场波动。"
    },
    {
        "id": 44,
        "name": "wr_hfq",
        "type": "float",
        "description": "W&R 威廉指标-CLOSE, HIGH, LOW, N=10, N1=6",
        "attribute": "超买超卖型",
        "introduction": "威廉指标（W&R），用于判断市场超买和超卖情况，通过比较当前价格与一定周期内的最高和最低价。"
    },
    {
        "id": 45,
        "name": "wr1_hfq",
        "type": "float",
        "description": "W&R 威廉指标-CLOSE, HIGH, LOW, N=10, N1=6",
        "attribute": "超买超卖型",
        "introduction": "廉指标的扩展版本，提供更多维度的市场强弱分析，以优化交易决策。"
    },
    {
        "id": 46,
        "name": "lowdays",
        "type": "float",
        "description": "LOWRANGE(LOW)表示当前最低价是近多少周期内最低价的最小值",
        "attribute": "超买超卖型",
        "introduction": "用于分析当前价格相对历史最低价的位置，判断是否超跌。"
    },
    {
        "id": 47,
        "name": "topdays",
        "type": "float",
        "description": "TOPRANGE(HIGH)表示当前最高价是近多少周期内最高价的最大值",
        "attribute": "超买超卖型",
        "introduction": "用于分析当前价格相对历史最高价的位置，判断是否超买。"
    },
    {
        "id": 48,
        "name": "bbi_hfq",
        "type": "float",
        "description": "BBI多空指标-CLOSE, M1=3, M2=6, M3=12, M4=21",
        "attribute": "均线型",
        "introduction": "BBI（Balance Bar Index）是一个综合性多空指标，通过对多个不同周期的移动平均进行平滑处理，帮助交易者判断市场的趋势和强度。"
    },
    {
        "id": 49,
        "name": "expma_12_hfq",
        "type": "float",
        "description": "EMA指数平均数指标-CLOSE, N1=12, N2=50",
        "attribute": "均线型",
        "introduction": "该指标使用指数加权法，给予近期价格更大的权重，反映出较短期的市场趋势，适合捕捉快速变化的市场信号。"
    },
    {
        "id": 50,
        "name": "expma_50_hfq",
        "type": "float",
        "description": "EMA指数平均数指标-CLOSE, N1=12, N2=50",
        "attribute": "均线型",
        "introduction": "该指标提供中期趋势的指示，平滑处理后能够有效过滤短期波动，适合用于趋势跟踪和支持/阻力分析。"
    },
    {
        "id": 51,
        "name": "ma_hfq_10",
        "type": "float",
        "description": "简单移动平均-N=10",
        "attribute": "均线型",
        "introduction": "该指标通过计算过去10个周期的平均值来确定价格趋势，常用于短期交易策略中，能够反映短期价格的波动。"
    },
    {
        "id": 52,
        "name": "ma_hfq_20",
        "type": "float",
        "description": "简单移动平均-N=20",
        "attribute": "均线型",
        "introduction": "此指标对过去20个周期的价格进行平均，有助于确认中期趋势，适合用于判断支撑和阻力位。"
    },
    {
        "id": 53,
        "name": "ma_hfq_250",
        "type": "float",
        "description": "简单移动平均-N=250",
        "attribute": "均线型",
        "introduction": "该指标提供长期趋势的视角，通过对250个周期的价格进行平滑处理，适合用于长期投资决策。"
    },
    {
        "id": 54,
        "name": "ma_hfq_30",
        "type": "float",
        "description": "简单移动平均-N=30",
        "attribute": "均线型",
        "introduction": "此指标通过对过去30个周期的价格进行平均，适合于中短期交易，能够有效识别价格波动的趋势。"
    },
    {
        "id": 55,
        "name": "ma_hfq_5",
        "type": "float",
        "description": "简单移动平均-N=5",
        "attribute": "均线型",
        "introduction": "该指标快速响应市场变化，通过对最近5个周期的价格进行平均，适合快速交易策略。"
    },
    {
        "id": 56,
        "name": "ma_hfq_60",
        "type": "float",
        "description": "简单移动平均-N=60",
        "attribute": "均线型",
        "introduction": "此指标提供较为平滑的趋势线，适合于中期投资，能够帮助交易者判断市场的整体方向。"
    },
    {
        "id": 57,
        "name": "ma_hfq_90",
        "type": "float",
        "description": "简单移动平均-N=90",
        "attribute": "均线型",
        "introduction": "该指标反映中长期的市场趋势，通过对过去90个周期的价格进行平均，适合长期投资决策。"
    },
    {
        "id": 58,
        "name": "ema_hfq_10",
        "type": "float",
        "description": "指数移动平均-N=10",
        "attribute": "均线型",
        "introduction": "该指标强调近期价格波动，通过对过去10个周期的价格赋予不同权重，适合短期交易者使用。"
    },
    {
        "id": 59,
        "name": "ema_hfq_20",
        "type": "float",
        "description": "指数移动平均-N=20",
        "attribute": "均线型",
        "introduction": "该指标适合中短期分析，通过对过去20个周期的价格进行指数加权，能够有效捕捉中期趋势变化。"
    },
    {
        "id": 60,
        "name": "ema_hfq_250",
        "type": "float",
        "description": "指数移动平均-N=250",
        "attribute": "均线型",
        "introduction": "该指标提供长期趋势视角，平滑处理后的数据帮助投资者判断市场的长期方向，适合长期投资策略。"
    },
    {
        "id": 61,
        "name": "ema_hfq_30",
        "type": "float",
        "description": "指数移动平均-N=30",
        "attribute": "均线型",
        "introduction": "此指标适合中期交易，通过对过去30个周期的价格进行加权处理，能够有效识别中期价格趋势。"
    },
    {
        "id": 62,
        "name": "ema_hfq_5",
        "type": "float",
        "description": "指数移动平均-N=5",
        "attribute": "均线型",
        "introduction": "该指标快速响应市场变化，适合短期交易者，能够迅速捕捉价格的短期波动。"
    },
    {
        "id": 63,
        "name": "ema_hfq_60",
        "type": "float",
        "description": "指数移动平均-N=60",
        "attribute": "均线型",
        "introduction": "此指标适合中期分析，通过对过去60个周期的价格进行加权处理，能够帮助识别中期趋势的变化。"
    },
    {
        "id": 64,
        "name": "ema_hfq_90",
        "type": "float",
        "description": "指数移动平均-N=90",
        "attribute": "均线型",
        "introduction": "该指标为投资者提供较为平滑的趋势线，适合于中长期投资，帮助交易者判断市场的整体方向。"
    },
    {
        "id": 65,
        "name": "boll_lower_hfq",
        "type": "float",
        "description": "BOLL指标，布林带-CLOSE, N=20, P=2",
        "attribute": "路径型",
        "introduction": "该指标代表布林带的下边界，用于识别价格的超卖状态，帮助交易者判断潜在的反弹机会。"
    },
    {
        "id": 66,
        "name": "boll_mid_hfq",
        "type": "float",
        "description": "BOLL指标，布林带-CLOSE, N=20, P=2",
        "attribute": "路径型",
        "introduction": "此指标为布林带的中间线，通常为20日简单移动平均，反映市场的中期趋势和均衡价格水平。"
    },
    {
        "id": 67,
        "name": "boll_upper_hfq",
        "type": "float",
        "description": "BOLL指标，布林带-CLOSE, N=20, P=2",
        "attribute": "路径型",
        "introduction": "该指标代表布林带的上边界，用于识别价格的超买状态，帮助交易者判断可能的回调风险。"
    },
    {
        "id": 68,
        "name": "xsii_td1_hfq",
        "type": "float",
        "description": "薛斯通道II-CLOSE, HIGH, LOW, N=102, M=7",
        "attribute": "路径型",
        "introduction": "该指标帮助识别价格趋势的强度和反转点，通过对多个价格数据的结合分析，为交易决策提供参考。"
    },
    {
        "id": 69,
        "name": "xsii_td2_hfq",
        "type": "float",
        "description": "薛斯通道II-CLOSE, HIGH, LOW, N=102, M=7",
        "attribute": "路径型",
        "introduction": "此指标提供额外的信号线，用于确认趋势的持续性和潜在的反转点，适合与其他指标结合使用。"
    },
    {
        "id": 70,
        "name": "xsii_td3_hfq",
        "type": "float",
        "description": "薛斯通道II-CLOSE, HIGH, LOW, N=102, M=7",
        "attribute": "路径型",
        "introduction": "该指标进一步细化趋势分析，帮助交易者识别市场动态，为短期交易提供决策依据。"
    },
    {
        "id": 71,
        "name": "xsii_td4_hfq",
        "type": "float",
        "description": "薛斯通道II-CLOSE, HIGH, LOW, N=102, M=7",
        "attribute": "路径型",
        "introduction": "此指标是对市场趋势的综合分析，帮助交易者洞察价格行为和波动，增强交易策略的有效性。"
    },
    {
        "id": 72,
        "name": "ktn_down_hfq",
        "type": "float",
        "description": "肯特纳交易通道, N选20日，ATR选10日-CLOSE, HIGH, LOW, N=20, M=10",
        "attribute": "路径型",
        "introduction": "显示价格波动范围，有助于判断支撑与阻力水平。"
    },
    {
        "id": 73,
        "name": "ktn_mid_hfq",
        "type": "float",
        "description": "肯特纳交易通道, N选20日，ATR选10日-CLOSE, HIGH, LOW, N=20, M=10",
        "attribute": "路径型",
        "introduction": "肯特纳通道的中线，反映市场的平均水平。"
    },
    {
        "id": 74,
        "name": "ktn_upper_hfq",
        "type": "float",
        "description": "肯特纳交易通道, N选20日，ATR选10日-CLOSE, HIGH, LOW, N=20, M=10",
        "attribute": "路径型",
        "introduction": "显示价格的上限，帮助识别超买区间。"
    },
    {
        "id": 75,
        "name": "taq_down_hfq",
        "type": "float",
        "description": "唐安奇通道(海龟)交易指标-HIGH, LOW, 20",
        "attribute": "路径型",
        "introduction": "用于判断价格下行趋势的指标，提供支撑位信息。"
    },
    {
        "id": 76,
        "name": "taq_mid_hfq",
        "type": "float",
        "description": "唐安奇通道(海龟)交易指标-HIGH, LOW, 20",
        "attribute": "路径型",
        "introduction": "反映价格走势的中间水平，提供参考线。"
    },
    {
        "id": 77,
        "name": "taq_up_hfq",
        "type": "float",
        "description": "唐安奇通道(海龟)交易指标-HIGH, LOW, 20",
        "attribute": "路径型",
        "introduction": "显示价格的上行趋势的上限，帮助识别超买区间。"
    },
    {
        "id": 78,
        "name": "brar_ar_hfq",
        "type": "float",
        "description": "BRAR情绪指标-OPEN, CLOSE, HIGH, LOW, M1=26",
        "attribute": "能量型",
        "introduction": "该指标用于衡量市场情绪，通过分析价格波动情况，帮助交易者判断买卖力量的变化。"
    },
    {
        "id": 79,
        "name": "brar_br_hfq",
        "type": "float",
        "description": "BRAR情绪指标-OPEN, CLOSE, HIGH, LOW, M1=26",
        "attribute": "能量型",
        "introduction": "此指标与AR指标配合使用，提供市场情绪的补充信息，帮助识别趋势强度和反转机会。"
    },
    {
        "id": 80,
        "name": "cr_hfq",
        "type": "float",
        "description": "CR价格动量指标-CLOSE, HIGH, LOW, N=20",
        "attribute": "能量型",
        "introduction": "该指标衡量价格动量，反映价格变化的速度和强度，适合用于捕捉市场趋势的转变。"
    },
    {
        "id": 81,
        "name": "mass_hfq",
        "type": "float",
        "description": "梅斯线-HIGH, LOW, N1=9, N2=25, M=6",
        "attribute": "能量型",
        "introduction": "该指标结合价格波动信息，帮助分析市场强度，通过提供趋势的确认信号，辅助交易决策。"
    },
    {
        "id": 82,
        "name": "ma_mass_hfq",
        "type": "float",
        "description": "梅斯线-HIGH, LOW, N1=9, N2=25, M=6",
        "attribute": "能量型",
        "introduction": "此指标是梅斯线的移动平均形式，更加平滑的数据显示了市场趋势，帮助交易者识别潜在的入场时机。"
    },
    {
        "id": 83,
        "name": "psy_hfq",
        "type": "float",
        "description": "投资者对股市涨跌产生心理波动的情绪指标-CLOSE, N=12, M=6",
        "attribute": "能量型",
        "introduction": "该指标反映投资者的心理情绪波动，帮助判断市场的过热或过冷状态，适合短期交易分析。"
    },
    {
        "id": 84,
        "name": "psyma_hfq",
        "type": "float",
        "description": "投资者对股市涨跌产生心理波动的情绪指标-CLOSE, N=12, M=6",
        "attribute": "能量型",
        "introduction": "此指标提供心理情绪的移动平均，帮助分析投资者情绪的持续性和趋势，为决策提供支持。"
    },
    {
        "id": 85,
        "name": "vr_hfq",
        "type": "float",
        "description": "VR容量比率-CLOSE, VOL, M1=26",
        "attribute": "能量型",
        "introduction": "该指标用于衡量市场的成交量与价格的关系，帮助交易者评估市场的强度和潜在的趋势变化。"
    },
    {
        "id": 86,
        "name": "obv_hfq",
        "type": "float",
        "description": "能量潮指标-CLOSE, VOL",
        "attribute": "能量型",
        "introduction": "能量潮指标通过将成交量与价格变动结合起来，帮助交易者判断市场趋势的强度，通常用于确认价格趋势或发现潜在的价格反转点。"
    },
    {
        "id": 87,
        "name": "dmi_adx_hfq",
        "type": "float",
        "description": "动向指标-CLOSE, HIGH, LOW, M1=14, M2=6",
        "attribute": "趋势型",
        "introduction": "该指标用于衡量趋势的强度，帮助交易者判断市场的趋势是否存在，适合趋势跟踪。"
    },
    {
        "id": 88,
        "name": "dmi_adxr_hfq",
        "type": "float",
        "description": "动向指标-CLOSE, HIGH, LOW, M1=14, M2=6",
        "attribute": "趋势型",
        "introduction": "此指标为ADX的平滑版本，提供更稳定的趋势强度信号，适合用于确认趋势的持续性。"
    },
    {
        "id": 89,
        "name": "dmi_mdi_hfq",
        "type": "float",
        "description": "动向指标-CLOSE, HIGH, LOW, M1=14, M2=6",
        "attribute": "趋势型",
        "introduction": "该指标提供负向动向（-DI）的计算，帮助识别市场的下行压力，适合趋势分析。"
    },
    {
        "id": 90,
        "name": "dmi_pdi_hfq",
        "type": "float",
        "description": "动向指标-CLOSE, HIGH, LOW, M1=14, M2=6",
        "attribute": "趋势型",
        "introduction": "此指标提供正向动向（+DI）的计算，帮助识别市场的上行机会，适合趋势判断。"
    },
    {
        "id": 91,
        "name": "dpo_hfq",
        "type": "float",
        "description": "区间震荡线-CLOSE, M1=20, M2=10, M3=6",
        "attribute": "趋势型",
        "introduction": "该指标帮助分析价格相对于其历史平均值的偏离程度，适合震荡市场的交易策略。"
    },
    {
        "id": 92,
        "name": "madpo_hfq",
        "type": "float",
        "description": "区间震荡线-CLOSE, M1=20, M2=10, M3=6",
        "attribute": "趋势型",
        "introduction": "此指标为DPO的移动平均版本，通过平滑处理提供更清晰的趋势信号，帮助交易者把握市场动态。"
    },
    {
        "id": 93,
        "name": "emv_hfq",
        "type": "float",
        "description": "简易波动指标-HIGH, LOW, VOL, N=14, M=9",
        "attribute": "趋势型",
        "introduction": "该指标衡量价格波动与成交量之间的关系，适合于评估市场活跃度和潜在趋势。"
    },
    {
        "id": 94,
        "name": "maemv_hfq",
        "type": "float",
        "description": "简易波动指标-HIGH, LOW, VOL, N=14, M=9",
        "attribute": "趋势型",
        "introduction": "此指标为EMV的移动平均形式，通过平滑处理显示市场波动的趋势，帮助交易者做出决策。"
    },
    {
        "id": 95,
        "name": "macd_hfq",
        "type": "float",
        "description": "MACD指标-CLOSE, SHORT=12, LONG=26, M=9",
        "attribute": "趋势型",
        "introduction": "该指标用于识别价格的动量变化，通过两条移动平均线的交叉信号帮助判断买卖时机。"
    },
    {
        "id": 96,
        "name": "macd_dea_hfq",
        "type": "float",
        "description": "MACD指标-CLOSE, SHORT=12, LONG=26, M=9",
        "attribute": "趋势型",
        "introduction": "此指标为MACD的信号线，通过平滑处理提供趋势确认，帮助交易者判断市场的潜在变化。"
    },
    {
        "id": 97,
        "name": "macd_dif_hfq",
        "type": "float",
        "description": "MACD指标-CLOSE, SHORT=12, LONG=26, M=9",
        "attribute": "趋势型",
        "introduction": "该指标表示短期和长期移动平均线的差值，反映市场的动量变化，适合于捕捉趋势转折点。"
    },
    {
        "id": 98,
        "name": "trix_hfq",
        "type": "float",
        "description": "三重指数平滑平均线-CLOSE, M1=12, M2=20",
        "attribute": "趋势型",
        "introduction": "该指标通过三次指数平滑处理消除价格噪音，帮助交易者识别中长期趋势的变化。"
    },
    {
        "id": 99,
        "name": "trma_hfq",
        "type": "float",
        "description": "三重指数平滑平均线-CLOSE, M1=12, M2=20",
        "attribute": "趋势型",
        "introduction": "此指标为TRIX的移动平均版本，提供更平滑的趋势信号，帮助交易者捕捉市场的长期动向。"
    },
    {
        "id": 100,
        "name": "dfma_dif_hfq",
        "type": "float",
        "description": "平行线差指标-CLOSE, N1=10, N2=50, M=10",
        "attribute": "趋势型",
        "introduction": "用于判断短期和长期趋势的差异，帮助交易决策。"
    },
    {
        "id": 101,
        "name": "dfma_difma_hfq",
        "type": "float",
        "description": "平行线差指标-CLOSE, N1=10, N2=50, M=10",
        "attribute": "趋势型",
        "introduction": "通过平行线差分析，帮助判断价格走势和买卖信号。"
    }
]