class CustomPandasData(bt.feeds.PandasData):
    """
    扩展Backtrader的PandasData，添加is_limit_up, is_limit_down, is_suspended字段。
    """
    lines = ('is_limit_up', 'is_limit_down', 'is_suspended',)
    params = (
        ('is_limit_up', -1),
        ('is_limit_down', -1),
        ('is_suspended', -1),
    )
