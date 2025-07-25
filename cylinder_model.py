import pandas as pd
from dateutil.relativedelta import relativedelta

def count_cylinders_by_type(df):
    '''
    统计每个企业的各类钢瓶数量。
    参数：
        df: 包含气瓶信息的 DataFrame，要求含有 'stationName' 和 'type' 字段
    返回：
        DataFrame：每个企业各类气瓶的数量
    '''
    # 确保时间字段格式正确
    df_type = df.groupby(['companyCode', 'status']).size().reset_index(name='count')
    return df_type

def check_overdue_next_check_date(df, current_time):
    """
    统计每个站点“在用”气瓶是否超过下次检验日期。
    参数：
        df: 包含气瓶信息的 DataFrame，要求含有 'stationName'、'status'、'nextCheckDate'、'id'
        current_time: 当前时间（datetime），可选，默认使用当前系统时间
    返回：
        DataFrame：每个站点超过下次检验日期的气瓶数量和钢印号列表
    """
    if current_time is None:
        current_time = pd.Timestamp.now()

    df_use = df[(df['status'] == 0) & df['nextCheckDate'].notna()].copy()     # 'status' == 0为在用气瓶
    df_use['nextCheckDate'] = pd.to_datetime(df_use['nextCheckDate'], errors='coerce')
    df_overdue = df_use[df_use['nextCheckDate'] < current_time + relativedelta(months=1)]

    return df_overdue.groupby('companyCode').agg(
        overdue_check_count=('id', 'count'),
        overdue_check_gasids=('id', lambda x: list(x))
    ).reset_index()


def check_overdue_scrap_date(df, current_time):
    """
    统计每个公司“在用”气瓶是否超过报废年月。
    参数：
        df: 包含气瓶信息的 DataFrame，要求含有 'stationName'、'status'、'scrapDate'、'id'
        current_time: 当前时间（datetime），可选，默认使用当前系统时间
    返回：
        DataFrame：每个站点报废气瓶数量和钢印号列表
    """
    if current_time is None:
        current_time = pd.Timestamp.now()

    df_use = df[(df['status'] == 0) & df['scrapDate'].notna()].copy()
    df_use['scrapDate'] = pd.to_datetime(df_use['scrapDate'], errors='coerce') + pd.offsets.MonthEnd(0)
    df_overdue = df_use[df_use['scrapDate'] < current_time + relativedelta(months=1)]

    return df_overdue.groupby('companyCode').agg(
        overdue_scrap_count=('id', 'count'),
        overdue_scrap_gasids=('id', lambda x: list(x))
    ).reset_index()
